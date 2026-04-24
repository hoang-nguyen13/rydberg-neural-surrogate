"""
Evaluation script for trained models.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.stats
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

from parse_jld2 import TrajectoryRecord
from dataset import TrajectoryDataset, create_splits, load_dataset, collate_fn
from transformer_surrogate import RydbergSurrogate

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 9

PHASE_THRESHOLD_RHO = 0.05  # Active if rho_ss > 0.05


def dtw_distance(x, y):
    """Simple Dynamic Time Warping distance."""
    n, m = len(x), len(y)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return dtw[n, m]


def evaluate_model(model, dataloader, device):
    model.eval()
    all_pred, all_true, all_omega, all_n_atoms = [], [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            omega = batch["omega"].to(device)
            n_atoms = batch["n_atoms"].to(device)
            inv_sqrt_n = batch["inv_sqrt_n"].to(device)
            t = batch["t"].to(device)
            sz_mean = batch["sz_mean"].to(device)
            
            pred = model(omega, n_atoms, inv_sqrt_n, t)
            
            all_pred.append(pred.cpu().numpy())
            all_true.append(sz_mean.cpu().numpy())
            all_omega.append(omega.cpu().numpy())
            all_n_atoms.append(n_atoms.cpu().numpy())
    
    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    all_omega = np.concatenate(all_omega)
    all_n_atoms = np.concatenate(all_n_atoms)
    
    mse = np.mean((all_pred - all_true) ** 2)
    mae = np.mean(np.abs(all_pred - all_true))
    max_err = np.max(np.abs(all_pred - all_true))
    
    norms = np.linalg.norm(all_true, axis=1)
    rel_l2 = np.mean(np.linalg.norm(all_pred - all_true, axis=1) / (norms + 1e-12))
    
    pearsons = []
    for i in range(len(all_pred)):
        if np.std(all_pred[i]) > 1e-6 and np.std(all_true[i]) > 1e-6:
            r, _ = scipy.stats.pearsonr(all_pred[i], all_true[i])
            pearsons.append(r)
    pearson_mean = np.mean(pearsons) if pearsons else 0.0
    
    dtw_vals = []
    for i in range(min(50, len(all_pred))):
        dtw_vals.append(dtw_distance(all_pred[i], all_true[i]))
    dtw_mean = np.mean(dtw_vals) if dtw_vals else 0.0
    
    pred_rho = (all_pred[:, -50:] + 1.0) / 2.0
    true_rho = (all_true[:, -50:] + 1.0) / 2.0
    pred_rho_ss = pred_rho.mean(axis=1)
    true_rho_ss = true_rho.mean(axis=1)
    rho_ss_mae = np.mean(np.abs(pred_rho_ss - true_rho_ss))
    
    pred_phase = pred_rho_ss > PHASE_THRESHOLD_RHO
    true_phase = true_rho_ss > PHASE_THRESHOLD_RHO
    phase_acc = np.mean(pred_phase == true_phase)
    
    ic_error = np.mean(np.abs(all_pred[:, 0] - all_true[:, 0]))
    bounds_violations = np.mean((all_pred < -1.0 - 1e-6) | (all_pred > 1.0 + 1e-6))
    
    metrics = {
        "mse": mse, "mae": mae, "max_error": max_err,
        "rel_l2": rel_l2, "pearson": pearson_mean, "dtw": dtw_mean,
        "rho_ss_mae": rho_ss_mae, "phase_acc": phase_acc,
        "ic_error": ic_error, "bounds_violations": bounds_violations,
    }
    
    return metrics, all_pred, all_true, all_omega, all_n_atoms


def critical_point_analysis(omegas, rho_ss_vals, n_bootstrap=1000):
    if len(omegas) < 3:
        return None, None, None
    
    idx = np.argsort(omegas)
    omegas = omegas[idx]
    rho_ss_vals = rho_ss_vals[idx]
    
    drho = np.diff(rho_ss_vals)
    domega = np.diff(omegas)
    derivative = drho / (domega + 1e-12)
    deriv_omegas = (omegas[:-1] + omegas[1:]) / 2
    
    if len(derivative) == 0:
        return None, None, None
    
    omega_c = deriv_omegas[np.argmax(derivative)]
    
    omega_c_boot = []
    for _ in range(n_bootstrap):
        idx_boot = np.random.choice(len(omegas), size=len(omegas), replace=True)
        omegas_boot = np.sort(omegas[idx_boot])
        rho_boot = rho_ss_vals[idx_boot]
        drho_b = np.diff(rho_boot)
        domega_b = np.diff(omegas_boot)
        deriv_b = drho_b / (domega_b + 1e-12)
        if len(deriv_b) > 0:
            deriv_omegas_b = (omegas_boot[:-1] + omegas_boot[1:]) / 2
            omega_c_boot.append(deriv_omegas_b[np.argmax(deriv_b)])
    
    omega_c_boot = np.array(omega_c_boot)
    ci_low = np.percentile(omega_c_boot, 2.5)
    ci_high = np.percentile(omega_c_boot, 97.5)
    
    return omega_c, ci_low, ci_high


def plot_results(records, all_pred, all_true, all_omega, all_n_atoms, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Trajectory comparison for N=1225 (interpolation test)
    n1225_mask = all_n_atoms == 1225
    if n1225_mask.sum() > 0:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
        axes = axes.flatten()
        
        omegas_1225 = all_omega[n1225_mask]
        pred_1225 = all_pred[n1225_mask]
        true_1225 = all_true[n1225_mask]
        idx = np.argsort(omegas_1225)
        
        for ax, i in zip(axes, idx[::max(1, len(idx)//6)][:6]):
            t = records[0].t_save
            ax.plot(t, true_1225[i], lw=2, label="True", color="steelblue")
            ax.plot(t, pred_1225[i], lw=2, ls="--", label="Predicted", color="coral")
            ax.set_title(f"N=1225, Omega={omegas_1225[i]:.2f}")
            ax.set_xlabel("Time")
            ax.set_ylabel("sz_mean")
            ax.set_ylim(-1.05, 1.05)
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / "trajectory_comparison_n1225.png")
        plt.close()
        print(f"Saved: {output_dir / 'trajectory_comparison_n1225.png'}")
    
    # 2. Size extrapolation visualization
    test_sizes = sorted(set(int(n) for n in all_n_atoms if n > 1225))
    if len(test_sizes) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
        axes = axes.flatten()
        
        for ax, n in zip(axes, test_sizes[:6]):
            mask = all_n_atoms == n
            if mask.sum() == 0:
                ax.axis("off")
                continue
            
            omegas_n = all_omega[mask]
            pred_n = all_pred[mask]
            true_n = all_true[mask]
            idx = np.argsort(omegas_n)
            
            i = idx[len(idx)//2] if len(idx) > 0 else 0
            t = records[0].t_save
            ax.plot(t, true_n[i], lw=2, label="True", color="steelblue")
            ax.plot(t, pred_n[i], lw=2, ls="--", label="Predicted", color="coral")
            ax.set_title(f"N={int(n)}, Omega={omegas_n[i]:.2f}")
            ax.set_xlabel("Time")
            ax.set_ylabel("sz_mean")
            ax.set_ylim(-1.05, 1.05)
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / "trajectory_comparison_extrapolation.png")
        plt.close()
        print(f"Saved: {output_dir / 'trajectory_comparison_extrapolation.png'}")
    
    # 3. Phase diagram: predicted vs true
    sizes = sorted(set(int(n) for n in all_n_atoms))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, label, data in [(axes[0], "True", all_true), (axes[1], "Predicted", all_pred)]:
        for n in sizes:
            mask = all_n_atoms == n
            omegas = all_omega[mask]
            rho_ss = (data[mask, -50:].mean(axis=1) + 1) / 2
            idx = np.argsort(omegas)
            ax.plot(omegas[idx], rho_ss[idx], "o-", markersize=2, label=f"N={n}")
        ax.set_xlabel("Omega")
        ax.set_ylabel("rho_ss")
        ax.set_title(f"{label} Phase Diagram")
        ax.axhline(PHASE_THRESHOLD_RHO, color="gray", ls="--", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / "phase_diagram_predicted_vs_true.png")
    plt.close()
    print(f"Saved: {output_dir / 'phase_diagram_predicted_vs_true.png'}")
    
    # 4. Critical point estimation per system size
    fig, ax = plt.subplots(figsize=(10, 6))
    omega_c_values = []
    omega_c_ci_low = []
    omega_c_ci_high = []
    sizes_with_data = []
    
    for n in sizes:
        mask = all_n_atoms == n
        if mask.sum() < 3:
            continue
        
        omegas = all_omega[mask]
        rho_ss = (all_true[mask, -50:].mean(axis=1) + 1) / 2
        
        omega_c, ci_low, ci_high = critical_point_analysis(omegas, rho_ss)
        if omega_c is not None:
            sizes_with_data.append(n)
            omega_c_values.append(omega_c)
            omega_c_ci_low.append(ci_low)
            omega_c_ci_high.append(ci_high)
            
            ax.errorbar(n, omega_c, yerr=[[omega_c - ci_low], [ci_high - omega_c]],
                       fmt="o", capsize=5, color="steelblue")
    
    ax.set_xlabel("System size N")
    ax.set_ylabel("Omega_c")
    ax.set_title("Critical Point vs. System Size (from True Data)")
    plt.tight_layout()
    plt.savefig(output_dir / "critical_points.png")
    plt.close()
    print(f"Saved: {output_dir / 'critical_points.png'}")
    
    # 5. ROC curve for phase classification
    pred_rho = (all_pred[:, -50:] + 1.0) / 2.0
    true_rho = (all_true[:, -50:] + 1.0) / 2.0
    pred_rho_ss = pred_rho.mean(axis=1)
    true_rho_ss = true_rho.mean(axis=1)
    
    fpr, tpr, thresholds = roc_curve(true_rho_ss > PHASE_THRESHOLD_RHO, pred_rho_ss)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, color="steelblue", label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Phase Classification ROC")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png")
    plt.close()
    print(f"Saved: {output_dir / 'roc_curve.png'}")
    
    return {
        "sizes_with_data": sizes_with_data,
        "omega_c_values": omega_c_values,
        "omega_c_ci_low": omega_c_ci_low,
        "omega_c_ci_high": omega_c_ci_high,
        "roc_auc": roc_auc,
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    records = load_dataset(args.data_path)
    train_r, val_r, test_r = create_splits(records)
    
    # Load model architecture from checkpoint
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model_args = checkpoint.get("args", {})
    
    model = RydbergSurrogate(
        n_layer=model_args.get("n_layer", 4),
        n_head=model_args.get("n_head", 4),
        n_embd=model_args.get("n_embd", 96),
        dropout=model_args.get("dropout", 0.2),
        mlp_ratio=model_args.get("mlp_ratio", 2),
    ).to(device)
    
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded model from: {args.model_path}")
    print(f"Trained for {checkpoint.get('epoch', '?')} epochs")
    
    # Evaluate on all splits
    from torch.utils.data import DataLoader
    
    for split_name, split_records in [("Train", train_r), ("Val", val_r), ("Test", test_r)]:
        ds = TrajectoryDataset(split_records, augment=False)
        loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        metrics, all_pred, all_true, all_omega, all_n_atoms = evaluate_model(model, loader, device)
        
        print(f"\n{split_name} metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")
        
        if split_name == "Test":
            crit_results = plot_results(split_records, all_pred, all_true, all_omega, all_n_atoms, args.output_dir)
            print(f"\nCritical points (true data):")
            for n, oc, cl, ch in zip(
                crit_results["sizes_with_data"],
                crit_results["omega_c_values"],
                crit_results["omega_c_ci_low"],
                crit_results["omega_c_ci_high"],
            ):
                print(f"  N={n}: Omega_c = {oc:.3f} [{cl:.3f}, {ch:.3f}]")
            print(f"\nROC AUC: {crit_results['roc_auc']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="outputs/rydberg_dataset.pkl")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation")
    args = parser.parse_args()
    main(args)
