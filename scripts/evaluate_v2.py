"""
Evaluation script for trained RydbergSurrogate (v2 dataset).

Generates paper figures and tables:
  Fig. 1: Trajectory overlays (interpolation)
  Fig. 2: Size extrapolation (N=4900)
  Fig. 3: γ transfer (N=3600, γ=5,10,20)
  Fig. 4: Critical scaling (data collapse + log-log decay)
  Table 1: Metrics across splits and models
  Table 2: Effective exponents

Also saves predicted trajectories to outputs/predictions/ for reuse by
critical exponent extraction scripts.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

from parse_jld2_v2 import TrajectoryRecord
from dataset_v2 import TrajectoryDataset, create_splits, load_dataset, collate_fn
from transformer_surrogate import RydbergSurrogate


OUTPUT_DIR = Path("outputs/evaluation")
PRED_DIR = Path("outputs/predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

# Physics constants
OMEGA_C = 11.2
BETA = 0.586
DELTA = 0.4577


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    model = RydbergSurrogate(
        n_layer=args.get("n_layer", 4),
        n_head=args.get("n_head", 4),
        n_embd=args.get("n_embd", 96),
        dropout=args.get("dropout", 0.2),
        mlp_ratio=args.get("mlp_ratio", 2),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def run_inference(model, records, device):
    """Run model on a list of records and return predictions + metrics."""
    ds = TrajectoryDataset(records, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    all_pred, all_true, all_meta = [], [], []
    total_mse = total_mae = total_rho_ss_mae = total_ic_error = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            omega = batch["omega"].to(device)
            n_atoms = batch["n_atoms"].to(device)
            inv_sqrt_n = batch["inv_sqrt_n"].to(device)
            gamma = batch["gamma"].to(device)
            dimension = batch["dimension"].to(device)
            t = batch["t"].to(device)
            sz_mean = batch["sz_mean"].to(device)

            pred = model(omega, n_atoms, inv_sqrt_n, gamma, dimension, t)

            mse = F.mse_loss(pred, sz_mean)
            mae = F.l1_loss(pred, sz_mean)

            pred_rho = (pred[:, -50:] + 1.0) / 2.0
            true_rho = (sz_mean[:, -50:] + 1.0) / 2.0
            pred_rho_ss = pred_rho.mean(dim=1)
            true_rho_ss = true_rho.mean(dim=1)
            rho_ss_mae = F.l1_loss(pred_rho_ss, true_rho_ss)

            ic_error = F.l1_loss(pred[:, 0], sz_mean[:, 0])

            total_mse += mse.item()
            total_mae += mae.item()
            total_rho_ss_mae += rho_ss_mae.item()
            total_ic_error += ic_error.item()
            n_batches += 1

            # Store per-sample (need to unbatch)
            for i in range(pred.size(0)):
                # Find actual length (non-padded). Padded t values are 0; actual t > 0.
                t_i = batch["t"][i].numpy()
                valid_len = np.sum(t_i > 0)
                if valid_len == 0:
                    valid_len = len(t_i)

                all_pred.append(pred[i, :valid_len].cpu().numpy())
                all_true.append(sz_mean[i, :valid_len].cpu().numpy())
                all_meta.append({
                    "omega": batch["omega"][i].item(),
                    "n_atoms": int(batch["n_atoms"][i].item()),
                    "gamma": batch["gamma"][i].item(),
                    "dimension": int(batch["dimension"][i].item()),
                    "rho_ss_true": true_rho_ss[i].item(),
                    "rho_ss_pred": pred_rho_ss[i].item(),
                })

    metrics = {
        "mse": total_mse / n_batches,
        "mae": total_mae / n_batches,
        "rho_ss_mae": total_rho_ss_mae / n_batches,
        "ic_error": total_ic_error / n_batches,
    }
    return all_pred, all_true, all_meta, metrics


def fig_trajectory_overlays(records, preds, trues, metas, suffix=""):
    """Fig. 1: Trajectory overlays for representative cases."""
    # Select 6 representative cases: 2 train, 2 val, 2 test
    # Sub-critical, near-critical, super-critical
    selected = []
    for target_n, target_omega_range, label in [
        (1600, (10.0, 10.5), "train_sub"),
        (2500, (11.1, 11.3), "train_near"),
        (2500, (12.5, 13.0), "train_super"),
        (3600, (10.0, 10.5), "val_sub"),
        (3600, (11.1, 11.3), "val_near"),
        (4900, (12.5, 13.0), "test_super"),
    ]:
        candidates = [
            (i, m) for i, m in enumerate(metas)
            if m["n_atoms"] == target_n and target_omega_range[0] <= m["omega"] <= target_omega_range[1]
        ]
        if candidates:
            # Pick closest to middle of omega range
            idx = min(candidates, key=lambda x: abs(x[1]["omega"] - (target_omega_range[0] + target_omega_range[1]) / 2))[0]
            selected.append(idx)

    if len(selected) < 6:
        # Fallback: just pick first 6
        selected = list(range(min(6, len(metas))))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    titles = ["Train sub-crit", "Train near-crit", "Train super-crit",
              "Val sub-crit", "Val near-crit", "Test super-crit"]

    for ax, idx, title in zip(axes, selected, titles):
        t = records[idx].t_save[:len(preds[idx])]
        ax.plot(t, trues[idx], lw=2, label="TWA", color="steelblue")
        ax.plot(t, preds[idx], lw=2, ls="--", label="Surrogate", color="coral")
        ax.set_title(f"{title}\nN={metas[idx]['n_atoms']}, Ω={metas[idx]['omega']:.2f}", fontsize=10)
        ax.set_xlabel(r"$t\,\Gamma$")
        ax.set_ylabel(r"$\langle S_z \rangle$")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"fig1_trajectory_overlays{suffix}.png", dpi=300)
    plt.close()
    print(f"Saved Fig. 1: trajectory overlays")


def fig_size_extrapolation(records, preds, trues, metas, suffix=""):
    """Fig. 2: Size extrapolation for N=4900."""
    n4900_indices = [i for i, m in enumerate(metas) if m["n_atoms"] == 4900]
    if not n4900_indices:
        print("Warning: no N=4900 predictions found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Dynamics for 3 representative Ω values
    ax = axes[0]
    target_omegas = [10.45, 11.2, 12.55]
    colors = ["steelblue", "darkorange", "forestgreen"]
    for target_omega, color in zip(target_omegas, colors):
        closest_idx = min(n4900_indices, key=lambda i: abs(metas[i]["omega"] - target_omega))
        t = records[closest_idx].t_save[:len(preds[closest_idx])]
        ax.plot(t, trues[closest_idx], lw=2, color=color, label=f"TWA Ω={metas[closest_idx]['omega']:.2f}")
        ax.plot(t, preds[closest_idx], lw=2, ls="--", color=color)
    ax.set_xlabel(r"$t\,\Gamma$")
    ax.set_ylabel(r"$\langle S_z \rangle$")
    ax.set_title("N=4900 dynamics (unseen size)")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Phase diagram ρ_ss vs Ω
    ax = axes[1]
    omegas = [metas[i]["omega"] for i in n4900_indices]
    rho_ss_true = [metas[i]["rho_ss_true"] for i in n4900_indices]
    rho_ss_pred = [metas[i]["rho_ss_pred"] for i in n4900_indices]
    idx = np.argsort(omegas)
    ax.plot(np.array(omegas)[idx], np.array(rho_ss_true)[idx], "o", color="steelblue", label="TWA", markersize=6)
    ax.plot(np.array(omegas)[idx], np.array(rho_ss_pred)[idx], "--", color="coral", label="Surrogate", lw=2)
    ax.axvline(OMEGA_C, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel(r"$\Omega$")
    ax.set_ylabel(r"$\rho_{\text{ss}}$")
    ax.set_title("N=4900 phase diagram")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"fig2_size_extrapolation{suffix}.png", dpi=300)
    plt.close()
    print(f"Saved Fig. 2: size extrapolation")


def fig_gamma_transfer(records, preds, trues, metas, suffix=""):
    """Fig. 3: γ transfer for N=3600."""
    n3600_indices = [i for i, m in enumerate(metas) if m["n_atoms"] == 3600]
    if not n3600_indices:
        print("Warning: no N=3600 predictions found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Dynamics for γ=5, 10, 20 at a fixed Ω near critical
    ax = axes[0]
    target_gamma_log = [np.log10(5.0), np.log10(10.0), np.log10(20.0)]
    gamma_labels = ["5", "10", "20"]
    colors = ["steelblue", "darkorange", "forestgreen"]
    for target_g, label, color in zip(target_gamma_log, gamma_labels, colors):
        candidates = [i for i in n3600_indices if abs(metas[i]["gamma"] - target_g) < 1e-3]
        # Pick one near Ω_c
        candidates_near = [i for i in candidates if 10.5 <= metas[i]["omega"] <= 12.0]
        if candidates_near:
            idx = candidates_near[0]
            t = records[idx].t_save[:len(preds[idx])]
            ax.plot(t, trues[idx], lw=2, color=color, label=f"TWA γ={label}")
            ax.plot(t, preds[idx], lw=2, ls="--", color=color)
    ax.set_xlabel(r"$t\,\Gamma$")
    ax.set_ylabel(r"$\langle S_z \rangle$")
    ax.set_title("N=3600 dynamics (unseen γ)")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Phase diagrams for all γ
    ax = axes[1]
    for target_g, label, color in zip(
        [np.log10(0.1), np.log10(5.0), np.log10(10.0), np.log10(20.0)],
        ["0.1", "5", "10", "20"],
        ["black", "steelblue", "darkorange", "forestgreen"]
    ):
        candidates = [i for i in n3600_indices if abs(metas[i]["gamma"] - target_g) < 1e-3]
        if not candidates:
            continue
        omegas = [metas[i]["omega"] for i in candidates]
        rho_true = [metas[i]["rho_ss_true"] for i in candidates]
        rho_pred = [metas[i]["rho_ss_pred"] for i in candidates]
        idx = np.argsort(omegas)
        ax.plot(np.array(omegas)[idx], np.array(rho_true)[idx], "o", color=color, markersize=4, alpha=0.7)
        ax.plot(np.array(omegas)[idx], np.array(rho_pred)[idx], "-", color=color, label=f"γ={label}", lw=1.5)
    ax.axvline(OMEGA_C, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel(r"$\Omega$")
    ax.set_ylabel(r"$\rho_{\text{ss}}$")
    ax.set_title("N=3600 phase diagrams")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"fig3_gamma_transfer{suffix}.png", dpi=300)
    plt.close()
    print(f"Saved Fig. 3: gamma transfer")


def fig_critical_scaling(records, preds, trues, metas, suffix=""):
    """Fig. 4: Data collapse and log-log decay from predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Data collapse: t^δ ρ vs t|Ω−Ω_c|^(β/δ)
    ax = axes[0]
    n3600_indices = [i for i, m in enumerate(metas) if m["n_atoms"] == 3600 and m["gamma"] < 0]
    # gamma is log10, so log10(0.1) = -1.0
    n3600_indices = [i for i, m in enumerate(metas) if m["n_atoms"] == 3600 and abs(m["gamma"] - (-1.0)) < 1e-3]

    for i in n3600_indices:
        delta_omega = abs(metas[i]["omega"] - OMEGA_C)
        if delta_omega < 1e-6:
            continue
        t = records[i].t_save[:len(preds[i])]
        sz = preds[i]
        rho = np.abs((1 + sz) / 2)
        x = t * (delta_omega ** (BETA / DELTA))
        y = (t ** DELTA) * rho
        ax.loglog(x, y, alpha=0.5, lw=1)

    ax.set_xlabel(r"$t\,|\Omega - \Omega_c|^{\beta/\delta}$")
    ax.set_ylabel(r"$t^{\delta}\,\rho(t)$")
    ax.set_title("Data collapse (predicted)")
    ax.grid(True, which="both", ls="--", alpha=0.3)

    # (b) Log-log decay at Ω_c = 11.2
    ax = axes[1]
    critical_indices = [i for i, m in enumerate(metas) if abs(m["omega"] - OMEGA_C) < 1e-3]
    for i in critical_indices:
        t = records[i].t_save[:len(preds[i])]
        sz = preds[i]
        rho = np.abs((1 + sz) / 2)
        rho = np.maximum(rho, 1e-12)
        ax.loglog(t, rho, alpha=0.6, lw=1.5, label=f"N={metas[i]['n_atoms']}")

    # Reference line: t^(-δ)
    t_ref = np.logspace(0, 3, 100)
    rho_ref = 0.5 * t_ref ** (-DELTA)
    ax.loglog(t_ref, rho_ref, "k--", lw=1, alpha=0.5, label=r"$t^{-\delta}$ ref")

    ax.set_xlabel(r"$t\,\Gamma$")
    ax.set_ylabel(r"$\rho(t)$")
    ax.set_title(f"Log-log decay at Ω_c={OMEGA_C} (predicted)")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"fig4_critical_scaling{suffix}.png", dpi=300)
    plt.close()
    print(f"Saved Fig. 4: critical scaling")


def save_predictions(records, preds, trues, metas, name):
    """Save predicted trajectories for reuse by exponent extraction scripts."""
    out = []
    for rec, pred, true, meta in zip(records, preds, trues, metas):
        out.append({
            "omega": meta["omega"],
            "n_atoms": meta["n_atoms"],
            "gamma": 10 ** meta["gamma"],  # convert back from log10
            "dimension": meta["dimension"],
            "t_save": rec.t_save[:len(pred)],
            "sz_mean_true": true,
            "sz_mean_pred": pred,
            "rho_ss_true": meta["rho_ss_true"],
            "rho_ss_pred": meta["rho_ss_pred"],
        })
    path = PRED_DIR / f"predictions_{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved predictions to {path}")


def print_table1(all_metrics):
    """Print Table 1: metrics across splits."""
    print("\n" + "=" * 70)
    print("TABLE 1: Metrics")
    print("=" * 70)
    print(f"{'Split':<25} {'MSE':>10} {'MAE':>10} {'ρ_ss MAE':>10} {'IC Error':>10}")
    print("-" * 70)
    for name, metrics in all_metrics.items():
        print(f"{name:<25} {metrics['mse']:>10.6f} {metrics['mae']:>10.6f} {metrics['rho_ss_mae']:>10.6f} {metrics['ic_error']:>10.6f}")
    print("=" * 70)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model, checkpoint = load_model(args.model_path, device)
    print(f"Loaded model from {args.model_path}")
    print(f"Trained for {checkpoint.get('epoch', '?')} epochs")

    records = load_dataset(args.data_path)
    train_r, val_r, test_sets = create_splits(records)

    all_metrics = {}
    all_preds, all_trues, all_metas = {}, {}, {}

    # Evaluate on all splits
    for name, split_records in [("Train", train_r), ("Val", val_r)] + list(test_sets.items()):
        print(f"\nEvaluating {name}...")
        preds, trues, metas, metrics = run_inference(model, split_records, device)
        all_metrics[name] = metrics
        all_preds[name] = preds
        all_trues[name] = trues
        all_metas[name] = metas
        save_predictions(split_records, preds, trues, metas, name)

    print_table1(all_metrics)

    # Generate figures
    print("\nGenerating figures...")

    # Fig 1: Trajectory overlays (use val for near-critical + test for extrapolation)
    combined_records = val_r + test_sets.get("size_extrapolation_2d", []) + test_sets.get("gamma_generalization_2d", [])
    combined_preds = all_preds["Val"] + all_preds.get("size_extrapolation_2d", []) + all_preds.get("gamma_generalization_2d", [])
    combined_trues = all_trues["Val"] + all_trues.get("size_extrapolation_2d", []) + all_trues.get("gamma_generalization_2d", [])
    combined_metas = all_metas["Val"] + all_metas.get("size_extrapolation_2d", []) + all_metas.get("gamma_generalization_2d", [])
    fig_trajectory_overlays(combined_records, combined_preds, combined_trues, combined_metas)

    # Fig 2: Size extrapolation
    if "size_extrapolation_2d" in test_sets:
        fig_size_extrapolation(test_sets["size_extrapolation_2d"],
                                all_preds["size_extrapolation_2d"],
                                all_trues["size_extrapolation_2d"],
                                all_metas["size_extrapolation_2d"])

    # Fig 3: Gamma transfer
    if "gamma_generalization_2d" in test_sets:
        fig_gamma_transfer(test_sets["gamma_generalization_2d"],
                           all_preds["gamma_generalization_2d"],
                           all_trues["gamma_generalization_2d"],
                           all_metas["gamma_generalization_2d"])

    # Fig 4: Critical scaling (use all available predictions at critical point)
    all_records_flat = []
    all_preds_flat = []
    all_trues_flat = []
    all_metas_flat = []
    split_records_map = {"Train": train_r, "Val": val_r, **test_sets}
    for name in all_preds:
        all_records_flat.extend(split_records_map[name])
        all_preds_flat.extend(all_preds[name])
        all_trues_flat.extend(all_trues[name])
        all_metas_flat.extend(all_metas[name])
    fig_critical_scaling(all_records_flat, all_preds_flat, all_trues_flat, all_metas_flat)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print(f"Predictions saved to {PRED_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data_path", type=str, default="outputs/rydberg_dataset_v2.pkl")
    args = parser.parse_args()
    main(args)
