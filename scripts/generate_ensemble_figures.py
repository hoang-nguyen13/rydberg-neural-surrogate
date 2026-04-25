"""
Generate ensemble-averaged comparison figures.
Loads 5 trained models, averages predictions, plots mean + uncertainty.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset_v2 import TrajectoryDataset, create_splits, load_dataset, collate_fn
from transformer_surrogate import RydbergSurrogate
from parse_jld2_v2 import TrajectoryRecord


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("outputs/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

C_TRUE = '#d62728'
C_TRANS = '#1f77b4'
C_ENS = '#2ca02c'


def load_model(path):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
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
    return model


def predict(model, records):
    ds = TrajectoryDataset(records, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    all_pred, all_true, all_meta = [], [], []
    with torch.no_grad():
        for batch in loader:
            omega = batch["omega"].to(device)
            n_atoms = batch["n_atoms"].to(device)
            inv_sqrt_n = batch["inv_sqrt_n"].to(device)
            gamma = batch["gamma"].to(device)
            dimension = batch["dimension"].to(device)
            t = batch["t"].to(device)
            sz_true = batch["sz_mean"].to(device)
            pred = model(omega, n_atoms, inv_sqrt_n, gamma, dimension, t)
            for i in range(len(omega)):
                all_pred.append(pred[i].cpu().numpy())
                all_true.append(sz_true[i].cpu().numpy())
                all_meta.append({
                    "omega": omega[i].item(),
                    "n_atoms": n_atoms[i].item(),
                    "gamma": gamma[i].item(),
                })
    return all_pred, all_true, all_meta


# Load dataset
records = load_dataset("data/rydberg_dataset_v2.pkl")
train_r, val_r, test_sets = create_splits(records)
test_records = test_sets["size_extrapolation_2d"]

# Load 5 ensemble models
ensemble_paths = [f"outputs/ensemble/model_seed{seed}/best_model.pt" for seed in [42, 43, 44, 45, 46]]
ensemble_models = []
for p in ensemble_paths:
    try:
        ensemble_models.append(load_model(p))
        print(f"Loaded {p}")
    except Exception as e:
        print(f"Failed {p}: {e}")

print(f"\nLoaded {len(ensemble_models)} models")

# Run predictions
all_preds = []
for i, model in enumerate(ensemble_models):
    print(f"Model {i+1}/{len(ensemble_models)}...")
    preds, trues, metas = predict(model, test_records)
    all_preds.append(preds)

# Stack predictions: (n_models, n_records, n_time)
pred_stack = np.stack(all_preds)
ens_mean = pred_stack.mean(axis=0)
ens_std = pred_stack.std(axis=0)

# === FIGURE 1: Trajectory overlays with ensemble bands ===
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

n = len(test_records)
selected = [int(i * (n - 1) / 5) for i in range(6)]

for idx, rec_idx in enumerate(selected):
    ax = axes[idx]
    t = np.linspace(0, 1000, 400)
    true = trues[rec_idx]
    mean = ens_mean[rec_idx]
    std = ens_std[rec_idx]
    meta = metas[rec_idx]

    ax.plot(t, true, "-", color=C_TRUE, lw=2, label="TWA", alpha=0.8)
    ax.plot(t, mean, "--", color=C_TRANS, lw=1.5, label="Ensemble mean")
    ax.fill_between(t, mean - std, mean + std, alpha=0.2, color=C_TRANS, label="±1σ")
    ax.set_title(f"Ω={meta['omega']:.2f}, N={int(meta['n_atoms'])}")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\langle S_z \rangle$")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(loc="upper right", fontsize=8)

plt.suptitle("Size Extrapolation (N=4900) — Ensemble", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ensemble_trajectories_size_extrapolation_2d.png", dpi=150)
plt.close()
print(f"Saved ensemble_trajectories_size_extrapolation_2d.png")

# === FIGURE 2: Phase diagram with error bars ===
fig, ax = plt.subplots(figsize=(10, 6))

omegas = [m["omega"] for m in metas]
rho_ss_true = [(trues[i][-50:].mean() + 1) / 2 for i in range(len(trues))]
rho_ss_mean = [(ens_mean[i][-50:].mean() + 1) / 2 for i in range(len(ens_mean))]
rho_ss_std = [(ens_std[i][-50:].mean()) / 2 for i in range(len(ens_std))]

idx = np.argsort(omegas)
ax.plot(np.array(omegas)[idx], np.array(rho_ss_true)[idx], "o", color=C_TRUE, markersize=8, label="TWA", zorder=5)
ax.errorbar(np.array(omegas)[idx], np.array(rho_ss_mean)[idx], yerr=np.array(rho_ss_std)[idx],
            fmt="s", color=C_TRANS, markersize=6, capsize=4, label="Ensemble mean ±σ", zorder=4)
ax.axhline(0.05, color="gray", ls="--", alpha=0.5, label=r"$\rho_{ss}=0.05$")
ax.set_xlabel(r"$\Omega$")
ax.set_ylabel(r"$\rho_{\text{ss}}$")
ax.set_title("N=4900 Phase Diagram — Ensemble")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "ensemble_phase_size_extrapolation_2d.png", dpi=150)
plt.close()
print(f"Saved ensemble_phase_size_extrapolation_2d.png")

# Print metrics
ens_mse = np.mean([(ens_mean[i] - trues[i])**2 for i in range(len(trues))])
ens_mae = np.mean([np.abs(ens_mean[i] - trues[i]) for i in range(len(trues))])
ens_rho_ss_mae = np.mean(np.abs(np.array(rho_ss_mean) - np.array(rho_ss_true)))
print(f"\nEnsemble metrics (N=4900):")
print(f"  MSE: {ens_mse:.6f}")
print(f"  MAE: {ens_mae:.6f}")
print(f"  ρ_ss MAE: {ens_rho_ss_mae:.6f}")

# Compare with single model
single_path = "outputs/models/best_model.pt"
single_model = load_model(single_path)
single_preds, _, _ = predict(single_model, test_records)
single_mse = np.mean([(single_preds[i] - trues[i])**2 for i in range(len(trues))])
single_rho = [((single_preds[i][-50:].mean() + 1) / 2) for i in range(len(trues))]
single_rho_mae = np.mean(np.abs(np.array(single_rho) - np.array(rho_ss_true)))
print(f"\nSingle model metrics (N=4900):")
print(f"  MSE: {single_mse:.6f}")
print(f"  ρ_ss MAE: {single_rho_mae:.6f}")
print(f"\nImprovement: MSE {single_mse/ens_mse:.2f}x, ρ_ss MAE {single_rho_mae/ens_rho_ss_mae:.2f}x")
