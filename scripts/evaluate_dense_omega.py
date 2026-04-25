"""
Evaluate the trained model on a dense omega grid for N=4900
and plot a smooth phase transition curve.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from transformer_surrogate import RydbergSurrogate
from dataset_v2 import load_dataset, create_splits
from parse_jld2_v2 import TrajectoryRecord


def load_best_model(path, device):
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


def predict_dense(model, device, n_atoms=4900, gamma=0.1, dimension=2,
                  omega_min=10.0, omega_max=13.0, n_omega=61, t_max=1000, n_time=400):
    """Predict trajectories on a dense omega grid."""
    omegas = np.linspace(omega_min, omega_max, n_omega)
    t = torch.linspace(0, t_max, n_time)
    
    omega_t = torch.tensor(omegas, dtype=torch.float32)
    n_t = torch.full((n_omega,), n_atoms, dtype=torch.float32)
    inv_sqrt_n_t = 1.0 / torch.sqrt(n_t)
    gamma_t = torch.full((n_omega,), np.log10(gamma), dtype=torch.float32)
    dim_t = torch.full((n_omega,), float(dimension), dtype=torch.float32)
    t_batch = t.unsqueeze(0).expand(n_omega, -1)
    
    with torch.no_grad():
        sz_pred = model(omega_t.to(device), n_t.to(device), inv_sqrt_n_t.to(device),
                        gamma_t.to(device), dim_t.to(device), t_batch.to(device))
    
    sz_pred = sz_pred.cpu().numpy()
    rho_pred = (sz_pred + 1.0) / 2.0
    rho_ss_pred = rho_pred[:, -50:].mean(axis=1)
    
    return omegas, sz_pred, rho_pred, rho_ss_pred


def plot_dense_phase_diagram(omegas, rho_ss_pred, test_records, output_dir):
    """Plot dense omega predictions overlaid with true test points."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract true test points
    true_omegas = []
    true_rho_ss = []
    for rec in test_records:
        true_omegas.append(rec.omega)
        true_rho_ss.append(rec.rho_ss)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # (a) Phase diagram ρ_ss vs Ω
    ax = axes[0]
    ax.plot(omegas, rho_ss_pred, "-", color="coral", lw=2, label="Surrogate (dense)")
    ax.scatter(true_omegas, true_rho_ss, color="steelblue", s=50, zorder=5, label="TWA")
    ax.axhline(0.05, color="gray", ls="--", alpha=0.5, label=r"$\rho_{ss}=0.05$")
    ax.set_xlabel(r"$\Omega$")
    ax.set_ylabel(r"$\rho_{\text{ss}}$")
    ax.set_title(f"N=4900 Phase Diagram (dense interpolation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) Full trajectories for a few omegas
    ax = axes[1]
    selected_idx = [0, 15, 30, 45, 60]
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_idx)))
    t = np.linspace(0, 1000, 400)
    for idx, color in zip(selected_idx, colors):
        ax.plot(t, rho_pred[idx], color=color, lw=1.5, label=f"Ω={omegas[idx]:.2f}")
    ax.set_xlabel(r"$t\,\Gamma$")
    ax.set_ylabel(r"$\rho(t)$")
    ax.set_title("Predicted dynamics (N=4900, dense Ω)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "dense_omega_phase_diagram.png", dpi=300)
    plt.close()
    print(f"Saved {output_dir / 'dense_omega_phase_diagram.png'}")
    
    # Also save a zoomed version near critical point
    fig, ax = plt.subplots(figsize=(8, 5))
    mask = (omegas >= 10.8) & (omegas <= 11.8)
    ax.plot(omegas[mask], rho_ss_pred[mask], "-", color="coral", lw=2, label="Surrogate (dense)")
    ax.scatter(np.array(true_omegas)[(np.array(true_omegas) >= 10.8) & (np.array(true_omegas) <= 11.8)],
               np.array(true_rho_ss)[(np.array(true_omegas) >= 10.8) & (np.array(true_omegas) <= 11.8)],
               color="steelblue", s=60, zorder=5, label="TWA")
    ax.axhline(0.05, color="gray", ls="--", alpha=0.5)
    ax.axvline(11.2, color="gray", ls="--", alpha=0.5, label=r"$\Omega_c=11.2$")
    ax.set_xlabel(r"$\Omega$")
    ax.set_ylabel(r"$\rho_{\text{ss}}$")
    ax.set_title("Critical Region (N=4900)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "dense_omega_critical_zoom.png", dpi=300)
    plt.close()
    print(f"Saved {output_dir / 'dense_omega_critical_zoom.png'}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = load_best_model("outputs/models/best_model.pt", device)
    print("Loaded model")
    
    records = load_dataset("data/rydberg_dataset_v2.pkl")
    _, _, test_sets = create_splits(records)
    test_records = test_sets.get("size_extrapolation_2d", [])
    
    print("Predicting on dense omega grid...")
    omegas, sz_pred, rho_pred, rho_ss_pred = predict_dense(model, device)
    
    print("Plotting...")
    plot_dense_phase_diagram(omegas, rho_ss_pred, test_records, "outputs/evaluation")
    
    print("Done!")
