"""
Log-log dynamics plot for selected Ω near the critical point.

Reproduces the thesis figure showing algebraic decay corresponding to δ.
Curves below Ω_c decay to zero (algebraic), curves above remain in active state.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np
import matplotlib.pyplot as plt
from data.dataset_v2 import TrajectoryRecord

OUTPUT_DIR = Path("outputs/critical_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OMEGA_C = 11.2
def load_data():
    with open("data/rydberg_dataset_v2.pkl", "rb") as f:
        data = pickle.load(f)
    return data


def main():
    data = load_data()

    # Collect 2D trajectories, gamma=0.1, selected omegas near critical point
    records = []
    for r in data:
        if r.dimension != 2 or r.gamma != 0.1:
            continue
        records.append(r)

    if not records:
        print("No records found!")
        return

    # Group by N
    from collections import defaultdict
    by_n = defaultdict(list)
    for r in records:
        by_n[r.n_atoms].append(r)

    # Plot for each N
    for n_atoms, n_records in sorted(by_n.items()):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Sort by omega
        n_records.sort(key=lambda r: r.omega)

        # Color gradient: dark blue (below critical) -> white (critical) -> dark red (above)
        omegas = np.array([r.omega for r in n_records])
        min_omega, max_omega = omegas.min(), omegas.max()

        for r in n_records:
            # Filter time <= 2000 (matches thesis)
            valid = r.t_save <= 2000
            t = r.t_save[valid]
            sz = r.sz_mean[valid]
            rho = np.abs((1 + sz) / 2)

            # Avoid log(0)
            rho = np.maximum(rho, 1e-12)

            # Color based on distance from critical point
            dist = (r.omega - OMEGA_C) / (max_omega - min_omega + 1e-9)
            if r.omega < OMEGA_C:
                color = plt.cm.Blues(0.3 + 0.7 * min(1.0, abs(dist) / (abs(min_omega - OMEGA_C) / (max_omega - min_omega + 1e-9) + 1e-9)))
            else:
                color = plt.cm.Reds(0.3 + 0.7 * min(1.0, abs(dist) / (abs(max_omega - OMEGA_C) / (max_omega - min_omega + 1e-9) + 1e-9)))

            label = f"Ω={r.omega:.2f}"
            ax.loglog(t, rho, color=color, alpha=0.8, linewidth=1.5, label=label)

        ax.axvline(x=OMEGA_C, color='black', linestyle='--', linewidth=1, alpha=0.5, label=f"Ω_c={OMEGA_C}")
        ax.set_xlabel(r"$t\,\Gamma$", fontsize=12)
        ax.set_ylabel(r"$\rho(t)$", fontsize=12)
        ax.set_title(f"Log-log dynamics — 2D, N={n_atoms}, γ=0.1", fontsize=13)
        ax.grid(True, which="both", ls="--", alpha=0.3)

        # Limit legend to a few representative curves
        handles, labels = ax.get_legend_handles_labels()
        # Select a subset
        step = max(1, len(handles) // 8)
        ax.legend(handles[::step], labels[::step], loc="lower left", fontsize=8)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"loglog_dynamics_N{n_atoms}.png", dpi=300)
        plt.close()
        print(f"Saved log-log dynamics for N={n_atoms}")

    # Also create a multi-panel figure for N=3600 (thesis figure equivalent)
    n3600_records = by_n.get(3600, [])
    if n3600_records:
        fig, ax = plt.subplots(figsize=(8, 6))
        n3600_records.sort(key=lambda r: r.omega)

        for r in n3600_records:
            valid = r.t_save <= 2000
            t = r.t_save[valid]
            sz = r.sz_mean[valid]
            rho = np.abs((1 + sz) / 2)
            rho = np.maximum(rho, 1e-12)

            if abs(r.omega - OMEGA_C) < 0.05:
                color = 'black'
                lw = 2.5
                alpha = 1.0
            elif r.omega < OMEGA_C:
                color = plt.cm.Blues(0.4 + 0.6 * (OMEGA_C - r.omega) / (OMEGA_C - 10.0))
                lw = 1.5
                alpha = 0.8
            else:
                color = plt.cm.Reds(0.4 + 0.6 * (r.omega - OMEGA_C) / (13.0 - OMEGA_C))
                lw = 1.5
                alpha = 0.8

            ax.loglog(t, rho, color=color, alpha=alpha, linewidth=lw)

        ax.set_xlabel(r"$t\,\Gamma$", fontsize=13)
        ax.set_ylabel(r"$\rho(t)$", fontsize=13)
        ax.set_title(r"Log-log dynamics near critical point — 2D, N=3600, $\gamma=0.1$", fontsize=14)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "loglog_dynamics_N3600_paper.png", dpi=300)
        plt.close()
        print("Saved paper-style log-log dynamics for N=3600")


if __name__ == "__main__":
    main()
