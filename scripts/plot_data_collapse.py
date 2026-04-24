"""
Data collapse for fixed N, varying Ω near critical point.

Plots t^δ × ρ(t) as a function of t × |Ω - Ω_c|^(β/δ).
Curves for different Ω should collapse onto a single master curve near criticality.

Reference: thesis section on 2D collapse (Fig. atoms_2D_collapse).
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
BETA = 0.586
DELTA = 0.4577
def load_data():
    with open("outputs/rydberg_dataset_v2.pkl", "rb") as f:
        data = pickle.load(f)
    return data


def main():
    data = load_data()

    # Target N values for collapse plots
    target_Ns = [225, 400, 900, 1600, 2500, 3600, 4900]

    for n_atoms in target_Ns:
        records = []
        for r in data:
            if r.dimension != 2 or r.gamma != 0.1 or r.n_atoms != n_atoms:
                continue
            records.append(r)

        if len(records) < 3:
            print(f"Skipping N={n_atoms}: only {len(records)} trajectories")
            continue

        records.sort(key=lambda r: r.omega)

        fig, ax = plt.subplots(figsize=(8, 6))

        for r in records:
            valid = r.t_save <= 2000
            t = r.t_save[valid]
            sz = r.sz_mean[valid]
            rho = np.abs((1 + sz) / 2)

            # Compute scaling variables
            delta_omega = abs(r.omega - OMEGA_C)
            if delta_omega < 1e-6:
                # At critical point, skip or handle separately
                continue

            x = t * (delta_omega ** (BETA / DELTA))
            y = (t ** DELTA) * rho

            # Color based on distance from critical point
            if r.omega < OMEGA_C:
                color = plt.cm.Blues(0.3 + 0.7 * min(1.0, delta_omega / 1.5))
            else:
                color = plt.cm.Reds(0.3 + 0.7 * min(1.0, delta_omega / 1.5))

            ax.loglog(x, y, color=color, alpha=0.7, linewidth=1.5,
                      label=f"Ω={r.omega:.2f}")

        ax.set_xlabel(r"$t\,|\Omega - \Omega_c|^{\beta/\delta}$", fontsize=13)
        ax.set_ylabel(r"$t^{\delta}\,\rho(t)$", fontsize=13)
        ax.set_title(f"Data collapse — 2D, N={n_atoms}, "
                     r"$\Omega_c={OMEGA_C}$, $\beta={BETA}$, $\delta={DELTA}$",
                     fontsize=13)
        ax.grid(True, which="both", ls="--", alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()
        step = max(1, len(handles) // 8)
        ax.legend(handles[::step], labels[::step], loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"data_collapse_N{n_atoms}.png", dpi=300)
        plt.close()
        print(f"Saved data collapse for N={n_atoms}")

    # Also create a focused plot near critical point for N=3600 (thesis style)
    n3600_records = [r for r in data
                     if r.dimension == 2 and r.gamma == 0.1 and r.n_atoms == 3600]

    if len(n3600_records) >= 3:
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes = axes.flatten()

        # Select 6 representative omegas near critical point
        near_critical = [r for r in n3600_records if abs(r.omega - OMEGA_C) < 1.0]
        near_critical.sort(key=lambda r: r.omega)

        for idx, r in enumerate(near_critical[:6]):
            ax = axes[idx]
            valid = r.t_save <= 2000
            t = r.t_save[valid]
            sz = r.sz_mean[valid]
            rho = np.abs((1 + sz) / 2)

            delta_omega = abs(r.omega - OMEGA_C)
            if delta_omega < 1e-6:
                x = t
                y = (t ** DELTA) * rho
                ax.loglog(x, y, 'k-', linewidth=2)
            else:
                x = t * (delta_omega ** (BETA / DELTA))
                y = (t ** DELTA) * rho
                ax.loglog(x, y, 'b-', linewidth=2)

            ax.set_title(f"Ω = {r.omega:.2f}", fontsize=12)
            ax.set_xlabel(r"$t\,|\Omega - \Omega_c|^{\beta/\delta}$", fontsize=10)
            ax.set_ylabel(r"$t^{\delta}\,\rho(t)$", fontsize=10)
            ax.grid(True, which="both", ls="--", alpha=0.3)

        # Hide unused subplots
        for idx in range(len(near_critical[:6]), 6):
            axes[idx].axis('off')

        plt.suptitle(f"Data collapse near critical point — 2D, N=3600, "
                     r"$\Omega_c={OMEGA_C}$", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "data_collapse_N3600_grid.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved grid data collapse for N=3600")


if __name__ == "__main__":
    main()
