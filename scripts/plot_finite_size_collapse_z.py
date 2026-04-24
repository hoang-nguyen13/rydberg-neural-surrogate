"""
Finite-size scaling collapse for tuning the dynamic exponent z.

Plots ρ(t) × t^δ versus t / N^(z/d) for various system sizes N at Ω = Ω_c.
By tuning z, curves from different N should collapse onto a single master curve.

Reference: thesis Fig. finite_size_samples (z tuning) and Fig. zbig.
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
DELTA = 0.4577
D = 2  # spatial dimension

# z values to try (thesis uses 1.20, 1.60, 1.70, 1.84, 1.90, 2.00, 2.02, 2.20, 3.00)
Z_VALUES_SMALL = [1.20, 1.60, 1.70, 1.84, 1.90, 2.00]
Z_VALUES_LARGE = [2.02, 2.20, 3.00]
Z_VALUES_ALL = [1.20, 1.60, 1.70, 1.84, 1.86, 1.90, 2.00, 2.02, 2.20, 3.00]


def load_data():
    with open("data/rydberg_dataset_v2.pkl", "rb") as f:
        data = pickle.load(f)
    return data


def plot_collapse_for_z(data, z_values, suffix=""):
    """Generate collapse plots for a list of z values."""
    # Collect records at Ω_c for different N
    records_by_n = {}
    for r in data:
        if r.dimension != 2 or r.gamma != 0.1:
            continue
        # Find records at or very near Ω_c
        if abs(r.omega - OMEGA_C) > 0.05:
            continue
        if r.n_atoms not in records_by_n:
            records_by_n[r.n_atoms] = []
        records_by_n[r.n_atoms].append(r)

    if not records_by_n:
        print("No records at Ω_c found!")
        return

    # For each N, pick the best record (or average if multiple)
    best_records = {}
    for n, recs in records_by_n.items():
        # Average if multiple trajectories at same parameters
        best_records[n] = recs[0]  # Data should have one per param set

    ns = sorted(best_records.keys())
    print(f"Available N values at Ω_c={OMEGA_C}: {ns}")

    # Use a color map for N values
    colors = plt.cm.viridis(np.linspace(0, 1, len(ns)))

    for z in z_values:
        fig, ax = plt.subplots(figsize=(8, 6))

        for n, color in zip(ns, colors):
            r = best_records[n]
            valid = r.t_save <= 2000
            t = r.t_save[valid]
            sz = r.sz_mean[valid]
            rho = np.abs((1 + sz) / 2)

            # Scaling variables
            x = t / (n ** (z / D))
            y = rho * (t ** DELTA)

            ax.loglog(x, y, color=color, alpha=0.8, linewidth=1.5, label=f"N={n}")

        ax.set_xlabel(r"$t / N^{z/d}$", fontsize=13)
        ax.set_ylabel(r"$\rho(t)\,t^{\delta}$", fontsize=13)
        ax.set_title(f"Finite-size collapse — 2D, $\Omega_c={OMEGA_C}$, $z={z}$, $\delta={DELTA}$",
                     fontsize=13)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"finite_size_collapse_z{z:.2f}{suffix}.png", dpi=300)
        plt.close()
        print(f"Saved finite-size collapse for z={z:.2f}")

    # Also create a grid figure (thesis style)
    if len(z_values) >= 6:
        n_cols = 3
        n_rows = (len(z_values) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

        for idx, z in enumerate(z_values[:n_rows * n_cols]):
            ax = axes[idx]
            for n, color in zip(ns, colors):
                r = best_records[n]
                valid = r.t_save <= 2000
                t = r.t_save[valid]
                sz = r.sz_mean[valid]
                rho = np.abs((1 + sz) / 2)
                x = t / (n ** (z / D))
                y = rho * (t ** DELTA)
                ax.loglog(x, y, color=color, alpha=0.8, linewidth=1.5, label=f"N={n}")

            ax.set_title(f"z = {z:.2f}", fontsize=12)
            ax.set_xlabel(r"$t / N^{z/d}$", fontsize=10)
            ax.set_ylabel(r"$\rho(t)\,t^{\delta}$", fontsize=10)
            ax.grid(True, which="both", ls="--", alpha=0.3)

        for idx in range(len(z_values[:n_rows * n_cols]), len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f"Finite-size scaling collapse — 2D, $\Omega_c={OMEGA_C}$, $\delta={DELTA}$",
                     fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"finite_size_collapse_grid{suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved grid finite-size collapse{suffix}")


def main():
    data = load_data()

    print("=" * 60)
    print("Small z values (thesis Fig. finite_size_samples)")
    print("=" * 60)
    plot_collapse_for_z(data, Z_VALUES_SMALL, suffix="_small")

    print("\n" + "=" * 60)
    print("Large z values (thesis Fig. zbig)")
    print("=" * 60)
    plot_collapse_for_z(data, Z_VALUES_LARGE, suffix="_large")

    print("\n" + "=" * 60)
    print("All z values")
    print("=" * 60)
    plot_collapse_for_z(data, Z_VALUES_ALL, suffix="_all")


if __name__ == "__main__":
    main()
