"""
Plot dynamics ρ(t) vs t for each system size N.
Shows all selected Ω values (10:0.15:13) per N.
"""
import sys
sys.path.insert(0, 'data')
from parse_jld2_v2 import TrajectoryRecord
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

with open('outputs/rydberg_dataset_v2.pkl', 'rb') as f:
    records = pickle.load(f)

nAtoms_values = [225, 400, 900, 1600, 2500, 3600, 4900]
omega_selected = [round(x, 8) for x in np.arange(10.0, 13.01, 0.15)]
gamma = 0.1
dimension = 2

# Group data: N -> {omega: (t, rho)}
data = {N: {} for N in nAtoms_values}
for rec in records:
    if (rec.dimension == dimension and
        abs(rec.gamma - gamma) < 0.001 and
        rec.n_atoms in nAtoms_values):
        o = round(rec.omega, 8)
        if o in omega_selected:
            # Thesis filtering: t <= 2000, then transform and abs
            mask = rec.t_save <= 2000
            t = rec.t_save[mask]
            rho = np.abs((rec.sz_mean[mask] + 1.0) / 2.0)
            data[rec.n_atoms][rec.omega] = (t, rho)

# One figure per N
for N in nAtoms_values:
    fig, ax = plt.subplots(figsize=(10, 6))

    omegas = sorted(data[N].keys())
    n_omegas = len(omegas)

    # Color map: low Ω (blue) → high Ω (red)
    colors = cm.coolwarm(np.linspace(0, 1, n_omegas))

    for i, o in enumerate(omegas):
        t, rho = data[N][o]
        ax.plot(t, rho, linewidth=1.5, color=colors[i], label=f'$\\Omega={o:.2f}$')

    ax.set_xlabel(r'$t\Gamma$', fontsize=14)
    ax.set_ylabel(r'$\rho(t)$', fontsize=14)
    ax.set_title(f'2D, N = {N}, $\\gamma$ = {gamma}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1000)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Legend: reduce if too many entries
    if n_omegas <= 15:
        ax.legend(loc='upper left', fontsize=7, ncol=2, frameon=True)
    else:
        ax.legend(loc='upper left', fontsize=6, ncol=3, frameon=True)

    plt.tight_layout()
    fname = f'outputs/finite_size/dynamics_N{N}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}  ({n_omegas} curves)")

print("\nAll dynamics plots saved to outputs/finite_size/")
