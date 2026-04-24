"""
Dynamics ρ(t) vs t for different γ values.
Fixed N = 3600, 2D, selected Ω = 10:0.15:13.
One plot per γ.
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

N = 3600
omega_selected = [round(x, 8) for x in np.arange(10.0, 13.01, 0.15)]
gamma_values = [0.1, 5.0, 10.0, 20.0]
dimension = 2

# Collect data: gamma -> {omega: (t, rho)}
data = {g: {} for g in gamma_values}
for rec in records:
    if (rec.dimension == dimension and
        rec.n_atoms == N and
        rec.gamma in gamma_values):
        o = round(rec.omega, 8)
        if o in omega_selected:
            mask = rec.t_save <= 2000
            t = rec.t_save[mask]
            rho = np.abs((rec.sz_mean[mask] + 1.0) / 2.0)
            data[rec.gamma][rec.omega] = (t, rho)

for g in gamma_values:
    fig, ax = plt.subplots(figsize=(10, 6))

    omegas = sorted(data[g].keys())
    n_omegas = len(omegas)
    colors = cm.coolwarm(np.linspace(0, 1, n_omegas))

    for i, o in enumerate(omegas):
        t, rho = data[g][o]
        ax.plot(t, rho, linewidth=1.5, color=colors[i], label=f'$\\Omega={o:.2f}$')

    ax.set_xlabel(r'$t\Gamma$', fontsize=14)
    ax.set_ylabel(r'$\rho(t)$', fontsize=14)
    ax.set_title(f'2D, N = {N}, $\\gamma$ = {g}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1000)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    if n_omegas <= 15:
        ax.legend(loc='upper left', fontsize=7, ncol=2, frameon=True)
    else:
        ax.legend(loc='upper left', fontsize=6, ncol=3, frameon=True)

    plt.tight_layout()
    fname = f'outputs/finite_size/dynamics_gamma{g}_N{N}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}  ({n_omegas} curves)")

print("\nAll gamma dynamics plots saved.")
