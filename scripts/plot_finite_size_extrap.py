"""
Finite-size scaling plot for 2D Rydberg facilitation.
Reproduces thesis plot_finite_extrap.ipynb Cell 2 exactly.
"""
import sys
sys.path.insert(0, 'data')
from parse_jld2_v2 import TrajectoryRecord
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load parsed data
with open('data/rydberg_dataset_v2.pkl', 'rb') as f:
    records = pickle.load(f)

# === Thesis parameters ===
nAtoms_values = [225, 400, 900, 1600, 2500, 3600, 4900]
omega_selected = [round(x, 8) for x in np.arange(10.0, 13.01, 0.15)]
gamma = 0.1
dimension = 2
omega_c = 11.2

# Collect data: N -> {omega: rho_ss}
data = {N: {} for N in nAtoms_values}
for rec in records:
    if (rec.dimension == dimension and
        abs(rec.gamma - gamma) < 0.001 and
        rec.n_atoms in nAtoms_values):
        o = round(rec.omega, 8)
        if o in omega_selected:
            # Thesis: last point only, abs value
            rho_ss = abs(rec.rho[-1])
            data[rec.n_atoms][rec.omega] = rho_ss

# Verify all N have data
for N in nAtoms_values:
    npts = len(data[N])
    omegas = sorted(data[N].keys())
    print(f"N = {N:4d}: {npts} points, Omega = {omegas[0]:.2f} ... {omegas[-1]:.2f}")

# === Plot ===
fig, ax = plt.subplots(figsize=(10, 6))

# Color gradient: lightpink -> darkred
for i, N in enumerate(nAtoms_values):
    t = i / max(1, len(nAtoms_values) - 1)
    r = 1.0 - t * 0.45
    g = 0.71 - t * 0.71
    b = 0.76 - t * 0.76
    color = (r, g, b)

    omegas = sorted(data[N].keys())
    rhos = [data[N][o] for o in omegas]

    ax.plot(omegas, rhos, linewidth=2, color=color, linestyle='-', zorder=2)
    ax.scatter(omegas, rhos, s=80, marker='o', color=color, edgecolors='none', zorder=3)

# Vertical line at critical point
ax.axvline(x=omega_c, linestyle='--', color='black', linewidth=2, zorder=1)

# Labels
ax.set_xlabel(r'$\Omega/\Gamma$', fontsize=16)
ax.set_ylabel(r'$\rho_{ss}$', fontsize=16)
ax.set_xlim(10.0, 13.0)
ax.set_xticks(np.arange(10.0, 13.25, 0.25))
ax.tick_params(labelsize=12)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Legend: reverse order so N=4900 is on top
handles = []
labels = []
for i, N in enumerate(nAtoms_values):
    t = i / max(1, len(nAtoms_values) - 1)
    r = 1.0 - t * 0.45
    g = 0.71 - t * 0.71
    b = 0.76 - t * 0.76
    color = (r, g, b)
    handles.append(plt.Line2D([0], [0], color=color, linewidth=2))
    labels.append(f'N = {N}')

# Add omega_c to legend
handles.append(plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--'))
labels.append(r'$\Omega_c = 11.2$')

ax.legend(handles[::-1], labels[::-1], loc='lower left', fontsize=10, frameon=True)

plt.tight_layout()
plt.savefig('outputs/finite_size/finite_size_extrap.png', dpi=300, bbox_inches='tight')
plt.savefig('outputs/finite_size/finite_size_extrap.pdf', bbox_inches='tight')
print("\nSaved to outputs/finite_size/finite_size_extrap.{png,pdf}")
