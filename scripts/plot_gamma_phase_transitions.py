"""
Phase transitions ρ_ss vs Ω for different γ (quantum vs classical dephasing).
Fixed N = 3600, 2D, selected Ω = 10:0.15:13.
"""
import sys
sys.path.insert(0, 'data')
from parse_jld2_v2 import TrajectoryRecord
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('outputs/rydberg_dataset_v2.pkl', 'rb') as f:
    records = pickle.load(f)

N = 3600
omega_selected = [round(x, 8) for x in np.arange(10.0, 13.01, 0.15)]
gamma_values = [0.1, 5.0, 10.0, 20.0]
dimension = 2

# Colors: γ=0.1 (dark red), γ=5 (blue), γ=10 (green), γ=20 (orange)
colors = ['#8B0000', '#4169E1', '#228B22', '#FF8C00']

# Collect data: gamma -> {omega: rho_ss}
data = {g: {} for g in gamma_values}
for rec in records:
    if (rec.dimension == dimension and
        rec.n_atoms == N and
        rec.gamma in gamma_values):
        o = round(rec.omega, 8)
        if o in omega_selected:
            rho_ss = abs(rec.rho[-1])
            data[rec.gamma][rec.omega] = rho_ss

fig, ax = plt.subplots(figsize=(10, 6))

for i, g in enumerate(gamma_values):
    omegas = sorted(data[g].keys())
    rhos = [data[g][o] for o in omegas]
    label = f'$\\gamma = {g}$'
    ax.plot(omegas, rhos, linewidth=2, color=colors[i], linestyle='-', zorder=2)
    ax.scatter(omegas, rhos, s=80, marker='o', color=colors[i],
               edgecolors='none', zorder=3, label=label)

ax.set_xlabel(r'$\Omega/\Gamma$', fontsize=16)
ax.set_ylabel(r'$\rho_{ss}$', fontsize=16)
ax.set_title(f'2D, N = {N}: Quantum vs Classical Dephasing', fontsize=14, fontweight='bold')
ax.set_xlim(10.0, 13.0)
ax.set_xticks(np.arange(10.0, 13.25, 0.25))
ax.tick_params(labelsize=12)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.legend(loc='upper left', fontsize=11, frameon=True)

plt.tight_layout()
plt.savefig('outputs/finite_size/gamma_phase_transitions.png', dpi=300, bbox_inches='tight')
plt.savefig('outputs/finite_size/gamma_phase_transitions.pdf', bbox_inches='tight')
print(f"Saved: outputs/finite_size/gamma_phase_transitions.png")
for g in gamma_values:
    print(f"  γ = {g}: {len(data[g])} points")
