"""
EXACT reproduction of thesis finite-size scaling plot.
Matches plot_finite_extrap.ipynb Cell 2 from MasterThesisHoang/Figures/finite_size_sims/
"""
import sys
sys.path.insert(0, 'data')
from parse_jld2_v2 import TrajectoryRecord
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load data
with open('data/rydberg_dataset_v2.pkl', 'rb') as f:
    records = pickle.load(f)

# === EXACT parameters from thesis code ===
nAtoms_values = [225, 400, 900, 1600, 2500, 3600, 4900]
gamma = 0.1
dimension = 2
chosen_omega = 11.2

# EXACT selected Omega values from thesis: 10:0.15:13
omega_selected = np.arange(10.0, 13.01, 0.15)
omega_selected = [round(o, 8) for o in omega_selected]  # avoid float issues

# Filter to 2D, gamma=0.1, N in list, Omega in SELECTED list only
data_dict = {}
for rec in records:
    if (rec.dimension == dimension and 
        abs(rec.gamma - gamma) < 0.001 and 
        rec.n_atoms in nAtoms_values):
        # Only use SELECTED omega values (thesis: 10:0.15:13)
        omega_rounded = round(rec.omega, 8)
        if omega_rounded not in omega_selected:
            continue
        if rec.n_atoms not in data_dict:
            data_dict[rec.n_atoms] = {}
        # EXACT computation from thesis:
        # Szs_mean_transformed = (1 .+ Szs_mean_filtered) / 2
        # data_dict stores abs.(Szs_mean_transformed)
        # Then: last_n = min(1, n_points); avg_value = mean(Szs_mean[end-last_n+1:end])
        # i.e., just the LAST point, with abs
        rho_thesis = abs(rec.rho[-1])  # last point, absolute value
        data_dict[rec.n_atoms][rec.omega] = rho_thesis

# Sort N values
nAtoms_values = sorted(data_dict.keys())

# === EXACT color scheme: cgrad([:lightpink, :darkred], length(nAtoms_values)) ===
# lightpink  = (1.0, 0.71, 0.76)  ≈ #FFB6C1
# darkred    = (0.55, 0.0, 0.0)   ≈ #8B0000
colors = []
for i in range(len(nAtoms_values)):
    t = i / max(1, len(nAtoms_values) - 1)
    r = 1.0 - t * (1.0 - 0.55)
    g = 0.71 - t * 0.71
    b = 0.76 - t * 0.76
    colors.append((r, g, b))

# === EXACT plot style ===
fig, ax = plt.subplots(figsize=(10, 6))  # size=(1000, 600) in pixels, dpi=100

for i, nAtoms in enumerate(nAtoms_values):
    omegas = sorted(data_dict[nAtoms].keys())
    rho_ss_values = [data_dict[nAtoms][o] for o in omegas]
    
    color = colors[i]
    label = f"N = {nAtoms}"
    
    # Line + scatter with circle marker (exactly as thesis)
    ax.plot(omegas, rho_ss_values, 
            linewidth=2, 
            color=color, 
            linestyle='-',
            label=label,
            zorder=2)
    ax.scatter(omegas, rho_ss_values,
               s=80,  # markersize=8 in Plots.jl ≈ s=64-100 in matplotlib
               marker='o',
               color=color,
               edgecolors='none',
               zorder=3)

# Vertical dashed line at chosen_omega
ax.axvline(x=chosen_omega, linestyle='--', color='black', linewidth=2, 
           label=f"$\\Omega = {chosen_omega}$", zorder=1)

# EXACT labels and style
ax.set_xlabel(r'$\Omega$', fontsize=16)
ax.set_ylabel(r'$\rho_{ss}$', fontsize=16)
ax.legend(loc='lower left', fontsize=10, frameon=True)
ax.set_xticks(np.arange(10, 13.5, 0.25))
ax.tick_params(labelsize=12)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_xlim(10.0, 13.0)

plt.tight_layout()
plt.savefig('outputs/phase_transitions/thesis_exact_finite_size.png', dpi=300, bbox_inches='tight')
plt.savefig('outputs/phase_transitions/thesis_exact_finite_size.pdf', bbox_inches='tight')
print("Saved: outputs/phase_transitions/thesis_exact_finite_size.png")
print(f"N values: {nAtoms_values}")
print(f"Total curves: {len(nAtoms_values)}")
