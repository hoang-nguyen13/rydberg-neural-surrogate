"""
Reproduce thesis-style phase transition plots for 2D Rydberg facilitation.

For each system size N, plots ρ_ss vs Ω in the critical region [10, 13].
Also generates a multi-size comparison plot showing finite-size scaling.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / 'data'))
from parse_jld2_v2 import TrajectoryRecord


def compute_rho_ss_thesis_style(rec):
    """
    Match thesis computation:
    ρ = abs((sz_mean + 1) / 2), steady-state = last point (not averaged).
    """
    rho = np.abs((rec.sz_mean + 1.0) / 2.0)
    return float(rho[-1])  # thesis uses last point only


def plot_single_size(records, dimension, gamma, n_atoms, omega_range=(10, 13), 
                     save_path=None, show_critical_line=True):
    """Plot phase transition for a single (dimension, gamma, N) combination."""
    recs = [r for r in records 
            if r.dimension == dimension 
            and abs(r.gamma - gamma) < 1e-10 
            and r.n_atoms == n_atoms
            and omega_range[0] <= r.omega <= omega_range[1]]
    
    if not recs:
        print(f"  No data: {dimension}D, N={n_atoms}, γ={gamma}, Ω∈{omega_range}")
        return None
    
    recs = sorted(recs, key=lambda r: r.omega)
    omegas = [r.omega for r in recs]
    rho_ss_vals = [compute_rho_ss_thesis_style(r) for r in recs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Thesis style: scatter with line
    ax.plot(omegas, rho_ss_vals, '-', color='lightcoral', alpha=0.6, linewidth=1.5, zorder=2)
    ax.scatter(omegas, rho_ss_vals, s=80, c='darkred', edgecolors='black', 
               linewidth=0.5, zorder=3, marker='o')
    
    # Phase threshold
    ax.axhline(y=0.05, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
    
    # Critical line (thesis value for 2D γ=0.1: Ω_c ≈ 11.2)
    if show_critical_line and dimension == 2 and abs(gamma - 0.1) < 1e-10:
        ax.axvline(x=11.2, color='black', linestyle='--', alpha=0.5, linewidth=1.5,
                   label=r'$\Omega_c \approx 11.2$')
    
    ax.set_xlabel(r'Driving strength $\Omega/\Gamma$', fontsize=16)
    ax.set_ylabel(r'Steady-state density $\rho_{ss}$', fontsize=16)
    ax.set_title(f'{dimension}D, N = {n_atoms}, γ = {gamma}', fontsize=16, fontweight='bold')
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(omega_range[0] - 0.1, omega_range[1] + 0.1)
    ax.set_ylim(-0.01, max(rho_ss_vals) * 1.15 + 0.02)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()
    return fig


def plot_multi_size_comparison(records, dimension, gamma, sizes, 
                                omega_range=(10, 13), save_path=None):
    """
    Plot multiple system sizes on the same axes to show finite-size scaling.
    This is the key figure from the thesis showing how the transition sharpens with N.
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sizes)))
    
    for idx, N in enumerate(sizes):
        recs = [r for r in records 
                if r.dimension == dimension 
                and abs(r.gamma - gamma) < 1e-10 
                and r.n_atoms == N
                and omega_range[0] <= r.omega <= omega_range[1]]
        
        if not recs:
            continue
        
        recs = sorted(recs, key=lambda r: r.omega)
        omegas = [r.omega for r in recs]
        rho_ss_vals = [compute_rho_ss_thesis_style(r) for r in recs]
        
        ax.plot(omegas, rho_ss_vals, '-', color=colors[idx], alpha=0.6, linewidth=1.5)
        ax.scatter(omegas, rho_ss_vals, s=50, color=colors[idx], 
                   edgecolors='black', linewidth=0.3, zorder=3,
                   label=f'N = {N}')
    
    # Phase threshold
    ax.axhline(y=0.05, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
    
    # Critical line
    if dimension == 2 and abs(gamma - 0.1) < 1e-10:
        ax.axvline(x=11.2, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel(r'Driving strength $\Omega/\Gamma$', fontsize=16)
    ax.set_ylabel(r'Steady-state density $\rho_{ss}$', fontsize=16)
    ax.set_title(f'{dimension}D, γ = {gamma}: Finite-Size Scaling', 
                 fontsize=16, fontweight='bold')
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=11, loc='upper left', title='System size', title_fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(omega_range[0] - 0.05, omega_range[1] + 0.05)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()
    return fig


def main(args):
    with open(args.data_path, 'rb') as f:
        records = pickle.load(f)
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loaded {len(records)} records")
    print(f"Output directory: {out_dir}")
    
    # === INDIVIDUAL PLOTS: 2D, γ=0.1, each N separately ===
    print("\n=== Individual Phase Diagrams (2D, γ=0.1, Ω∈[10,13]) ===")
    sizes_2d = [225, 400, 900, 1225, 1600, 2500, 3025, 3600, 4900]
    
    for N in sizes_2d:
        recs = [r for r in records 
                if r.dimension == 2 and abs(r.gamma - 0.1) < 1e-10 
                and r.n_atoms == N and 10 <= r.omega <= 13]
        if recs:
            plot_single_size(records, 2, 0.1, N, omega_range=(10, 13),
                           save_path=out_dir / f'phase_2D_N{N}_gamma0.1.png')
        else:
            print(f"  Skipping N={N} (no data in Ω∈[10,13])")
    
    # === MULTI-SIZE COMPARISON: All N on same plot ===
    print("\n=== Multi-Size Comparison (2D, γ=0.1, Ω∈[10,13]) ===")
    sizes_with_data = []
    for N in sizes_2d:
        recs = [r for r in records 
                if r.dimension == 2 and abs(r.gamma - 0.1) < 1e-10 
                and r.n_atoms == N and 10 <= r.omega <= 13]
        if recs:
            sizes_with_data.append(N)
    
    if sizes_with_data:
        plot_multi_size_comparison(records, 2, 0.1, sizes_with_data, omega_range=(10, 13),
                                   save_path=out_dir / 'phase_2D_all_N_gamma0.1.png')
    
    # === CLASSICAL γ COMPARISON: γ=0.1, 5, 10, 20 on same plot (N=3600) ===
    print("\n=== γ Comparison (2D, N=3600, Ω∈[10,13]) ===")
    gammas = [0.1, 5.0, 10.0, 20.0]
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ['darkred', 'blue', 'green', 'orange']
    
    for gamma, color in zip(gammas, colors):
        recs = [r for r in records 
                if r.dimension == 2 and abs(r.gamma - gamma) < 1e-10 
                and r.n_atoms == 3600 and 10 <= r.omega <= 13]
        if not recs:
            continue
        recs = sorted(recs, key=lambda r: r.omega)
        omegas = [r.omega for r in recs]
        rho_ss_vals = [compute_rho_ss_thesis_style(r) for r in recs]
        
        ax.plot(omegas, rho_ss_vals, '-', color=color, alpha=0.6, linewidth=1.5)
        ax.scatter(omegas, rho_ss_vals, s=50, color=color, 
                   edgecolors='black', linewidth=0.3, label=f'γ = {gamma}')
    
    ax.axhline(y=0.05, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel(r'Driving strength $\Omega/\Gamma$', fontsize=16)
    ax.set_ylabel(r'Steady-state density $\rho_{ss}$', fontsize=16)
    ax.set_title('2D, N = 3600: Quantum vs Classical Dephasing', fontsize=16, fontweight='bold')
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(9.9, 13.1)
    plt.tight_layout()
    plt.savefig(out_dir / 'phase_2D_N3600_gamma_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {out_dir / 'phase_2D_N3600_gamma_comparison.png'}")
    plt.show()
    
    print(f"\nAll plots saved to {out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='outputs/rydberg_dataset_v2.pkl')
    parser.add_argument('--output_dir', type=str, default='outputs/phase_transitions')
    args = parser.parse_args()
    main(args)
