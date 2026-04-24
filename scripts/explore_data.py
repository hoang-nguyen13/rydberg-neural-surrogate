"""
Reproduce thesis-style figures from parsed data.

Generates:
1. Phase diagram: ρ_ss vs Ω (per dimension, per γ)
2. Dynamics: ρ(t) vs t for selected Ω values
3. Log-log dynamics: log(ρ) vs log(t) — power-law decay
4. Critical exponent β: log(ρ_ss) vs log(|Ω - Ω_c|)
5. Data collapse: t^δ * ρ_ss vs t * |Ω - Ω_c|^(β/δ)
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).parent.parent / 'data'))
from parse_jld2_v2 import TrajectoryRecord


def plot_phase_diagram(records, dimension, gamma, save_path=None):
    """Plot phase diagram: steady-state density ρ_ss vs driving strength Ω."""
    recs = [r for r in records if r.dimension == dimension and abs(r.gamma - gamma) < 1e-10]
    if not recs:
        print(f"No data for {dimension}D, γ={gamma}")
        return
    
    # Sort by omega
    recs = sorted(recs, key=lambda r: r.omega)
    omegas = [r.omega for r in recs]
    rho_ss_vals = [r.rho_ss for r in recs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(omegas, rho_ss_vals, s=50, c='darkred', edgecolors='black', linewidth=0.5, zorder=3)
    ax.plot(omegas, rho_ss_vals, '-', color='lightcoral', alpha=0.5, zorder=2)
    
    ax.axhline(y=0.05, color='gray', linestyle=':', alpha=0.7, label='Phase threshold')
    ax.set_xlabel(r'Driving strength $\Omega/\Gamma$', fontsize=14)
    ax.set_ylabel(r'Steady-state density $\rho_{ss}$', fontsize=14)
    ax.set_title(f'{dimension}D, N={recs[0].n_atoms}, γ={gamma}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, max(rho_ss_vals) * 1.1 + 0.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def plot_dynamics(records, dimension, gamma, n_atoms, selected_omegas, save_path=None):
    """Plot ρ(t) vs t for selected Ω values."""
    recs = [r for r in records 
            if r.dimension == dimension and abs(r.gamma - gamma) < 1e-10 and r.n_atoms == n_atoms
            and r.omega in selected_omegas]
    recs = sorted(recs, key=lambda r: r.omega)
    
    if not recs:
        print(f"No matching data")
        return
    
    n_plots = len(recs)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True, sharey=True)
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_plots))
    
    for idx, rec in enumerate(recs):
        ax = axes[idx]
        rho = rec.rho
        ax.plot(rec.t_save, rho, color=colors[idx], linewidth=1.5)
        ax.axvspan(rec.t_save[-50], rec.t_save[-1], alpha=0.1, color='blue')
        ax.set_title(f'Ω={rec.omega:.2f}', fontsize=11)
        ax.set_xlabel(r'Time $t/\Gamma$', fontsize=10)
        ax.set_ylabel(r'$\rho(t)$', fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'{dimension}D, N={n_atoms}, γ={gamma}', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def plot_log_log_dynamics(records, dimension, gamma, n_atoms, selected_omegas, 
                          delta=0.47, save_path=None):
    """Plot log(ρ) vs log(t) to show power-law decay."""
    recs = [r for r in records 
            if r.dimension == dimension and abs(r.gamma - gamma) < 1e-10 and r.n_atoms == n_atoms
            and r.omega in selected_omegas]
    recs = sorted(recs, key=lambda r: r.omega)
    
    if not recs:
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(recs)))
    
    for idx, rec in enumerate(recs):
        rho = rec.rho
        t = rec.t_save
        # Only plot where rho > 0 (avoid log(0))
        mask = rho > 1e-6
        ax.loglog(t[mask], rho[mask], '-', color=colors[idx], linewidth=1.5, 
                  label=f'Ω={rec.omega:.2f}')
    
    # Add reference line for δ = 0.47: ρ ~ t^(-δ)
    t_ref = np.logspace(0, 3, 100)
    rho_ref = t_ref ** (-delta)
    ax.loglog(t_ref, rho_ref, 'k--', alpha=0.5, linewidth=1, label=f'$t^{{-{delta}}}$')
    
    ax.set_xlabel(r'Time $t/\Gamma$ (log)', fontsize=14)
    ax.set_ylabel(r'$\rho(t)$ (log)', fontsize=14)
    ax.set_title(f'{dimension}D, N={n_atoms}, γ={gamma}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def extract_critical_exponent_beta(records, dimension, gamma, n_atoms, omega_c):
    """
    Extract critical exponent β by fitting log(ρ_ss) vs log(|Ω - Ω_c|).
    ρ_ss ~ (Ω - Ω_c)^β for Ω > Ω_c.
    """
    recs = [r for r in records 
            if r.dimension == dimension and abs(r.gamma - gamma) < 1e-10 
            and r.n_atoms == n_atoms and r.omega > omega_c + 0.01]
    
    if len(recs) < 3:
        print(f"Not enough data above Ω_c={omega_c}")
        return None
    
    recs = sorted(recs, key=lambda r: r.omega)
    omegas = np.array([r.omega for r in recs])
    rho_ss = np.array([r.rho_ss for r in recs])
    
    # Only use points where ρ_ss > 0
    mask = rho_ss > 1e-6
    if mask.sum() < 3:
        print("Not enough non-zero ρ_ss points")
        return None
    
    x = np.log(np.abs(omegas[mask] - omega_c))
    y = np.log(rho_ss[mask])
    
    # Linear fit: y = β * x + const
    coeffs = np.polyfit(x, y, 1)
    beta = coeffs[0]
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=80, c='darkred', edgecolors='black', linewidth=0.5, zorder=3)
    
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = coeffs[0] * x_fit + coeffs[1]
    ax.plot(x_fit, y_fit, 'r--', linewidth=2, label=f'Fit: β = {beta:.3f}')
    
    ax.set_xlabel(r'$\ln|\Omega - \Omega_c|$', fontsize=14)
    ax.set_ylabel(r'$\ln \rho_{ss}$', fontsize=14)
    ax.set_title(f'{dimension}D, N={n_atoms}, γ={gamma}, Ω_c={omega_c}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Extracted β = {beta:.4f} (expected ~0.588 for 2D DP)")
    return beta


def plot_data_collapse(records, dimension, gamma, n_atoms, omega_c, beta=0.588, delta=0.47, 
                       save_path=None):
    """
    Plot data collapse: t^δ * ρ_ss vs t * |Ω - Ω_c|^(β/δ).
    All curves should collapse onto a single universal function near criticality.
    """
    recs = [r for r in records 
            if r.dimension == dimension and abs(r.gamma - gamma) < 1e-10 
            and r.n_atoms == n_atoms]
    recs = sorted(recs, key=lambda r: r.omega)
    
    if not recs:
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(recs)))
    
    for idx, rec in enumerate(recs):
        t = rec.t_save
        rho = rec.rho
        omega = rec.omega
        
        # Compute scaled variables
        delta_omega = abs(omega - omega_c)
        x = t * (delta_omega ** (beta / delta))
        y = (t ** delta) * rho
        
        ax.loglog(x, y, '-', color=colors[idx], linewidth=1.2, 
                  label=f'Ω={omega:.2f}')
    
    ax.set_xlabel(r'$t |\Omega - \Omega_c|^{\beta/\delta}$', fontsize=14)
    ax.set_ylabel(r'$t^\delta \rho(t)$', fontsize=14)
    ax.set_title(f'Data Collapse: {dimension}D, N={n_atoms}, γ={gamma}, ' 
                 f'Ω_c={omega_c}, β={beta}, δ={delta}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=2, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def plot_critical_point_vs_size(records, dimension, gamma, save_path=None):
    """Plot critical point Ω_c(N) vs system size N."""
    recs = [r for r in records 
            if r.dimension == dimension and abs(r.gamma - gamma) < 1e-10]
    
    from collections import defaultdict
    by_size = defaultdict(list)
    for r in recs:
        by_size[r.n_atoms].append(r)
    
    # Estimate Ω_c for each size using max derivative method
    sizes = []
    omega_cs = []
    
    for N in sorted(by_size.keys()):
        size_recs = sorted(by_size[N], key=lambda r: r.omega)
        omegas = np.array([r.omega for r in size_recs])
        rho_ss = np.array([r.rho_ss for r in size_recs])
        
        if len(omegas) < 3:
            continue
        
        dr = np.diff(rho_ss) / np.diff(omegas)
        omega_mid = (omegas[:-1] + omegas[1:]) / 2
        
        if len(dr) > 0:
            c_idx = np.argmax(dr)
            omega_c = omega_mid[c_idx]
            sizes.append(N)
            omega_cs.append(omega_c)
    
    if len(sizes) < 2:
        print("Not enough sizes to plot Ω_c(N)")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sizes, omega_cs, 'ko-', markersize=8, linewidth=1.5)
    
    # Fit: Ω_c(N) = Ω_inf + A * N^(-1/ν)
    # For 2D DP, ν ≈ 0.73, so exponent ≈ -1.37
    try:
        sizes_arr = np.array(sizes)
        omega_cs_arr = np.array(omega_cs)
        
        def fit_func(N, omega_inf, A, nu_inv):
            return omega_inf + A * N ** (-nu_inv)
        
        popt, _ = curve_fit(fit_func, sizes_arr, omega_cs_arr, 
                           p0=[omega_cs[-1], 1.0, 1.0],
                           maxfev=10000)
        
        N_fit = np.logspace(np.log10(min(sizes)), np.log10(max(sizes)*2), 100)
        ax.plot(N_fit, fit_func(N_fit, *popt), 'r--', linewidth=2,
                label=f'Fit: Ω_∞={popt[0]:.2f}, A={popt[1]:.2f}, 1/ν={popt[2]:.2f}')
        ax.legend(fontsize=11)
    except Exception as e:
        print(f"Fit failed: {e}")
    
    ax.set_xlabel(r'System size $N$', fontsize=14)
    ax.set_ylabel(r'Critical point $\Omega_c(N)$', fontsize=14)
    ax.set_title(f'{dimension}D, γ={gamma}', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def main(args):
    with open(args.data_path, 'rb') as f:
        records = pickle.load(f)
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loaded {len(records)} records")
    
    # === 1. Phase diagrams for key parameter combinations ===
    print("\n=== Phase Diagrams ===")
    
    # 2D, γ=0.1 — main story
    for N in [225, 400, 900, 3600, 4900, 10000]:
        recs = [r for r in records if r.dimension == 2 and abs(r.gamma - 0.1) < 1e-10 and r.n_atoms == N]
        if recs:
            plot_phase_diagram(records, 2, 0.1, 
                              save_path=out_dir / f'phase_diagram_2D_N{N}_gamma0.1.png')
    
    # 2D, γ=5.0, 10.0, 20.0 — classical comparison
    for g in [5.0, 10.0, 20.0]:
        recs = [r for r in records if r.dimension == 2 and abs(r.gamma - g) < 1e-10]
        if recs:
            plot_phase_diagram(records, 2, g,
                              save_path=out_dir / f'phase_diagram_2D_gamma{g}.png')
    
    # 1D and 3D
    plot_phase_diagram(records, 1, 0.1, save_path=out_dir / 'phase_diagram_1D_gamma0.1.png')
    plot_phase_diagram(records, 3, 0.1, save_path=out_dir / 'phase_diagram_3D_gamma0.1.png')
    plot_phase_diagram(records, 3, 1e-5, save_path=out_dir / 'phase_diagram_3D_gamma1e-5.png')
    
    # === 2. Dynamics for selected cases ===
    print("\n=== Dynamics ===")
    
    # 2D, N=3600, γ=0.1 — critical region
    recs_2d = [r for r in records if r.dimension == 2 and r.n_atoms == 3600 and abs(r.gamma - 0.1) < 1e-10]
    if recs_2d:
        omegas = sorted(set(r.omega for r in recs_2d))
        # Select: below, near, above critical
        selected = [omegas[0], omegas[len(omegas)//3], omegas[2*len(omegas)//3], omegas[-1]]
        plot_dynamics(records, 2, 0.1, 3600, selected,
                     save_path=out_dir / 'dynamics_2D_N3600_gamma0.1.png')
    
    # === 3. Critical exponent extraction ===
    print("\n=== Critical Exponent β ===")
    
    # 2D, N=3600, γ=0.1 — thesis critical point Ω_c ≈ 11.2
    extract_critical_exponent_beta(records, 2, 0.1, 3600, omega_c=11.2)
    
    # 1D, N=3000, γ=0.1 — thesis critical point Ω_c ≈ 29.85
    extract_critical_exponent_beta(records, 1, 0.1, 3000, omega_c=29.85)
    
    # === 4. Data collapse ===
    print("\n=== Data Collapse ===")
    
    plot_data_collapse(records, 2, 0.1, 3600, omega_c=11.2, beta=0.588, delta=0.47,
                      save_path=out_dir / 'collapse_2D_N3600_gamma0.1.png')
    
    # === 5. Critical point vs size ===
    print("\n=== Critical Point vs Size ===")
    
    plot_critical_point_vs_size(records, 2, 0.1,
                               save_path=out_dir / 'critical_point_vs_size_2D_gamma0.1.png')
    
    print(f"\nAll plots saved to {out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='outputs/rydberg_dataset_v2.pkl')
    parser.add_argument('--output_dir', type=str, default='outputs/data_exploration')
    args = parser.parse_args()
    main(args)
