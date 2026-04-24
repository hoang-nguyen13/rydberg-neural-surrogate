"""
Data quality checks and visualization for Rydberg facilitation dataset.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent / 'data'))
from parse_jld2 import TrajectoryRecord

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 9


def check_anomalies(records):
    """Check for NaN, Inf, and shape anomalies."""
    print("=" * 60)
    print("ANOMALY CHECK")
    print("=" * 60)
    
    issues = []
    for rec in records:
        if np.any(np.isnan(rec.sz_mean)):
            issues.append(f"NaN in sz_mean: N={rec.n_atoms}, Omega={rec.omega}")
        if np.any(np.isinf(rec.sz_mean)):
            issues.append(f"Inf in sz_mean: N={rec.n_atoms}, Omega={rec.omega}")
        if len(rec.sz_mean) != 400:
            issues.append(f"Wrong sz_mean length: N={rec.n_atoms}, Omega={rec.omega}, len={len(rec.sz_mean)}")
        if len(rec.t_save) != 400:
            issues.append(f"Wrong t_save length: N={rec.n_atoms}, Omega={rec.omega}, len={len(rec.t_save)}")
        if rec.sz_mean[0] < 0.99:
            issues.append(f"sz_mean[0] != +1.0: N={rec.n_atoms}, Omega={rec.omega}, sz0={rec.sz_mean[0]:.4f}")
    
    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues[:20]:
            print(f"  {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
    else:
        print("No anomalies detected.")
    print()


def plot_representative_trajectories(records, output_dir):
    """Plot trajectories for absorbing, near-critical, and active phases."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n1225 = [r for r in records if r.n_atoms == 1225]
    n1225 = sorted(n1225, key=lambda r: r.omega)
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    
    examples = [
        (n1225[0], "Absorbing (low Omega)"),
        (n1225[len(n1225)//3], "Near-critical (low)"),
        (n1225[len(n1225)//2], "Near-critical (mid)"),
        (n1225[2*len(n1225)//3], "Near-critical (high)"),
        (n1225[-2], "Active (high Omega)"),
        (n1225[-1], "Active (max Omega)"),
    ]
    
    for ax, (rec, title) in zip(axes, examples):
        ax.plot(rec.t_save, rec.sz_mean, lw=1.5, color='steelblue')
        ax.axhline(-1.0, color='gray', ls='--', alpha=0.5, label='Absorbing state')
        ax.set_title(f"N={rec.n_atoms}, Omega={rec.omega:.2f}\n{title}")
        ax.set_xlabel('Time')
        ax.set_ylabel('sz_mean')
        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'representative_trajectories.png')
    plt.close()
    print(f"Saved: {output_dir / 'representative_trajectories.png'}")


def plot_phase_diagram(records, output_dir):
    """Plot rho_ss vs Omega for each system size."""
    output_dir = Path(output_dir)
    
    sizes = sorted(set(r.n_atoms for r in records))
    n_sizes = len(sizes)
    n_cols = 3
    n_rows = (n_sizes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3*n_rows), squeeze=False)
    colors = plt.cm.viridis(np.linspace(0, 1, n_sizes))
    
    for idx, n in enumerate(sizes):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        recs = [r for r in records if r.n_atoms == n]
        recs = sorted(recs, key=lambda r: r.omega)
        
        omegas = [r.omega for r in recs]
        rho_ss_vals = [(np.mean(r.sz_mean[-50:]) + 1) / 2 for r in recs]
        
        ax.plot(omegas, rho_ss_vals, 'o-', color=colors[idx], markersize=3, lw=1)
        ax.set_title(f"N = {n} ({int(np.sqrt(n))}x{int(np.sqrt(n))})")
        ax.set_xlabel('Omega')
        ax.set_ylabel('rho_ss')
        ax.set_ylim(-0.05, 0.55)
        ax.axhline(0.0, color='gray', ls='--', alpha=0.3)
    
    for idx in range(n_sizes, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase_diagram_by_size.png')
    plt.close()
    print(f"Saved: {output_dir / 'phase_diagram_by_size.png'}")


def plot_combined_phase_diagram(records, output_dir):
    """Plot all sizes on a single figure."""
    output_dir = Path(output_dir)
    
    sizes = sorted(set(r.n_atoms for r in records))
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(sizes), vmax=max(sizes))
    
    for n in sizes:
        recs = [r for r in records if r.n_atoms == n]
        recs = sorted(recs, key=lambda r: r.omega)
        
        omegas = [r.omega for r in recs]
        rho_ss_vals = [(np.mean(r.sz_mean[-50:]) + 1) / 2 for r in recs]
        
        color = cmap(norm(n))
        ax.plot(omegas, rho_ss_vals, 'o-', color=color, markersize=2, lw=1, label=f'N={n}')
    
    ax.set_xlabel('Omega')
    ax.set_ylabel('rho_ss')
    ax.set_title('Phase Diagram: Steady-State Density vs. Rabi Frequency')
    ax.axhline(0.0, color='gray', ls='--', alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('System size N')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase_diagram_combined.png')
    plt.close()
    print(f"Saved: {output_dir / 'phase_diagram_combined.png'}")


def plot_sz_distribution(records, output_dir):
    """Histogram of all sz_mean values."""
    output_dir = Path(output_dir)
    
    all_sz = np.concatenate([r.sz_mean for r in records])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax = axes[0]
    ax.hist(all_sz, bins=100, color='steelblue', edgecolor='white', alpha=0.7)
    ax.set_xlabel('sz_mean')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of all sz_mean values')
    ax.axvline(-1.0, color='red', ls='--', alpha=0.5, label='Absorbing state')
    ax.legend()
    
    ax = axes[1]
    ax.hist(all_sz, bins=100, color='steelblue', edgecolor='white', alpha=0.7)
    ax.set_xlabel('sz_mean')
    ax.set_ylabel('Count (log)')
    ax.set_yscale('log')
    ax.set_title('Distribution of all sz_mean values (log scale)')
    ax.axvline(-1.0, color='red', ls='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sz_distribution.png')
    plt.close()
    print(f"Saved: {output_dir / 'sz_distribution.png'}")


def plot_data_coverage(records, output_dir):
    """Plot Omega coverage for each system size."""
    output_dir = Path(output_dir)
    
    sizes = sorted(set(r.n_atoms for r in records))
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_positions = range(len(sizes))
    
    for y, n in zip(y_positions, sizes):
        recs = [r for r in records if r.n_atoms == n]
        omegas = sorted([r.omega for r in recs])
        ax.scatter(omegas, [y] * len(omegas), s=30, alpha=0.7, label=f'N={n}')
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f'N={n}' for n in sizes])
    ax.set_xlabel('Omega')
    ax.set_title('Data Coverage: Omega Values per System Size')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'data_coverage.png')
    plt.close()
    print(f"Saved: {output_dir / 'data_coverage.png'}")


def print_summary_stats(records):
    """Print summary statistics."""
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    sizes = sorted(set(r.n_atoms for r in records))
    print(f"Total records: {len(records)}")
    print(f"System sizes: {sizes}")
    print(f"Time steps per trajectory: {len(records[0].t_save)}")
    print(f"Time range: [{records[0].t_save[0]:.1f}, {records[0].t_save[-1]:.1f}]")
    
    all_sz = np.concatenate([r.sz_mean for r in records])
    print(f"sz_mean range: [{all_sz.min():.4f}, {all_sz.max():.4f}]")
    print(f"sz_mean mean: {all_sz.mean():.4f}")
    print(f"sz_mean std: {all_sz.std():.4f}")
    
    # Count absorbing vs active
    n_absorbing = sum(1 for r in records if r.rho_ss < -0.95)
    n_active = sum(1 for r in records if r.rho_ss >= -0.95)
    print(f"Absorbing trajectories (rho_ss < -0.95): {n_absorbing}")
    print(f"Active trajectories (rho_ss >= -0.95): {n_active}")
    print()


def main():
    pkl_path = Path(__file__).parent.parent / 'outputs' / 'rydberg_dataset.pkl'
    output_dir = Path(__file__).parent.parent / 'outputs' / 'data_checks'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        records = pickle.load(f)
    
    print_summary_stats(records)
    check_anomalies(records)
    
    print("Generating plots...")
    plot_representative_trajectories(records, output_dir)
    plot_phase_diagram(records, output_dir)
    plot_combined_phase_diagram(records, output_dir)
    plot_sz_distribution(records, output_dir)
    plot_data_coverage(records, output_dir)
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
