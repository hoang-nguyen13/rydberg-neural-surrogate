"""
Visualize GP baseline predictions with publication-quality figures.

Generates:
1. Trajectory comparison (predicted vs true) for sample test cases
2. Phase diagram: rho_ss vs Omega (predicted vs true)
3. Error distribution by system size
4. Size extrapolation panel
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent / 'data'))
from parse_jld2 import TrajectoryRecord
from dataset import create_splits


def plot_trajectory_comparison(gp, records, n_show=6, save_path=None):
    """Plot predicted vs true trajectories for sample test cases."""
    # Select diverse samples: mix of sizes and phases
    samples = []
    sizes = sorted(set(r.n_atoms for r in records))
    for size in sizes:
        size_recs = [r for r in records if r.n_atoms == size]
        if size_recs:
            # Pick one near critical and one far from critical
            size_recs_sorted = sorted(size_recs, key=lambda r: r.omega)
            samples.append(size_recs_sorted[len(size_recs_sorted)//3])
            if len(size_recs_sorted) > 1:
                samples.append(size_recs_sorted[2*len(size_recs_sorted)//3])
    
    samples = samples[:n_show]
    n_rows = (len(samples) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3*n_rows))
    axes = axes.flatten() if len(samples) > 1 else [axes]
    
    for idx, rec in enumerate(samples):
        ax = axes[idx]
        X_test = np.array([[rec.omega, rec.n_atoms, t] for t in rec.t_save])
        y_pred, y_std = gp.predict(X_test, return_std=True)
        y_true = rec.sz_mean
        
        ax.plot(rec.t_save, y_true, 'k-', linewidth=1.5, label='True (TWA)')
        ax.plot(rec.t_save, y_pred, 'r--', linewidth=1.5, label='GP prediction')
        ax.fill_between(rec.t_save, y_pred - 2*y_std, y_pred + 2*y_std, 
                        alpha=0.2, color='red', label='±2σ')
        
        # Mark steady-state region
        ax.axvspan(rec.t_save[-50], rec.t_save[-1], alpha=0.1, color='blue', label='Steady-state')
        
        mse = np.mean((y_pred - y_true)**2)
        ax.set_title(f'N={rec.n_atoms}, Ω={rec.omega:.2f}, MSE={mse:.4f}', fontsize=10)
        ax.set_xlabel('Time t/Γ')
        ax.set_ylabel('⟨S_z⟩')
        ax.set_ylim(-1.1, 1.1)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved trajectory comparison to {save_path}")
    plt.show()
    return fig


def plot_phase_diagram(gp, records, save_path=None):
    """Plot phase diagram: steady-state density vs driving strength."""
    # Collect true and predicted rho_ss
    results = {}
    for rec in records:
        N = rec.n_atoms
        if N not in results:
            results[N] = {'omega': [], 'true_rho': [], 'pred_rho': []}
        
        X_test = np.array([[rec.omega, rec.n_atoms, t] for t in rec.t_save])
        y_pred = gp.predict(X_test)
        
        true_rho_ss = np.mean((rec.sz_mean[-50:] + 1) / 2)
        pred_rho_ss = np.mean((y_pred[-50:] + 1) / 2)
        
        results[N]['omega'].append(rec.omega)
        results[N]['true_rho'].append(true_rho_ss)
        results[N]['pred_rho'].append(pred_rho_ss)
    
    n_sizes = len(results)
    fig, axes = plt.subplots(1, n_sizes, figsize=(4*n_sizes, 4), sharey=True)
    if n_sizes == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_sizes))
    
    for idx, (N, data) in enumerate(sorted(results.items())):
        ax = axes[idx]
        sort_idx = np.argsort(data['omega'])
        omega = np.array(data['omega'])[sort_idx]
        true_rho = np.array(data['true_rho'])[sort_idx]
        pred_rho = np.array(data['pred_rho'])[sort_idx]
        
        ax.plot(omega, true_rho, 'ko-', markersize=4, linewidth=1.5, label='True')
        ax.plot(omega, pred_rho, 'rs--', markersize=4, linewidth=1.5, label='GP')
        
        ax.axhline(y=0.05, color='gray', linestyle=':', alpha=0.5, label='Phase threshold')
        ax.set_xlabel('Driving strength Ω')
        ax.set_ylabel('Steady-state density ρ_ss')
        ax.set_title(f'N = {N}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    
    plt.suptitle('Phase Diagram: GP vs True (TWA)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved phase diagram to {save_path}")
    plt.show()
    return fig


def plot_error_by_size(gp, records, save_path=None):
    """Box plot of MSE and rho_ss error by system size."""
    errors = {}
    for rec in records:
        N = rec.n_atoms
        if N not in errors:
            errors[N] = {'mse': [], 'rho_err': [], 'mae': []}
        
        X_test = np.array([[rec.omega, rec.n_atoms, t] for t in rec.t_save])
        y_pred = gp.predict(X_test)
        y_true = rec.sz_mean
        
        mse = np.mean((y_pred - y_true)**2)
        mae = np.mean(np.abs(y_pred - y_true))
        
        pred_rho_ss = np.mean((y_pred[-50:] + 1) / 2)
        true_rho_ss = np.mean((y_true[-50:] + 1) / 2)
        rho_err = np.abs(pred_rho_ss - true_rho_ss)
        
        errors[N]['mse'].append(mse)
        errors[N]['mae'].append(mae)
        errors[N]['rho_err'].append(rho_err)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    sizes = sorted(errors.keys())
    mse_data = [errors[N]['mse'] for N in sizes]
    mae_data = [errors[N]['mae'] for N in sizes]
    rho_data = [errors[N]['rho_err'] for N in sizes]
    
    # Color by seen vs unseen
    train_sizes = {225, 400, 900}
    colors = ['#2ecc71' if N in train_sizes else '#e74c3c' for N in sizes]
    
    bp1 = axes[0].boxplot(mse_data, labels=[f'N={N}' for N in sizes], patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_ylabel('MSE')
    axes[0].set_title('MSE by System Size')
    axes[0].grid(True, alpha=0.3)
    
    bp2 = axes[1].boxplot(mae_data, labels=[f'N={N}' for N in sizes], patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1].set_ylabel('MAE')
    axes[1].set_title('MAE by System Size')
    axes[1].grid(True, alpha=0.3)
    
    bp3 = axes[2].boxplot(rho_data, labels=[f'N={N}' for N in sizes], patch_artist=True)
    for patch, color in zip(bp3['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[2].set_ylabel('|Δρ_ss|')
    axes[2].set_title('Steady-State Density Error')
    axes[2].grid(True, alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.6, label='Training sizes'),
        Patch(facecolor='#e74c3c', alpha=0.6, label='Unseen sizes')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.02))
    
    plt.suptitle('GP Prediction Errors by System Size', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved error analysis to {save_path}")
    plt.show()
    return fig


def plot_scatter_predictions(gp, records, save_path=None):
    """Scatter plot: predicted vs true rho_ss."""
    true_vals = []
    pred_vals = []
    sizes = []
    
    for rec in records:
        X_test = np.array([[rec.omega, rec.n_atoms, t] for t in rec.t_save])
        y_pred = gp.predict(X_test)
        
        true_rho_ss = np.mean((rec.sz_mean[-50:] + 1) / 2)
        pred_rho_ss = np.mean((y_pred[-50:] + 1) / 2)
        
        true_vals.append(true_rho_ss)
        pred_vals.append(pred_rho_ss)
        sizes.append(rec.n_atoms)
    
    true_vals = np.array(true_vals)
    pred_vals = np.array(pred_vals)
    sizes = np.array(sizes)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Color by size
    scatter = ax.scatter(true_vals, pred_vals, c=sizes, cmap='viridis', 
                         s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    max_val = max(true_vals.max(), pred_vals.max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='Perfect prediction')
    
    # ±10% error bands
    ax.fill_between([0, max_val], [0, 0.9*max_val], [0, 1.1*max_val], 
                    alpha=0.1, color='red', label='±10%')
    
    ax.set_xlabel('True ρ_ss (TWA)', fontsize=12)
    ax.set_ylabel('Predicted ρ_ss (GP)', fontsize=12)
    ax.set_title('Steady-State Density: Predicted vs True', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, max_val + 0.05)
    ax.set_ylim(-0.05, max_val + 0.05)
    ax.set_aspect('equal')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('System size N', fontsize=11)
    
    # Add Pearson correlation
    from scipy.stats import pearsonr
    r, _ = pearsonr(true_vals, pred_vals)
    ax.text(0.05, 0.95, f'Pearson r = {r:.3f}', transform=ax.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved scatter plot to {save_path}")
    plt.show()
    return fig


def main(args):
    # Load data
    with open(args.data_path, 'rb') as f:
        records = pickle.load(f)
    
    # Create splits
    train_r, val_r, test_r = create_splits(records)
    
    # Load GP model
    print(f"Loading GP model from {args.model_path}...")
    gp = joblib.load(args.model_path)
    print(f"Model loaded. Kernel: {gp.kernel_}")
    
    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    print("\n=== Generating Plots ===\n")
    
    print("1. Trajectory comparison (test set)...")
    plot_trajectory_comparison(gp, test_r, n_show=6, 
                                save_path=out_dir / 'gp_trajectories_test.png')
    
    print("\n2. Phase diagram (test set)...")
    plot_phase_diagram(gp, test_r, save_path=out_dir / 'gp_phase_diagram_test.png')
    
    print("\n3. Error by system size (test set)...")
    plot_error_by_size(gp, test_r, save_path=out_dir / 'gp_errors_by_size_test.png')
    
    print("\n4. Scatter: predicted vs true ρ_ss (test set)...")
    plot_scatter_predictions(gp, test_r, save_path=out_dir / 'gp_scatter_test.png')
    
    # Also evaluate on training set to show interpolation quality
    print("\n=== Training Set (Interpolation) ===\n")
    
    print("5. Phase diagram (train set)...")
    plot_phase_diagram(gp, train_r, save_path=out_dir / 'gp_phase_diagram_train.png')
    
    print("\n6. Scatter: predicted vs true ρ_ss (train set)...")
    plot_scatter_predictions(gp, train_r, save_path=out_dir / 'gp_scatter_train.png')
    
    print(f"\n=== All plots saved to {out_dir}/ ===")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='outputs/rydberg_dataset.pkl')
    parser.add_argument('--model_path', type=str, default='outputs/baselines/gp_model.pkl')
    parser.add_argument('--output_dir', type=str, default='outputs/baselines/plots')
    args = parser.parse_args()
    main(args)
