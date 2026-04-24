"""
Comprehensive GP baseline evaluation report.
Generates detailed tables and metrics broken down by system size, 
driving strength, and phase region.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import joblib
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent / 'data'))
from parse_jld2 import TrajectoryRecord
from dataset import create_splits


def classify_phase(rho_ss, threshold=0.05):
    """Classify as absorbing (0) or active (1)."""
    return 1 if rho_ss >= threshold else 0


def evaluate_gp_detailed(gp, records, label=""):
    """Detailed evaluation with breakdowns."""
    
    # Overall metrics
    all_mse = []
    all_mae = []
    all_rho_ss_err = []
    all_true_rho = []
    all_pred_rho = []
    
    # Per-size metrics
    per_size = {}
    
    # Per-Omega metrics
    per_omega = {}
    
    # Phase classification
    true_phases = []
    pred_phases = []
    
    # Critical point estimation (per size)
    critical_points = {}
    
    for rec in records:
        N = rec.n_atoms
        omega = rec.omega
        
        X_test = np.array([[rec.omega, rec.n_atoms, t] for t in rec.t_save])
        y_pred = gp.predict(X_test)
        y_true = rec.sz_mean
        
        mse = np.mean((y_pred - y_true)**2)
        mae = np.mean(np.abs(y_pred - y_true))
        
        true_rho_ss = np.mean((y_true[-50:] + 1) / 2)
        pred_rho_ss = np.mean((y_pred[-50:] + 1) / 2)
        rho_ss_err = np.abs(pred_rho_ss - true_rho_ss)
        
        true_phase = classify_phase(true_rho_ss)
        pred_phase = classify_phase(pred_rho_ss)
        
        all_mse.append(mse)
        all_mae.append(mae)
        all_rho_ss_err.append(rho_ss_err)
        all_true_rho.append(true_rho_ss)
        all_pred_rho.append(pred_rho_ss)
        true_phases.append(true_phase)
        pred_phases.append(pred_phase)
        
        # Per-size
        if N not in per_size:
            per_size[N] = {'mse': [], 'mae': [], 'rho_err': [], 
                          'true_rho': [], 'pred_rho': [], 'omega': [],
                          'true_phase': [], 'pred_phase': []}
        per_size[N]['mse'].append(mse)
        per_size[N]['mae'].append(mae)
        per_size[N]['rho_err'].append(rho_ss_err)
        per_size[N]['true_rho'].append(true_rho_ss)
        per_size[N]['pred_rho'].append(pred_rho_ss)
        per_size[N]['omega'].append(omega)
        per_size[N]['true_phase'].append(true_phase)
        per_size[N]['pred_phase'].append(pred_phase)
        
        # Per-omega (binned)
        omega_bin = round(omega, 1)
        if omega_bin not in per_omega:
            per_omega[omega_bin] = {'mse': [], 'mae': [], 'rho_err': [],
                                   'true_rho': [], 'pred_rho': [], 'N': [],
                                   'true_phase': [], 'pred_phase': []}
        per_omega[omega_bin]['mse'].append(mse)
        per_omega[omega_bin]['mae'].append(mae)
        per_omega[omega_bin]['rho_err'].append(rho_ss_err)
        per_omega[omega_bin]['true_rho'].append(true_rho_ss)
        per_omega[omega_bin]['pred_rho'].append(pred_rho_ss)
        per_omega[omega_bin]['N'].append(N)
        per_omega[omega_bin]['true_phase'].append(true_phase)
        per_omega[omega_bin]['pred_phase'].append(pred_phase)
    
    all_true_rho = np.array(all_true_rho)
    all_pred_rho = np.array(all_pred_rho)
    true_phases = np.array(true_phases)
    pred_phases = np.array(pred_phases)
    
    # Overall
    r, _ = pearsonr(all_true_rho, all_pred_rho)
    phase_acc = np.mean(true_phases == pred_phases)
    
    print(f"\n{'='*70}")
    print(f"  GP BASELINE DETAILED REPORT — {label}")
    print(f"{'='*70}")
    
    print(f"\n📊 OVERALL METRICS ({len(records)} trajectories)")
    print(f"  MSE:        {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}")
    print(f"  MAE:        {np.mean(all_mae):.6f} ± {np.std(all_mae):.6f}")
    print(f"  ρ_ss MAE:   {np.mean(all_rho_ss_err):.6f} ± {np.std(all_rho_ss_err):.6f}")
    print(f"  Pearson r:  {r:.4f}")
    print(f"  Phase Acc:  {phase_acc:.1%} ({np.sum(true_phases==pred_phases)}/{len(true_phases)})")
    
    # ROC-like analysis
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(true_phases, all_pred_rho)
        print(f"  ROC AUC:    {auc:.4f}")
    except:
        print(f"  ROC AUC:    N/A (only one class present)")
    
    # Per-size breakdown
    print(f"\n📐 PER-SYSTEM-SIZE BREAKDOWN")
    print(f"  {'N':>8} {'N_traj':>8} {'MSE':>10} {'MAE':>10} {'ρ_err':>10} {'r':>8} {'PhaseAcc':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
    for N in sorted(per_size.keys()):
        d = per_size[N]
        mse = np.mean(d['mse'])
        mae = np.mean(d['mae'])
        rho_err = np.mean(d['rho_err'])
        r_n, _ = pearsonr(np.array(d['true_rho']), np.array(d['pred_rho']))
        phase_acc_n = np.mean(np.array(d['true_phase']) == np.array(d['pred_phase']))
        train_marker = "✓" if N in {225, 400, 900} else "✗"
        print(f"  {N:>8} {len(d['mse']):>8} {mse:>10.6f} {mae:>10.6f} {rho_err:>10.6f} {r_n:>8.3f} {phase_acc_n:>9.1%} {train_marker}")
    
    # Per-Omega breakdown (critical region)
    print(f"\n🔬 PER-DRIVING-STRENGTH BREAKDOWN (critical region Ω ≈ 10-13)")
    print(f"  {'Ω':>8} {'N_traj':>8} {'Trueρ':>10} {'Predρ':>10} {'ρ_err':>10} {'MSE':>10} {'PhaseAcc':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for omega in sorted(per_omega.keys()):
        if 9.5 <= omega <= 13.5:
            d = per_omega[omega]
            mse = np.mean(d['mse'])
            true_rho = np.mean(d['true_rho'])
            pred_rho = np.mean(d['pred_rho'])
            rho_err = np.mean(d['rho_err'])
            phase_acc_o = np.mean(np.array(d['true_phase']) == np.array(d['pred_phase']))
            print(f"  {omega:>8.1f} {len(d['mse']):>8} {true_rho:>10.4f} {pred_rho:>10.4f} {rho_err:>10.4f} {mse:>10.6f} {phase_acc_o:>9.1%}")
    
    # Phase transition analysis: find critical point per size
    print(f"\n🎯 CRITICAL POINT ESTIMATION (from GP predictions)")
    print(f"  {'N':>8} {'Ω_c(true)':>12} {'Ω_c(pred)':>12} {'Error':>10} {'Method':>20}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*20}")
    for N in sorted(per_size.keys()):
        d = per_size[N]
        omegas = np.array(d['omega'])
        true_rhos = np.array(d['true_rho'])
        pred_rhos = np.array(d['pred_rho'])
        
        if len(omegas) < 3:
            continue
        
        sort_idx = np.argsort(omegas)
        omegas = omegas[sort_idx]
        true_rhos = true_rhos[sort_idx]
        pred_rhos = pred_rhos[sort_idx]
        
        # True critical point: max derivative
        if len(omegas) > 1:
            true_dr = np.diff(true_rhos) / np.diff(omegas)
            true_omega_mid = (omegas[:-1] + omegas[1:]) / 2
            if len(true_dr) > 0:
                true_c_idx = np.argmax(true_dr)
                true_c = true_omega_mid[true_c_idx]
            else:
                true_c = np.nan
            
            pred_dr = np.diff(pred_rhos) / np.diff(omegas)
            pred_omega_mid = (omegas[:-1] + omegas[1:]) / 2
            if len(pred_dr) > 0:
                pred_c_idx = np.argmax(pred_dr)
                pred_c = pred_omega_mid[pred_c_idx]
            else:
                pred_c = np.nan
            
            if not np.isnan(true_c) and not np.isnan(pred_c):
                err = abs(pred_c - true_c)
                print(f"  {N:>8} {true_c:>12.2f} {pred_c:>12.2f} {err:>10.2f} {'max(dρ/dΩ)':>20}")
    
    # Worst cases
    print(f"\n🔴 WORST CASES (highest MSE)")
    worst_idx = np.argsort(all_mse)[-5:][::-1]
    print(f"  {'Rank':>6} {'N':>8} {'Ω':>10} {'MSE':>10} {'Trueρ':>10} {'Predρ':>10}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for rank, idx in enumerate(worst_idx, 1):
        rec = records[idx]
        print(f"  {rank:>6} {rec.n_atoms:>8} {rec.omega:>10.2f} {all_mse[idx]:>10.6f} {all_true_rho[idx]:>10.4f} {all_pred_rho[idx]:>10.4f}")
    
    # Best cases
    print(f"\n🟢 BEST CASES (lowest MSE)")
    best_idx = np.argsort(all_mse)[:5]
    print(f"  {'Rank':>6} {'N':>8} {'Ω':>10} {'MSE':>10} {'Trueρ':>10} {'Predρ':>10}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for rank, idx in enumerate(best_idx, 1):
        rec = records[idx]
        print(f"  {rank:>6} {rec.n_atoms:>8} {rec.omega:>10.2f} {all_mse[idx]:>10.6f} {all_true_rho[idx]:>10.4f} {all_pred_rho[idx]:>10.4f}")
    
    return {
        'overall_mse': np.mean(all_mse),
        'overall_mae': np.mean(all_mae),
        'overall_rho_err': np.mean(all_rho_ss_err),
        'pearson_r': r,
        'phase_acc': phase_acc,
        'per_size': per_size,
        'per_omega': per_omega,
    }


def main(args):
    with open(args.data_path, 'rb') as f:
        records = pickle.load(f)
    
    train_r, val_r, test_r = create_splits(records)
    
    print(f"Loading GP model from {args.model_path}...")
    gp = joblib.load(args.model_path)
    print(f"Kernel: {gp.kernel_}")
    
    # Evaluate on all sets
    train_metrics = evaluate_gp_detailed(gp, train_r, "TRAINING SET (Interpolation)")
    val_metrics = evaluate_gp_detailed(gp, val_r, "VALIDATION SET")
    test_metrics = evaluate_gp_detailed(gp, test_r, "TEST SET (Size Extrapolation)")
    
    # Summary comparison
    print(f"\n{'='*70}")
    print(f"  SUMMARY COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Metric':>15} {'Train':>12} {'Val':>12} {'Test':>12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'MSE':>15} {train_metrics['overall_mse']:>12.6f} {val_metrics['overall_mse']:>12.6f} {test_metrics['overall_mse']:>12.6f}")
    print(f"  {'MAE':>15} {train_metrics['overall_mae']:>12.6f} {val_metrics['overall_mae']:>12.6f} {test_metrics['overall_mae']:>12.6f}")
    print(f"  {'ρ_ss MAE':>15} {train_metrics['overall_rho_err']:>12.6f} {val_metrics['overall_rho_err']:>12.6f} {test_metrics['overall_rho_err']:>12.6f}")
    print(f"  {'Pearson r':>15} {train_metrics['pearson_r']:>12.4f} {val_metrics['pearson_r']:>12.4f} {test_metrics['pearson_r']:>12.4f}")
    print(f"  {'Phase Acc':>15} {train_metrics['phase_acc']:>11.1%} {val_metrics['phase_acc']:>11.1%} {test_metrics['phase_acc']:>11.1%}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='outputs/rydberg_dataset.pkl')
    parser.add_argument('--model_path', type=str, default='outputs/baselines/gp_model.pkl')
    args = parser.parse_args()
    main(args)
