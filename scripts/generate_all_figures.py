"""
Generate comprehensive paper figures comparing Transformer, FNO, and Ensemble.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import torch.nn.functional as F

from dataset_v2 import TrajectoryDataset, create_splits, load_dataset, collate_fn
from transformer_surrogate import RydbergSurrogate
from parse_jld2_v2 import TrajectoryRecord

# Device
device = torch.device('cpu')

# Load dataset
records = load_dataset('data/rydberg_dataset_v2.pkl')
train_r, val_r, test_sets = create_splits(records)

# Colors
C_TRANS = '#1f77b4'
C_FNO = '#ff7f0e'
C_ENS = '#2ca02c'
C_TRUE = '#d62728'

def load_model(path, model_cls, **kwargs):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if model_cls == RydbergSurrogate:
        model_args = checkpoint.get('args', {})
        model = RydbergSurrogate(
            n_layer=model_args.get('n_layer', 4),
            n_head=model_args.get('n_head', 4),
            n_embd=model_args.get('n_embd', 96),
            n_time=400,
            dropout=model_args.get('dropout', 0.2),
            mlp_ratio=model_args.get('mlp_ratio', 2),
        )
    else:
        model = model_cls(**kwargs)
    state_dict = checkpoint['model']
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('_')}
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model


def predict_trajectories(model, records_list, is_transformer=True):
    """Generate predictions for a list of records."""
    ds = TrajectoryDataset(records_list, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    results = []
    with torch.no_grad():
        for batch in loader:
            omega = batch['omega'].to(device)
            n_atoms = batch['n_atoms'].to(device)
            inv_sqrt_n = batch['inv_sqrt_n'].to(device)
            gamma = batch['gamma'].to(device)
            dimension = batch['dimension'].to(device)
            t = batch['t'].to(device)
            sz_true = batch['sz_mean'].to(device)
            
            pred = model(omega, n_atoms, inv_sqrt_n, gamma, dimension, t)
            
            for i in range(len(omega)):
                valid = t[i] > 0 if t[i][0] == 0 else t[i] >= 0
                t_valid = t[i][t[i] > 0].cpu().numpy() if t[i][0] == 0 else t[i].cpu().numpy()
                results.append({
                    'omega': omega[i].item(),
                    'n_atoms': int(n_atoms[i].item()),
                    'gamma': gamma[i].item(),
                    'dimension': int(dimension[i].item()),
                    't': t[i].cpu().numpy(),
                    'sz_true': sz_true[i].cpu().numpy(),
                    'sz_pred': pred[i].cpu().numpy(),
                })
    return results


# Load Transformer
print("Loading Transformer...")
trans_model = load_model('outputs/models/best_model.pt', RydbergSurrogate)

# Toggle baselines on/off
SHOW_FNO = False
SHOW_ENSEMBLE = False

# Load FNO
HAS_FNO = False
if SHOW_FNO:
    try:
        from baselines.fno_baseline_v2 import FNOBaselineV2
        print("Loading FNO...")
        fno_checkpoint = torch.load('outputs/models/fno/fno_best.pt', map_location=device, weights_only=False)
        fno_model = FNOBaselineV2(
            n_modes=(fno_checkpoint['args'].get('n_modes', 32),),
            hidden_channels=fno_checkpoint['args'].get('hidden_channels', 64),
            n_layers=fno_checkpoint['args'].get('n_layers', 4),
        ).to(device)
        fno_state = {k: v for k, v in fno_checkpoint['model'].items() if not k.startswith('_')}
        fno_model.load_state_dict(fno_state, strict=False)
        fno_model.eval()
        HAS_FNO = True
    except Exception as e:
        print(f"FNO load failed: {e}")

# Load Ensemble
ensemble_models = []
if SHOW_ENSEMBLE:
    print("Loading Ensemble...")
    for seed in [42, 43, 44, 45, 46]:
        try:
            m = load_model(f'outputs/ensemble/model_seed{seed}/best_model.pt', RydbergSurrogate)
            ensemble_models.append(m)
        except Exception as e:
            print(f"  Seed {seed} failed: {e}")
    print(f"Loaded {len(ensemble_models)} ensemble models")

# Generate predictions on test sets
print("\nGenerating predictions...")

SKIP_GAMMA = True  # Exclude gamma generalization from paper figures
for test_name, test_records in test_sets.items():
    if SKIP_GAMMA and test_name == 'gamma_generalization_2d':
        continue
    print(f"\n{test_name}: {len(test_records)} records")
    
    # Transformer
    trans_preds = predict_trajectories(trans_model, test_records)
    
    # FNO
    if HAS_FNO:
        fno_preds = predict_trajectories(fno_model, test_records, is_transformer=False)
    
    # Ensemble
    ens_preds_all = []
    for m in ensemble_models:
        ens_preds_all.append(predict_trajectories(m, test_records))
    
    # === FIGURE 1: Trajectory overlays for a few examples ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    selected = [0, len(test_records)//3, 2*len(test_records)//3, len(test_records)-1]
    if len(test_records) < 4:
        selected = list(range(min(6, len(test_records))))
    
    for idx, rec_idx in enumerate(selected[:6]):
        ax = axes[idx]
        r = trans_preds[rec_idx]
        t = r['t']
        
        ax.plot(t, r['sz_true'], '-', color=C_TRUE, linewidth=2, label='TWA', alpha=0.8)
        ax.plot(t, r['sz_pred'], '--', color=C_TRANS, linewidth=1.5, label='Transformer')
        
        if HAS_FNO:
            ax.plot(t, fno_preds[rec_idx]['sz_pred'], ':', color=C_FNO, linewidth=1.5, label='FNO')
        
        # Ensemble mean and std
        if ens_preds_all:
            ens_stack = np.stack([ens_preds_all[i][rec_idx]['sz_pred'] for i in range(len(ens_preds_all))])
            ens_mean = ens_stack.mean(axis=0)
            ens_std = ens_stack.std(axis=0)
            ax.plot(t, ens_mean, '-.', color=C_ENS, linewidth=1.5, label='Ensemble mean')
            ax.fill_between(t, ens_mean - ens_std, ens_mean + ens_std, alpha=0.2, color=C_ENS)
        
        ax.set_title(f'Ω={r["omega"]:.2f}, N={r["n_atoms"]}, γ={10**r["gamma"]:.1f}')
        ax.set_xlabel('t')
        ax.set_ylabel(r'$\langle S_z \rangle$')
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle(f'{test_name.replace("_", " ").title()} — Trajectory Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'outputs/evaluation/comparison_trajectories_{test_name}.png', dpi=150)
    plt.close()
    print(f"  Saved comparison_trajectories_{test_name}.png")
    
    # === FIGURE 2: ρ_ss vs Ω phase diagram ===
    fig, ax = plt.subplots(figsize=(10, 6))
    
    trans_omegas = [r['omega'] for r in trans_preds]
    trans_rho_ss = [(r['sz_true'][-50:].mean() + 1) / 2 for r in trans_preds]
    trans_rho_ss_pred = [(r['sz_pred'][-50:].mean() + 1) / 2 for r in trans_preds]
    
    ax.scatter(trans_omegas, trans_rho_ss, color=C_TRUE, s=60, alpha=0.7, label='TWA', zorder=5)
    ax.scatter(trans_omegas, trans_rho_ss_pred, color=C_TRANS, s=60, marker='x', alpha=0.7, label='Transformer')
    
    if HAS_FNO:
        fno_rho_ss_pred = [(r['sz_pred'][-50:].mean() + 1) / 2 for r in fno_preds]
        ax.scatter(trans_omegas, fno_rho_ss_pred, color=C_FNO, s=40, marker='^', alpha=0.6, label='FNO')
    
    if ens_preds_all:
        ens_rho_ss_all = []
        for i in range(len(ens_preds_all)):
            ens_rho_ss_all.append([(r['sz_pred'][-50:].mean() + 1) / 2 for r in ens_preds_all[i]])
        ens_rho_ss_mean = np.mean(ens_rho_ss_all, axis=0)
        ens_rho_ss_std = np.std(ens_rho_ss_all, axis=0)
        ax.errorbar(trans_omegas, ens_rho_ss_mean, yerr=ens_rho_ss_std, fmt='o', color=C_ENS, 
                   markersize=4, capsize=3, alpha=0.7, label='Ensemble')
    
    ax.axhline(0.05, color='gray', linestyle='--', alpha=0.5, label='ρ_ss = 0.05')
    ax.set_xlabel('Ω', fontsize=12)
    ax.set_ylabel(r'$\rho_{ss}$', fontsize=12)
    ax.set_title(f'{test_name.replace("_", " ").title()} — Phase Diagram', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'outputs/evaluation/comparison_phase_{test_name}.png', dpi=150)
    plt.close()
    print(f"  Saved comparison_phase_{test_name}.png")

# === FIGURE 3: (removed error comparison, kept for future use) ===

# === FIGURE 4: Ensemble spread visualization (only if ensemble loaded) ===
if SHOW_ENSEMBLE and ensemble_models:
    print("\nGenerating ensemble spread figure...")
    
    test_name = 'size_extrapolation_2d'
    test_records = test_sets[test_name]
    ens_preds_all = [predict_trajectories(m, test_records) for m in ensemble_models]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx in range(min(6, len(test_records))):
        ax = axes[idx]
        r = ens_preds_all[0][idx]
        t = r['t']
        
        # True
        ax.plot(t, r['sz_true'], '-', color=C_TRUE, linewidth=2.5, label='TWA', zorder=5)
        
        # Individual ensemble members
        for i, preds in enumerate(ens_preds_all):
            ax.plot(t, preds[idx]['sz_pred'], '-', color=C_ENS, alpha=0.3, linewidth=0.8)
        
        # Ensemble mean and std
        ens_stack = np.stack([ens_preds_all[i][idx]['sz_pred'] for i in range(len(ens_preds_all))])
        ens_mean = ens_stack.mean(axis=0)
        ens_std = ens_stack.std(axis=0)
        ax.plot(t, ens_mean, '-', color=C_ENS, linewidth=2, label='Ensemble mean')
        ax.fill_between(t, ens_mean - 2*ens_std, ens_mean + 2*ens_std, alpha=0.2, color=C_ENS, label='±2σ')
        
        ax.set_title(f'Ω={r["omega"]:.2f}, N={r["n_atoms"]}')
        ax.set_xlabel('t')
        ax.set_ylabel(r'$\langle S_z \rangle$')
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Ensemble Uncertainty Quantification (N=4900)', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/evaluation/ensemble_spread.png', dpi=150)
    plt.close()
    print("Saved ensemble_spread.png")
else:
    print("\nSkipping ensemble spread figure (ensemble not loaded).")

print("\nAll figures generated in outputs/evaluation/")
