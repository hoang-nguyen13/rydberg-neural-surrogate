"""
PyTorch Dataset for multi-parameter Rydberg facilitation dynamics.

Supports: dimension (1D/2D/3D), system size N, driving Ω, dephasing γ.
Splits designed for multiple generalization experiments:
  1. Size extrapolation within 2D γ=0.1
  2. γ-generalization (quantum → classical)
  3. Dimension generalization (2D → 1D/3D)
"""

import random
import pickle
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent))
from parse_jld2_v2 import TrajectoryRecord


class TrajectoryDataset(Dataset):
    """Dataset returning trajectories with full parameter conditioning."""
    
    def __init__(self, records, augment=True, omega_jitter=0.05, 
                 traj_noise_std=0.005, seed=None):
        self.records = records
        self.augment = augment
        self.omega_jitter = omega_jitter
        self.traj_noise_std = traj_noise_std
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        rec = self.records[idx]
        
        omega = rec.omega
        n_atoms = rec.n_atoms
        gamma = rec.gamma
        dimension = rec.dimension
        
        if self.augment:
            omega = omega + np.random.normal(0, self.omega_jitter)
        
        t = torch.tensor(rec.t_save, dtype=torch.float32)
        sz_mean = torch.tensor(rec.sz_mean, dtype=torch.float32)
        
        if self.augment and self.traj_noise_std > 0:
            noise = torch.randn_like(sz_mean) * self.traj_noise_std
            sz_mean = sz_mean + noise
        
        return {
            'omega': torch.tensor(omega, dtype=torch.float32),
            'n_atoms': torch.tensor(n_atoms, dtype=torch.float32),
            'inv_sqrt_n': torch.tensor(1.0 / np.sqrt(n_atoms), dtype=torch.float32),
            'gamma': torch.tensor(np.log10(gamma + 1e-10), dtype=torch.float32),  # log scale
            'dimension': torch.tensor(dimension, dtype=torch.float32),
            't': t,
            'sz_mean': sz_mean,
            'rho': torch.tensor(rec.rho, dtype=torch.float32),
            'rho_ss': torch.tensor(rec.rho_ss, dtype=torch.float32),
        }


def collate_fn(batch):
    """Pad/truncate to same time length and stack."""
    max_len = max(b['t'].shape[0] for b in batch)
    
    out = {}
    for key in ['omega', 'n_atoms', 'inv_sqrt_n', 'gamma', 'dimension', 'rho_ss']:
        out[key] = torch.stack([b[key] for b in batch])
    
    # Pad time-dependent arrays
    for key in ['t', 'sz_mean', 'rho']:
        padded = []
        for b in batch:
            val = b[key]
            if val.shape[0] < max_len:
                pad = torch.zeros(max_len - val.shape[0], dtype=val.dtype)
                val = torch.cat([val, pad])
            padded.append(val)
        out[key] = torch.stack(padded)
    
    return out


def create_splits(records: List, omega_selected=None) -> Tuple[List, List, Dict[str, List]]:
    """
    Create splits for multiple generalization experiments.
    
    Returns: (train_records, val_records, test_sets_dict)
    """
    import numpy as np
    if omega_selected is None:
        omega_selected = [round(x, 8) for x in np.arange(10.0, 13.01, 0.15)]
    
    # Sort records for deterministic splitting
    records = sorted(records, key=lambda r: (r.dimension, r.gamma, r.n_atoms, r.omega))
    
    train = []
    val = []
    test_sets = {}
    
    # === PRIMARY: 2D, γ=0.1, SELECTED Ω only ===
    recs_2d_q = [r for r in records if r.dimension == 2 and abs(r.gamma - 0.1) < 1e-6]
    # Filter to selected Ω values (the ones in our plots)
    recs_2d_q = [r for r in recs_2d_q if round(r.omega, 8) in omega_selected]
    
    # Training: N = 225, 400, 900, 1600, 2500
    train_2d = [r for r in recs_2d_q if r.n_atoms in {225, 400, 900, 1600, 2500}]
    
    # Validation: N = 3600
    val_2d = [r for r in recs_2d_q if r.n_atoms == 3600]
    
    # Test: N = 4900 only (6400/10000/21025 excluded — insufficient data)
    test_sets['size_extrapolation_2d'] = [
        r for r in recs_2d_q if r.n_atoms == 4900
    ]
    
    train.extend(train_2d)
    val.extend(val_2d)
    
    # === γ-generalization (2D, N=3600, ALL γ values) ===
    # Test ALL dephasing rates at fixed N=3600
    recs_2d_gamma = [r for r in records if r.dimension == 2 and r.n_atoms == 3600]
    # Filter to selected Ω for γ=0.1; keep all available Ω for other γ
    gamma_test = []
    for r in recs_2d_gamma:
        if r.gamma == 0.1:
            if round(r.omega, 8) in omega_selected:
                gamma_test.append(r)
        else:
            gamma_test.append(r)
    test_sets['gamma_generalization_2d'] = gamma_test
    
    # Print summary
    print("=" * 70)
    print("DATASET SPLITS")
    print("=" * 70)
    print(f"\nTrain: {len(train)} trajectories")
    train_summary = {}
    for r in train:
        key = f"{r.dimension}D, N={r.n_atoms}, γ={r.gamma}"
        train_summary[key] = train_summary.get(key, 0) + 1
    for key, count in sorted(train_summary.items()):
        print(f"  {key}: {count}")
    
    print(f"\nValidation: {len(val)} trajectories")
    val_summary = {}
    for r in val:
        key = f"{r.dimension}D, N={r.n_atoms}, γ={r.gamma}"
        val_summary[key] = val_summary.get(key, 0) + 1
    for key, count in sorted(val_summary.items()):
        print(f"  {key}: {count}")
    
    print(f"\nTest sets:")
    for name, recs in test_sets.items():
        print(f"\n  {name}: {len(recs)} trajectories")
        test_summary = {}
        for r in recs:
            key = f"{r.dimension}D, N={r.n_atoms}, γ={r.gamma}"
            test_summary[key] = test_summary.get(key, 0) + 1
        for key, count in sorted(test_summary.items()):
            print(f"    {key}: {count}")
    
    total = len(train) + len(val) + sum(len(v) for v in test_sets.values())
    print(f"\nTotal used: {total} / {len(records)} trajectories")
    print(f"Unused: {len(records) - total} trajectories")
    print("=" * 70)
    
    return train, val, test_sets


def load_dataset(path: str):
    """Load pickled dataset."""
    with open(path, 'rb') as f:
        records = pickle.load(f)
    return records


def worker_init_fn(worker_id):
    """Seed RNG per DataLoader worker."""
    np.random.seed(np.random.get_state()[1][0] + worker_id)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/rydberg_dataset_v2.pkl')
    args = parser.parse_args()
    
    records = load_dataset(args.data_path)
    train, val, test_sets = create_splits(records)
