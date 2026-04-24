"""PyTorch Dataset and data utilities for Rydberg facilitation dynamics."""

import pickle
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent))
from parse_jld2 import TrajectoryRecord


def worker_init_fn(worker_id):
    """Seed each DataLoader worker independently for reproducible augmentation."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for Rydberg facilitation trajectories."""
    
    def __init__(
        self,
        records: List,
        augment: bool = False,
        omega_jitter: float = 0.05,
        trajectory_noise_std: float = 0.005,
    ):
        self.records = records
        self.augment = augment
        self.omega_jitter = omega_jitter
        self.trajectory_noise_std = trajectory_noise_std
        
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        rec = self.records[idx]
        
        omega = rec.omega
        n_atoms = rec.n_atoms
        inv_sqrt_n = 1.0 / np.sqrt(n_atoms)
        t = rec.t_save.astype(np.float32)
        sz_mean = rec.sz_mean.astype(np.float32)
        
        # Data augmentation (physically valid only)
        if self.augment:
            omega = omega + random.gauss(0, self.omega_jitter)
            noise = np.random.normal(0, self.trajectory_noise_std, size=sz_mean.shape).astype(np.float32)
            sz_mean = sz_mean + noise
            # NOTE: No hard clipping — creates systematic bias in absorbing phase
        
        # Compute order parameter and steady-state values
        rho = (sz_mean + 1.0) / 2.0
        sz_ss = float(np.mean(sz_mean[-50:]))
        rho_ss = float(np.mean(rho[-50:]))  # CORRECT: use rho, not sz_mean
        
        return {
            'omega': torch.tensor(omega, dtype=torch.float32),
            'n_atoms': torch.tensor(n_atoms, dtype=torch.float32),
            'inv_sqrt_n': torch.tensor(inv_sqrt_n, dtype=torch.float32),
            't': torch.from_numpy(t),
            'sz_mean': torch.from_numpy(sz_mean),
            'rho': torch.from_numpy(rho),
            'rho_ss': torch.tensor(rho_ss, dtype=torch.float32),
            'sz_ss': torch.tensor(sz_ss, dtype=torch.float32),
        }


def create_splits(
    records: List,
    train_sizes: List[int] = None,
    val_size: int = 1225,
    val_omega_indices: str = 'even',
    test_sizes: List[int] = None,
) -> Tuple[List, List, List]:
    """Create parameter-based train/val/test splits."""
    assert val_omega_indices in ('even', 'odd'), \
        f"val_omega_indices must be 'even' or 'odd', got {val_omega_indices}"
    
    if train_sizes is None:
        train_sizes = [225, 400, 900]
    if test_sizes is None:
        test_sizes = [1600, 2500, 3025, 3600, 4900]
    
    train_records = [r for r in records if r.n_atoms in train_sizes]
    
    val_candidates = [r for r in records if r.n_atoms == val_size]
    val_candidates = sorted(val_candidates, key=lambda r: r.omega)
    
    if val_omega_indices == 'even':
        val_records = [r for i, r in enumerate(val_candidates) if i % 2 == 0]
        test_interpolation = [r for i, r in enumerate(val_candidates) if i % 2 == 1]
    else:
        val_records = [r for i, r in enumerate(val_candidates) if i % 2 == 1]
        test_interpolation = [r for i, r in enumerate(val_candidates) if i % 2 == 0]
    
    test_size_records = [r for r in records if r.n_atoms in test_sizes]
    test_records = test_interpolation + test_size_records
    
    print(f"Split summary:")
    print(f"  Train: {len(train_records)} records, sizes={sorted(set(r.n_atoms for r in train_records))}")
    print(f"  Val:   {len(val_records)} records, sizes={sorted(set(r.n_atoms for r in val_records))}")
    print(f"  Test:  {len(test_records)} records, sizes={sorted(set(r.n_atoms for r in test_records))}")
    
    return train_records, val_records, test_records


def load_dataset(pkl_path: str) -> List:
    """Load parsed records from pickle file."""
    with open(pkl_path, 'rb') as f:
        records = pickle.load(f)
    return records


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    omega = torch.stack([b['omega'] for b in batch])
    n_atoms = torch.stack([b['n_atoms'] for b in batch])
    inv_sqrt_n = torch.stack([b['inv_sqrt_n'] for b in batch])
    t = torch.stack([b['t'] for b in batch])
    sz_mean = torch.stack([b['sz_mean'] for b in batch])
    rho = torch.stack([b['rho'] for b in batch])
    rho_ss = torch.stack([b['rho_ss'] for b in batch])
    sz_ss = torch.stack([b['sz_ss'] for b in batch])
    
    return {
        'omega': omega,
        'n_atoms': n_atoms,
        'inv_sqrt_n': inv_sqrt_n,
        't': t,
        'sz_mean': sz_mean,
        'rho': rho,
        'rho_ss': rho_ss,
        'sz_ss': sz_ss,
    }


if __name__ == '__main__':
    pkl_path = Path(__file__).parent.parent / 'outputs' / 'rydberg_dataset.pkl'
    records = load_dataset(str(pkl_path))
    
    train_r, val_r, test_r = create_splits(records)
    
    train_ds = TrajectoryDataset(train_r, augment=True)
    val_ds = TrajectoryDataset(val_r, augment=False)
    
    print(f"\nTrain dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")
    
    sample = train_ds[0]
    print(f"\nSample keys: {sample.keys()}")
    for k, v in sample.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
