"""
Parse ALL JLD2 files from Downloads/Rydberg_facilitation.

Extracts: Ω, Δ, γ, N, dimension, sz_mean, tSave
Handles: 1D, 2D, 3D data with mixed γ values and system sizes.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np


@dataclass
class TrajectoryRecord:
    omega: float
    delta: float
    gamma: float
    n_atoms: int
    dimension: int
    t_save: np.ndarray
    sz_mean: np.ndarray
    
    @property
    def rho(self):
        """Order parameter: ρ = (sz + 1) / 2."""
        return (self.sz_mean + 1.0) / 2.0
    
    @property
    def rho_ss(self):
        """Steady-state density: mean of last 50 time points."""
        return float(np.mean(self.rho[-50:]))
    
    @property
    def inv_sqrt_n(self):
        return 1.0 / np.sqrt(self.n_atoms)


def parse_filename(fname: str) -> Optional[dict]:
    """Parse parameters from filename like 'ρ_ss_2D,Ω=10.15,Δ=2000.0,γ=0.1.jld2'."""
    # Remove extension
    base = fname.replace('.jld2', '')
    
    # Extract dimension: ρ_ss_2D or ρ_ss_1D or ρ_ss_3D
    dim_match = re.search(r'ρ_ss_(\d)D', base)
    if not dim_match:
        return None
    dimension = int(dim_match.group(1))
    
    # Extract Ω
    omega_match = re.search(r'Ω=([0-9]+(?:\.[0-9]+)?)', base)
    if not omega_match:
        return None
    omega = float(omega_match.group(1))
    
    # Extract Δ
    delta_match = re.search(r'Δ=([0-9]+(?:\.[0-9]+)?)', base)
    if not delta_match:
        return None
    delta = float(delta_match.group(1))
    
    # Extract γ (handles scientific notation like 1.0e-5)
    gamma_match = re.search(r'γ=([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)', base)
    if not gamma_match:
        return None
    gamma = float(gamma_match.group(1))
    
    return {
        'dimension': dimension,
        'omega': omega,
        'delta': delta,
        'gamma': gamma,
    }


def parse_directory_name(dname: str) -> Optional[dict]:
    """Parse parameters from directory name like 'atoms=225,Δ=2000.0,γ=0.1'."""
    n_match = re.search(r'atoms=(\d+)', dname)
    if not n_match:
        return None
    n_atoms = int(n_match.group(1))
    
    delta_match = re.search(r'Δ=([0-9]+(?:\.[0-9]+)?)', dname)
    if not delta_match:
        return None
    delta = float(delta_match.group(1))
    
    gamma_match = re.search(r'γ=([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)', dname)
    if not gamma_match:
        return None
    gamma = float(gamma_match.group(1))
    
    return {
        'n_atoms': n_atoms,
        'delta': delta,
        'gamma': gamma,
    }


def parse_jld2_file(path: Path) -> Optional[TrajectoryRecord]:
    """Read a single JLD2 file and return TrajectoryRecord."""
    try:
        with h5py.File(path, 'r') as h:
            if 'sz_mean' not in h or 'tSave' not in h:
                return None
            
            sz_mean = np.array(h['sz_mean'])
            t_save = np.array(h['tSave'])
            
            if len(sz_mean) == 0 or len(t_save) == 0:
                return None
            
            # Validate shape
            if sz_mean.ndim != 1 or t_save.ndim != 1:
                return None
            
            # Parse filename
            fname_params = parse_filename(path.name)
            if fname_params is None:
                return None
            
            # Parse directory
            dir_params = parse_directory_name(path.parent.name)
            if dir_params is None:
                return None
            
            # Cross-check: filename and directory should agree on gamma and delta
            if abs(fname_params['gamma'] - dir_params['gamma']) > 1e-10:
                # This shouldn't happen, but handle gracefully
                pass
            if abs(fname_params['delta'] - dir_params['delta']) > 1e-10:
                pass
            
            return TrajectoryRecord(
                omega=fname_params['omega'],
                delta=dir_params['delta'],
                gamma=dir_params['gamma'],
                n_atoms=dir_params['n_atoms'],
                dimension=fname_params['dimension'],
                t_save=t_save,
                sz_mean=sz_mean.astype(np.float32),
            )
    except Exception as e:
        return None


def parse_all_data(data_root: str) -> List[TrajectoryRecord]:
    """Walk data_root and parse all readable JLD2 files."""
    data_root = Path(data_root)
    records = []
    failed = []
    
    for root, dirs, files in os.walk(data_root):
        for fname in files:
            if not fname.endswith('.jld2'):
                continue
            path = Path(root) / fname
            rec = parse_jld2_file(path)
            if rec is not None:
                records.append(rec)
            else:
                failed.append(str(path))
    
    print(f"Parsed {len(records)} files successfully.")
    if failed:
        print(f"Failed to parse {len(failed)} files.")
        for f in failed[:5]:
            print(f"  - {f}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
    
    return records


def summarize_records(records: List[TrajectoryRecord]):
    """Print summary of parsed records."""
    from collections import Counter
    
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total trajectories: {len(records)}")
    
    # By dimension
    dims = Counter(r.dimension for r in records)
    print(f"\nBy dimension:")
    for d in sorted(dims.keys()):
        print(f"  {d}D: {dims[d]} trajectories")
    
    # By gamma
    gammas = Counter(r.gamma for r in records)
    print(f"\nBy γ (dephasing rate):")
    for g in sorted(gammas.keys()):
        print(f"  γ={g}: {gammas[g]} trajectories")
    
    # By (dimension, N, gamma)
    print(f"\nBy (dimension, N, γ):")
    triples = Counter((r.dimension, r.n_atoms, r.gamma) for r in records)
    for (dim, N, g), count in sorted(triples.items()):
        print(f"  {dim}D, N={N:>6}, γ={g:>8.2e}: {count:>3} trajectories")
    
    # Omega ranges
    print(f"\nΩ ranges per (dimension, N, γ):")
    from itertools import groupby
    keyfunc = lambda r: (r.dimension, r.n_atoms, r.gamma)
    for key, group in groupby(sorted(records, key=keyfunc), key=keyfunc):
        omegas = sorted([r.omega for r in group])
        print(f"  {key[0]}D, N={key[1]:>6}, γ={key[2]:>8.2e}: Ω={omegas[0]:.2f} to {omegas[-1]:.2f} ({len(omegas)} points)")
    
    # Delta values
    deltas = set(r.delta for r in records)
    print(f"\nΔ values: {sorted(deltas)}")
    
    # Time array shape
    t_lengths = Counter(len(r.t_save) for r in records)
    print(f"\nTime array lengths: {dict(t_lengths)}")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, 
                        default='/Users/hoangnguyen/Downloads/Rydberg_facilitation/results_data_mean',
                        help='Root directory containing JLD2 files')
    parser.add_argument('--output', type=str, default='data/rydberg_dataset_v2.pkl',
                        help='Output pickle file path')
    args = parser.parse_args()
    
    records = parse_all_data(args.data_root)
    summarize_records(records)
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(records, f)
    print(f"Saved {len(records)} records to {args.output}")
