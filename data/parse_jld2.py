"""
Parse Rydberg facilitation JLD2 files into a clean dataset.

Reads all JLD2 files from Rydberg_facilitation/results_data_mean/,
extracts sz_mean(t) and tSave, parses parameters from filenames,
and saves a consolidated dataset as a pickle file.

Known issues: N=1089 (6 files), N=3481 (3 files), N=9 (3 files) have
JLD2 encoding errors and are skipped.
"""

import os
import re
import pickle
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import h5py

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

N_TIME_STEPS = 400  # Expected number of time steps per trajectory
T_MAX = 1000.0      # Final simulation time


@dataclass
class TrajectoryRecord:
    """A single parsed trajectory record."""
    omega: float
    delta: float
    gamma: float
    V: float
    Gamma: float
    n_atoms: int
    lattice_size: int
    dimension: int
    sz_mean: np.ndarray
    t_save: np.ndarray
    rho: np.ndarray
    rho_ss: float
    sz_ss: float
    file_path: str


def parse_directory_name(dir_name: str) -> dict:
    """Parse parameters from directory name like 'atoms=1225,=2000.0,=0.1'."""
    m = re.search(r'atoms=(\d+)', dir_name)
    n_atoms = int(m.group(1)) if m else None
    
    m = re.search(r'\u0394=([0-9]+(?:\.[0-9]+)?)', dir_name)
    delta = float(m.group(1)) if m else None
    
    m = re.search(r'\u03B3=([0-9]+(?:\.[0-9]+)?)', dir_name)
    gamma = float(m.group(1)) if m else None
    
    return {'n_atoms': n_atoms, 'delta': delta, 'gamma': gamma}


def parse_filename(fname: str) -> dict:
    """Parse parameters from filename like 'ss_2D,=10.15,=2000.0,=0.1.jld2'."""
    m = re.search(r'ss_(\d)D', fname)
    dimension = int(m.group(1)) if m else 2
    
    # CRITICAL FIX: Use explicit number format [0-9]+(?:\.[0-9]+)?
    # This correctly matches integers (10) and decimals (10.15, 0.1)
    m = re.search(r'\u03A9=([0-9]+(?:\.[0-9]+)?)', fname)
    omega = float(m.group(1)) if m else None
    
    m = re.search(r'\u0394=([0-9]+(?:\.[0-9]+)?)', fname)
    delta = float(m.group(1)) if m else None
    
    m = re.search(r'\u03B3=([0-9]+(?:\.[0-9]+)?)', fname)
    gamma = float(m.group(1)) if m else None
    
    return {'dimension': dimension, 'omega': omega, 'delta': delta, 'gamma': gamma}


def read_jld2_file(filepath: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read a single JLD2 file, returning (sz_mean, tSave) or None on error."""
    try:
        with h5py.File(str(filepath), 'r') as f:
            if 'sz_mean' not in f or 'tSave' not in f:
                logger.warning(f"Missing keys in {filepath}: {list(f.keys())}")
                return None
            sz_mean = np.array(f['sz_mean'])
            t_save = np.array(f['tSave'])
            return sz_mean, t_save
    except (OSError, KeyError, RuntimeError) as e:
        logger.warning(f"Failed to read {filepath}: {e}")
        return None


def parse_all_data(data_root: str) -> List[TrajectoryRecord]:
    """Parse all readable JLD2 files from the data directory."""
    data_root = Path(data_root)
    records: List[TrajectoryRecord] = []
    
    total_files = 0
    success_files = 0
    skip_files = 0
    
    for dir_path in sorted(data_root.iterdir()):
        if not dir_path.is_dir():
            continue
        
        dir_params = parse_directory_name(dir_path.name)
        n_atoms = dir_params['n_atoms']
        
        for file_path in sorted(dir_path.iterdir()):
            if not file_path.suffix == '.jld2':
                continue
            
            total_files += 1
            
            file_params = parse_filename(file_path.name)
            
            result = read_jld2_file(file_path)
            if result is None:
                skip_files += 1
                continue
            
            sz_mean, t_save = result
            
            # Validate shapes
            if len(sz_mean) != N_TIME_STEPS or len(t_save) != N_TIME_STEPS:
                logger.warning(
                    f"Unexpected shape in {file_path}: "
                    f"sz_mean={sz_mean.shape}, tSave={t_save.shape}"
                )
                skip_files += 1
                continue
            
            # Compute derived quantities
            rho = (sz_mean + 1.0) / 2.0  # DP order parameter
            sz_ss = float(np.mean(sz_mean[-50:]))
            rho_ss = float(np.mean(rho[-50:]))  # CORRECT: use rho, not sz_mean
            
            # Validate lattice geometry: must be perfect square for 2D
            lattice_size = int(np.sqrt(n_atoms)) if n_atoms is not None else None
            if lattice_size is not None and lattice_size * lattice_size != n_atoms:
                logger.warning(
                    f"Non-square lattice: N={n_atoms}, sqrt={np.sqrt(n_atoms):.2f}"
                )
            
            record = TrajectoryRecord(
                omega=file_params['omega'],
                delta=file_params['delta'] or dir_params['delta'],
                gamma=file_params['gamma'] or dir_params['gamma'],
                V=dir_params['delta'] or 2000.0,
                Gamma=1.0,
                n_atoms=n_atoms,
                lattice_size=lattice_size,
                dimension=file_params['dimension'],
                sz_mean=sz_mean,
                t_save=t_save,
                rho=rho,
                rho_ss=rho_ss,
                sz_ss=sz_ss,
                file_path=str(file_path),
            )
            
            records.append(record)
            success_files += 1
    
    logger.info(f"Total files scanned: {total_files}")
    logger.info(f"Successfully parsed: {success_files}")
    logger.info(f"Skipped/Failed: {skip_files}")
    logger.info(f"Unique system sizes: {sorted(set(r.n_atoms for r in records))}")
    
    return records


def save_dataset(records: List[TrajectoryRecord], output_path: str):
    """Save parsed records to a pickle file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"Saved {len(records)} records to {output_path}")
    
    summary = {
        'n_records': len(records),
        'n_atoms_values': sorted(set(r.n_atoms for r in records)),
        'omega_ranges': {n: (min(r.omega for r in records if r.n_atoms == n),
                             max(r.omega for r in records if r.n_atoms == n))
                        for n in sorted(set(r.n_atoms for r in records))},
        'records_per_size': {n: sum(1 for r in records if r.n_atoms == n)
                            for n in sorted(set(r.n_atoms for r in records))},
    }
    
    summary_path = output_path.with_suffix('.summary.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"Saved summary to {summary_path}")
    for n, count in summary['records_per_size'].items():
        omega_min, omega_max = summary['omega_ranges'][n]
        logger.info(f"  N={n:4d}: {count:2d} records, Omega=[{omega_min:.2f}, {omega_max:.2f}]")


def main():
    data_root = Path(__file__).parent.parent / 'Rydberg_facilitation' / 'results_data_mean'
    output_path = Path(__file__).parent.parent / 'outputs' / 'rydberg_dataset.pkl'
    
    logger.info(f"Parsing data from: {data_root}")
    records = parse_all_data(str(data_root))
    save_dataset(records, str(output_path))


if __name__ == '__main__':
    main()
