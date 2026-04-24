"""
Gaussian Process baseline for Rydberg facilitation dynamics.

Maps (Omega, N, t) -> sz_mean via kernel regression.
Gold standard for small-data interpolation.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent / 'data'))
from parse_jld2 import TrajectoryRecord


def prepare_data(records, time_subsample=1):
    """Flatten trajectories into (Omega, N, t) -> sz_mean pairs."""
    X = []
    y = []
    
    for rec in records:
        n = rec.n_atoms
        omega = rec.omega
        for i in range(0, len(rec.t_save), time_subsample):
            X.append([omega, n, rec.t_save[i]])
            y.append(rec.sz_mean[i])
    
    return np.array(X), np.array(y)


def train_gp(train_records, kernel=None, time_subsample=5):
    """Train Gaussian Process on training data."""
    X_train, y_train = prepare_data(train_records, time_subsample)
    
    if kernel is None:
        # Default kernel: constant * RBF + white noise
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * \
                 RBF(length_scale=[1.0, 100.0, 10.0],
                     length_scale_bounds=(1e-2, 1e3)) + \
                 WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    
    print(f"Training GP on {len(X_train)} points...")
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=2,
        normalize_y=True,
        alpha=1e-6,
    )
    gp.fit(X_train, y_train)
    print(f"Optimized kernel: {gp.kernel_}")
    
    return gp


def evaluate_gp(gp, records):
    """Evaluate GP on test trajectories."""
    mse_list = []
    mae_list = []
    rho_ss_mae_list = []
    
    for rec in records:
        X_test = np.array([[rec.omega, rec.n_atoms, t] for t in rec.t_save])
        y_pred, y_std = gp.predict(X_test, return_std=True)
        y_true = rec.sz_mean
        
        mse = np.mean((y_pred - y_true) ** 2)
        mae = np.mean(np.abs(y_pred - y_true))
        
        pred_rho_ss = np.mean(y_pred[-50:])
        true_rho_ss = np.mean(y_true[-50:])
        rho_ss_mae = np.abs(pred_rho_ss - true_rho_ss)
        
        mse_list.append(mse)
        mae_list.append(mae)
        rho_ss_mae_list.append(rho_ss_mae)
    
    return {
        'mse': np.mean(mse_list),
        'mae': np.mean(mae_list),
        'rho_ss_mae': np.mean(rho_ss_mae_list),
        'mse_std': np.std(mse_list),
    }


def main(args):
    # Load data
    with open(args.data_path, 'rb') as f:
        records = pickle.load(f)
    
    # Create splits
    from dataset import create_splits
    train_r, val_r, test_r = create_splits(records)
    
    # Train GP
    gp = train_gp(train_r, time_subsample=args.time_subsample)
    
    # Evaluate
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_gp(gp, val_r)
    print(f"Val MSE: {val_metrics['mse']:.6f} +/- {val_metrics['mse_std']:.6f}")
    print(f"Val MAE: {val_metrics['mae']:.6f}")
    print(f"Val rho_ss MAE: {val_metrics['rho_ss_mae']:.6f}")
    
    print("\nEvaluating on test set...")
    test_metrics = evaluate_gp(gp, test_r)
    print(f"Test MSE: {test_metrics['mse']:.6f} +/- {test_metrics['mse_std']:.6f}")
    print(f"Test MAE: {test_metrics['mae']:.6f}")
    print(f"Test rho_ss MAE: {test_metrics['rho_ss_mae']:.6f}")
    
    # Save model
    save_path = Path(args.output_dir) / 'gp_model.pkl'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(gp, save_path)
    print(f"\nGP model saved to: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='outputs/rydberg_dataset.pkl')
    parser.add_argument('--time_subsample', type=int, default=5, help='Subsample time points for GP training')
    parser.add_argument('--output_dir', type=str, default='outputs/baselines')
    args = parser.parse_args()
    main(args)
