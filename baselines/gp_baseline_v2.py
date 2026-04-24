"""Gaussian Process baseline for Rydberg facilitation dynamics (v2 dataset)."""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from parse_jld2_v2 import TrajectoryRecord
from dataset_v2 import create_splits, load_dataset


def prepare_data(records, time_subsample=1):
    """Flatten trajectories into (Omega, N, 1/sqrt(N), log10_gamma, dimension, t) -> sz_mean pairs."""
    X = []
    y = []

    for rec in records:
        n = rec.n_atoms
        omega = rec.omega
        inv_sqrt_n = 1.0 / np.sqrt(n)
        log10_gamma = np.log10(rec.gamma + 1e-10)
        dimension = rec.dimension
        for i in range(0, len(rec.t_save), time_subsample):
            X.append([omega, n, inv_sqrt_n, log10_gamma, dimension, rec.t_save[i]])
            y.append(rec.sz_mean[i])

    return np.array(X), np.array(y)


def train_gp(train_records, kernel=None, time_subsample=5):
    """Train Gaussian Process on training data."""
    X_train, y_train = prepare_data(train_records, time_subsample)

    if kernel is None:
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * \
                 RBF(length_scale=[1.0, 100.0, 0.1, 1.0, 1.0, 10.0],
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
    ic_error_list = []

    for rec in records:
        n = rec.n_atoms
        inv_sqrt_n = 1.0 / np.sqrt(n)
        log10_gamma = np.log10(rec.gamma + 1e-10)
        dimension = rec.dimension
        X_test = np.array([[rec.omega, n, inv_sqrt_n, log10_gamma, dimension, t] for t in rec.t_save])
        y_pred, y_std = gp.predict(X_test, return_std=True)
        y_true = rec.sz_mean

        mse = np.mean((y_pred - y_true) ** 2)
        mae = np.mean(np.abs(y_pred - y_true))

        pred_rho_ss = np.mean((y_pred[-50:] + 1.0) / 2.0)
        true_rho_ss = np.mean((y_true[-50:] + 1.0) / 2.0)
        rho_ss_mae = np.abs(pred_rho_ss - true_rho_ss)

        ic_error = np.abs(y_pred[0] - y_true[0])

        mse_list.append(mse)
        mae_list.append(mae)
        rho_ss_mae_list.append(rho_ss_mae)
        ic_error_list.append(ic_error)

    return {
        "mse": np.mean(mse_list),
        "mae": np.mean(mae_list),
        "rho_ss_mae": np.mean(rho_ss_mae_list),
        "ic_error": np.mean(ic_error_list),
    }


def main(args):
    records = load_dataset(args.data_path)
    train_r, val_r, test_sets = create_splits(records)

    gp = train_gp(train_r, time_subsample=args.time_subsample)

    save_path = Path(args.output_dir) / "gp_model.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(gp, save_path)
    print(f"Saved GP model to {save_path}")

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    for split_name, split_records in [("Train", train_r), ("Val", val_r)] + list(test_sets.items()):
        metrics = evaluate_gp(gp, split_records)
        print(f"\n{split_name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/rydberg_dataset_v2.pkl")
    parser.add_argument("--output_dir", type=str, default="outputs/models")
    parser.add_argument("--time_subsample", type=int, default=5)
    args = parser.parse_args()
    main(args)
