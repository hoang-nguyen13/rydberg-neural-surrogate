"""Fourier Neural Operator (FNO) baseline for Rydberg facilitation dynamics (v2 dataset)."""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from parse_jld2_v2 import TrajectoryRecord
from dataset_v2 import TrajectoryDataset, create_splits, load_dataset, collate_fn

try:
    from neuralop.models import FNO
except ImportError:
    raise ImportError("neuraloperator not installed. Run: pip install neuraloperator")


class FNOBaselineV2(nn.Module):
    """FNO for 1D temporal operator learning with 5-parameter conditioning (v2)."""

    def __init__(self, n_modes=(32,), hidden_channels=64, n_layers=4, n_time=400):
        super().__init__()
        self.n_time = n_time
        # in_channels=6: omega, n_atoms, inv_sqrt_n, log10_gamma, dimension, t
        self.fno = FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=6,
            out_channels=1,
            n_layers=n_layers,
            positional_embedding="grid",
        )

    def forward(self, omega, n_atoms, inv_sqrt_n, gamma, dimension, t):
        batch_size = t.size(0)
        n_time = t.size(1)

        omega_ch = omega.view(batch_size, 1, 1).expand(batch_size, 1, n_time)
        n_ch = n_atoms.view(batch_size, 1, 1).expand(batch_size, 1, n_time)
        inv_sqrt_n_ch = inv_sqrt_n.view(batch_size, 1, 1).expand(batch_size, 1, n_time)
        gamma_ch = gamma.view(batch_size, 1, 1).expand(batch_size, 1, n_time)
        dim_ch = dimension.view(batch_size, 1, 1).expand(batch_size, 1, n_time)
        t_ch = t.view(batch_size, 1, n_time)

        x = torch.cat([omega_ch, n_ch, inv_sqrt_n_ch, gamma_ch, dim_ch, t_ch], dim=1)
        y = self.fno(x)
        return y.squeeze(1)


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        omega = batch["omega"].to(device)
        n_atoms = batch["n_atoms"].to(device)
        inv_sqrt_n = batch["inv_sqrt_n"].to(device)
        gamma = batch["gamma"].to(device)
        dimension = batch["dimension"].to(device)
        t = batch["t"].to(device)
        sz_mean = batch["sz_mean"].to(device)

        optimizer.zero_grad()
        pred = model(omega, n_atoms, inv_sqrt_n, gamma, dimension, t)
        loss = F.mse_loss(pred, sz_mean)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, dataloader, device):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_rho_ss_mae = 0.0
    total_ic_error = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            omega = batch["omega"].to(device)
            n_atoms = batch["n_atoms"].to(device)
            inv_sqrt_n = batch["inv_sqrt_n"].to(device)
            gamma = batch["gamma"].to(device)
            dimension = batch["dimension"].to(device)
            t = batch["t"].to(device)
            sz_mean = batch["sz_mean"].to(device)

            pred = model(omega, n_atoms, inv_sqrt_n, gamma, dimension, t)

            mse = F.mse_loss(pred, sz_mean)
            mae = F.l1_loss(pred, sz_mean)

            pred_rho = (pred[:, -50:] + 1.0) / 2.0
            true_rho = (sz_mean[:, -50:] + 1.0) / 2.0
            pred_rho_ss = pred_rho.mean(dim=1)
            true_rho_ss = true_rho.mean(dim=1)
            rho_ss_mae = F.l1_loss(pred_rho_ss, true_rho_ss)

            ic_error = F.l1_loss(pred[:, 0], sz_mean[:, 0])

            total_mse += mse.item()
            total_mae += mae.item()
            total_rho_ss_mae += rho_ss_mae.item()
            total_ic_error += ic_error.item()
            n_batches += 1

    return {
        "mse": total_mse / n_batches,
        "mae": total_mae / n_batches,
        "rho_ss_mae": total_rho_ss_mae / n_batches,
        "ic_error": total_ic_error / n_batches,
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    records = load_dataset(args.data_path)
    train_r, val_r, test_sets = create_splits(records)

    train_ds = TrajectoryDataset(train_r, augment=True)
    val_ds = TrajectoryDataset(val_r, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    test_loaders = {}
    for name, test_r in test_sets.items():
        test_ds = TrajectoryDataset(test_r, augment=False)
        test_loaders[name] = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = FNOBaselineV2(
        n_modes=(args.n_modes,),
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FNO parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.lr * 0.01)

    best_val_mse = float("inf")
    patience_counter = 0
    save_path = None

    for epoch in range(args.max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)

        if epoch % args.log_interval == 0:
            print(f"Epoch {epoch:3d} | train_loss={train_loss:.6f} | val_mse={val_metrics['mse']:.6f} | val_rho_ss_mae={val_metrics['rho_ss_mae']:.6f}")

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            patience_counter = 0
            save_path = Path(args.output_dir) / "fno_best.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_mse": best_val_mse, "args": vars(args)}, save_path)
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nTraining complete. Best val MSE: {best_val_mse:.6f}")

    if save_path is not None:
        print("\n" + "=" * 60)
        print("TEST EVALUATION")
        print("=" * 60)
        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model"]
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_")}
        model.load_state_dict(state_dict)

        for name, loader in test_loaders.items():
            metrics = evaluate(model, loader, device)
            print(f"\n{name}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/rydberg_dataset_v2.pkl")
    parser.add_argument("--output_dir", type=str, default="outputs/models")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--n_modes", type=int, default=32)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=4)
    args = parser.parse_args()
    main(args)
