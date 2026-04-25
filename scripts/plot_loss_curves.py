"""
Plot training and validation loss curves from training log.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_path):
    """Parse train_mse, val_mse, val_rho_ss_mae, val_ic_error, lr from log."""
    epochs, train_mse, val_mse, val_rho_ss_mae, val_ic_error, lrs = [], [], [], [], [], []
    with open(log_path) as f:
        for line in f:
            m = re.search(
                r"Epoch\s+(\d+)\s+\|\s+train_mse=([\d.e+-]+)\s+\|\s+val_mse=([\d.e+-]+)\s+\|\s+val_rho_ss_mae=([\d.e+-]+)\s+\|\s+val_ic_error=([\d.e+-]+)\s+\|\s+lr=([\d.e+-]+)",
                line,
            )
            if m:
                epochs.append(int(m.group(1)))
                train_mse.append(float(m.group(2)))
                val_mse.append(float(m.group(3)))
                val_rho_ss_mae.append(float(m.group(4)))
                val_ic_error.append(float(m.group(5)))
                lrs.append(float(m.group(6)))
    return {
        "epochs": np.array(epochs),
        "train_mse": np.array(train_mse),
        "val_mse": np.array(val_mse),
        "val_rho_ss_mae": np.array(val_rho_ss_mae),
        "val_ic_error": np.array(val_ic_error),
        "lr": np.array(lrs),
    }


def plot_loss_curves(data, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Fig A: Train vs Val MSE ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(data["epochs"], data["train_mse"], "-", color="steelblue", lw=1.5, label="Train MSE")
    ax.semilogy(data["epochs"], data["val_mse"], "-", color="coral", lw=1.5, label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Training and Validation MSE")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(output_dir / "loss_mse.png", dpi=300)
    plt.close(fig)
    print(f"Saved {output_dir / 'loss_mse.png'}")

    # --- Fig B: Val metrics breakdown ---
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "steelblue"
    ax1.semilogy(data["epochs"], data["val_mse"], "-", color=color1, lw=1.5, label="Val MSE")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val MSE", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3, which="both")

    ax2 = ax1.twinx()
    color2 = "coral"
    ax2.plot(data["epochs"], data["val_rho_ss_mae"], "--", color=color2, lw=1.5, label="Val ρ_ss MAE")
    ax2.set_ylabel("Val ρ_ss MAE", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Validation Metrics")
    fig.tight_layout()
    fig.savefig(output_dir / "loss_val_metrics.png", dpi=300)
    plt.close(fig)
    print(f"Saved {output_dir / 'loss_val_metrics.png'}")

    # --- Fig C: Learning rate schedule ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data["epochs"], data["lr"], "-", color="darkgreen", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_lr.png", dpi=300)
    plt.close(fig)
    print(f"Saved {output_dir / 'loss_lr.png'}")


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/long_train_1000ep_gpu.log"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs/evaluation"
    data = parse_log(log_path)
    print(f"Parsed {len(data['epochs'])} epochs from {log_path}")
    plot_loss_curves(data, out_dir)
