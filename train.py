"""
Training script for RydbergSurrogate transformer model (v2 dataset).

Usage:
    python train.py --max_epochs 500 --patience 50 --batch_size 32 --lr 1e-3
"""

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

import sys
sys.path.insert(0, str(Path(__file__).parent / 'data'))
sys.path.insert(0, str(Path(__file__).parent / 'models'))

from dataset_v2 import TrajectoryDataset, create_splits, load_dataset, collate_fn, worker_init_fn
from parse_jld2_v2 import TrajectoryRecord  # needed for pickle unpickling
from transformer_surrogate import RydbergSurrogate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class EMA:
    """Exponential Moving Average of model parameters."""
    
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
    
    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def physics_informed_loss(pred, target, bounds_weight=0.1, smoothness_weight=0.01):
    """MSE loss with regularization constraints."""
    mse = F.mse_loss(pred, target)
    
    # Bounds regularization (soft, not hard constraint)
    lower_violation = torch.relu(-1.0 - pred)
    upper_violation = torch.relu(pred - 1.0)
    bounds_penalty = (lower_violation + upper_violation).mean()
    
    # Smoothness regularization
    if pred.size(1) > 2:
        d2 = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
        smoothness_penalty = (d2 ** 2).mean()
    else:
        smoothness_penalty = torch.tensor(0.0, device=pred.device)
    
    total_loss = mse + bounds_weight * bounds_penalty + smoothness_weight * smoothness_penalty
    
    return {
        'loss': total_loss,
        'mse': mse,
        'bounds_penalty': bounds_penalty,
        'smoothness_penalty': smoothness_penalty,
    }


def configure_optimizer(model, lr, weight_decay):
    """Configure AdamW with separate weight decay for decayable params."""
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2 and p.requires_grad]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2 and p.requires_grad]
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]
    
    return AdamW(param_groups, lr=lr)


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    n_batches = 0
    
    for batch in dataloader:
        omega = batch['omega'].to(device)
        n_atoms = batch['n_atoms'].to(device)
        inv_sqrt_n = batch['inv_sqrt_n'].to(device)
        gamma = batch['gamma'].to(device)
        dimension = batch['dimension'].to(device)
        t = batch['t'].to(device)
        sz_mean = batch['sz_mean'].to(device)
        
        optimizer.zero_grad()
        pred = model(omega, n_atoms, inv_sqrt_n, gamma, dimension, t)
        
        loss_dict = physics_informed_loss(pred, sz_mean)
        loss = loss_dict['loss']
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += loss_dict['mse'].item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
    }


def evaluate(model, dataloader, device):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_rho_ss_mae = 0.0
    total_ic_error = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            omega = batch['omega'].to(device)
            n_atoms = batch['n_atoms'].to(device)
            inv_sqrt_n = batch['inv_sqrt_n'].to(device)
            gamma = batch['gamma'].to(device)
            dimension = batch['dimension'].to(device)
            t = batch['t'].to(device)
            sz_mean = batch['sz_mean'].to(device)
            
            pred = model(omega, n_atoms, inv_sqrt_n, gamma, dimension, t)
            
            mse = F.mse_loss(pred, sz_mean)
            mae = F.l1_loss(pred, sz_mean)
            
            # Steady-state MAE on rho (last 50 points)
            pred_rho = (pred[:, -50:] + 1.0) / 2.0
            true_rho = (sz_mean[:, -50:] + 1.0) / 2.0
            pred_rho_ss = pred_rho.mean(dim=1)
            true_rho_ss = true_rho.mean(dim=1)
            rho_ss_mae = F.l1_loss(pred_rho_ss, true_rho_ss)
            
            # Initial condition error
            ic_error = F.l1_loss(pred[:, 0], sz_mean[:, 0])
            
            total_mse += mse.item()
            total_mae += mae.item()
            total_rho_ss_mae += rho_ss_mae.item()
            total_ic_error += ic_error.item()
            n_batches += 1
    
    return {
        'mse': total_mse / n_batches,
        'mae': total_mae / n_batches,
        'rho_ss_mae': total_rho_ss_mae / n_batches,
        'ic_error': total_ic_error / n_batches,
    }


def main(args):
    set_seed(args.seed)
    
    # Device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading dataset from: {args.data_path}")
    records = load_dataset(args.data_path)
    train_r, val_r, test_sets = create_splits(records)
    
    train_ds = TrajectoryDataset(train_r, augment=True)
    val_ds = TrajectoryDataset(val_r, augment=False)
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, worker_init_fn=worker_init_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn,
    )
    
    # Test loaders
    test_loaders = {}
    for name, test_r in test_sets.items():
        test_ds = TrajectoryDataset(test_r, augment=False)
        test_loaders[name] = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn,
        )
    
    # Initialize model
    model = RydbergSurrogate(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_time=400,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
    ).to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Optimizer and scheduler
    optimizer = configure_optimizer(model, args.lr, args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.lr * 0.01)
    
    # EMA
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    
    # Initialize W&B
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name or f"surrogate_l{args.n_layer}_h{args.n_head}_d{args.n_embd}_seed{args.seed}",
            config=vars(args),
        )
        wandb.watch(model, log="gradients", log_freq=100)
    
    # Training loop
    best_val_mse = float('inf')
    patience_counter = 0
    save_path = None
    
    for epoch in range(args.max_epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        
        if ema is not None:
            ema.update(model)
        
        val_metrics = evaluate(model, val_loader, device)
        
        # Log metrics
        log_dict = {
            'epoch': epoch,
            'train/loss': train_metrics['loss'],
            'train/mse': train_metrics['mse'],
            'val/mse': val_metrics['mse'],
            'val/mae': val_metrics['mae'],
            'val/rho_ss_mae': val_metrics['rho_ss_mae'],
            'val/ic_error': val_metrics['ic_error'],
            'lr': optimizer.param_groups[0]['lr'],
        }
        
        if args.use_wandb:
            wandb.log(log_dict, step=epoch)
        
        if epoch % args.log_interval == 0:
            print(f"Epoch {epoch:3d} | "
                  f"train_mse={train_metrics['mse']:.6f} | "
                  f"val_mse={val_metrics['mse']:.6f} | "
                  f"val_rho_ss_mae={val_metrics['rho_ss_mae']:.6f} | "
                  f"val_ic_error={val_metrics['ic_error']:.6f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping and checkpointing
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            patience_counter = 0
            
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'val_mse': best_val_mse,
                'args': vars(args),
            }
            if ema is not None:
                checkpoint['ema_shadow'] = ema.shadow
            
            save_path = Path(args.output_dir) / 'best_model.pt'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, save_path)
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"\nTraining complete. Best val MSE: {best_val_mse:.6f}")
    if save_path is not None:
        print(f"Best model saved to: {save_path}")
    
    # Final evaluation on all test sets
    if save_path is not None:
        print("\n" + "=" * 70)
        print("TEST SET EVALUATION")
        print("=" * 70)
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        
        for name, loader in test_loaders.items():
            metrics = evaluate(model, loader, device)
            print(f"\n{name}:")
            print(f"  MSE:        {metrics['mse']:.6f}")
            print(f"  MAE:        {metrics['mae']:.6f}")
            print(f"  ρ_ss MAE:   {metrics['rho_ss_mae']:.6f}")
            print(f"  IC error:   {metrics['ic_error']:.6f}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--data_path', type=str, default='data/rydberg_dataset_v2.pkl')
    parser.add_argument('--output_dir', type=str, default='outputs/models')
    
    # Model
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_embd', type=int, default=96)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--mlp_ratio', type=int, default=2)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=10)
    
    # Regularization
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='rydberg-surrogate')
    parser.add_argument('--run_name', type=str, default=None)
    
    args = parser.parse_args()
    main(args)
