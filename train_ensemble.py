"""
Deep ensemble training script.

Trains N models with different random seeds and saves all checkpoints.
Used for uncertainty quantification via ensemble variance.

Usage:
    python train_ensemble.py --n_models 5 --max_epochs 500 --patience 50
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main(args):
    base_output = Path(args.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)
    
    for i in range(args.n_models):
        seed = args.base_seed + i
        output_dir = base_output / f"model_seed{seed}"
        
        cmd = [
            sys.executable, "train.py",
            "--seed", str(seed),
            "--output_dir", str(output_dir),
            "--max_epochs", str(args.max_epochs),
            "--patience", str(args.patience),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--weight_decay", str(args.weight_decay),
            "--n_layer", str(args.n_layer),
            "--n_head", str(args.n_head),
            "--n_embd", str(args.n_embd),
            "--dropout", str(args.dropout),
            "--mlp_ratio", str(args.mlp_ratio),
            "--use_ema",
            "--ema_decay", str(args.ema_decay),
        ]
        
        if args.use_wandb:
            cmd.extend(["--use_wandb", "--wandb_project", args.wandb_project])
        
        print(f"\n{'='*60}")
        print(f"Training model {i+1}/{args.n_models} with seed {seed}")
        print(f"{'='*60}")
        
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        if result.returncode != 0:
            print(f"WARNING: Model {i+1} training failed with code {result.returncode}")
    
    print(f"\n{'='*60}")
    print(f"Ensemble training complete. {args.n_models} models saved to:")
    print(f"  {base_output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_models', type=int, default=5)
    parser.add_argument('--base_seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='outputs/ensemble')
    
    # Training hyperparameters (passed through to train.py)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # Model architecture
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_embd', type=int, default=96)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--mlp_ratio', type=int, default=2)
    
    # EMA
    parser.add_argument('--ema_decay', type=float, default=0.999)
    
    # W&B
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='rydberg-surrogate-ensemble')
    
    args = parser.parse_args()
    main(args)
