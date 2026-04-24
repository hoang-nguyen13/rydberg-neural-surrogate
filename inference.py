"""
Inference script for trained RydbergSurrogate model.

Usage:
    python inference.py --model_path outputs/models/best_model.pt --omega 11.5 --n_atoms 1225
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent / 'models'))
from transformer_surrogate import RydbergSurrogate


def load_model(model_path, device='cpu'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_args = checkpoint.get("args", {})
    
    model = RydbergSurrogate(
        n_layer=model_args.get("n_layer", 4),
        n_head=model_args.get("n_head", 4),
        n_embd=model_args.get("n_embd", 96),
        dropout=model_args.get("dropout", 0.2),
        mlp_ratio=model_args.get("mlp_ratio", 2),
    ).to(device)
    
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    return model, checkpoint


def predict(model, omega, n_atoms, t_max=1000.0, n_time=400, device='cpu'):
    """
    Predict sz_mean(t) for given parameters.
    
    Args:
        omega: Rabi frequency
        n_atoms: Number of atoms (system size)
        t_max: Final time
        n_time: Number of time steps
        device: torch device
        
    Returns:
        t: (n_time,) time array
        sz_pred: (n_time,) predicted sz_mean
    """
    t = torch.linspace(0, t_max, n_time, dtype=torch.float32).unsqueeze(0).to(device)
    omega_t = torch.tensor([omega], dtype=torch.float32).to(device)
    n_atoms_t = torch.tensor([n_atoms], dtype=torch.float32).to(device)
    inv_sqrt_n = torch.tensor([1.0 / np.sqrt(n_atoms)], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        sz_pred = model(omega_t, n_atoms_t, inv_sqrt_n, t).cpu().numpy()[0]
    
    return t.cpu().numpy()[0], sz_pred


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model, checkpoint = load_model(args.model_path, device)
    print(f"Loaded model from: {args.model_path}")
    print(f"Model trained for {checkpoint.get('epoch', '?')} epochs")
    
    # Predict
    t, sz_pred = predict(model, args.omega, args.n_atoms, device=device)
    
    # Compute derived quantities
    rho_pred = (sz_pred + 1.0) / 2.0
    rho_ss = np.mean(rho_pred[-50:])
    sz_ss = np.mean(sz_pred[-50:])
    
    print(f"\nParameters: Omega={args.omega}, N={args.n_atoms}")
    print(f"Predicted steady-state rho_ss: {rho_ss:.6f}")
    print(f"Predicted steady-state sz_ss:  {sz_ss:.6f}")
    print(f"Phase: {'ACTIVE' if rho_ss > 0.05 else 'ABSORBING'}")
    
    # Speed benchmark
    n_runs = 100
    start = time.time()
    for _ in range(n_runs):
        _ = predict(model, args.omega, args.n_atoms, device=device)
    elapsed = time.time() - start
    print(f"\nInference time: {elapsed/n_runs*1000:.3f} ms per trajectory")
    
    # Save prediction
    if args.output_path:
        np.savez(args.output_path, t=t, sz_mean=sz_pred, rho=rho_pred,
                 omega=args.omega, n_atoms=args.n_atoms)
        print(f"Saved prediction to: {args.output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--omega', type=float, required=True)
    parser.add_argument('--n_atoms', type=int, required=True)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()
    main(args)
