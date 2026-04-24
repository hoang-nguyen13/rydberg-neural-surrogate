# Neural Surrogate for Rydberg Atom Dynamics

Neural network surrogate model for predicting the dynamics of Rydberg atom arrays under quantum driving and dephasing. Trains on DTWA (Discrete Truncated Wigner Approximation) simulation data and generalizes across system sizes, dephasing rates, and dimensions.

## Overview

This project trains neural surrogates to predict the time evolution of Rydberg excitation density ρ(t) for driven-dissipative Rydberg atom systems. The surrogate conditions on physical parameters (Ω, Δ, γ, N, dimension) and predicts the full trajectory, enabling:

- **Size extrapolation**: Train on small systems, predict large ones
- **Dephasing generalization**: Quantum → classical dephasing rates
- **Dimension transfer**: 1D, 2D, 3D with shared representations

## Repository Structure

```
data/
  parse_jld2.py          # Parse JLD2 simulation files
  parse_jld2_v2.py       # Extended parser (multi-γ, multi-dim)
  dataset.py             # PyTorch Dataset (3 params)
  dataset_v2.py          # PyTorch Dataset (5 params, 5 test sets)

models/
  transformer_surrogate.py   # Transformer-based surrogate (310K params)

baselines/
  gp_baseline.py         # Gaussian Process baseline
  fno_baseline.py        # Fourier Neural Operator baseline

scripts/
  plot_finite_size_extrap.py   # Finite-size scaling plots
  evaluate.py            # Evaluation metrics
  check_data.py          # Data quality checks
  explore_data.py        # Data exploration

train.py               # Main training loop
train_ensemble.py      # Ensemble training
inference.py           # Inference script
requirements.txt       # Python dependencies
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Simulation data is in JLD2 format from Julia DTWA code. The parser extracts:
- `sz_mean`: Time-dependent Rydberg population (shape: 400,)
- `tSave`: Saved time points (shape: 400,)
- Parameters: Ω, Δ, γ, N, dimension

## Training

```bash
python train.py --max_epochs 500 --patience 50 --batch_size 32 --lr 1e-3
```

## License

MIT
