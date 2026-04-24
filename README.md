# Neural Surrogate for Rydberg Non-Equilibrium Phase Transitions

A compact transformer-based neural surrogate that learns ensemble-averaged Truncated Wigner Approximation (TWA) dynamics for driven-dissipative Rydberg lattices. Trained on only 105 trajectories, it generalizes to unseen system sizes and dephasing rates, reproduces effective critical scaling, and runs ~10⁴× faster than direct TWA simulation.

## Repository Structure

```
.
├── data/
│   ├── rydberg_dataset_v2.pkl      # Parsed dataset (759 trajectories)
│   ├── dataset_v2.py                # PyTorch Dataset & splits
│   └── parse_jld2_v2.py             # JLD2 → pickle parser
├── models/
│   └── transformer_surrogate.py     # 5-parameter transformer model
├── baselines/
│   ├── fno_baseline_v2.py           # Fourier Neural Operator baseline
│   └── gp_baseline_v2.py            # Gaussian Process baseline
├── scripts/
│   ├── evaluate_v2.py               # Main evaluation (generates paper figures)
│   ├── extract_critical_exponents.py # β, δ extraction from trajectories
│   ├── plot_loglog_dynamics.py      # Log-log dynamics plots
│   ├── plot_data_collapse.py        # Data collapse t^δ ρ vs t|Ω−Ω_c|^(β/δ)
│   ├── plot_finite_size_collapse_z.py # Finite-size z-tuning collapse
│   ├── plot_finite_size_extrap.py   # Finite-size scaling extrapolation
│   ├── plot_dynamics_per_N.py       # Dynamics ρ(t) per N
│   ├── plot_gamma_phase_transitions.py # Phase transitions vs γ
│   └── plot_gamma_dynamics.py       # Dynamics comparison across γ
├── train.py                         # Main training script
├── train_ensemble.py                # Deep ensemble training
├── inference.py                     # Single-trajectory inference
├── requirements.txt                 # Python dependencies
└── PAPER_OUTLINE_FINAL.md           # Paper structure and figure plan
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
# Single model
python train.py --max_epochs 1000 --patience 100 --batch_size 32 --lr 1e-3 --use_ema

# Deep ensemble (5 models, for uncertainty quantification)
python train_ensemble.py --n_models 5 --max_epochs 1000 --patience 100
```

### Evaluation

```bash
# Generate all paper figures and tables
python scripts/evaluate_v2.py --model_path outputs/models/best_model.pt

# Extract critical exponents from predictions
python scripts/extract_critical_exponents.py
```

### Inference

```bash
python inference.py --model_path outputs/models/best_model.pt --omega 11.5 --n_atoms 3600 --gamma 0.1 --dimension 2
```

## Dataset

The dataset `data/rydberg_dataset_v2.pkl` contains 759 trajectories parsed from Julia TWA simulations:

| Parameter | Values |
|-----------|--------|
| Dimension | 1D, 2D, 3D |
| System size N | 100–21,025 |
| Rabi frequency Ω | 0–29 (selected: 10.0–13.0 step 0.15) |
| Dephasing γ | 0.1, 5.0, 10.0, 20.0 |
| Time range | tΓ ∈ [0, 1000] |

**Training split (2D only, γ=0.1):**
- Train: N ∈ {225, 400, 900, 1600, 2500} → 105 trajectories
- Val: N = 3600 → 21 trajectories
- Test size: N = 4900 → 19 trajectories (unseen size)
- Test γ: N = 3600, γ ∈ {5, 10, 20} → 85 trajectories (unseen dephasing)

## Model Architecture

- **Input:** 5 scalars (Ω, N, 1/√N, log₁₀γ, d) + time array t
- **Backbone:** 4-layer, 4-head transformer, n_embd=96, bidirectional attention
- **Output:** sz_mean(t) for all time points (direct regression)
- **Parameters:** ~310K
- **Loss:** MSE + soft bounds penalty + smoothness penalty

## Key Results (from thesis)

For 2D low-dephasing Rydberg lattices:
- Critical point: Ω_c ≈ 11.2
- Effective exponents: β ≈ 0.586, δ ≈ 0.4577, z ≈ 1.86
- Quantum and classical NEPT map one-to-one (no d→d+1 shift)

## Paper

See `PAPER_OUTLINE_FINAL.md` for the full paper structure, figure plan, and narrative arc.

## Citation

If you use this code, please cite the original thesis on TWA for Rydberg non-equilibrium phase transitions.

## License

MIT
