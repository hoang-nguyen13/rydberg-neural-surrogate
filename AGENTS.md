# AGENTS.md -- Rydberg Neural Surrogate Project

**Purpose:** This document is a cold-start guide for AI agents (including future instances of Kimi) working on this project. It explains the research goal, the codebase architecture, what has been built, how to run everything, what can go wrong, and how to fix it.

**Last updated:** 2026-04-24
**Current status:** Infrastructure complete, ready for GPU training
**Hardware needed:** Any CUDA-capable GPU (~1GB VRAM). A100/V100/RTX 4090 all work. Even an RTX 3060 (12GB) is overkill.

**MANDATORY:** Read this entire file before modifying any code. When in doubt, grep the codebase rather than guessing.

---

## Table of Contents

1. [Research Objective](#1-research-objective)
2. [The Physics Problem](#2-the-physics-problem)
3. [The Method](#3-the-method)
4. [What Has Been Built](#4-what-has-been-built)
5. [Project Structure](#5-project-structure)
6. [How to Set Up](#6-how-to-set-up)
7. [How to Run](#7-how-to-run)
8. [Current State & Known Issues](#8-current-state--known-issues)
9. [Next Steps](#9-next-steps)
10. [Key Design Decisions](#10-key-design-decisions)
11. [Common Pitfalls and How to Avoid Them](#11-common-pitfalls-and-how-to-avoid-them)
12. [Expected Outputs and How to Interpret Them](#12-expected-outputs-and-how-to-interpret-them)
13. [Troubleshooting Guide](#13-troubleshooting-guide)
14. [What to Do If Something Breaks](#14-what-to-do-if-something-breaks)
15. [File-by-File Detailed Reference](#15-file-by-file-detailed-reference)
16. [Appendices](#16-appendices)

---

## 1. Research Objective

Build a **transformer-based neural surrogate** for ensemble-averaged Rydberg facilitation dynamics. Given Hamiltonian parameters (Rabi frequency Omega, system size N), the model predicts the full time evolution of the mean excitation density sz_mean(t) in a single forward pass. The goal is to:

1. **Accelerate simulation:** Predict dynamics in milliseconds vs. minutes for TWA simulation
2. **Generalize across parameter space:** Interpolate and extrapolate to unseen Omega and N
3. **Extract physical observables:** Phase diagrams, critical points, effective exponents
4. **Compare against baselines:** FNO, DeepONet, Gaussian Process

**What this is NOT:** A foundation model. It is a specialized neural surrogate trained on 290 trajectories from a single physical system with fixed Hamiltonian parameters (only Omega and N vary).

**Publication target:** Physical Review Research or Machine Learning: Science and Technology (MLST). NeurIPS/ICML AI for Science if methodological novelty is strong.

---

## 2. The Physics Problem

### 2.1 Rydberg Facilitation

A 2D lattice of atoms where each atom can be ground or Rydberg excited. An excited atom **facilitates** (helps) its neighbors to also excite, creating an avalanche of excitations.

### 2.2 The Phase Transition

Depending on driving strength (Rabi frequency Omega):
- **Low Omega:** Avalanche dies out -> all atoms return to ground state -> **absorbing phase** (rho = 0)
- **High Omega:** Avalanche sustains -> finite excitation density -> **active phase** (rho > 0)
- **Critical Omega_c:** Boundary between phases -> **directed percolation (DP) universality class**

### 2.3 The Data

- **Source:** Master's thesis simulations using TWA (Truncated Wigner Approximation)
- **Format:** 290 JLD2 files containing ensemble-averaged sz_mean[t] (400 time steps)
- **Parameters fixed:** V = Delta = 2000, Gamma = 1, gamma = 0.1, t_max = 1000
- **Parameters varying:** Omega (0-30), N (100-4900), all 2D square lattices
- **System sizes:** N = 100, 225, 400, 900, 1225, 1600, 2500, 3025, 3600, 4900

### 2.4 Critical Physics Caveats

**IMPORTANT -- read before making physics claims:**

1. **Ensemble-averaged data cannot reveal true DP exponents.** DP transitions are defined by a bimodal trajectory distribution (some die, some survive). Averaging destroys this. True critical exponents require per-trajectory survival probabilities.
2. **TWA may not reproduce DP exponents.** It is a semiclassical approximation. If exponents deviate from DP, the default explanation is 'TWA breaks down,' not 'new universality class.'
3. **t_max = 1000 may be too short for large systems near criticality.** With z ~ 1.77, relaxation time for L=70 is tau ~ 70^1.77 ~ 1700. The steady-state from last 50 points may be a transient.

**Therefore:** The paper should frame results as 'neural surrogate for mean-field non-equilibrium dynamics' with effective exponents and explicit caveats. Do NOT claim precise DP critical exponents from mean-field data alone.

---

## 3. The Method

### 3.1 Architecture: Direct Regression Transformer

**Key decision:** Abandoned autoregressive token generation (v1.0) in favor of direct regression (v2.0).

```
Input:  [Omega, N, 1/sqrt(N)]  +  [t_0, t_1, ..., t_399]
         |____________________|     |____________________|
           parameter embedding         time embedding

Output: [sz_0, sz_1, ..., sz_399]  (all 400 steps predicted in parallel)

Loss: MSE + bounds penalty + smoothness penalty
```

- **No causal mask:** Full bidirectional self-attention
- **No tokenization:** Continuous scalar outputs
- **Single forward pass:** ~ms inference per trajectory

### 3.2 Model Specs

| Hyperparameter | Value |
|---|---|
| n_layer | 4 |
| n_head | 4 |
| n_embd | 96 |
| mlp_ratio | 2 |
| dropout | 0.2 |
| **Total params** | **~309K** |

### 3.3 Data Splits

| Split | System Sizes | Purpose |
|---|---|---|
| Train | N=225, 400, 900 | Learn diverse dynamics |
| Validation | N=1225 (even Omega indices) | Early stopping, hyperparameter tuning |
| Test (interpolation) | N=1225 (odd Omega indices) | Interpolation within critical region |
| Test (size extrapolation) | N=1600, 2500, 3025, 3600, 4900 | Zero-shot size generalization |

**N=100 dropped** -- too small, limited coverage, may confuse model with finite-size artifacts.

### 3.4 Physics-Informed Loss

```python
loss = MSE(pred, true)
       + 0.1 * bounds_penalty  # penalize predictions outside [-1, 1]
       + 0.01 * smoothness_penalty  # penalize large second derivatives
```

Note: The smoothness penalty is generic Tikhonov regularization, NOT a true physics-informed term from the master equation.

### 3.5 Regularization

- Dropout: 0.2
- Weight decay: 1e-4 (applied only to dim >= 2 params, not biases/LayerNorm)
- Gradient clipping: max_norm = 1.0
- Early stopping: patience = 100 epochs
- Optional EMA: decay = 0.999
- Data augmentation: parameter jittering (+/-0.05 Omega), small trajectory noise (std=0.005)

---

## 4. What Has Been Built

### Phase 0: Data Archaeology
- Parsed 290/302 JLD2 files (12 unreadable due to encoding errors)
- Confirmed fixed parameters: V=Delta=2000, Gamma=1, gamma=0.1, t_max=1000, n_T=400
- Verified phase transition signal in data

### Phase 1: Data Engineering
- data/parse_jld2.py -- Robust parser with correct regex for decimal parameters
- data/dataset.py -- PyTorch Dataset with parameter-based splits, augmentation, worker_init_fn
- scripts/check_data.py -- Quality checks + 5 visualization plots

### Phase 2: Model Architecture
- models/transformer_surrogate.py -- Direct regression transformer (~309K params)

### Phase 3: Baselines
- baselines/gp_baseline.py -- Gaussian Process (scikit-learn)
- baselines/fno_baseline.py -- Fourier Neural Operator (neuraloperator library)
- DeepONet baseline: **NOT IMPLEMENTED** (was planned but deprioritized)

### Phase 4: Training
- train.py -- Full training loop with AdamW, cosine LR, physics-informed loss, EMA, W&B logging
- train_ensemble.py -- Deep ensemble wrapper (trains N models with different seeds)

### Phase 5: Evaluation
- scripts/evaluate.py -- Comprehensive evaluation:
  - Trajectory metrics: MSE, MAE, rel L2, Pearson, DTW
  - Phase classification: accuracy, confusion matrix, ROC, AUC
  - Critical point estimation: derivative method + bootstrap CIs
  - Visualizations: trajectory comparisons, phase diagrams, size extrapolation, critical points

### Phase 6: Inference
- inference.py -- Load model, predict on new parameters, compute phase, benchmark speed

### Infrastructure
- requirements.txt -- All Python dependencies

---

## 5. Project Structure

```
nanoGPT/
├── data/
│   ├── parse_jld2.py          # JLD2 parser (CRITICAL: regex fixed for decimals)
│   └── dataset.py              # PyTorch Dataset + splits + augmentation
├── models/
│   └── transformer_surrogate.py # Main model (~309K params, direct regression)
├── baselines/
│   ├── gp_baseline.py          # Gaussian Process baseline
│   └── fno_baseline.py         # Fourier Neural Operator baseline
├── scripts/
│   ├── check_data.py           # Data quality checks + plots
│   └── evaluate.py             # Full evaluation pipeline
├── configs/                     # EMPTY -- Hydra configs not yet created
├── outputs/
│   ├── rydberg_dataset.pkl      # Parsed dataset (290 records)
│   ├── rydberg_dataset.summary.pkl
│   ├── data_checks/             # Quality check plots
│   ├── models/                  # Trained model checkpoints
│   ├── baselines/               # Baseline checkpoints
│   └── evaluation/              # Evaluation plots
├── train.py                     # Main training script
├── train_ensemble.py            # Deep ensemble wrapper
├── inference.py                 # Inference script
├── requirements.txt             # Python dependencies
├── AGENTS.md                    # THIS FILE
├── PROJECT_PLAN.md              # Original v1.0 plan
├── PROJECT_PLAN_REVISED.md      # v2.0 revised plan
├── AUDIT_PHYSICS.md             # Physics audit report
├── AUDIT_ENGINEERING.md         # Engineering audit report
├── AUDIT_LOGIC.md               # Logic audit report
└── Rydberg_facilitation/        # Raw thesis data
    └── results_data_mean/       # 302 JLD2 files
```

---

## 6. How to Set Up

### 6.1 Prerequisites

- Python 3.10+
- CUDA-capable GPU (any modern NVIDIA GPU works; ~1GB VRAM needed)
- ~100MB disk space for data + checkpoints
- 5–10GB to install PyTorch + dependencies

### 6.2 Install Dependencies

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **CRITICAL:** Every `python3` command in this document assumes the virtual environment above is **active**. If you open a new terminal, you MUST run `source .venv/bin/activate` first or commands will fail with `ModuleNotFoundError`.

**Dependency list (requirements.txt):**
| Package | Minimum Version | Purpose |
|---|---|---|
| torch | 2.0.0 | Transformer model, training loop |
| numpy | 1.24.0 | Numerical arrays |
| scipy | 1.10.0 | Pearson correlation, statistics |
| pandas | 2.0.0 | Data manipulation (if needed) |
| matplotlib | 3.7.0 | Plotting |
| seaborn | 0.12.0 | Plotting style |
| h5py | 3.8.0 | Reading JLD2 files (HDF5-based) |
| scikit-learn | 1.3.0 | GP baseline, ROC/AUC, confusion matrix |
| wandb | 0.15.0 | Experiment tracking (optional) |
| hydra-core | 1.3.0 | Config management (currently unused) |
| torchmetrics | 1.0.0 | Metrics (currently unused) |
| neuraloperator | 1.0.0 | FNO baseline |

### 6.3 Verify Data Exists

```bash
# Should show 290 records
python3 -c "
import pickle, sys
sys.path.insert(0, 'data')
from parse_jld2 import TrajectoryRecord
with open('outputs/rydberg_dataset.pkl', 'rb') as f:
    records = pickle.load(f)
print(f'Loaded {len(records)} records')
print(f'System sizes: {sorted(set(r.n_atoms for r in records))}')
"
```

If outputs/rydberg_dataset.pkl is missing, regenerate it:
```bash
python3 data/parse_jld2.py
```

---

## 7. How to Run

### 7.1 Quick Sanity Check (CPU, ~1 minute)

```bash
# Train for 5 epochs to verify pipeline works
python3 train.py --max_epochs 5 --patience 10 --batch_size 16

# Evaluate
python3 scripts/evaluate.py --model_path outputs/models/best_model.pt

# Inference
python3 inference.py --model_path outputs/models/best_model.pt --omega 11.5 --n_atoms 1225
```

### 7.2 Full Transformer Training (GPU, ~2-6 hours)

```bash
python3 train.py \
    --max_epochs 500 \
    --patience 50 \
    --batch_size 32 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --n_layer 4 \
    --n_head 4 \
    --n_embd 96 \
    --dropout 0.2 \
    --use_ema \
    --ema_decay 0.999 \
    --use_wandb \
    --wandb_project rydberg-surrogate \
    --run_name surrogate_run1
```

### 7.3 Deep Ensemble Training (GPU, ~10-30 hours for 5 models)

```bash
python3 train_ensemble.py \
    --n_models 5 \
    --base_seed 42 \
    --max_epochs 500 \
    --patience 50 \
    --output_dir outputs/ensemble
```

This trains 5 models with seeds 42, 43, 44, 45, 46. Checkpoints saved to outputs/ensemble/model_seed{seed}/best_model.pt.

### 7.4 FNO Baseline (GPU, ~2-6 hours)

```bash
python3 baselines/fno_baseline.py \
    --epochs 500 \
    --patience 50 \
    --batch_size 16 \
    --n_modes 32 \
    --output_dir outputs/baselines
```

### 7.5 GP Baseline (CPU, ~5-30 minutes)

```bash
# Note: GP is trained on time-subsampled data (every 5th point)
# because full 44K points is intractable for scikit-learn GP
python3 baselines/gp_baseline.py --time_subsample 5
```

### 7.6 Full Evaluation

```bash
python3 scripts/evaluate.py \
    --model_path outputs/models/best_model.pt \
    --output_dir outputs/evaluation
```

Produces:
- trajectory_comparison_n1225.png -- True vs predicted for interpolation test
- trajectory_comparison_extrapolation.png -- True vs predicted for size extrapolation
- phase_diagram_predicted_vs_true.png -- Side-by-side phase diagrams
- critical_points.png -- Omega_c vs N with bootstrap CIs
- roc_curve.png -- Phase classification ROC

---

## 8. Current State & Known Issues

### 8.1 What Works

| Component | Status |
|---|---|
| Data pipeline | Complete and tested |
| Transformer model | Complete and tested |
| Training loop | Complete and tested |
| Evaluation pipeline | Complete and tested |
| FNO baseline | Complete, needs GPU testing |
| GP baseline | Complete, runs on CPU |
| Inference script | Complete and tested |
| Ensemble training | Script complete, needs GPU run |

### 8.2 Known Issues / TODOs

| Issue | Severity | Status |
|---|---|---|
| Model only trained for 5 epochs (quick test) | **Critical** | Needs full GPU training |
| DeepONet baseline not implemented | Medium | Planned but deprioritized; can be added later |
| Hydra config files not created | Low | Hardcoded args work fine |
| No Docker/environment file | Low | requirements.txt is sufficient |
| Spatial extension (Phase 6) not started | Medium | Deferred to post-mean-field paper |
| Speedup benchmark vs TWA not measured | Medium | Needs timed Julia TWA run for comparison |

### 8.3 Physics Limitations (Acceptable for Paper)

1. Mean-field data only -- no per-trajectory survival probabilities
2. Fixed Hamiltonian parameters -- model cannot generalize to different gamma, Delta, V
3. t_max = 1000 may be insufficient for largest systems near criticality
4. All systems are 2D square lattices

These should be explicitly stated as limitations in the paper.

---

## 9. Next Steps

### Immediate (GPU Machine)

1. **Train main model:** python3 train.py --max_epochs 500 --patience 50 --use_ema
2. **Train ensemble:** python3 train_ensemble.py --n_models 5
3. **Train FNO baseline:** python3 baselines/fno_baseline.py --epochs 500
4. **Evaluate all:** python3 scripts/evaluate.py --model_path outputs/models/best_model.pt
5. **Compare metrics:** Tabulate transformer vs FNO vs GP on all test sets

### Medium-Term (After Training)

6. **Hyperparameter sweep:** Try n_embd={64,96,128}, dropout={0.1,0.2,0.3}, mlp_ratio={2,4}
7. **Ablation studies:** Remove parameter conditioning, remove time embedding, test causality
8. **Linear probing:** Train classifiers on frozen hidden states to test physics encoding
9. **Speedup benchmark:** Time TWA simulation vs NN inference
10. **Spatial extension:** Generate per-atom data, train patch-based ViT (Phase 6)

### Publication

11. **Write paper:** Target PR Research or MLST
12. **Release code:** GitHub repo with model weights and inference script
13. **arXiv preprint**

---

## 10. Key Design Decisions

### Why Direct Regression instead of Autoregressive?

The data is deterministic ensemble averages. Autoregressive token generation introduces:
- Quantization error (discretizing continuous sz_mean)
- Error accumulation (400 sequential predictions)
- 400x slower inference
- Unnecessary stochasticity from softmax sampling

Direct regression predicts all 400 time steps in one forward pass with MSE loss.

### Why ~309K Parameters?

Original plan targeted ~500K, but audit found 810K was too many for 110 training examples. Reduced to 309K by:
- n_embd: 128 -> 96
- mlp_ratio: 4 -> 2

This gives ~2,800 params/example, a sane ratio for small-data scientific ML.

### Why Drop N=100 from Training?

N=100 has only 21 examples with Omega in [10,13] only. Its strong finite-size artifacts could confuse the model. Better to train on larger systems (N=225,400,900) that show cleaner dynamics.

### Why 1/sqrt(N) as a Feature?

Finite-size scaling suggests the critical point shifts with system size. 1/sqrt(N) is a simple scaling variable. Note: The true DP scaling exponent is nu_perp ~ 0.734, so N^{-1/(2*nu_perp)} ~ N^{-0.682} would be more accurate. 1/sqrt(N) is a pragmatic approximation.

### Why No Causal Mask?

The task is function regression: sz(t) = f(Omega, N, t). The target at time t is a fixed deterministic value, not a random variable conditioned on the past. Full bidirectional attention is appropriate and more expressive.

---

## 11. Common Pitfalls and How to Avoid Them

### Pitfall 1: Confusing sz_mean and rho

**What goes wrong:** sz_mean ranges from -1 to +1. The DP order parameter rho = (sz_mean + 1) / 2 ranges from 0 to 1. Steady-state analysis MUST use rho_ss, NOT sz_ss. A previous bug used sz_mean to compute rho_ss, giving nonsensical values.

**How to avoid:**
- Always compute rho = (sz_mean + 1.0) / 2.0 before taking means or thresholds
- Phase threshold is rho_ss > 0.05 (equivalent to sz_ss > -0.9)
- The TrajectoryRecord dataclass already computes both correctly -- use rec.rho_ss, not manual re-computation

**Verification:**
```python
# CORRECT
rho_ss = np.mean(rho[-50:])      # rho = (sz_mean + 1) / 2

# WRONG -- this was a previous bug
rho_ss = np.mean(sz_mean[-50:])  # sz_mean is in [-1, 1], not [0, 1]
```

### Pitfall 2: Filename Regex Truncating Decimals

**What goes wrong:** Original regex r'\u03A9=([0-9]+)' only matched integers, so Omega=10.15 was parsed as 10.0, corrupting the parameter space.

**How to avoid:** The fix r'\u03A9=([0-9]+(?:\.[0-9]+)?)' is already in data/parse_jld2.py. Never revert to integer-only regex.

**Verification:** Check that parsed Omega values include decimals:
```bash
python3 -c "
import pickle
with open('outputs/rydberg_dataset.pkl','rb') as f:
    records = pickle.load(f)
import numpy as np
omegas = [r.omega for r in records]
print(f'Min Omega: {min(omegas)}, Max Omega: {max(omegas)}')
print(f'Unique Omegas (sample): {sorted(set(omegas))[:10]}')
"
```
If all values are integers, the regex is broken.

### Pitfall 3: Using save_path Before Definition

**What goes wrong:** In train.py, if validation never improves (e.g., NaN loss), save_path remains None and a NameError crashes on the final print statement.

**How to avoid:** Already fixed -- the code now checks 'if save_path is not None' before printing. If you see 'WARNING: No model was saved (validation never improved)', investigate training instability (NaN loss, bad data, etc.).

### Pitfall 4: Data Augmentation Creating Systematic Bias

**What goes wrong:** The TrajectoryDataset adds Gaussian noise to sz_mean during augmentation. Hard-clipping to [-1, 1] would create systematic bias in the absorbing phase (where true values cluster near -1.0). The current code intentionally does NOT clip augmented values.

**How to avoid:** Do NOT add hard clipping to the augmentation path. The bounds penalty in the loss function handles out-of-range predictions during training.

### Pitfall 5: Failing to Re-seed DataLoader Workers

**What goes wrong:** Without worker_init_fn, all DataLoader workers share the same random state, producing identical augmentations across workers.

**How to avoid:** The TrajectoryDataset uses worker_init_fn which seeds each worker from torch.initial_seed(). Always pass it to DataLoader:
```python
DataLoader(dataset, worker_init_fn=worker_init_fn, ...)
```

### Pitfall 6: Loading Checkpoints with Wrong Architecture

**What goes wrong:** If you instantiate RydbergSurrogate with default hyperparameters but the checkpoint was trained with different values, load_state_dict() will fail with size mismatches.

**How to avoid:** Always reconstruct the model from checkpoint args:
```python
checkpoint = torch.load(path, map_location=device, weights_only=False)
model_args = checkpoint.get('args', {})
model = RydbergSurrogate(
    n_layer=model_args.get('n_layer', 4),
    n_head=model_args.get('n_head', 4),
    n_embd=model_args.get('n_embd', 96),
    dropout=model_args.get('dropout', 0.2),
    mlp_ratio=model_args.get('mlp_ratio', 2),
)
model.load_state_dict(checkpoint['model'])
```
Both scripts/evaluate.py and inference.py already do this correctly.

### Pitfall 7: Running Evaluation on Augmented Data

**What goes wrong:** Using augment=True during evaluation produces noisy metrics that do not reflect true model performance.

**How to avoid:** Always set augment=False for validation and test sets:
```python
train_ds = TrajectoryDataset(train_r, augment=True)
val_ds = TrajectoryDataset(val_r, augment=False)   # NO augmentation
test_ds = TrajectoryDataset(test_r, augment=False) # NO augmentation
```

### Pitfall 8: Forgetting to Call model.eval() Before Inference

**What goes wrong:** If model remains in train mode, dropout is active and predictions become stochastic.

**How to avoid:** Both evaluate.py and inference.py call model.eval(). If writing custom inference code, always call it:
```python
model.eval()
with torch.no_grad():
    pred = model(...)
```

### Pitfall 9: Mismatched Data Path After Moving Code

**What goes wrong:** Several scripts use relative paths like 'outputs/rydberg_dataset.pkl'. If you run scripts from the wrong working directory, FileNotFoundError occurs.

**How to avoid:** Always run scripts from the project root (nanoGPT/). The scripts use Path(__file__) to compute absolute paths for imports, but data paths are relative.

### Pitfall 10: Assuming FNO Installs Easily

**What goes wrong:** The neuraloperator package can have compilation issues on some systems.

**How to avoid:** If pip install neuraloperator fails, try:
```bash
pip install --no-build-isolation neuraloperator
# Or install from source:
git clone https://github.com/neuraloperator/neuraloperator.git
cd neuraloperator && pip install -e .
```

---

## 12. Expected Outputs and How to Interpret Them

### 12.1 Training Output (train.py)

**Console output:**
```
Using device: cuda
Loading dataset from: outputs/rydberg_dataset.pkl
Model parameters: 308,737
Epoch   0 | train_mse=0.123456 | val_mse=0.098765 | val_rho_ss_mae=0.012345 | val_ic_error=0.001234 | lr=1.00e-03
Epoch  10 | train_mse=0.045678 | val_mse=0.032109 | val_rho_ss_mae=0.008901 | val_ic_error=0.000567 | lr=9.90e-04
...
Early stopping at epoch 247
Training complete. Best val MSE: 0.008234
Best model saved to: outputs/models/best_model.pt
```

**What to expect:**
- Initial train MSE: ~0.05-0.2 (random init)
- Final train MSE: ~0.001-0.01 (well-trained)
- Val MSE should track train MSE closely; large gaps indicate overfitting
- val_rho_ss_mae < 0.01 is excellent, < 0.05 is acceptable
- val_ic_error should be very small (< 0.001) since all trajectories start at sz=+1
- lr decreases from 1e-3 to 1e-5 via cosine annealing

**Checkpoint file:** outputs/models/best_model.pt
Contains: model state_dict, optimizer state, scheduler state, epoch, val_mse, args dict, and optional EMA shadow weights.

### 12.2 Evaluation Output (scripts/evaluate.py)

**Console output:**
```
Using device: cuda
Loaded model from: outputs/models/best_model.pt
Trained for 247 epochs

Train metrics:
  mse: 0.001234
  mae: 0.023456
  max_error: 0.345678
  rel_l2: 0.012345
  pearson: 0.998765
  dtw: 1.234567
  rho_ss_mae: 0.004321
  phase_acc: 0.987654
  ic_error: 0.000123
  bounds_violations: 0.000000

Val metrics:
  ...

Test metrics:
  ...

Critical points (true data):
  N=225: Omega_c = 10.150 [9.800, 10.500]
  N=400: Omega_c = 10.800 [10.500, 11.100]
  ...

ROC AUC: 0.995
```

**Metric definitions:**
| Metric | Definition | Good Value | Bad Value |
|---|---|---|---|
| mse | Mean squared error over all (time, sample) | < 0.01 | > 0.1 |
| mae | Mean absolute error over all (time, sample) | < 0.05 | > 0.2 |
| max_error | Worst-case absolute error | < 0.5 | > 1.0 |
| rel_l2 | L2 norm of error / L2 norm of target | < 0.05 | > 0.2 |
| pearson | Mean Pearson r across trajectories | > 0.99 | < 0.95 |
| dtw | Dynamic Time Warping distance (50 samples) | < 5 | > 20 |
| rho_ss_mae | MAE of steady-state order parameter | < 0.01 | > 0.05 |
| phase_acc | Accuracy of active/absorbing classification | > 0.95 | < 0.90 |
| ic_error | Mean absolute error at t=0 | < 0.001 | > 0.01 |
| bounds_violations | Fraction of predictions outside [-1, 1] | 0.0 | > 0.001 |

**Plot files in outputs/evaluation/:**
- trajectory_comparison_n1225.png: Should show predicted dashed lines closely tracking solid true lines
- trajectory_comparison_extrapolation.png: Same for N > 1225; may show more error for largest N
- phase_diagram_predicted_vs_true.png: Both panels should show similar S-curves shifting left with increasing N
- critical_points.png: Omega_c should decrease with N (finite-size scaling toward thermodynamic limit)
- roc_curve.png: Should hug the top-left corner; AUC > 0.95 expected

### 12.3 Inference Output (inference.py)

**Console output:**
```
Using device: cuda
Loaded model from: outputs/models/best_model.pt
Model trained for 247 epochs

Parameters: Omega=11.5, N=1225
Predicted steady-state rho_ss: 0.123456
Predicted steady-state sz_ss:  -0.753088
Phase: ACTIVE

Inference time: 0.852 ms per trajectory
```

**Interpretation:**
- rho_ss > 0.05 means ACTIVE phase (finite excitation density)
- rho_ss < 0.05 means ABSORBING phase (all atoms ground state)
- Inference time ~1 ms is the target; > 10 ms suggests CPU fallback or batching issues

### 12.4 Baseline Outputs

**FNO baseline:** Saves to outputs/baselines/fno_best.pt. Expect similar metrics to transformer but often slightly worse on extrapolation.

**GP baseline:** Saves to outputs/baselines/gp_model.pkl. Expect excellent interpolation but poor extrapolation. Runs on CPU only.

---

## 13. Troubleshooting Guide

### Problem: NaN Loss During Training

**Symptoms:** train_mse becomes nan, val_mse becomes nan, model stops improving.

**Causes & Fixes:**
1. **Learning rate too high:** Reduce --lr to 5e-4 or 1e-4.
2. **Bad data point:** Run scripts/check_data.py and inspect for NaN/Inf in sz_mean.
3. **Gradient explosion:** Gradient clipping (max_norm=1.0) is already enabled. If NaN persists, reduce to 0.5.
4. **Bad initialization:** The model uses std=0.02 init. If problems persist, try torch.nn.init.xavier_uniform_.

**Diagnostic command:**
```bash
python3 -c "
import pickle
with open('outputs/rydberg_dataset.pkl','rb') as f:
    records = pickle.load(f)
import numpy as np
for r in records:
    if np.any(np.isnan(r.sz_mean)) or np.any(np.isinf(r.sz_mean)):
        print(f'BAD: N={r.n_atoms}, Omega={r.omega}')
print('Check complete')
"
```

### Problem: Validation MSE Plateaus High

**Symptoms:** val_mse stays > 0.1 even after 100+ epochs.

**Causes & Fixes:**
1. **Model too small:** Try n_embd=128, mlp_ratio=4.
2. **Not enough training data:** Only 110 training examples. Data augmentation helps but cannot create new physics. Consider adding N=100 back (with caution).
3. **Wrong loss weights:** The bounds and smoothness penalties may dominate. Try reducing bounds_weight to 0.01 or smoothness_weight to 0.001.
4. **Scheduler issue:** CosineAnnealingLR may decay too fast. Try T_max = 2 * max_epochs for slower decay.

### Problem: Overfitting (Train MSE << Val MSE)

**Symptoms:** train_mse ~ 0.001, val_mse ~ 0.05+

**Causes & Fixes:**
1. **Too many parameters:** Already reduced to ~309K. If still overfitting, reduce n_embd to 64.
2. **Insufficient regularization:** Increase dropout to 0.3, increase weight_decay to 1e-3.
3. **Too few training epochs with augmentation:** The model may memorize training data. Ensure augment=True and increase omega_jitter.

### Problem: Phase Classification Accuracy is Low

**Symptoms:** phase_acc < 0.90, ROC AUC < 0.95.

**Causes & Fixes:**
1. **rho_ss threshold too strict/loose:** The default 0.05 is empirically reasonable. Check if true data has trajectories near threshold (rho_ss ~ 0.05) where small errors flip classification.
2. **Model misses late-time behavior:** The last 50 points determine phase. Check rho_ss_mae metric; if high, the model fails to capture steady-state.
3. **Critical region underrepresented:** Most training data may be far from Omega_c. Ensure training sizes cover the critical region adequately.

### Problem: Size Extrapolation Fails

**Symptoms:** Good interpolation (N=1225) but poor extrapolation (N=4900).

**Causes & Fixes:**
1. **Feature engineering:** Ensure inv_sqrt_n is passed to the model. Without it, N is treated as an absolute number rather than a scaling variable.
2. **Training range too narrow:** Training on N=225,400,900 may not span enough dynamic range. Consider adding N=1600 to training (but then lose it as test).
3. **Architecture limit:** Transformers extrapolate better than FNOs but still struggle with out-of-distribution sizes. This is a known limitation -- report it honestly.

### Problem: Evaluation Script Crashes

**Symptoms:** scripts/evaluate.py throws KeyError, RuntimeError, or FileNotFoundError.

**Causes & Fixes:**
1. **Missing checkpoint:** Ensure outputs/models/best_model.pt exists. If training was interrupted early, check outputs/models/ for partial files.
2. **Checkpoint args missing:** Old checkpoints may not have the 'args' key. Update evaluate.py to use hardcoded defaults if args is absent.
3. **Matplotlib backend issues:** The script uses matplotlib.use('Agg') for headless environments. If running with display, remove that line.

### Problem: W&B Login Fails

**Symptoms:** wandb.init() hangs or asks for API key.

**Fix:** Either:
- Run `wandb login` and paste your API key, OR
- Omit --use_wandb flag to disable logging entirely

### Problem: FNO Baseline Import Error

**Symptoms:** from neuralop.models import FNO raises ImportError.

**Fix:** Install neuraloperator: `pip install neuraloperator`. If compilation fails, see Pitfall 10.

---

## 14. What to Do If Something Breaks

This section is a decision tree for when things go wrong. Follow it step by step.

### Scenario A: Training crashes immediately

1. **Check Python environment:** which python3 should point to your venv. If not, run source .venv/bin/activate.
2. **Check dependencies:** pip list | grep torch should show torch>=2.0.0. If missing, run pip install -r requirements.txt.
3. **Check data file:** ls outputs/rydberg_dataset.pkl. If missing, run python3 data/parse_jld2.py.
4. **Check GPU:** python3 -c 'import torch; print(torch.cuda.is_available())' should print True. If False, training runs on CPU (slower but functional).
5. **Check for syntax errors:** If you modified code, run python3 -m py_compile train.py to verify syntax.
6. **Run minimal test:** python3 train.py --max_epochs 1 --batch_size 2 to isolate the crash.

### Scenario B: Training runs but metrics are terrible

1. **Verify data integrity:** python3 scripts/check_data.py. Inspect outputs/data_checks/ for anomalies.
2. **Verify splits:** Run a small Python script to check train/val/test sizes match expectations (train ~110, val ~15-20, test ~160).
3. **Check initial predictions:** After 1 epoch, predictions should be random but bounded. If they are all identical, the model may not be learning.
4. **Compare against random baseline:** A model predicting the mean training sz_mean should achieve MSE ~ 0.05. If your model is worse than this after 50 epochs, something is fundamentally broken.
5. **Inspect loss components:** Add print statements in physics_informed_loss() to see if bounds_penalty or smoothness_penalty dominates MSE.

### Scenario C: Evaluation produces blank or nonsensical plots

1. **Check model loaded correctly:** The evaluate.py output should say 'Loaded model from: ...' and 'Trained for X epochs'. If epochs=0, the checkpoint is corrupt.
2. **Check checkpoint integrity:** Load the checkpoint manually and verify it contains 'model', 'args', 'epoch', and 'val_mse' keys.
3. **Check output directory permissions:** ls -la outputs/evaluation/. If permission denied, fix with chmod 755 outputs/evaluation.
4. **Check matplotlib backend:** If running on a server without display, matplotlib.use('Agg') is correct. If running locally with display and plots do not show, that is expected -- they are saved to files.

### Scenario D: Inference gives wrong phase prediction

1. **Check parameter range:** Omega=11.5, N=1225 is near the critical point. Small errors can flip classification. Try Omega=5 (definitely absorbing) or Omega=20 (definitely active).
2. **Check rho_ss calculation:** inference.py computes rho_ss = np.mean(rho_pred[-50:]). Verify this matches the training definition.
3. **Compare with evaluation:** Run evaluate.py and look at the phase_acc metric. If phase_acc is high but inference is wrong for a specific point, that point may be genuinely ambiguous.

### Scenario E: Ensemble training fails partway through

1. **Check disk space:** Each model checkpoint is ~2MB. 5 models = ~10MB. Ensure outputs/ensemble/ has space.
2. **Check individual model logs:** train_ensemble.py runs train.py as subprocess. If one model fails, the rest continue. Check the specific model directory for error messages.
3. **Resume partially completed ensemble:** If model 3 of 5 failed, manually run train.py with seed=44 and output_dir=outputs/ensemble/model_seed44, then proceed.

### Scenario F: Data parsing produces wrong number of records

1. **Expected:** 290 records. If fewer, some JLD2 files failed to read.
2. **Check logs:** parse_jld2.py prints 'Successfully parsed: X' and 'Skipped/Failed: Y'.
3. **Check raw data:** ls Rydberg_facilitation/results_data_mean/ should show multiple subdirectories with .jld2 files.
4. **If 0 records parsed:** h5py may not be installed, or the JLD2 files are not HDF5-compatible. The thesis data uses JLD2 format v0.2 which is HDF5-based. Very old JLD2 files may not work.

---

## 15. File-by-File Detailed Reference

This section describes every file an agent needs to understand. Read this before modifying anything.

### data/parse_jld2.py

**Purpose:** Parse raw JLD2 thesis data into a structured Python dataset.

**Key classes:**
- TrajectoryRecord (dataclass): Stores omega, delta, gamma, V, Gamma, n_atoms, lattice_size, dimension, sz_mean, t_save, rho, rho_ss, sz_ss, file_path

**Key functions:**
- parse_directory_name(dir_name): Extracts n_atoms, delta, gamma from directory names like 'atoms=1225,=2000.0,=0.1'
- parse_filename(fname): Extracts dimension, omega, delta, gamma from filenames like 'ss_2D,=10.15,=2000.0,=0.1.jld2'
- read_jld2_file(filepath): Uses h5py to read sz_mean and tSave arrays. Returns None on error.
- parse_all_data(data_root): Iterates over all directories and files, skipping unreadable ones. Returns List[TrajectoryRecord].
- save_dataset(records, output_path): Saves records to pickle + a summary pickle with per-size statistics.

**CRITICAL implementation details:**
- Regex uses [0-9]+(?:\.[0-9]+)? to capture both integers and decimals. Previous integer-only regex was a severe bug.
- JLD2 files are read via h5py because JLD2 v0.2+ is HDF5-compatible.
- N_TIME_STEPS = 400 and T_MAX = 1000.0 are hardcoded constants.
- rho = (sz_mean + 1.0) / 2.0 is computed correctly.
- rho_ss uses rho (not sz_mean) averaged over last 50 points.
- Non-square lattices trigger a warning but are still included.

**Output:** outputs/rydberg_dataset.pkl, outputs/rydberg_dataset.summary.pkl

**When to modify:** Only if the data source changes (e.g., new JLD2 files with different naming conventions).

---

### data/dataset.py

**Purpose:** PyTorch Dataset and data splitting utilities.

**Key classes:**
- TrajectoryDataset(Dataset): __getitem__ returns a dict with keys: omega, n_atoms, inv_sqrt_n, t, sz_mean, rho, rho_ss, sz_ss

**Key functions:**
- worker_init_fn(worker_id): Seeds numpy and random per DataLoader worker for reproducible augmentation.
- create_splits(records, train_sizes, val_size, val_omega_indices, test_sizes): Splits data by system size. Default train_sizes=[225,400,900], val_size=1225, test_sizes=[1600,2500,3025,3600,4900].
- load_dataset(pkl_path): Simple pickle loader.
- collate_fn(batch): Stacks batch dicts into tensors.

**CRITICAL implementation details:**
- augment=True adds Gaussian noise to omega (+/-0.05 std) and sz_mean (std=0.005). NO hard clipping.
- inv_sqrt_n = 1.0 / np.sqrt(n_atoms) is computed on-the-fly.
- val_omega_indices='even' means even indices (0,2,4...) go to validation, odd to test interpolation.
- All returned tensors are float32.

**When to modify:** If you want to change splits, add new features, or modify augmentation.

---

### models/transformer_surrogate.py

**Purpose:** The main neural surrogate model.

**Key classes:**
- SelfAttention(nn.Module): Multi-head self-attention WITHOUT causal masking. Uses qkv fused linear + proj.
- TransformerBlock(nn.Module): Pre-norm transformer block: ln1 -> attn -> residual -> ln2 -> mlp -> residual.
- RydbergSurrogate(nn.Module): Main model.

**RydbergSurrogate architecture:**
1. param_embed: nn.Sequential(Linear(3->n_embd), GELU, Linear(n_embd->n_embd)) embeds [omega, n_atoms, inv_sqrt_n]
2. time_embed: Linear(1->n_embd) embeds time coordinates t
3. x = param_embed.unsqueeze(1) + time_embed  # broadcasted addition
4. Pass through n_layer TransformerBlocks
5. ln_f + head(Linear(n_embd->1)) -> squeeze -> sz_pred shape (batch, n_time)

**CRITICAL implementation details:**
- No causal mask. Full bidirectional attention.
- Weight init: Linear weights ~ N(0, 0.02^2), biases = 0, LayerNorm weight = 1, bias = 0.
- count_parameters() sums p.numel() for p.requires_grad.
- Default n_time=400 matches the dataset.

**When to modify:** If changing architecture (e.g., adding causal mask, changing embedding strategy, adding new parameter inputs).

---

### train.py

**Purpose:** Main training loop for the transformer surrogate.

**Key classes:**
- EMA: Exponential Moving Average of model parameters. Stores shadow weights and can apply/restore them.

**Key functions:**
- set_seed(seed): Sets random, numpy, torch, and cudnn seeds for reproducibility.
- physics_informed_loss(pred, target, bounds_weight, smoothness_weight): Computes MSE + bounds penalty + smoothness penalty. Returns dict with 'loss', 'mse', 'bounds_penalty', 'smoothness_penalty'.
- configure_optimizer(model, lr, weight_decay): AdamW with separate weight decay for dim>=2 params (0.0 for biases/LayerNorm).
- train_epoch(model, dataloader, optimizer, device): Single training epoch. Returns {'loss', 'mse'}.
- evaluate(model, dataloader, device): Validation/test evaluation. Returns {'mse', 'mae', 'rho_ss_mae', 'ic_error'}.
- main(args): Full training loop with early stopping, checkpointing, W&B logging, EMA, and cosine LR scheduling.

**CRITICAL implementation details:**
- Gradient clipping: max_norm=1.0 applied every step.
- Scheduler: CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr*0.01).
- Early stopping: patience counter increments when val_mse does not improve. Stops when patience_counter >= patience.
- Best checkpoint saved as outputs/models/best_model.pt (or args.output_dir/best_model.pt).
- Checkpoint includes model, optimizer, scheduler, epoch, val_mse, args, and optionally ema_shadow.
- W&B logging only if --use_wandb is passed.
- num_workers=0 in DataLoader by default (safe for CPU-only machines; increase for GPU servers).

**Command-line args:**
- --data_path: default 'outputs/rydberg_dataset.pkl'
- --output_dir: default 'outputs/models'
- --n_layer, --n_head, --n_embd, --dropout, --mlp_ratio: model architecture
- --batch_size, --lr, --weight_decay, --max_epochs, --patience: training hyperparameters
- --use_ema, --ema_decay: EMA settings
- --use_wandb, --wandb_project, --run_name: W&B settings
- --seed: random seed (default 42)

**When to modify:** If changing loss function, optimizer, scheduler, logging, or checkpointing strategy.

---

### train_ensemble.py

**Purpose:** Wrapper that trains N models with different seeds for uncertainty quantification.

**How it works:**
- Loops from i=0 to n_models-1
- seed = base_seed + i
- output_dir = output_dir/model_seed{seed}
- Constructs a command list and runs train.py via subprocess.run()
- If a model fails (returncode != 0), prints a warning and continues

**CRITICAL implementation details:**
- Does NOT use multiprocessing or joblib. Runs models sequentially.
- Always passes --use_ema to individual train.py calls.
- W&B can be enabled with --use_wandb but run names are not customized per model.
- If interrupted, you must manually identify which seeds completed and resume the rest.

**When to modify:** If you want parallel ensemble training, custom per-model hyperparameters, or ensemble aggregation logic.

---

### scripts/evaluate.py

**Purpose:** Comprehensive evaluation of a trained model on all data splits.

**Key functions:**
- dtw_distance(x, y): Simple O(n*m) Dynamic Time Warping implementation. Only computed on first 50 test samples for speed.
- evaluate_model(model, dataloader, device): Returns metrics dict + all_pred, all_true, all_omega, all_n_atoms arrays.
- critical_point_analysis(omegas, rho_ss_vals, n_bootstrap): Finds Omega_c by maximum derivative of rho_ss vs Omega. Bootstrap resamples to compute 95% CI.
- plot_results(...): Generates 5 plots (trajectory comparison, extrapolation, phase diagram, critical points, ROC curve).

**Metrics computed:**
- mse, mae, max_error, rel_l2, pearson, dtw, rho_ss_mae, phase_acc, ic_error, bounds_violations

**CRITICAL implementation details:**
- Loads model architecture from checkpoint['args']. Falls back to defaults if missing.
- PHASE_THRESHOLD_RHO = 0.05 hardcoded.
- Plots only generated for Test split (not Train/Val) to avoid clutter.
- Critical points computed from TRUE data, not predictions. This estimates the ground-truth phase boundary.
- ROC curve uses pred_rho_ss as score and true phase as label.
- matplotlib.use('Agg') ensures headless compatibility.

**When to modify:** If adding new metrics, new plots, or changing evaluation strategy.

---

### inference.py

**Purpose:** Load a trained model and predict sz_mean(t) for arbitrary (Omega, N) parameters.

**Key functions:**
- load_model(model_path, device): Loads checkpoint, reconstructs model from args, loads state_dict, sets eval mode.
- predict(model, omega, n_atoms, t_max, n_time, device): Returns (t_array, sz_pred_array) for a single parameter point.
- main(args): Loads model, runs prediction, computes rho_ss/sz_ss/phase, benchmarks inference speed over 100 runs, optionally saves to .npz.

**CRITICAL implementation details:**
- Always constructs t = torch.linspace(0, t_max, n_time) with dtype float32.
- inv_sqrt_n is computed as 1.0 / np.sqrt(n_atoms) automatically.
- Phase threshold: rho_ss > 0.05 -> ACTIVE, else ABSORBING.
- Speed benchmark runs 100 forward passes and reports mean ms per trajectory.
- Output .npz contains: t, sz_mean, rho, omega, n_atoms.

**When to modify:** If you want batch inference, different output formats, or additional derived quantities.

---

### baselines/fno_baseline.py

**Purpose:** Fourier Neural Operator baseline for comparison.

**Key classes:**
- FNOBaseline(nn.Module): Wraps neuralop.models.FNO with 4 input channels (omega, n_atoms, inv_sqrt_n, t) and 1 output channel (sz_mean).

**CRITICAL implementation details:**
- Uses positional_embedding='grid' (FNO default).
- Loss is pure MSE (no physics-informed penalties unlike train.py).
- Gradient clipping max_norm=1.0 applied.
- Saves best checkpoint to output_dir/fno_best.pt.
- Test evaluation is performed automatically at the end.

**When to modify:** If changing FNO hyperparameters (n_modes, hidden_channels) or adding physics-informed loss.

---

### baselines/gp_baseline.py

**Purpose:** Gaussian Process baseline using scikit-learn.

**Key functions:**
- prepare_data(records, time_subsample): Flattens trajectories into (omega, n, t) -> sz_mean pairs. time_subsample=5 by default.
- train_gp(train_records, kernel, time_subsample): Fits GaussianProcessRegressor with ConstantKernel * RBF + WhiteKernel.
- evaluate_gp(gp, records): Evaluates per-trajectory MSE, MAE, rho_ss_MAE.

**CRITICAL implementation details:**
- Kernel: C(1.0, (1e-3, 1e3)) * RBF([1.0, 100.0, 10.0], (1e-2, 1e3)) + WhiteKernel(1e-5, (1e-10, 1e-1))
- normalize_y=True standardizes targets.
- n_restarts_optimizer=2 for kernel hyperparameter optimization.
- Saved via joblib.dump() to outputs/baselines/gp_model.pkl.
- Only evaluates on validation and test sets (no training set evaluation to save time).

**When to modify:** If changing kernel, adding more training data, or switching to GPyTorch for GPU acceleration.

---

### scripts/check_data.py

**Purpose:** Data quality checks and visualizations.

**Key functions:**
- check_anomalies(records): Checks for NaN, Inf, wrong shapes, and incorrect initial conditions (sz_mean[0] should be ~+1.0).
- plot_representative_trajectories(records, output_dir): 6-panel plot showing absorbing, near-critical, and active trajectories for N=1225.
- plot_phase_diagram(records, output_dir): Grid of rho_ss vs Omega plots, one per system size.
- plot_combined_phase_diagram(records, output_dir): Single figure with all sizes color-coded.
- plot_sz_distribution(records, output_dir): Histogram of all sz_mean values (linear and log scale).
- plot_data_coverage(records, output_dir): Scatter plot showing Omega coverage per system size.
- print_summary_stats(records): Prints total records, sizes, time range, sz_mean statistics, absorbing vs active counts.

**Output:** 5 PNG files in outputs/data_checks/.

**When to modify:** If adding new data quality checks or visualizations.

---

## 16. Appendices

### Appendix A: Audit History

Three independent audits were conducted on 2026-04-24:

1. **Physics Audit** (AUDIT_PHYSICS.md) -- Found rho_ss naming bug, mixup creating unphysical trajectories, unjustified steady-state extraction
2. **Engineering Audit** (AUDIT_ENGINEERING.md) -- Found regex truncating decimals, broken LR scheduler, save_path NameError, unfair baseline comparisons
3. **Logic Audit** (AUDIT_LOGIC.md) -- Found parameter count overrun (~810K vs ~500K), missing deep ensembles, missing critical exponent extraction, plan-to-code gaps

All critical issues were fixed in the same session.

### Appendix B: Quick Reference

#### Order Parameter Definitions

```python
rho = (sz_mean + 1.0) / 2.0      # DP order parameter: 0 = absorbing, >0 = active
rho_ss = np.mean(rho[-50:])       # Steady-state order parameter
sz_ss = np.mean(sz_mean[-50:])    # Steady-state sz_mean
```

#### Phase Classification

```python
PHASE_THRESHOLD_RHO = 0.05        # Active if rho_ss > 0.05 (equivalent to sz_ss > -0.9)
```

#### DP Exponents (2D, for reference)

| Exponent | Symbol | Value |
|---|---|---|
| Order parameter | beta | 0.583 |
| Spatial correlation | nu_perp | 0.733 |
| Temporal correlation | nu_parallel | 1.295 |
| Dynamic | z | 1.766 |
| Survival probability | delta | 0.450 |

#### Fixed Simulation Parameters

| Parameter | Value |
|---|---|
| V (interaction) | 2000 |
| Delta (detuning) | 2000 |
| Gamma (spontaneous emission) | 1 |
| gamma (dephasing) | 0.1 |
| t_max | 1000 |
| n_T (time steps) | 400 |
| Dimension | 2D |
| Trajectories per point | 500 |

### Appendix C: Glossary for AI Agents

| Term | Meaning |
|---|---|
| TWA | Truncated Wigner Approximation -- semiclassical simulation method |
| DP | Directed Percolation -- universality class of the phase transition |
| rho | Order parameter = (sz_mean + 1) / 2, ranges [0, 1] |
| rho_ss | Steady-state value of rho, averaged over last 50 time points |
| Omega | Rabi frequency -- driving strength parameter |
| N | Number of atoms (system size) |
| sz_mean | Mean spin-z expectation value, ranges [-1, 1] |
| JLD2 | Julia data format (HDF5-based) used for raw simulation outputs |
| FNO | Fourier Neural Operator -- baseline architecture |
| GP | Gaussian Process -- baseline method |
| EMA | Exponential Moving Average of model weights for inference |
| DTW | Dynamic Time Warping -- trajectory similarity metric |
| ROC AUC | Area Under the Receiver Operating Characteristic curve |

### Appendix D: Contact and Context

- This project lives in /Users/hoangnguyen/nanoGPT/
- The nanoGPT repo was repurposed from Andrej Karpathy's nanoGPT; original GPT-2 code is still present (model.py, sample.py, config/) but unused for this project.
- Original Rydberg simulation code is NOT in this repo. Only the post-processed mean-field data is present.
- If you need to regenerate raw data, you need the original Julia TWA code (not available in this repo).

---

## 17. Publication Feasibility & Success Criteria

**This section tells you what targets the model MUST hit for the paper to be publishable. Read this before training.**

### 17.1 The Brutal Truth

This project is **scientifically sound but scientifically thin.** The methodology is rigorous, the physics is real, but the scientific payload — what we learn that physicists didn't already know — depends entirely on quantitative results.

A reviewer will ask: *"Why should I care about a neural network that reproduces 500-trajectory TWA averages when I can just run the TWA?"*

Your answer must demonstrate **at least two of these three:**
1. **Speed** — ms vs. minutes enables rapid parameter screening
2. **Generalization** — zero-shot prediction of unseen system sizes (N=4900 from training on N≤900)
3. **Insight** — model encodes physical phases in its representations

### 17.2 Venue-Specific Probabilities

| Venue | Probability | Why |
|---|---|---|
| **Physical Review Research** | ~35% | Needs strong size extrapolation + baseline wins |
| **Machine Learning: Science and Technology** | ~55% | Better target — values methodology over physics novelty |
| **PRX Quantum** | <10% | Not feasible without spatial data + new physics discovery |
| **Nature Physics** | <5% | Not feasible |
| **NeurIPS/ICML AI4Science** | ~30% | Needs architectural novelty + larger scale |

**Recommendation:** Train first, measure results, then decide. If size extrapolation is strong → PR Research. If moderate → MLST. If weak → arXiv methods note only.

### 17.3 Minimum Viable Paper (PR Research)

These are the hard numbers you MUST achieve:

| Metric | Target | What it proves |
|---|---|---|
| Test MSE (interpolation, N=1225) | < 0.01 | Model learns the critical region accurately |
| Test MSE (size extrapolation, N=4900) | < 0.02 | **The killer result** — generalizes across 5x size gap |
| Phase accuracy | > 90% | Can classify absorbing vs. active reliably |
| ROC AUC | > 0.85 | Phase classification is robust |
| Beats FNO | Yes, on ≥3 metrics | Transformer choice is justified |
| Beats GP (interpolation) | Yes | Beats gold-standard interpolation baseline |
| Speedup vs. TWA | > 100x | Practical utility for parameter screening |
| Relative error at N=4900 | < 10% | Size extrapolation is not just noise |

### 17.4 If Results Are Strong (Go for PR Research)

**Title:** "A Transformer-Based Neural Surrogate for Non-Equilibrium Rydberg Facilitation Dynamics"

**Key claims:**
- Accurate prediction of mean-field dynamics across parameter space
- Zero-shot generalization to unseen system sizes
- 100x+ speedup over TWA with <5% error
- Competitive with or superior to neural operators on small data

**Need to add:**
- Linear interpolation baseline (simple but necessary)
- Data efficiency curve (train on 25%, 50%, 75% of data)
- Ablations (remove parameter conditioning, replace transformer with MLP)

### 17.5 If Results Are Moderate (Go for MLST)

**Title:** "Small-Data Operator Learning for Quantum Dynamics: A Comparative Study"

**Key claims:**
- Systematic comparison of transformers, FNOs, and GPs on small-data regime
- Data efficiency analysis (how many trajectories are needed?)
- Uncertainty quantification via deep ensembles
- Calibration plots

**This framing does NOT require strong size extrapolation.** It values rigorous benchmarking over physics novelty.

### 17.6 If Results Are Weak (Do Not Publish in Journal)

**Do NOT submit to PR Research or MLST.**

Options:
1. Write an arXiv preprint as a methods note
2. Use the codebase as a starting point for Phase 6 (spatial extension)
3. Collect more data (vary gamma, Delta, V) and retry

### 17.7 Red Flags That Kill the Paper

If you see any of these after training, the paper is in serious trouble:

| Red Flag | Severity | What to do |
|---|---|---|
| Test MSE > 0.05 on interpolation | **Critical** | Model did not learn basic dynamics. Check for bugs. |
| Test MSE > 0.10 on size extrapolation | **Critical** | Size extrapolation failed. Switch to MLST framing or don't publish. |
| FNO beats transformer on ALL metrics | **Critical** | Transformer choice is unjustified. Need architectural innovation to justify it. |
| Ensemble variance > 20% of prediction | **High** | Model is not confident. Need more data or stronger regularization. |
| Phase accuracy < 80% | **High** | Cannot reliably classify phases. The transition is the main feature. |
| No speedup vs. TWA measured | **Medium** | Practical utility claim is unsubstantiated. |
| Model predicts values outside [-1, 1] frequently | **Medium** | Physics constraints not learned. Add tanh output or stronger bounds penalty. |

### 17.8 The One Thing That Matters Most

**Size extrapolation to N=4900.**

This is the only result that a physicist would find surprising. Everything else (interpolation, speedup, phase classification) is expected from a well-trained neural network.

If the model trained on N=225, 400, 900 can predict N=4900 accurately, you have a paper. If it cannot, you have a methods note.

**Measure this first. Everything else is secondary.**

---

*End of AGENTS.md*
