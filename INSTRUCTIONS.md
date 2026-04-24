# Instructions for PC Kimi

## 1. What This Project Is

This is a **neural surrogate for Rydberg non-equilibrium phase transitions**. The goal is to train a small transformer (~310K parameters) to predict the full time evolution ⟨S_z(t)⟩ of driven-dissipative Rydberg lattices, given physical parameters (Ω, N, γ, dimension).

The surrogate learns from **ensemble-averaged TWA (Truncated Wigner Approximation) trajectories**. TWA is expensive (hours per phase diagram). The surrogate runs ~10⁴× faster.

**The paper arc:** Physics problem → TWA breakthrough → TWA bottleneck → surrogate solution. See `PAPER_OUTLINE_FINAL.md` for the full structure.

---

## 2. Environment Setup

```bash
cd rydberg-neural-surrogate
pip install -r requirements.txt
```

**Expected packages:** PyTorch 2.x, numpy, scipy, matplotlib, scikit-learn, joblib, neuraloperator (for FNO baseline).

**Key file:** `data/rydberg_dataset_v2.pkl` (3.6MB) — contains 759 parsed trajectories. This is already in the repo.

---

## 3. Codebase Structure

| File/Dir | Purpose |
|----------|---------|
| `data/rydberg_dataset_v2.pkl` | Dataset (759 trajectories, parsed from Julia JLD2) |
| `data/dataset_v2.py` | PyTorch Dataset, `create_splits()`, `collate_fn` |
| `data/parse_jld2_v2.py` | Parser (already run — dataset is pre-built) |
| `models/transformer_surrogate.py` | `RydbergSurrogate` — 4-layer transformer, ~310K params |
| `train.py` | Main training script |
| `train_ensemble.py` | Trains 5 models with different seeds (uncertainty quantification) |
| `inference.py` | Single-trajectory inference + speed benchmark |
| `scripts/evaluate_v2.py` | **Main evaluation** — generates all paper figures + tables |
| `scripts/extract_critical_exponents.py` | Extracts β, δ from trajectories |
| `scripts/plot_*.py` | Individual plotting scripts for specific analyses |
| `baselines/fno_baseline_v2.py` | FNO baseline training |
| `baselines/gp_baseline_v2.py` | GP baseline training |

---

## 4. Dataset Splits (CRITICAL — do not change)

The `create_splits()` function in `data/dataset_v2.py` enforces these splits:

| Split | Conditions | Count |
|-------|-----------|-------|
| **Train** | 2D, γ=0.1, N∈{225,400,900,1600,2500} | 105 |
| **Val** | 2D, γ=0.1, N=3600 | 21 |
| **Test (size)** | 2D, γ=0.1, N=4900 | 19 |
| **Test (γ)** | 2D, N=3600, γ∈{0.1,5,10,20} | 85 |

**Scope is strictly 2D.** Do not add 1D/3D to training or testing.

---

## 5. Training

### 5.1 Single Model

```bash
python train.py \
  --max_epochs 1000 \
  --patience 100 \
  --batch_size 32 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --use_ema \
  --output_dir outputs/models
```

**Expected:** Best val MSE should drop below 0.005. The 41-epoch test run on MPS got 0.0087. On GPU with 1000 epochs, target is ~0.003–0.005.

**Checkpoint saved to:** `outputs/models/best_model.pt`

### 5.2 Deep Ensemble (5 models)

```bash
python train_ensemble.py --n_models 5 --max_epochs 1000 --patience 100 --use_ema
```

**Checkpoints saved to:** `outputs/models/model_seed{42,43,44,45,46}/best_model.pt`

**Why ensemble:** Uncertainty quantification. The evaluation script can load all 5 and compute mean ± std.

### 5.3 Training Tips

- If val MSE plateaus early, increase `patience` or reduce `lr` to 5e-4
- If overfitting (train MSE << val MSE), increase `dropout` to 0.3 or reduce model size
- The dataset is small (105 train) — regularization matters. EMA is strongly recommended

---

## 6. Evaluation (Generate Paper Figures)

### 6.1 Main Evaluation Script

```bash
python scripts/evaluate_v2.py --model_path outputs/models/best_model.pt
```

**This generates 4 paper figures + metrics table + saved predictions:**

| Output | File | Description |
|--------|------|-------------|
| **Fig. 1** | `outputs/evaluation/fig1_trajectory_overlays.png` | True vs predicted trajectories (6 panels) |
| **Fig. 2** | `outputs/evaluation/fig2_size_extrapolation.png` | N=4900 dynamics + phase diagram |
| **Fig. 3** | `outputs/evaluation/fig3_gamma_transfer.png` | γ=5,10,20 dynamics + phase diagrams |
| **Fig. 4** | `outputs/evaluation/fig4_critical_scaling.png` | Data collapse + log-log decay |
| **Table 1** | printed to console | MSE, MAE, ρ_ss MAE, IC error per split |
| **Predictions** | `outputs/predictions/predictions_*.pkl` | Saved trajectories for reuse |

### 6.2 Critical Exponent Extraction

```bash
python scripts/extract_critical_exponents.py
```

**This extracts β and δ** from the raw dataset (not predictions). To extract from predictions, you would need to point it at `outputs/predictions/` — but the script is currently hardcoded to `data/rydberg_dataset_v2.pkl`. Modify if needed.

### 6.3 Individual Plotting Scripts

```bash
# Log-log dynamics
python scripts/plot_loglog_dynamics.py

# Data collapse (t^δ ρ vs t|Ω−Ω_c|^(β/δ))
python scripts/plot_data_collapse.py

# Finite-size z-tuning collapse
python scripts/plot_finite_size_collapse_z.py

# Finite-size scaling extrapolation
python scripts/plot_finite_size_extrap.py

# Dynamics per N
python scripts/plot_dynamics_per_N.py

# Gamma phase transitions
python scripts/plot_gamma_phase_transitions.py

# Gamma dynamics
python scripts/plot_gamma_dynamics.py
```

**All outputs go to `outputs/critical_analysis/` or `outputs/finite_size/`.**

---

## 7. Baselines (for Table 1/Table 2 comparison)

### 7.1 FNO Baseline

```bash
python baselines/fno_baseline_v2.py \
  --max_epochs 1000 \
  --patience 100 \
  --batch_size 32 \
  --lr 1e-3 \
  --output_dir outputs/models/fno
```

### 7.2 GP Baseline

```bash
python baselines/gp_baseline_v2.py \
  --time_subsample 5 \
  --output_dir outputs/models/gp
```

**Note:** GP trains on CPU. Time subsample 5 means it uses every 5th time point. Lower = more accurate but slower.

---

## 8. Inference (Speed Benchmark)

```bash
python inference.py \
  --model_path outputs/models/best_model.pt \
  --omega 11.5 \
  --n_atoms 3600 \
  --gamma 0.1 \
  --dimension 2
```

**Expected output:** ~1 ms per trajectory on GPU. Compare to TWA: ~minutes per trajectory.

---

## 9. Paper Figures and Tables (Checklist)

### Main Text Figures (4)

| # | Script | What to verify |
|---|--------|---------------|
| **Fig. 1** | `scripts/evaluate_v2.py` | Trajectory overlays match TWA closely |
| **Fig. 2** | `scripts/evaluate_v2.py` | N=4900 phase diagram shows proper transition (not flat) |
| **Fig. 3** | `scripts/evaluate_v2.py` | γ transfer shows critical point shift left with increasing γ |
| **Fig. 4** | `scripts/evaluate_v2.py` | Data collapse curves actually collapse; log-log decay follows t^(−δ) |

### Main Text Tables (2)

| # | Source | Content |
|---|--------|---------|
| **Table 1** | `scripts/evaluate_v2.py` console output | MSE, MAE, Pearson r, ρ_ss MAE, phase accuracy, IC error. Merge with baseline metrics. |
| **Table 2** | Run `scripts/extract_critical_exponents.py` on predictions | β, δ, Ω_c with 95% CIs. Columns: Thesis TWA \| Transformer \| FNO \| GP. |

### Supplementary Figures

- `plot_loglog_dynamics.py` → log-log plots for all N
- `plot_data_collapse.py` → collapse for each N
- `plot_finite_size_collapse_z.py` → z-tuning plots
- `plot_finite_size_extrap.py` → finite-size scaling

---

## 10. Common Issues and Fixes

### Issue: `ModuleNotFoundError: Can't get attribute 'TrajectoryRecord'`
**Fix:** Add this before `pickle.load()`:
```python
import sys
sys.path.insert(0, '.')
from data.parse_jld2_v2 import TrajectoryRecord
```

### Issue: Model predicts flat ρ_ss (no phase transition)
**Cause:** Undertrained. The 41-epoch test run showed this.
**Fix:** Train for 500+ epochs. The phase transition structure emerges with more training.

### Issue: `IndexError` in `evaluate_v2.py`
**Fix:** Make sure `all_records_flat` is constructed correctly in `main()`. Use the `split_records_map` dict.

### Issue: Out of memory during training
**Fix:** Reduce `--batch_size` to 16 or 8. The model is small (310K params) but padding to max time length can use memory.

---

## 11. What Success Looks Like

After full GPU training (~500–1000 epochs), you should see:

| Metric | Target |
|--------|--------|
| Val MSE | < 0.005 |
| Test (N=4900) MSE | < 0.01 |
| Test (γ) MSE | < 0.01 |
| ρ_ss MAE | < 0.03 |
| Phase accuracy | > 95% |
| IC error | < 0.05 |

**Visual checks:**
- Fig 1: Surrogate curves (dashed) nearly overlap TWA (solid)
- Fig 2 (N=4900): Phase diagram shows S-curve transition around Ω_c ≈ 11.2
- Fig 3 (γ transfer): Higher γ → lower Ω_c and broader transition
- Fig 4: Data collapse curves overlap; log-log decay slope ≈ −δ = −0.4577

---

## 12. Next Steps After Training

1. **Train the main model** → `outputs/models/best_model.pt`
2. **Train the ensemble** → 5 checkpoints in `outputs/models/model_seed*/`
3. **Run baselines** → FNO + GP checkpoints
4. **Run evaluation** → generate all figures
5. **Extract exponents** → from predictions, compare to thesis
6. **Write paper** → follow `PAPER_OUTLINE_FINAL.md`

---

## 13. Contact / Context

- This repo was prepared on macOS (MPS) but designed for CUDA training
- The dataset comes from a Master Thesis on TWA for Rydberg NEPT
- Thesis reference values: Ω_c = 11.2, β = 0.586, δ = 0.4577, z = 1.86 (2D low-γ)
- All exponents are **effective exponents** from mean-field TWA data, not true DP exponents

**Do not modify the dataset splits without asking.** The 2D-only, γ=0.1 training setup is intentional.
