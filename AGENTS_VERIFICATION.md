# AGENTS.md Technical Verification Report

**Verifier:** Senior ML Engineer (skeptical mode)  
**Date:** 2026-04-24  
**Method:** Read every claim in `AGENTS.md`, cross-referenced against actual source code, executed commands where possible.

---

## Executive Summary

`AGENTS.md` is **mostly accurate** for the Rydberg-specific codebase, but contains **one critical execution failure** (`python3` without venv), **wildly inflated hardware requirements**, and an **incomplete file inventory**. An agent following this guide blindly would waste time on the first command if the venv isn't active, and might provision unnecessary GPU memory.

**Overall grade: B+ (good for cold-start, needs corrections before trusting at scale)**

---

## 1. Parameter Count (~309K)

| Claim | Status |
|---|---|
| "Total params: ~309K" with n_layer=4, n_head=4, n_embd=96, mlp_ratio=2 | **PASS** |

**Verification:**
```bash
$ python models/transformer_surrogate.py
Model parameters: 309,313
```

The count is exact (309,313). The hyperparameter table in section 3.2 matches `RydbergSurrogate.__init__` defaults.

---

## 2. Data Split Description

| Claim | Status |
|---|---|
| Train: N=225, 400, 900 | **PASS** |
| Val: N=1225 (even Omega indices) | **PASS** |
| Test (interpolation): N=1225 (odd Omega indices) | **PASS** |
| Test (size extrapolation): N=1600, 2500, 3025, 3600, 4900 | **PASS** |
| N=100 dropped from training | **PASS** |

**Verification:** Checked `data/dataset.py` `create_splits()` defaults:
```python
train_sizes = [225, 400, 900]
val_size = 1225
val_omega_indices = 'even'
test_sizes = [1600, 2500, 3025, 3600, 4900]
```
N=100 exists in the raw parsed data (21 records) but is absent from all three splits.

---

## 3. Loss Function Description

| Claim | Status |
|---|---|
| `loss = MSE + 0.1 * bounds_penalty + 0.01 * smoothness_penalty` | **PASS** |
| Bounds penalty is soft (not hard constraint) | **PASS** |
| Smoothness penalty is generic Tikhonov, NOT true physics-informed | **PASS** |

**Verification:** `train.py` line 70-93:
```python
def physics_informed_loss(pred, target, bounds_weight=0.1, smoothness_weight=0.01):
    mse = F.mse_loss(pred, target)
    lower_violation = torch.relu(-1.0 - pred)
    upper_violation = torch.relu(pred - 1.0)
    bounds_penalty = (lower_violation + upper_violation).mean()
    d2 = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    smoothness_penalty = (d2 ** 2).mean()
    total_loss = mse + bounds_weight * bounds_penalty + smoothness_weight * smoothness_penalty
```
The code comments explicitly state "generic Tikhonov, not physics-informed" — matching the document's physics caveats.

---

## 4. Bash Commands in "How to Run"

### 4.1 Quick Sanity Check (Section 7.1)

| Command | Status | Evidence |
|---|---|---|
| `python3 train.py --max_epochs 5 --patience 10 --batch_size 16` | **PASS** | Executed with `--max_epochs 2`, completed successfully |
| `python3 scripts/evaluate.py --model_path outputs/models/best_model.pt` | **PASS** | Executed successfully, produced all outputs |
| `python3 inference.py --model_path outputs/models/best_model.pt --omega 11.5 --n_atoms 1225` | **PASS** | Executed successfully, produced prediction and benchmark |

### 4.2 Full Transformer Training (Section 7.2)

| Argument | Exists in `train.py`? | Status |
|---|---|---|
| `--max_epochs 500` | Yes | PASS |
| `--patience 50` | Yes | PASS |
| `--batch_size 32` | Yes | PASS |
| `--lr 1e-3` | Yes | PASS |
| `--weight_decay 1e-4` | Yes | PASS |
| `--n_layer 4` | Yes | PASS |
| `--n_head 4` | Yes | PASS |
| `--n_embd 96` | Yes | PASS |
| `--dropout 0.2` | Yes | PASS |
| `--use_ema` | Yes (store_true) | PASS |
| `--ema_decay 0.999` | Yes | PASS |
| `--use_wandb` | Yes (store_true) | PASS |
| `--wandb_project rydberg-surrogate` | Yes | PASS |
| `--run_name surrogate_run1` | Yes | PASS |

### 4.3 Deep Ensemble Training (Section 7.3)

| Argument | Exists in `train_ensemble.py`? | Status |
|---|---|---|
| `--n_models 5` | Yes | PASS |
| `--base_seed 42` | Yes | PASS |
| `--output_dir outputs/ensemble` | Yes | PASS |

Seed logic verified: `train_ensemble.py` uses `seed = args.base_seed + i` for `i in range(args.n_models)`, producing seeds 42, 43, 44, 45, 46. Output path `outputs/ensemble/model_seed{seed}/best_model.pt` matches the code.

### 4.4 FNO Baseline (Section 7.4)

| Argument | Exists in `baselines/fno_baseline.py`? | Status |
|---|---|---|
| `--epochs 500` | Yes | PASS |
| `--patience 50` | Yes | PASS |
| `--batch_size 16` | Yes | PASS |
| `--n_modes 32` | Yes | PASS |
| `--output_dir outputs/baselines` | Yes | PASS |

### 4.5 GP Baseline (Section 7.5)

| Command | Status |
|---|---|
| `python3 baselines/gp_baseline.py --time_subsample 5` | **PASS** |

Verified: `gp_baseline.py` accepts `--time_subsample` with default 5. The comment "every 5th point" is accurate.

### 4.6 Full Evaluation (Section 7.6)

| Command | Status |
|---|---|
| `python3 scripts/evaluate.py --model_path outputs/models/best_model.pt --output_dir outputs/evaluation` | **PASS** |

Output files claimed:
- `trajectory_comparison_n1225.png` — **produced** ✓
- `trajectory_comparison_extrapolation.png` — **produced** ✓
- `phase_diagram_predicted_vs_true.png` — **produced** ✓
- `critical_points.png` — **produced** ✓
- `roc_curve.png` — **produced** ✓

---

## 5. Commands That Would Fail If Run As Written

### 🔴 CRITICAL: `python3` requires active venv

**Claim:** All commands use `python3`.
**Reality:** On the test system (macOS), `python3` outside the activated virtual environment resolves to system Python, which lacks all installed packages.

**Reproduction:**
```bash
$ python3 -c "import torch"
ModuleNotFoundError: No module named 'torch'
```

Inside the venv it works, but the document only shows activation once at the top of section 6.2. If an agent or user opens a fresh terminal for section 7, every command fails immediately.

**Correction needed:** Add a reminder before section 7: "Ensure your virtual environment is activated: `source .venv/bin/activate`" OR change all commands to `.venv/bin/python3 ...` for robustness.

### 🟡 Minor: `python3 data/parse_jld2.py` works but regenerates existing data

Not a failure, but worth noting: if `outputs/rydberg_dataset.pkl` already exists, the script overwrites it silently. The document says "If missing, regenerate it," which is safe.

---

## 6. File Structure Diagram Accuracy

### What the diagram gets RIGHT:
- All Rydberg-specific files and directories are listed and exist.
- `configs/` is correctly labeled as EMPTY.
- `outputs/` subdirectories (`data_checks`, `models`, `baselines`, `evaluation`) all exist.
- `Rydberg_facilitation/results_data_mean/` exists with 302 JLD2 files.

### What the diagram gets WRONG or OMITS:

| Missing Item | Significance |
|---|---|
| `config/` (singular) directory | Contains 7 original nanoGPT config files. Could be confused with `configs/` (plural, empty). |
| `assets/` directory | Contains project images (`nanogpt.jpg`, `gpt2_124M_loss.png`). |
| `data/openwebtext/`, `data/shakespeare/`, `data/shakespeare_char/` | Original nanoGPT data directories still present. |
| `README.md` | Original nanoGPT README. A human/agent seeing this first would think it's a pure nanoGPT repo. |
| `LICENSE` | MIT license from nanoGPT. |
| `bench.py`, `configurator.py`, `model.py`, `sample.py` | Original nanoGPT source files at repo root. |
| `scaling_laws.ipynb`, `transformer_sizing.ipynb` | Original nanoGPT notebooks. |
| `MasterThesis/` directory | Contains the full LaTeX thesis, figures, and raw materials. Large and significant. |
| `PHYSICS_REVIEW.md`, `PROJECT_REVIEW.md` | Additional review documents not mentioned. |
| `write_audit.py`, `gptqe.pdf`, `long_test*.md` | Helper/artifact files. |

**Status:** PARTIAL FAIL. The diagram is accurate for what it shows, but silently omits ~30% of the repository. An agent might delete "unused" original nanoGPT files thinking they're safe to remove.

---

## 7. Outdated References to Removed Features

### Mixup
- **AGENTS.md claim:** Appendix A says Physics Audit "Found ... mixup creating unphysical trajectories."
- **Current code:** `data/dataset.py` contains ZERO references to mixup. Verified with `grep -i mixup data/dataset.py` — no matches.
- **Status:** PASS. The reference is historical (audit found it, it was fixed), not a live feature.

### Tokenization
- **AGENTS.md claim:** Section 3.1 says "No tokenization: Continuous scalar outputs."
- **Current code:** `models/transformer_surrogate.py` uses continuous `nn.Linear` regression. No token embedding layer, no vocabulary, no softmax.
- **Status:** PASS. The claim is accurate.

### Autoregressive generation
- **AGENTS.md claim:** "Abandoned autoregressive token generation (v1.0) in favor of direct regression (v2.0)."
- **Current code:** No autoregressive code exists in the active Rydberg files.
- **Status:** PASS (historical claim, not contradicted by current code).

---

## 8. Does the Document Mention ALL Files in the Repo?

**NO.**

Complete inventory of files/directories at repo root NOT mentioned in AGENTS.md:

```
assets/
bench.py
config/
configurator.py
data/openwebtext/
data/shakespeare/
data/shakespeare_char/
gptqe.pdf
LICENSE
long_test.md
long_test2.md
long_test3.md
long_test4.md
long_test5.md
MasterThesis/
model.py
PHYSICS_REVIEW.md
PROJECT_REVIEW.md
README.md
sample.py
scaling_laws.ipynb
shell_test.md
test_write.md
transformer_sizing.ipynb
write_audit.py
```

**Risk:** An agent operating under the assumption that AGENTS.md is exhaustive might treat these as "safe to delete" or "irrelevant." In reality, `README.md` and `LICENSE` should not be removed, and `MasterThesis/` contains source data/figures that may be needed for the paper.

---

## 9. Additional Technical Claims Verified

| Claim | Source | Verification | Status |
|---|---|---|---|
| "~10GB GPU memory for training" | Section 6.1 | Model is 309K params. At batch_size=32, this uses <1GB VRAM. | **FAIL** |
| "~5GB disk space for data + checkpoints" | Section 6.1 | Dataset pickle = 2.8MB. Checkpoint = 3.8MB. Total with all outputs <100MB. | **FAIL** |
| "Early stopping: patience = 100 epochs" | Section 3.5 table | `train.py` default is 100. | **PASS** |
| But example uses `--patience 50` | Section 7.2 | Contradicts table without explanation. | **WARNING** |
| "Dropout: 0.2" | Section 3.5 | `train.py` default `--dropout 0.2`. | **PASS** |
| "Gradient clipping: max_norm = 1.0" | Section 3.5 | `train.py` line 129: `clip_grad_norm_(..., max_norm=1.0)`. | **PASS** |
| "Weight decay: 1e-4 (dim >= 2 only)" | Section 3.5 | `configure_optimizer()` applies WD only to `p.dim() >= 2`. | **PASS** |
| "Optional EMA: decay = 0.999" | Section 3.5 | `train.py` default `--ema_decay 0.999`. | **PASS** |
| "Data augmentation: parameter jittering (±0.05 Omega), small trajectory noise (std=0.005)" | Section 3.5 | `dataset.py` defaults: `omega_jitter=0.05`, `trajectory_noise_std=0.005`. | **PASS** |
| "290/302 JLD2 files parsed" | Section 4 Phase 0 | `parse_jld2.py` output: "302 scanned, 290 success, 12 skipped." | **PASS** |
| "12 unreadable due to encoding errors" | Section 4 Phase 0 | Actual skips: N=1089 (6 files), N=3481 (3 files), N=9 (3 files) = 12. | **PASS** |
| "5 visualization plots" from `check_data.py` | Section 4 Phase 1 | Counted 5 `plt.savefig()` calls in `scripts/check_data.py`. | **PASS** |
| "Input: [Omega, N, 1/sqrt(N)] + [t_0, ..., t_399]" | Section 3.1 | `transformer_surrogate.py` line 125 concatenates exactly these 3 params. | **PASS** |
| "Output: [sz_0, ..., sz_399]" | Section 3.1 | Model returns `(batch, n_time)` with `n_time=400`. | **PASS** |
| "No causal mask" | Section 3.1 | `SelfAttention.forward()` has no causal masking logic. | **PASS** |
| "GP trained on time-subsampled data (every 5th point)" | Section 7.5 | `gp_baseline.py` default `time_subsample=5`. 110 × 400 = 44K points. | **PASS** |
| "Full 44K points is intractable" | Section 7.5 | 44K points for scikit-learn GP is indeed intractable. | **PASS** |
| Fixed params: V=Δ=2000, Γ=1, γ=0.1, t_max=1000, n_T=400 | Section 2.3 & Appendix B | `parse_jld2.py` defaults and filename parsing confirm. | **PASS** |
| System sizes list | Section 2.3 | Parsed data confirms: [100, 225, 400, 900, 1225, 1600, 2500, 3025, 3600, 4900]. | **PASS** |
| `rho = (sz_mean + 1.0) / 2.0` | Appendix B | `dataset.py` line 59 and `parse_jld2.py` line 138 both use this exact formula. | **PASS** |
| `PHASE_THRESHOLD_RHO = 0.05` | Appendix B | `scripts/evaluate.py` line 30: `PHASE_THRESHOLD_RHO = 0.05`. | **PASS** |
| DP exponents table | Appendix B | Values match standard 2D DP literature (β≈0.583, ν⊥≈0.733, etc.). | **PASS** |

---

## 10. Specific Corrections Needed

### Must Fix (would waste GPU time or cause immediate failures)

1. **Add venv activation reminder before Section 7 commands.**
   ```markdown
   > **Note:** The commands below assume your virtual environment is active.
   > If you opened a new terminal, run `source .venv/bin/activate` first.
   ```

2. **Correct hardware requirements.**
   - Change "~10GB GPU memory" to "~1GB GPU memory (any CUDA-capable GPU works)"
   - Change "~5GB disk space" to "~100MB disk space"

### Should Fix (confusion or incompleteness)

3. **Clarify early stopping patience.**
   - In section 3.5 table, add: "patience = 100 (default), 50 recommended for full training runs"

4. **Expand file structure diagram or add disclaimer.**
   ```markdown
   **Note:** This diagram shows only the Rydberg surrogate files. The repo also
   contains the original nanoGPT codebase (`model.py`, `sample.py`, `config/`, etc.)
   and a `MasterThesis/` directory with source figures.
   ```

5. **Mention `README.md` exists and describes nanoGPT.**
   - An agent reading `README.md` first might be confused about the project goal.

### Nice to Have

6. **Distinguish `config/` vs `configs/`** to prevent confusion.
7. **Add `MasterThesis/` to the diagram** with a note that it contains the original thesis figures and source.

---

## Final Verdict

| Category | Score | Notes |
|---|---|---|
| Code accuracy | A | Every code claim checked against source — all match. |
| Command correctness | B+ | Commands work, but `python3` without venv is a landmine. |
| Hardware specs | D | 10x-100x inflated. Could cause unnecessary cloud provisioning. |
| File completeness | C | Omits original nanoGPT files and other significant directories. |
| Physics claims | A | Caveats are appropriately conservative. |
| Historical accuracy | A | References to removed features (mixup, tokenization) are correct. |

**Recommendation:** Apply the 5 corrections above before distributing this as a canonical cold-start guide. The core technical content is solid, but the execution environment and resource claims need tightening.
