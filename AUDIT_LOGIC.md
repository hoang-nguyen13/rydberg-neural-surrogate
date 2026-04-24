# Logic & Coherence Audit: Rydberg Neural Surrogate Project

**Auditor:** Research Methodology Expert (Physics + CS)  
**Date:** 2026-04-24  
**Plan reviewed:** `PROJECT_PLAN_REVISED.md` (v2.0)  
**Code reviewed:** `parse_jld2.py`, `dataset.py`, `transformer_surrogate.py`, `train.py`, `check_data.py`, `evaluate.py`, `gp_baseline.py`, `fno_baseline.py`

---

## Executive Summary

The project successfully pivoted from the flawed v1.0 autoregressive-token paradigm to the v2.0 direct-regression paradigm. The core model architecture and data pipeline are logically sound and physically appropriate. **However, the implementation contains a significant parameter-count overrun, a persistent naming/physics bug in the order parameter computation, a learning-rate scheduler bug, and massive gaps in evaluation code relative to the plan's promises.** The project cannot currently support the physics claims in the revised plan (critical exponents, uncertainty quantification, finite-size scaling) because the code to produce those results is simply missing.

**Overall Rating: NEEDS_FIX**

---

## 1. Direct Regression: CORRECTLY Implemented

**Plan (v2.0, Section 4.1):** "Parameter embedding + time coordinate -> full attention -> predict all 400 time steps in parallel. No causal mask needed. Single forward pass. No discretization."

**Code (`models/transformer_surrogate.py`):**
- Line 12: `SelfAttention` -- **no causal mask** is applied. `attn = F.softmax(attn, dim=-1)` allows full bidirectional attention across all time steps.
- Line 140: `sz_pred = self.head(x).squeeze(-1)` outputs shape `(batch, n_time)`, predicting all 400 steps simultaneously.
- Line 102-141: `forward()` takes continuous parameters and time coordinates, combines them via broadcasted addition, and runs through standard (non-causal) transformer blocks.

**Verdict:** The autoregressive elements have been fully removed. This is the single most important architectural change from v1.0 to v2.0, and it is implemented faithfully.

---

## 2. Parameter Count Discrepancy: MEANINGFUL

**Plan (v2.0, Section 4.2, Table):** "Total params ~500K"  
**Plan rationale (Section 4.2, and v1.0->v2.0 diff):** "20,000 params/example = overfitting; reduce to ~500K."

**Actual count from `transformer_surrogate.py`:**

| Component | Calculation | Parameters |
|---|---|---|
| `param_embed` | `Linear(3,128)` + `Linear(128,128)` | 17,024 |
| `time_embed` | `Linear(1,128)` | 256 |
| 4x TransformerBlock | Per block: `qkv`(49,536) + `proj`(16,512) + `mlp`(131,712) + 2x `LayerNorm`(512) | 198,272 x 4 = **793,088** |
| `ln_f` | `LayerNorm(128)` | 256 |
| `head` | `Linear(128,1)` | 129 |
| **TOTAL** | | **~810,753** |

**Discrepancy:** ~810K vs. ~500K = **62% overrun**.

**Why this matters:** The revised plan explicitly identified over-parameterization as a critical risk (Risk Register: "Transformer overfits on 110 examples | High | Critical"). With ~110 training examples, the plan targets ~4,500 params/example. The actual model gives ~7,400 params/example. While far better than v1.0's ~6M params (~55K/example), this overrun directly undermines the project's primary anti-overfitting mitigation. The MLP expansion ratio of 4x inside each transformer block (`n_embd -> 4*n_embd`, line 55-57) is the main driver.

**Fix:** Reduce `n_embd` to 96 (would yield ~457K params) or shrink MLP ratio to 2x.

---

## 3. Continuous Parameter Embeddings: CORRECTLY Implemented

**Plan (v2.0, Section 4.1):** "Input: [Omega_embed, N_embed, 1/sqrt(N)_embed, t_0_embed, ..., t_399_embed]"

**Code (`transformer_surrogate.py`, lines 75-79, 102-132):**
```python
self.param_embed = nn.Sequential(
    nn.Linear(3, n_embd),  # [Omega, N, 1/sqrt(N)]
    nn.GELU(), nn.Linear(n_embd, n_embd)
)
...
params = torch.cat([omega, n_atoms, inv_sqrt_n], dim=-1)
param_emb = self.param_embed(params)  # (batch, n_embd)
```

**Verdict:** The code correctly accepts three continuous scalars and projects them into the embedding space. This replaces the v1.0 discrete token bins and preserves metric structure. **Note:** The plan does not specify whether parameter and time embeddings should be *added* (as in line 132: `param_emb.unsqueeze(1) + time_emb`) or *concatenated*. Addition is an implicit design choice with representational consequences -- it forces parameter and time embeddings to live in the same vector space -- but it is not a bug.

---

## 4. Deep Ensembles for Uncertainty Quantification: NOT IMPLEMENTED

**Plan (v2.0, Section 6.2):** "Deep ensemble (5 models)"  
**Plan (v2.0, Section 7.4):** "Deep ensemble: 5 models, report mean +/- std"  
**Project Review (`PROJECT_REVIEW.md`, Section 6.1):** "Use a deep ensemble (5-10 models trained from different initializations) to estimate epistemic uncertainty."

**Code (`train.py`):** Trains a single model with a single seed (`args.seed`, default 42). No ensemble loop, no multi-seed training, no aggregation of predictions.

**Gap severity:** CRITICAL for scientific credibility. The plan elevated UQ from "nice to have" (missing in v1.0) to a core requirement in v2.0. Without ensembles, there are no prediction intervals, no calibration checks, and no principled way to flag out-of-distribution failures.

---

## 5. Bootstrap Confidence Intervals on Critical Exponents: NOT IMPLEMENTED

**Plan (v2.0, Section 7.3):** "All fits report bootstrap 95% confidence intervals."  
**Plan (v2.0, Section 7.4):** "Bootstrap on critical exponents: 1000 resamples"

**Code (`scripts/evaluate.py`):** The evaluation script computes basic trajectory metrics (MSE, MAE, phase accuracy) but contains **zero critical-exponent analysis**. There is no:
- Derivative method for Omega_c(L)
- Power-law fitting for beta
- Finite-size scaling for Omega_c(inf)
- Data collapse visualization
- Bootstrap resampling

**Gap severity:** CRITICAL. The physics narrative of the paper depends entirely on these analyses.

---

## 6. The `rho_ss` Naming Bug and Threshold Mismatch

This is a persistent, cross-file logical error that corrupts the order parameter definition used for phase classification.

**Plan (v2.0, Section 3.1):** "Compute order parameter: rho = (sz_mean + 1) / 2"  
**Plan (v2.0, Section 7.2):** "Absorbing: rho_ss < 0.05 vs. Active: rho_ss > 0.05"

**Bug location 1 (`data/parse_jld2.py`, line 154-155):**
```python
rho = (sz_mean + 1.0) / 2.0  # DP order parameter: correct
rho_ss = float(np.mean(sz_mean[-50:]))  # BUG: this is sz_ss, NOT rho_ss
```

**Bug location 2 (`data/dataset.py`, line 99-100):**
```python
rho = (sz_mean + 1.0) / 2.0
rho_ss = float(np.mean(sz_mean[-50:]))  # BUG: same error
```

**Bug location 3 (`scripts/evaluate.py`, line 56-61):**
```python
pred_rho_ss = np.mean(all_pred[:, -50:], axis=1)   # Actually sz_ss
pred_phase = pred_rho_ss > -0.95                    # Threshold on sz, not rho
```

**Bug location 4 (`scripts/check_data.py`, line 238-239):**
```python
n_absorbing = sum(1 for r in records if r.rho_ss < -0.95)  # Threshold on sz_ss
```

**Analysis:** The code correctly computes `rho = (sz_mean + 1) / 2` but then computes the steady-state value from `sz_mean` instead of `rho`. The variable name `rho_ss` is therefore a **lie** -- it actually contains `sz_ss` (steady-state sz_mean). Since `sz_ss` for an absorbing state is approximately -1.0, the threshold `-0.95` corresponds to `rho = 0.025`, not the planned `rho = 0.05`. The correct computation should be:
```python
rho_ss = float(np.mean(rho[-50:]))  # = (np.mean(sz_mean[-50:]) + 1.0) / 2.0
```

**Impact:** Phase classification is internally inconsistent with the plan's own definition. The threshold is stricter than documented (0.025 vs 0.05). All downstream analyses using `rho_ss` (phase diagrams, critical point estimation) are semantically confused.

**Fix:** Replace all `rho_ss = float(np.mean(sz_mean[-50:]))` with `rho_ss = float(np.mean(rho[-50:]))` and change the phase threshold from `-0.95` to `0.05` (or `-0.9` if operating on `sz_ss`).

---

## 7. Learning-Rate Scheduler Bug

**Plan (v2.0, Section 6.1):** "scheduler = CosineAnnealingLR(optimizer, T_max=5000, eta_min=1e-5)"  
**Expected behavior:** Cosine annealing over 5000 **steps**.

**Code (`train.py`, line 168, 186):**
```python
scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=args.lr * 0.01)
...
for epoch in range(args.max_epochs):
    train_metrics = train_epoch(model, train_loader, optimizer, device)
    scheduler.step()  # Called once per EPOCH, not per step
```

**The bug:** With batch_size=32 and ~110 training examples, each epoch is ~4 steps. `max_steps` defaults to 5000, but `scheduler.step()` is called only once per epoch. Over the default `max_epochs=1000`, the scheduler receives only 1000 steps, and the cosine curve is stretched over 5000 intended steps. The learning rate barely decays. For a small-data regime where overfitting is the primary risk, an improperly decayed learning rate is dangerous.

**Fix:** Either step the scheduler per batch (inside `train_epoch`), or set `T_max=args.max_epochs`.

---

## 8. Evaluation Script: Severely Incomplete

**Plan (v2.0, Section 7) promises:**
- Trajectory metrics: MSE, Relative L2, Max error, MAE on rho_ss, Pearson correlation, Dynamic Time Warping (DTW)
- Phase classification: Confusion matrix, ROC, AUC, calibration plot
- Critical point estimation: Derivative method, power-law fit, finite-size scaling, data collapse
- Uncertainty quantification: Deep ensemble, bootstrap CIs, calibration check
- Generalization tests: Interpolation, extrapolation, size extrapolation, cross-parameter
- Speedup benchmark: TWA time vs. NN inference time

**Code (`scripts/evaluate.py`) delivers:**
- MSE, MAE, max error, relative L2 (lines 51-54)
- rho_ss MAE (line 58) -- but computed on `sz_ss`, not `rho_ss` (see Section 6)
- Phase accuracy with a single hard threshold (lines 60-62)
- Trajectory comparison plots for N=1225 only (lines 71-102)

**Missing entirely:** Pearson correlation, DTW, confusion matrix, ROC, AUC, calibration, critical point estimation, power-law fitting, finite-size scaling, data collapse, bootstrap, speedup measurement, ensemble aggregation, cross-parameter tests.

**Gap severity:** CRITICAL. The evaluation script is a diagnostic dashboard, not the comprehensive physics analysis pipeline promised in the plan. Without it, the paper has no results section.

---

## 9. Missing Baselines and Infrastructure

| Promise (Plan v2.0) | Status | File |
|---|---|---|
| FNO baseline | Implemented | `baselines/fno_baseline.py` |
| GP baseline | Implemented | `baselines/gp_baseline.py` |
| DeepONet baseline | **MISSING** | Not in repo |
| Linear interpolation baseline | **MISSING** | Not in repo |
| Hydra config management | **MISSING** | `configs/` directory exists but is empty |
| `requirements.txt` | **MISSING** | Not in repo |
| Inference script | **MISSING** | Not in repo |
| EMA (decay=0.999) | **MISSING** | Not in `train.py` |
| Stochastic depth (layer dropout 0.1) | **MISSING** | Not in `train.py` |
| Weights & Biases logging | Implemented | `train.py` lines 171-177 |

**DeepONet omission severity:** HIGH. The Project Review (Section 6.3) calls this omission "fatal" for ML venue credibility, and the revised plan (Section 5.3) explicitly promises it.

**EMA / Stochastic depth severity:** MEDIUM. These were listed as regularization measures in the plan (Section 6.2) to combat overfitting. Their omission weakens the anti-overfitting protocol, compounding the parameter-count overrun issue.

---

## 10. FNO Baseline: Logically Questionable Formulation

**Code (`baselines/fno_baseline.py`, lines 41-51):**
```python
omega_ch = omega.view(batch_size, 1, 1).expand(batch_size, 1, n_time)
n_ch = n_atoms.view(batch_size, 1, 1).expand(batch_size, 1, n_time)
inv_sqrt_n_ch = inv_sqrt_n.view(batch_size, 1, 1).expand(batch_size, 1, n_time)
x = torch.cat([omega_ch, n_ch, inv_sqrt_n_ch], dim=1)
y = self.fno(x)
```

**Issue:** The FNO receives three input channels that are **constant across the temporal dimension**. The time coordinate `t` is **not fed as an input**. The model relies entirely on `positional_embedding='grid'` to learn temporal structure. For a 1D temporal operator, this is an unusual formulation -- the "trunk" input (time) is implicit rather than explicit. The FNO is effectively learning a mapping from parameter-conditioned constant fields to temporal profiles, which may work but is not the standard operator-learning formulation where time is an explicit input coordinate.

**Verdict:** Not a bug, but an undocumented and potentially suboptimal design choice that makes the FNO baseline harder to interpret and potentially weaker than it should be.

---

## 11. Logical Path from "Train Model" to "Publish Paper"

The following table maps each paper claim to its supporting code:

| Paper Claim (Plan v2.0) | Supporting Code? | Gap |
|---|---|---|
| "Transformer surrogate predicts sz_mean(t)" | Yes: `transformer_surrogate.py`, `train.py` | -- |
| "Faster than TWA" | **No** | No speedup benchmark anywhere |
| "Accurate interpolation and extrapolation" | Partial: `evaluate.py` evaluates splits but does not produce interpolation/extrapolation analysis figures | No systematic generalization plots |
| "Phase diagram: absorbing vs. active" | Partial: `check_data.py` plots ground truth; `evaluate.py` does not plot predicted phase diagram | No predicted-vs-true phase diagram |
| "Effective exponents with error bars" | **No** | No exponent fitting, no bootstrap CIs |
| "Uncertainty quantification via deep ensemble" | **No** | No ensemble training, no calibration |
| "Comparison to FNO, GP, DeepONet" | Partial: GP and FNO exist; DeepONet missing | Missing DeepONet |
| "Finite-size scaling and data collapse" | **No** | No FSS code |
| "Reproducibility: seeds, splits, configs" | Partial: seed and splits are fixed | No YAML configs, no requirements.txt, no inference script |

**What is missing for publication:**
1. **Critical exponent extraction pipeline** (derivative method + power-law fitting + FSS)
2. **Bootstrap confidence intervals** on all fits
3. **Deep ensemble training script** (loop over 5 seeds, aggregate predictions)
4. **Comparative evaluation dashboard** (transformer vs. FNO vs. GP on all metrics)
5. **Speedup benchmark** (timed inference vs. TWA simulation)
6. **Predicted phase diagram** (model outputs overlaid on ground truth)
7. **Generalization analysis** (size-extrapolation-specific metrics and visualization)
8. **DeepONet baseline** (or explicit justification for dropping it)
9. **Inference script** (load model, predict on new parameters)
10. **Reproducibility package** (requirements.txt, Hydra configs, exact environment)

---

## 12. Implicit Assumptions Not Documented

1. **Additive embedding combination (`param_emb + time_emb`):** The plan says "Input: [Omega_embed, N_embed, 1/sqrt(N)_embed, t_0_embed, ...]" which implies concatenation or a sequence. The code uses element-wise addition. This is a reasonable choice but changes the inductive bias: the parameter embedding must be directly additively compatible with every time-step embedding.

2. **Unnormalized time coordinates:** `t` ranges from 0 to 1000 and is fed directly into `nn.Linear(1, n_embd)`. No normalization (e.g., to [0,1]) is applied. The model must learn to handle large-magnitude inputs internally.

3. **Phase threshold of -0.95 on `sz_mean`:** The code implicitly assumes that `sz_ss > -0.95` defines the active phase. The plan says `rho_ss > 0.05`, which is equivalent to `sz_ss > -0.9`. The code is stricter (0.025 vs 0.05) and undocumented.

4. **Square-lattice assumption:** `parse_jld2.py` line 158 computes `lattice_size = int(np.sqrt(n_atoms))`. This assumes all systems are perfect square 2D lattices. If non-square data appears, this breaks silently.

5. **GP kernel length scales are hand-tuned:** The GP baseline uses `length_scale=[1.0, 100.0, 10.0]` for `[Omega, N, t]`. N ranges from 225 to 4900; a length scale of 100 means the GP assumes smoothness over ~100 atoms. For size extrapolation to N=4900 (unseen in training), this is an untested assumption.

6. **Training epochs vs. steps confusion:** The plan specifies "max_steps = 5000" and "patience = 500" (steps). `train.py` implements `max_epochs=1000` and `patience=100` (epochs). With ~110 examples and batch_size=32, 100 epochs ~ 400 steps. The intended early-stopping horizon (~500 steps) is approximately matched, but the training protocol is epoch-based rather than step-based as the plan describes.

---

## 13. Logical Fallacies and Methodological Risks

### 13.1 No Circular Reasoning Detected
The train/val/test splits are parameter-based (by system size and Omega index), not random. There is no direct data leakage. The evaluation computes metrics on all three splits, which is acceptable for diagnostics as long as test-set results are not used for model selection.

### 13.2 Testing on Training Data Indirectly
`evaluate.py` line 124 evaluates the model on the **training set** and prints metrics. This is fine for detecting underfitting, but the script does not clearly flag that train-set metrics are optimistic and not generalization estimates. A naive reader might be misled.

### 13.3 The "Killer Test" is Underserved
The v1.0 plan (and v2.0 implicitly) identifies size extrapolation (N=1600-4900) as the most compelling generalization claim. `evaluate.py` only plots trajectory comparisons for N=1225 (the validation size, line 76). It does **not** generate any visualization or dedicated metric report for the size-extrapolation test set. This weakens the strongest claim the paper could make.

### 13.4 Phase Classification Without Probabilistic Threshold
The plan promises ROC, AUC, and calibration plots (Section 7.2). The code implements only a hard threshold at -0.95. For a phase transition where the system hovers near criticality, a probabilistic classifier (or at least threshold sweeping) is physically more appropriate. A hard threshold collapses the continuous order parameter into a binary label without capturing the transition width.

---

## 14. Specific File-by-File Issue Summary

### `data/parse_jld2.py`
- **Line 155:** `rho_ss` computed from `sz_mean` instead of `rho`. Variable name is semantically incorrect.
- **Line 158:** Assumes square lattice (`sqrt(n_atoms)`). Undocumented assumption.
- **Line 164:** `V=dir_params['delta'] or 2000.0` -- fallback to 2000.0 is arbitrary and not explained.

### `data/dataset.py`
- **Line 100:** Same `rho_ss` bug as above.
- **Line 95:** Mixup interpolates `n_atoms` to non-integer values then casts to `int`. This creates fictitious system sizes during augmentation. Acceptable for regularization but physically nonsensical.
- **Lines 113-162:** Split logic correctly drops N=100 per revised plan, and correctly implements even/odd Omega splitting for N=1225.

### `models/transformer_surrogate.py`
- **Lines 55-57:** MLP expansion ratio of 4x drives parameter count to ~810K vs. planned ~500K.
- **Line 132:** Additive combination of parameter and time embeddings is undocumented.
- **Lines 102-141:** Architecture is otherwise faithful to plan.

### `train.py`
- **Line 168, 186:** LR scheduler stepped per epoch instead of per step / per batch.
- **Lines 171-177:** WandB logging present.
- **Line 164:** Prints parameter count (will show ~810K, not ~500K).
- **Lines 39-63:** Physics-informed loss correctly implements bounds and smoothness penalties as specified (Section 4.4).
- **Missing:** EMA, stochastic depth, deep ensemble loop.

### `scripts/evaluate.py`
- **Lines 56-62:** `rho_ss` naming bug and hardcoded threshold.
- **Lines 71-102:** Only visualizes N=1225. No size-extrapolation visualization.
- **Missing:** All critical-exponent analysis, bootstrap, ROC/AUC, DTW, Pearson, speedup, ensemble aggregation.

### `baselines/gp_baseline.py`
- **Lines 43-46:** Kernel hyperparameters are hand-tuned and not validated.
- **Line 51:** `n_restarts_optimizer=2` (plan specifies 10).

### `baselines/fno_baseline.py`
- **Lines 41-51:** Time coordinate `t` is not passed to the FNO. Relies on grid positional embedding.

---

## 15. Conclusion and Recommendations

### What Works
- The direct-regression architecture is correct and free of autoregressive baggage.
- Continuous parameter embeddings replace the flawed token-based conditioning.
- Physics-informed loss constraints (bounds, smoothness) are implemented.
- Data augmentation (jitter, noise, mixup) is present.
- The train/val/test split strategy is parameter-based and leak-free.

### What Must Be Fixed (Critical)
1. **Fix `rho_ss` computation** in `parse_jld2.py`, `dataset.py`, and `evaluate.py`. Change threshold from `-0.95` (on `sz_ss`) to `0.05` (on true `rho_ss`).
2. **Fix LR scheduler** in `train.py` -- step per batch or adjust `T_max` to match epochs.
3. **Add deep ensemble training** -- a wrapper script that trains 5 models with different seeds and saves them.
4. **Build the critical exponent pipeline** -- derivative method, power-law fit, finite-size scaling, bootstrap CIs. Without this, there is no physics results section.
5. **Add comprehensive evaluation** -- ROC/AUC, confusion matrix, calibration, DTW, Pearson, speedup benchmark, size-extrapolation visualization.

### What Should Be Fixed (High Priority)
6. **Reduce parameter count** to match the ~500K target (reduce `n_embd` or MLP ratio).
7. **Implement DeepONet baseline** or explicitly document why it was dropped.
8. **Add EMA and/or stochastic depth** as promised regularization.
9. **Create `requirements.txt`**, Hydra configs, and a standalone `inference.py` script.
10. **Add predicted-vs-true phase diagram plots** to the evaluation script.

### Overall Project Coherence
The **conceptual coherence** is strong: the revised plan correctly identified the methodological flaws of v1.0 and prescribed appropriate fixes. The **implementation coherence** is weak: the core model trains, but the evaluation and scientific analysis layers are essentially missing. The project is currently at the "working prototype" stage, not the "publication-ready pipeline" stage.

**Final Rating: NEEDS_FIX**
