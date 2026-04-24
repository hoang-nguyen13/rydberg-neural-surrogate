# Publication Feasibility Assessment

**Date:** 2026-04-24
**Project:** Transformer-Based Neural Surrogate for Rydberg Facilitation Dynamics
**Assessment:** Rigorous, evidence-based evaluation of publishability

---

## Executive Summary

**Verdict: FEASIBLE for Physical Review Research or MLST, with conditions.**
**NOT feasible for PRX Quantum or Nature Physics without major additions.**

The project has a coherent scientific narrative, a working codebase, and a defensible methodology. Whether it produces a publishable paper depends entirely on the **quantitative results of GPU training**. The current state is a promising prototype with no trained model.

---

## 1. What Evidence Do We Currently Have?

### 1.1 Data Evidence

| Claim | Evidence | Strength |
|---|---|---|
| Phase transition exists | Phase diagram plots show clear absorbing→active transition around Omega~10-13 | **Strong** |
| Finite-size effect visible | Larger systems have sharper transitions | **Strong** |
| 290 trajectories available | Parsed from JLD2, confirmed readable | **Strong** |
| Clean data | No NaN, no Inf, all trajectories start at sz=+1 | **Strong** |
| Parameter space coverage | 10 system sizes, Omega 0-30 | **Moderate** |

### 1.2 Model Evidence

| Claim | Evidence | Strength |
|---|---|---|
| Model compiles and runs | 5-epoch test completed successfully | **Weak** — trivial |
| Model beats GP baseline | GP val MSE ~0.035; 5-epoch transformer val MSE ~0.015 | **Weak** — not trained |
| Model generalizes to unseen sizes | NOT TESTED | **None** |
| Model extrapolates in Omega | NOT TESTED | **None** |
| Speedup vs TWA | NOT MEASURED | **None** |

### 1.3 Baseline Evidence

| Baseline | Status | Evidence |
|---|---|---|
| GP | Runs, val MSE ~0.035 | Moderate — subsampled to 20% of time points |
| FNO | Code complete, NOT TRAINED | None |
| DeepONet | NOT IMPLEMENTED | None |
| Linear interpolation | NOT IMPLEMENTED | None |

---

## 2. What Would Make This Publishable in Physical Review Research?

Physical Review Research publishes rigorous methods papers and applications of ML to physics. The bar is:

1. **Scientific question is well-posed** ✅
2. **Method is appropriate and novel enough** ✅ (conditional)
3. **Results are quantitatively strong** ❌ (unknown until training)
4. **Baselines are fair and comprehensive** ⚠️ (DeepONet missing)
5. **Limitations are discussed honestly** ✅ (built into the plan)

### 2.1 The Minimum Viable Paper

To be competitive at PR Research, the paper needs:

**A. Strong quantitative results:**
- Test MSE < 0.01 on interpolation (N=1225, unseen Omega)
- Test MSE < 0.02 on size extrapolation (N=1600-4900)
- Phase accuracy > 90%
- ROC AUC > 0.85
- Transformer beats FNO and GP on at least 3 of 5 metrics

**B. Meaningful generalization:**
- Model trained on N≤900 predicts N=4900 with <10% relative error
- This is the "killer result" — if it works, the paper is compelling

**C. Speedup benchmark:**
- TWA: ~minutes per parameter set (500 trajectories)
- NN: ~milliseconds per parameter set
- Speedup factor > 100x with <5% error

**D. Physical insight:**
- Critical point Omega_c(N) extracted from predicted trajectories
- Finite-size scaling trend visible (even if exponents are "effective")
- Bootstrap confidence intervals show statistical rigor

### 2.2 Honest Assessment of Each Requirement

| Requirement | Probability of Success | Why |
|---|---|---|
| Strong test MSE | **60%** | 290 examples is few, but the mapping is smooth and deterministic |
| Size extrapolation N=4900 | **30%** | Very hard. Training on N≤900 to predict N=4900 is a 5x linear size gap |
| Phase accuracy > 90% | **70%** | The phase transition is sharp; even a coarse model should classify correctly |
| Beats FNO | **50%** | FNO is strong for smooth PDEs; transformer may not win |
| Beats GP on interpolation | **80%** | GP is gold standard for interpolation; beating it is realistic |
| Speedup > 100x | **95%** | Almost guaranteed — any NN is faster than 500-trajectory TWA |

**Overall probability of meeting minimum viable paper: ~45%**

This is not pessimistic — it's realistic. The project is a gamble on whether the transformer generalizes across system sizes.

---

## 3. What Would Make This Publishable in MLST?

Machine Learning: Science and Technology is more methods-focused and accepts rigorous benchmarking papers. The bar is:

1. **Methodological contribution** ✅ (transformer vs. FNO for small-data operator learning)
2. **Comprehensive baselines** ⚠️ (need FNO + GP + at least one more)
3. **Ablations** ❌ (not done)
4. **Reproducibility** ✅ (code + checkpoints planned)

### 3.1 MLST-Specific Requirements

**A. Systematic comparison:**
- Table comparing transformer, FNO, GP, and ideally DeepONet or Neural ODE
- Metrics: MSE, MAE, inference time, training time, data efficiency

**B. Ablation studies:**
- Remove parameter conditioning → measure degradation
- Replace transformer with MLP → measure degradation
- Vary model size → plot scaling curve

**C. Data efficiency analysis:**
- Train on 25%, 50%, 75% of data → plot test MSE vs. training set size
- This is very compelling for small-data physics

**D. Uncertainty quantification:**
- Deep ensemble (5 models) → prediction intervals
- Calibration plot: does ground truth fall within predicted intervals?

### 3.2 Honest Assessment

| Requirement | Probability | Why |
|---|---|---|
| Systematic comparison | **70%** | Code exists; just needs training runs |
| Ablations | **60%** | Easy to implement; need compute time |
| Data efficiency curve | **80%** | Trivial to implement; very compelling result |
| Uncertainty quantification | **70%** | Ensemble script exists; needs 5 training runs |

**Overall probability of meeting MLST bar: ~60%**

MLST is a better target than PR Research because it values methodology over physics novelty.

---

## 4. What Would NOT Be Publishable?

### 4.1 PRX Quantum — Probability: <10%

PRX Quantum requires:
- Exceptional novelty in quantum science
- Broad impact on the field
- Discovery of new physics OR paradigm-shifting method

Our project:
- Applies a standard transformer to a specific simulation dataset
- No new physics discovered
- Does not solve a problem that physicists currently struggle with
- The TWA simulation is already fast enough for most purposes

**To reach PRX Quantum, we would need:**
- Spatial data + true critical exponent extraction matching DP theory
- Discovery that the transformer predicts critical points more accurately than traditional fitting
- Generalization across Hamiltonian parameters (gamma, Delta, V) — not just Omega and N
- Demonstration that the surrogate reveals physics that TWA misses

### 4.2 Nature Physics — Probability: <5%

Nature Physics requires results that change how physicists think about a problem. A neural surrogate that reproduces existing simulations does not meet this bar.

### 4.3 NeurIPS/ICML AI for Science — Probability: ~30%

ML venues expect:
- Methodological novelty (new architecture, new training technique)
- Large-scale experiments (hundreds of models, ablations, scaling laws)
- Comparison to state-of-the-art

Our project:
- Uses a vanilla transformer (no architectural novelty)
- Dataset is tiny (290 examples)
- No theoretical analysis (no NTK, no generalization bounds)

**To reach NeurIPS, we would need:**
- A novel architecture (e.g., physics-informed attention, Hamiltonian-conditioned transformer)
- Theoretical insight (e.g., why transformers beat FNO on this problem)
- Much larger scale (multiple systems, not just one)

---

## 5. The Honest Strengths of This Project

1. **Real physical data from a real master's thesis** — not synthetic toy data
2. **Clean phase transition** — the physics is unambiguous
3. **Rigorous methodology** — baselines, bootstrap CIs, proper splits, uncertainty quantification
4. **Honest framing** — "neural surrogate" not "foundation model" or "AI physicist"
5. **Open science** — code, checkpoints, and data will be released
6. **Speedup** — if it works, it's genuinely useful for parameter screening

---

## 6. The Honest Weaknesses

1. **Tiny dataset** — 290 examples from a single system. The model is not learning "physics"; it's learning an interpolation surface.
2. **Fixed parameters** — Delta, gamma, V are all fixed. The model cannot generalize to different Hamiltonians.
3. **No spatial data** — The surrogate predicts only the global mean. It misses all spatial correlations, cluster structure, and critical fluctuations.
4. **TWA limitations** — Even if the surrogate is perfect, it only reproduces TWA. TWA may not be the ground truth.
5. **No new physics** — The paper will report "we can predict dynamics faster," not "we discovered X."
6. **Overfitting risk** — 309K parameters on 110 training examples is still aggressive. The model may memorize.

---

## 7. Recommendations

### 7.1 Immediate (Before Writing a Word)

1. **Train the model on GPU.** Run full training. If test MSE > 0.05, the project is in trouble.
2. **Train the ensemble.** If ensemble variance is large (>20% of prediction), the model is not confident.
3. **Measure size extrapolation.** If N=4900 MSE is >0.05, the "killer result" is dead.
4. **Train FNO baseline.** If FNO beats the transformer on all metrics, the transformer choice is unjustified.

### 7.2 If Results Are Strong

**Target Physical Review Research.** Frame as:
> "A Transformer-Based Neural Surrogate for Non-Equilibrium Rydberg Facilitation Dynamics"

Key claims:
- Accurate prediction of mean-field dynamics across parameter space
- Zero-shot generalization to unseen system sizes (if it works)
- 100x+ speedup over TWA with <5% error
- Competitive with or superior to neural operators on small data

### 7.3 If Results Are Moderate

**Target MLST or APL Machine Learning.** Frame as:
> "Small-Data Operator Learning for Quantum Dynamics: A Comparative Study"

Key claims:
- Systematic comparison of transformers, FNOs, and GPs on small-data regime
- Data efficiency analysis (how many trajectories are needed?)
- Uncertainty quantification via deep ensembles

### 7.4 If Results Are Weak

**Do not publish.** Write an arXiv preprint as a methods note and move on. The physics insights are not strong enough to justify a journal paper if the model does not generalize.

---

## 8. The Brutal Truth

This project is **scientifically sound but scientifically thin.**

The methodology is rigorous. The codebase is clean. The physics is real. But the **scientific payload** — what we actually learn that physicists didn't already know — is minimal.

A physicist reading this paper will ask:
> "Why should I care about a neural network that reproduces 500-trajectory TWA averages when I can just run the TWA?"

The answer must be:
1. **Speed** — ms vs. minutes enables parameter screening
2. **Generalization** — predicts unseen sizes/parameters accurately
3. **Insight** — the model's internal representations encode physical phases

If we cannot demonstrate at least two of these three convincingly, the paper will be rejected.

**Probability of eventual publication in a peer-reviewed journal: ~50%**
**Probability of publication in PR Research specifically: ~35%**
**Probability of publication in MLST: ~55%**

These are not bad odds for a 4-month project. But they are not guaranteed.

---

*Assessment prepared by independent analysis of codebase, plan, and audit reports.*
