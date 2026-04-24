# Training Plan — Rydberg Neural Surrogate

## Goal
Train a neural surrogate that predicts ρ(t) from (Ω, N, γ, dimension) and generalizes to unseen sizes, dephasing rates, and dimensions.

---

## Phase 1: Fix Training Infrastructure (1–2 hours)

1. **Update `train.py`** to use `dataset_v2.py`:
   - Load `rydberg_dataset_v2.pkl`
   - Use 5-parameter input: `[omega, n_atoms, inv_sqrt_n, gamma, dimension]`
   - Use `create_splits_v2()` for train/val/test
   - Update `collate_fn` to handle variable-length trajectories

2. **Verify model compatibility**:
   - `RydbergSurrogate` already accepts 5 params — confirm forward pass works
   - Check that loss computation handles full trajectories (400 time points)

3. **Add test-set evaluation**:
   - Per-test-set MSE
   - Per-test-set ρ_ss error
   - Phase accuracy (absorbing vs active)

---

## Phase 2: Train Core Model (CPU first, then GPU)

**Experiment A: Baseline Transformer**
- Model: 4 layers, 4 heads, n_embd=96, ~310K params
- Train on: 2D γ=0.1, N=100–2500 (325 trajectories)
- Validate on: N=3025, 3600 (94 trajectories)
- Loss: MSE on ρ(t) + physics bounds + smoothness
- Metrics: Trajectory MSE, ρ_ss MAE, phase accuracy
- Target: Val MSE < 0.01, ρ_ss MAE < 0.005

**Experiment B: FNO Baseline**
- Use `baselines/fno_baseline.py`
- Same train/val split
- Compare against transformer

**Experiment C: GP Baseline (already done)**
- Use existing GP results for comparison

---

## Phase 3: Generalization Tests (use trained model)

Run inference on each test set WITHOUT retraining:

| Test Set | Data | What it tests |
|----------|------|---------------|
| size_extrapolation_2d | N=4900 (19 traj) | Can it predict larger N? |
| gamma_classical_2d | γ=5,10,20 at N=3600 (64 traj) | Can it predict classical dephasing? |
| dimension_1d | 1D N=3000 (57 traj) | Can it do 1D? |
| dimension_3d | 3D N=3375 γ=0.1 (14 traj) | Can it do 3D? |
| gamma_quantum_3d | 3D γ=1e-5 (102 traj) | Can it do quantum 3D? |

For each: compute MSE, ρ_ss error, phase accuracy.

---

## Phase 4: Physics Extraction (if model is good enough)

1. **Dense Ω grid prediction**: Use trained model to predict ρ_ss at Ω=10:0.01:13 for each N
2. **Critical point extraction**: Find Ω_c(N) = argmax(dρ_ss/dΩ) from predicted curves
3. **Finite-size scaling**: Fit Ω_c(N) = Ω_c(∞) + a·N^(-1/2)
4. **β exponent**: Fit ρ_ss ~ |Ω - Ω_c|^β near critical point
5. **Compare** extracted exponents against known 2D Ising values

---

## Phase 5: Paper Figures

1. Training curves (loss vs epoch)
2. Predicted vs true trajectories (sampled)
3. ρ_ss vs Ω (model prediction overlaid on data)
4. Finite-size scaling of Ω_c(N)
5. β extraction plot
6. Generalization bar chart (MSE per test set)

---

## Deliverables

- Trained model checkpoint
- Test set evaluation report
- Physics extraction results
- Publication-ready figures

---

## Open Questions

1. Do you want ensemble training (3 models, average predictions)?
2. Do you want to compare transformer vs FNO vs GP in the paper?
3. Do you have GPU access for full training, or should I optimize for CPU?
