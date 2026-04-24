# Foundation Model for Non-Equilibrium Phase Transitions in Rydberg Facilitation Dynamics

## REVISED Implementation Plan (v2.0)

**Reviews incorporated:** Physics review, Engineering review, ML Research review
**Core decision:** Keep the transformer backbone, abandon autoregressive token generation, switch to direct regression.

---

## Summary of Changes from v1.0

| Issue (v1.0) | Fix (v2.0) | Why |
|---|---|---|
| Foundation model framing | Neural surrogate | 290 trajectories / fixed Hamiltonian = emulator, not foundation model |
| Autoregressive next-token prediction | Direct regression: full trajectory in one forward pass | Data is deterministic; autoregression = quantization + error accumulation + 400x slower |
| 256-bin discretization of sz_mean | Continuous scalar output, MSE loss | Discretization destroys critical physics |
| 6M parameters | ~500K parameters | 20,000 params/example = overfitting |
| Token-based parameter conditioning | Continuous parameter embeddings (Omega, N, 1/sqrt(N)) | Preserves metric structure |
| Sigmoid fit for critical exponents | Power-law fit + derivative method + data collapse | Sigmoid has analytic inflection, DP requires power laws |
| Missing baselines | Add FNO, DeepONet, GP | Standard in scientific ML |
| 50K training steps | 5K-10K steps with aggressive early stopping | 11K epochs = memorization |
| No uncertainty quantification | Deep ensemble (5 models) + bootstrap CIs | Required for scientific credibility |
| 64-dim CNN autoencoder for spatial | Patch-based ViT or ConvLSTM | Autoencoder destroys critical correlations |
| PRX Quantum target | Physical Review Research or MLST | Realistic for methods paper |
| Claiming DP exponents from mean data | Effective exponents with caveats | Mean data cannot reveal true DP behavior |

---

## 1. The Honest Framing

### 1.1 What This Project Actually Is

A transformer-based neural surrogate for ensemble-averaged Rydberg facilitation dynamics. Given Hamiltonian parameters, it predicts the full time evolution of the mean excitation density in a single forward pass. Fast inference (~ms) vs. TWA simulation (~minutes).

### 1.2 What This Project Is NOT

- Not a foundation model (no broad pre-training, no emergent transfer)
- Not a tool for extracting precise critical exponents from mean-field data alone
- Not a replacement for TWA when exact critical behavior is needed

### 1.3 What Physics We CAN Claim (Mean-Field Phase)

1. Interpolation and prediction of ensemble-averaged sz_mean(t)
2. Qualitative phase diagram -- absorbing vs. active
3. Finite-size trends
4. Computational speedup
5. Effective exponents with explicit error bars

### 1.4 What Physics Requires Spatial/Per-Trajectory Data (Future Work)

True DP exponents, survival probability, avalanche statistics, spatial correlations, susceptibility.

---

## 2. Data Foundation (Phase 0 -- COMPLETE)

Same as v1.0: 290 usable JLD2 files. Fixed parameters confirmed.

**New addition:** Extract per-trajectory survival flags if raw data exists.

---

## 3. Phase 1: Data Engineering (Weeks 1-2)

### 3.1 JLD2 Parsing

Write data/parse_jld2.py:
- Read all 290 files with h5py
- Extract sz_mean (400,), tSave (400,)
- Build DataFrame: [omega, n_atoms, lattice_size, sz_mean, tSave]
- Compute order parameter: rho = (sz_mean + 1) / 2

### 3.2 Train/Val/Test Split (Revised)

| Split | Sizes | Omega | Examples | Purpose |
|---|---|---|---|---|
| Train | N=225, 400, 900 | All | ~110 | Learn diverse dynamics |
| Validation | N=1225 | Even-indexed | ~10 | Early stopping, hyperparameter tuning |
| Test (interpolation) | N=1225 | Odd-indexed | ~10 | Interpolation within critical region |
| Test (size extrapolation) | N=1600, 2500, 3025, 3600, 4900 | All available | ~105 | Zero-shot size generalization |

Drop N=100 -- too small, limited coverage, may confuse the model.

### 3.3 Data Augmentation (New)

1. Parameter jittering: Perturb Omega by +/- 0.05-0.1 during training
2. Trajectory noise: Add Gaussian noise (std ~0.01) to sz_mean(t)
3. Mixup: Interpolate trajectories in parameter space with lambda ~ Beta(0.4, 0.4)

### 3.4 Baseline Data Preparation

Prepare data format for Gaussian Process, FNO, and DeepONet baselines.

---

## 4. Phase 2: Architecture (Weeks 3-4)

### 4.1 The Big Change: From Autoregressive to Direct Regression

**Old (v1.0):** Token sequence -> causal attention -> next-token prediction -> 400 forward passes

**New (v2.0):** Parameter embedding + time coordinate -> full attention -> predict all 400 time steps in parallel

Input: [Omega_embed, N_embed, 1/sqrt(N)_embed, t_0_embed, ..., t_399_embed]
Output: [sz_0_pred, sz_1_pred, ..., sz_399_pred]  (400 scalars)
Loss: MSE(output, true_sz_mean)

No causal mask needed. Single forward pass. No discretization.

### 4.2 Model Architecture (Revised)

```python
class RydbergSurrogate(nn.Module):
    def __init__(self, n_layer=4, n_head=4, n_embd=128, n_time=400):
        self.param_embed = nn.Sequential(
            nn.Linear(3, n_embd),  # [Omega, N, 1/sqrt(N)]
            nn.GELU(), nn.Linear(n_embd, n_embd)
        )
        self.time_embed = nn.Linear(1, n_embd)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout=0.2)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, 1)
```

| Hyperparameter | v1.0 | v2.0 |
|---|---|---|
| n_layer | 6 | 4 |
| n_head | 6 | 4 |
| n_embd | 384 | 128 |
| dropout | 0.1 | 0.2 |
| block_size | 512 | 401 |
| Total params | ~6M | ~500K |

### 4.3 Why Keep the Transformer?

We implement FNO and DeepONet as baselines. If the transformer wins, we have a methodological contribution. If it loses, we still have a solid benchmark paper. The transformer is also more flexible for future extensions.

### 4.4 Physics-Informed Constraints (New)

```python
total_loss = MSE(pred, true)
penalty = torch.relu(pred - 1.0) + torch.relu(-1.0 - pred)
total_loss += 0.1 * penalty.mean()
d2_pred = pred[:, 2:] - 2*pred[:, 1:-1] + pred[:, :-2]
total_loss += 0.01 * (d2_pred ** 2).mean()
```

---

## 5. Phase 3: Baselines (Week 5)

### 5.1 Gaussian Process
```python
kernel = ConstantKernel() * RBF(length_scale=[1.0, 100.0, 10.0]) + WhiteKernel()
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
```
Gold standard for small-data regression.

### 5.2 Fourier Neural Operator (FNO)
```python
fno = FNO(n_modes=(16,), hidden_channels=64, in_channels=3, out_channels=1)
```
State-of-the-art for operator learning.

### 5.3 DeepONet
Mathematically natural for parameter-to-function mappings.

### 5.4 Linear Interpolation
Simple baseline for nearest-neighbor interpolation.

---

## 6. Phase 4: Transformer Training (Weeks 6-7)

### 6.1 Training Configuration
```python
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=5000, eta_min=1e-5)
max_steps = 5000
batch_size = 32
gradient_clip = 1.0
patience = 500  # early stopping
```
Training time: 2-6 hours on A100.

### 6.2 Regularization
- Dropout 0.2
- Stochastic depth (layer dropout 0.1)
- Weight decay 1e-4
- Gradient clipping max_norm=1.0
- Early stopping 500 steps
- EMA decay=0.999
- Deep ensemble (5 models)

### 6.3 Experiment Tracking
Use Weights & Biases. Log hyperparameters, loss curves, sample trajectories, metrics. Use Hydra for config management.

---

## 7. Phase 5: Evaluation & Physics (Weeks 8-10)

### 7.1 Trajectory Metrics
- MSE, Relative L2, Max error
- MAE on rho_ss (last 50 points)
- Pearson correlation
- Dynamic Time Warping (DTW)

### 7.2 Phase Classification
- Absorbing: rho_ss < 0.05 vs. Active: rho_ss > 0.05
- Confusion matrix, ROC, AUC, calibration plot

### 7.3 Critical Point Estimation (Revised)
1. **Derivative method:** Omega_c(L) = argmax(d rho_ss / d Omega)
2. **Power-law fit:** rho = A * (Omega - Omega_c(L))^beta for Omega > Omega_c
3. **Finite-size scaling:** Omega_c(L) = Omega_c(inf) + A * L^{-1/nu_perp}
4. **Data collapse:** rho * L^{beta/nu_perp} vs. (Omega - Omega_c) * L^{1/nu_perp}

All fits report bootstrap 95% confidence intervals.

### 7.4 Uncertainty Quantification (New)
- Deep ensemble: 5 models, report mean +/- std
- Bootstrap on critical exponents: 1000 resamples
- Calibration check

### 7.5 Generalization Tests
- Interpolation (N=1225, unseen Omega)
- Extrapolation (Omega > 25)
- Size extrapolation (N=1600 to 4900)
- Cross-parameter (gamma, Delta if data exists)

### 7.6 Speedup Benchmark
Measure TWA time vs. NN inference time per parameter set.

---

## 8. Phase 6: Spatial Extension (Weeks 11-14)

### 8.1 Spatial Data Generation
Generate per-atom sz_atoms[k, t] for N=225, 900, 1225 at 5 Omega values. Also save per-trajectory survival flags.

### 8.2 Spatial Model (Revised)

**Abandon 64-dim CNN autoencoder.**

**Option A: Patch-based ViT**
- 5x5 patches from LxL lattice
- Learned patch embeddings
- Temporal transformer across patch sequences
- ~1960 tokens (manageable)

**Option B: ConvLSTM**
- 3D convolutions + LSTM
- Proven for spatiotemporal forecasting
- Handles 400 timesteps natively

**Option C: Hybrid**
- Transformer for mean trajectory
- Conditional U-Net for spatial residuals

Start with Option A; fallback to B if unstable.

### 8.3 Critical Analysis from Spatial Data
With per-atom, per-trajectory data:
- Survival probability P_surv(t) ~ t^{-delta}
- Susceptibility peak chi = N * (<rho^2> - <rho>^2)
- Spatial correlation C(r) and correlation length xi
- Cluster size distribution P(S) ~ S^{-tau_s}
- True finite-size scaling with genuine critical exponents

---

## 9. Phase 7: Interpretability (Weeks 15-16)

### 9.1 Linear Probing
Freeze encoder weights. Train linear classifier on hidden states to predict phase, distance to criticality, system size.

### 9.2 Attention Analysis (Revised Claims)
Frame as exploratory hypothesis generation only. Do NOT claim attention weights = physical correlations.

### 9.3 Ablation Studies
- Remove each attention head
- Randomize parameter embeddings
- Test parameter usage vs. memorization

---

## 10. Phase 8: Paper Writing (Weeks 17-20)

### 10.1 Revised Title Options
1. "A Transformer-Based Neural Surrogate for Non-Equilibrium Rydberg Facilitation Dynamics"
2. "Learning Non-Equilibrium Phase Transitions: A Transformer Surrogate for Rydberg Atom Arrays"

### 10.2 Target Venues
| Venue | Realism |
|---|---|
| Physical Review Research | High |
| Machine Learning: Science and Technology | High |
| NeurIPS/ICML AI for Science | Medium-High |
| APL Machine Learning | Medium |
| PRX Quantum | Low |
| Nature Physics | Very Low |

### 10.3 Paper Structure
1. Introduction: Motivation, surrogate framing
2. Model: Architecture, baselines (FNO, DeepONet)
3. Data: Rydberg facilitation, parameters
4. Results: Trajectory accuracy, phase diagram, effective exponents, speedup
5. Interpretability: Linear probing, attention (exploratory)
6. Discussion: Limitations (mean-field data, fixed Hamiltonian)
7. Methods: Training protocol, reproducibility checklist

### 10.4 Reproducibility Checklist
- Fixed random seeds
- Exact split indices
- Model checkpoints
- Hyperparameter YAML files
- requirements.txt
- Inference script

---

## 11. Hardware & Dependencies

### 11.1 Compute Estimates
| Task | Resource | Time |
|---|---|---|
| Data preprocessing | CPU | Hours |
| Baselines | 1x GPU | 1-2 hours |
| Transformer training | 1x A100 | 2-6 hours |
| Spatial simulation | 20 CPU cores | 1-2 days |
| Spatial training | 1x A100 | 2-4 days |
| **Total GPU** | 1x A100 | **~3-5 days** |

### 11.2 Dependencies
```
Python 3.10+, PyTorch 2.0+, numpy, scipy, pandas, matplotlib, h5py,
scikit-learn, neuraloperator, wandb, hydra-core, torchmetrics
```

---

## 12. Risk Register (Revised)

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Transformer overfits on 110 examples | High | Critical | 500K params, dropout 0.2, augmentation, early stopping, ensemble |
| FNO/DeepONet outperform transformer | Medium | High | Expected and fine -- publish the comparison |
| TWA data does not exhibit DP universality | High | Critical | Reframe as surrogate for TWA dynamics |
| t_max=1000 insufficient for large systems | High | High | Convergence check; flag non-converged points |
| Ensemble-averaged data prevents true critical analysis | Certain | Critical | Acknowledge explicitly; defer to spatial follow-up |
| Spatial extension too complex | Medium | Medium | Mean-field surrogate is already complete standalone |
| Reviewers say "just curve fitting" | Medium | High | Emphasize generalization, speedup, baselines, UQ |
| NOQS authors publish competing work | Medium | Critical | Differentiate by focusing on surrogate aspect |

---

## Appendix: Key Insights from Reviews

### From Physics Review
- Ensemble-averaged data cannot reveal true DP critical exponents (bimodal distribution destroyed by averaging)
- TWA may not reproduce DP exponents anyway (semiclassical approximation)
- Sigmoid fit is wrong functional form; use power law
- t_max=1000 likely too short for L=70 near criticality (tau ~ L^z ~ 1700)
- 64-dim autoencoder destroys power-law correlations and fractal clusters
- Order parameter must be rho = (sz_mean + 1)/2, not sz_mean directly
- Must distinguish nu_perp (spatial) and nu_parallel (temporal)

### From Engineering Review
- 6M params on 290 examples = severe overfitting; reduce to 300K-800K
- 50K steps = 11K epochs = memorization; reduce to 5K-10K steps
- Cross-entropy on continuous physics is wrong; use MSE regression
- Token-based conditioning destroys smoothness; use continuous embeddings
- Missing W&B, Hydra, checkpoint management
- Compute estimate was 10x too pessimistic for mean-field
- Need gradient clipping, EMA, deeper ensembles

### From ML Research Review
- "Foundation model" framing is scientifically inaccurate; use "neural surrogate"
- Autoregressive token generation for deterministic data is methodological mismatch
- Must compare against FNO and DeepONet or be rejected by ML reviewers
- Need deep ensemble UQ, bootstrap CIs on exponents
- Attention analysis is not scientifically defensible as physical insight
- Add linear probing for minimum interpretability bar
- Target PR Research or MLST, not PRX Quantum or Nature Physics

---

*Plan version 2.0 -- 2026-04-24*
*Next step: Await user approval before any code is written*
