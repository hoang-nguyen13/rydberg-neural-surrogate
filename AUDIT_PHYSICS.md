# Physics Audit: Neural Surrogate for Rydberg Facilitation Dynamics

**Auditor:** Senior Theoretical Physicist (Non-Equilibrium Statistical Mechanics, Quantum Optics, Rydberg Arrays)  
**Date:** 2026-04-24  
**Scope:** Full codebase review with emphasis on physical correctness, conceptual consistency, and hidden assumptions.

---

## Executive Summary

The codebase implements a neural surrogate for the **mean magnetization** `sz_mean(t)` of a Rydberg facilitation model. While the high-level goal is reasonable, the implementation contains **multiple conceptual errors, physically invalid data-augmentation procedures, and misapplications of statistical-mechanics principles**. The most severe issue is the **mixup augmentation** (`data/dataset.py`), which generates training samples that violate the underlying master equation and invent non-existent system sizes. Other critical concerns include:

- A **naming/definition bug** where `rho_ss` is actually the steady-state `sz` (not the DP order parameter `rho`), propagating confusion through training and evaluation.
- **Unjustified steady-state extraction** via a flat average over the last 50 time points, which fails near the critical point due to **critical slowing down**.
- **Misleading "physics-informed" loss** that is merely a generic smoothness regularizer and can suppress physically valid sharp transients.
- **Fixed phase-classification threshold** (`sz < -0.95`) that ignores finite-size scaling and misclassifies weakly active trajectories.
- **Incorrect finite-size scaling feature** (`inv_sqrt_n = 1/sqrt(N)`) hard-coded as a model input, contradicting the DP correlation-length exponent `nu_perp ~ 0.734`.
- **Misapplication of FNO** to a temporal domain with a fixed initial condition, violating the periodicity assumption of the Fourier basis.

| File | Rating | Key Issue |
|------|--------|-----------|
| `data/parse_jld2.py` | **NEEDS_FIX** | `rho_ss` stores `sz_ss`; hardcoded 400-step shape; undocumented parameter assumptions. |
| `data/dataset.py` | **CRITICAL** | Mixup creates unphysical trajectories and fictitious `N`; noise+clipping biases absorbing state. |
| `models/transformer_surrogate.py` | **NEEDS_FIX** | No hard physics constraints (IC, bounds, asymptotics); linear time embedding is inadequate for multi-scale relaxation. |
| `train.py` | **NEEDS_FIX** | Smoothness penalty is not physics-informed; no critical-region weighting; steady-state assumption baked in. |
| `scripts/check_data.py` | **NEEDS_FIX** | Arbitrary phase threshold; no stationarity test; no monotonicity/bounds checks. |
| `scripts/evaluate.py` | **NEEDS_FIX** | Missing all physics observables (critical exponents, FSS, relaxation times); unstable `rel_l2`; no extrapolation visuals. |
| `baselines/gp_baseline.py` | **NEEDS_FIX** | Kernel mis-specified for finite-size scaling; flattens trajectories (destroys temporal correlations); unfair subsampling. |
| `baselines/fno_baseline.py` | **NEEDS_FIX** | Fourier truncation destroys early-time transients; periodicity assumption violated by fixed initial condition. |

---

## 1. `data/parse_jld2.py` -- **NEEDS_FIX**

### 1.1 Order parameter `rho` is correctly defined, but `rho_ss` is a conceptual bug
**Lines 154-155:**
```python
rho = (sz_mean + 1.0) / 2.0  # DP order parameter: 0 = absorbing, >0 = active
rho_ss = float(np.mean(sz_mean[-50:]))  # steady-state from last 50 points
```
- `rho = (sz + 1)/2` **is correct** *if and only if* `sz = -1` corresponds to the all-ground (absorbing) state. The initial condition check in `check_data.py` (expecting `sz[0] ~ +1`) confirms the convention: the system starts **fully excited** and relaxes toward `sz = -1`. Thus `rho` is the excitation density, and `rho = 0` is indeed the absorbing state. *However*, the code does **not document this unconventional initial condition** (most facilitation studies start from the vacuum).
- **Critical naming error:** `rho_ss` is defined as the mean of the *raw* `sz_mean`, not of `rho`. It should be:
  ```python
  rho_ss = float(np.mean(rho[-50:]))  # or equivalently (np.mean(sz_mean[-50:]) + 1)/2
  ```
  Because the field is named `rho_ss`, downstream code (`dataset.py`, `train.py`, `evaluate.py`) treats it as the DP order parameter, but it is actually `sz_ss`. This propagates a **factor-of-2 offset and sign confusion** through all evaluation metrics.

### 1.2 Hardcoded trajectory length with no physics justification
**Line 148:**
```python
if len(sz_mean) != 400 or len(t_save) != 400:
```
The code enforces exactly 400 saved times. There is no justification for this number. Near the critical point, relaxation can be algebraic; a finer or coarser grid may be needed. The parser should validate the time grid (e.g., monotonicity, coverage of `t_max`) rather than enforcing a magic number.

### 1.3 Undocumented physical parameter assumptions
**Lines 164-165:**
```python
V=dir_params['delta'] or 2000.0,  # V = Delta per run script
Gamma=1.0,  # From run script
```
The facilitation condition `V = Delta` is hardcoded without comment. `Gamma = 1.0` sets the energy/time unit. These are fundamental model choices that must be documented in the parser docstring or a metadata file. A user who wants to generalize the surrogate to `V != Delta` will have no warning that the data was generated under this constraint.

### 1.4 Lattice geometry assumption
**Line 158:**
```python
lattice_size = int(np.sqrt(n_atoms)) if n_atoms is not None else None
```
This silently assumes a **square 2D lattice**. If the data ever includes rectangular or higher-dimensional geometries, `int(sqrt(N))` will produce a nonsensical length. There is no assertion that `n_atoms` is a perfect square.

---

## 2. `data/dataset.py` -- **CRITICAL**

### 2.1 Mixup augmentation is physically invalid
**Lines 85-96:**
```python
if random.random() < 0.2 and len(self.records) > 1:
    j = random.randint(0, len(self.records) - 1)
    rec2 = self.records[j]
    lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
    sz_mean = lam * sz_mean + (1.0 - lam) * sz_mean2
    omega = lam * omega + (1.0 - lam) * rec2.omega
    n_atoms = int(lam * n_atoms + (1.0 - lam) * rec2.n_atoms)
    inv_sqrt_n = 1.0 / np.sqrt(n_atoms)
```
This is the **most severe physics error** in the codebase.
- **Trajectory interpolation:** The master equation for `sz_mean(t)` is **non-linear**. A convex combination of two solutions for different parameters is **not** a solution for any intermediate parameter. The interpolated `sz_mean` does not correspond to any physical dynamics.
- **Fictitious system sizes:** `n_atoms` is interpolated and cast to `int`. For example, mixing `N=225` and `N=400` with `lambda=0.5` yields `N=312`, a lattice size that does not exist. The model is then trained on a system with `inv_sqrt_n = 1/sqrt(312)`, learning a mapping from a **non-existent finite-size scaling variable**.
- **Recommendation:** Remove mixup entirely. If data augmentation is needed, use physically valid perturbations (e.g., small parameter jitters within the experimental uncertainty, or resampling the stochastic trajectories that generated the mean).

### 2.2 Trajectory noise + clamping creates a systematic bias in the absorbing phase
**Lines 79-83:**
```python
noise = np.random.normal(0, self.trajectory_noise_std, size=sz_mean.shape).astype(np.float32)
sz_mean = sz_mean + noise
# Clamp to physical bounds
sz_mean = np.clip(sz_mean, -1.0, 1.0)
```
- For trajectories in the **absorbing phase** (`sz ~ -1`), symmetric Gaussian noise is truncated from below. The mean of the clipped noise is **positive**, so the augmented data systematically shifts the absorbing state upward, introducing a **spurious active bias**.
- The standard deviation `0.01` is also large relative to the statistical precision of a well-converged ensemble average. If the original JLD2 data was averaged over many quantum trajectories, the true error bar on `sz_mean` may be `O(1e-3)` or smaller. Adding `0.01` noise overwrites the physical signal.

### 2.3 `rho_ss` is still misnamed
**Line 100:**
```python
rho_ss = float(np.mean(sz_mean[-50:]))
```
Same bug as in the parser: `rho_ss` is `sz_ss`, not the order parameter.

### 2.4 Train/val/test splits lack critical-region stratification
**Lines 113-162:**
The split is parameter-based (by `N`), which is sound for testing size extrapolation. However:
- The validation set uses **even/odd indices** on sorted `Omega`. If the `Omega` grid is non-uniform (e.g., denser near the critical point), this can leave large gaps in the training coverage of the most important region.
- There is **no stratification by phase or distance to criticality**. The critical point is the most challenging and physically interesting region. If most near-critical points land in the test set, the model will fail to learn the universality class.
- The training set sizes `[225, 400, 900]` are all relatively small. The test set includes `N` up to `4900`. Extrapolation over more than a factor of 5 in linear system size is extremely demanding for a black-box surrogate; finite-size scaling laws are subtle.

### 2.5 Finite-size scaling feature is wrong
**Line 69:**
```python
inv_sqrt_n = 1.0 / np.sqrt(n_atoms)
```
For 2D directed percolation (the expected universality class), finite-size scaling of the order parameter near criticality follows
```
rho_ss(L, Omega) ~ L^{-beta/nu_perp} f( (Omega - Omega_c) L^{1/nu_perp} )
```
with `nu_perp ~ 0.734` and `beta ~ 0.583`. The natural scaling variable is `L^{-1/nu_perp} ~ N^{-0.682}`, not `N^{-0.5}`. By feeding `1/sqrt(N)` as a hand-engineered feature, the authors **bake in an incorrect scaling exponent**. The Transformer may learn to correct this, but the feature engineering contradicts known critical exponents.

---

## 3. `models/transformer_surrogate.py` -- **NEEDS_FIX**

### 3.1 Linear time embedding is inadequate for multi-scale relaxation
**Line 81:**
```python
self.time_embed = nn.Linear(1, n_embd)
```
The dynamics span `t in [0, 1000]` and exhibit **multiple time scales**: fast initial transients (Rabi oscillations or rapid decay), intermediate relaxation, and slow critical algebraic tails. A linear embedding of absolute time cannot represent logarithmic or power-law time dependence. A **sinusoidal positional encoding** (like in the original Transformer) or a **log-time embedding** would be far more appropriate.

### 3.2 No hard physics constraints
The architecture is a vanilla Transformer regressor. It does **not enforce**:
- **Initial condition:** `sz(0) = +1` is guaranteed by the data but not by the model. The surrogate can predict arbitrary values at `t=0`.
- **Bounds:** `sz in [-1, 1]` is not enforced (the head is linear). The model can and will predict unphysical values during extrapolation.
- **Asymptotics:** For `Omega` in the absorbing phase, the model should approach `sz -> -1` as `t -> infinity`. There is no structural bias toward this.

For a physics-respecting surrogate, consider:
- An output activation `tanh` or a soft clamp.
- An initial-condition residual: `sz_pred(t) = +1 + t * network(t, ...)` so that `sz_pred(0) = +1` exactly.
- A long-time bias term that forces the model to a learned steady state.

### 3.3 Non-causal attention is acceptable but not exploited
**Lines 12-43:**
The self-attention is non-causal. For a **deterministic function regression** task `sz(t) = f(t; Omega, N)`, this is not a physics error. The target at time `t` is not a random variable conditioned only on the past; it is a fixed value. However, the architecture could exploit causality (e.g., via causal masking) to ensure predictions at early times do not depend on unobserved late times, which is conceptually cleaner for dynamical systems.

---

## 4. `train.py` -- **NEEDS_FIX**

### 4.1 Smoothness penalty is NOT physics-informed
**Lines 49-54:**
```python
# Smoothness penalty: penalize large second derivatives in time
if pred.size(1) > 2:
    d2 = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    smoothness_penalty = (d2 ** 2).mean()
```
This is a **generic Tikhonov regularizer**, not a physics-informed term. The true dynamics can have:
- **Rabi oscillations** at early times (large `d2sz/dt2`).
- **Inflection points** during relaxation.
- **Algebraic approach** to steady state near criticality (`sz(t) - sz_ss ~ t^{-delta}`), where the second derivative decays as a power law and is not negligible.

Penalizing curvature will bias the model away from these physically correct features. If regularization is needed, use a small weight and **do not call it physics-informed**.

### 4.2 Bounds penalty is too weak
**Lines 45-47:**
```python
lower_violation = torch.relu(-1.0 - pred)
upper_violation = torch.relu(pred - 1.0)
bounds_penalty = (lower_violation + upper_violation).mean()
```
With `bounds_weight=0.1`, the penalty is soft and easily overwhelmed by the MSE term, especially during early training or size extrapolation. A hard constraint (e.g., output `tanh`) would be more robust.

### 4.3 Steady-state assumption is baked into evaluation
**Lines 122-125:**
```python
pred_rho_ss = pred[:, -50:].mean(dim=1)
true_rho_ss = sz_mean[:, -50:].mean(dim=1)
```
This assumes the system is stationary over the last 50 points. **Near the critical point, critical slowing down causes the relaxation time to diverge**. For `t_max = 1000`, the system may still be evolving algebraically at late times. The evaluation should:
- Fit the late-time tail to an exponential or power law and extrapolate to `t -> infinity`.
- Or at minimum, report the **trend** (slope) of the last 50 points as a diagnostic.

### 4.4 No loss weighting for the critical region
The MSE loss treats all `(Omega, N, t)` equally. In statistical mechanics, the **critical region** is the hardest to learn and the most important. A weighted loss that up-weights near-critical trajectories (e.g., those where the steady-state density is small but non-zero) would improve physical fidelity.

### 4.5 Missing initial-condition loss term
There is no penalty for `|pred(t=0) - 1.0|`. The model should be forced to respect the exact initial state.

---

## 5. `scripts/check_data.py` -- **NEEDS_FIX**

### 5.1 Phase threshold is arbitrary and physically unjustified
**Lines 238-241:**
```python
n_absorbing = sum(1 for r in records if r.rho_ss < -0.95)
n_active = sum(1 for r in records if r.rho_ss >= -0.95)
```
Since `r.rho_ss` is actually `sz_ss`, the threshold `sz = -0.95` corresponds to `rho = 0.025`. This threshold is:
- **Fixed for all N**, ignoring finite-size scaling. For large `N`, the active density just above `Omega_c` can be much smaller than 0.025.
- **Not derived from any physical criterion** (e.g., crossing of susceptibility peak, Binder cumulant, or scaling function).

A proper classification would use a **size-dependent threshold** or fit the data to a scaling form.

### 5.2 Plots use the correct `rho_ss` but the data structure does not
**Line 108:**
```python
rho_ss_vals = [(np.mean(r.sz_mean[-50:]) + 1) / 2 for r in recs]
```
This correctly computes the order parameter for visualization, highlighting that the stored `rho_ss` field is wrong.

### 5.3 Missing physical consistency checks
The script checks for NaN, Inf, shape, and initial condition. It does **not** check:
- Whether `sz_mean` stays within `[-1, 1]` at all times.
- Whether trajectories are monotonic (if the physical model predicts monotonic decay from the fully excited state).
- Whether the last 50 points are stationary (e.g., linear regression slope ~ 0).
- Whether the phase diagram shape is sensible (e.g., a critical `Omega_c` that shifts with `N`).

---

## 6. `scripts/evaluate.py` -- **NEEDS_FIX**

### 6.1 Same threshold bug and missing observables
**Lines 56-62:**
```python
pred_rho_ss = np.mean(all_pred[:, -50:], axis=1)
true_rho_ss = np.mean(all_true[:, -50:], axis=1)
pred_phase = pred_rho_ss > -0.95
true_phase = true_rho_ss > -0.95
```
Same issue: `pred_rho_ss` is `sz_ss`, and the threshold is arbitrary.

**Missing physics observables:**
- **Critical point estimator:** `Omega_c(N)` from the crossing of curves or inflection point of `rho_ss(Omega)`.
- **Finite-size scaling collapse:** Plot `N^{beta/nu_perp} rho_ss` vs `N^{1/nu_perp}(Omega - Omega_c)` to test whether the model learns the correct universality class.
- **Relaxation time:** Fit `sz(t) - sz_ss` to an exponential `exp(-t/tau)` or power law `t^{-delta}` and compare `tau` or `delta` between prediction and truth.
- **Initial condition error:** `|pred(t=0) - 1.0|`.
- **Bounds violation rate:** Fraction of predictions outside `[-1, 1]`.
- **Dynamic susceptibility:** If trajectory ensembles were available, `chi(t) = N (<sz^2> - <sz>^2)`.

### 6.2 Relative L2 is numerically unstable
**Line 54:**
```python
rel_l2 = np.mean(np.linalg.norm(all_pred - all_true, axis=1) / np.linalg.norm(all_true, axis=1))
```
For trajectories in the absorbing phase, `||true||` is non-zero but small variations can cause large relative errors. More importantly, relative metrics are not standard in this field; absolute errors in `rho` are more interpretable.

### 6.3 Visualization ignores extrapolation test set
**Line 82:**
```python
n1225 = [r for r in records if r.n_atoms == 1225]
```
The trajectory comparison plot is only for `N=1225` (interpolation test). It does **not** visualize the size-extrapolation test set (`N = 1600, 2500, 3025, 3600, 4900`), which is the main scientific question.

---

## 7. `baselines/gp_baseline.py` -- **NEEDS_FIX**

### 7.1 Kernel mis-specified for finite-size scaling
**Lines 43-46:**
```python
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * \
         RBF(length_scale=[1.0, 100.0, 10.0],
             length_scale_bounds=(1e-2, 1e3)) + \
         WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
```
- The length scale for `N` is `100.0`. With `N in [225, 4900]`, this implies the GP expects `sz` to vary significantly over `Delta N ~ 100`. In reality, finite-size scaling is smooth; a much larger length scale (comparable to the range of `N`) is more appropriate.
- The kernel is a simple product of stationary RBFs. It cannot capture the **non-analytic behavior** (algebraic singularity) at the critical point.

### 7.2 Flattening destroys temporal correlations
**Lines 22-34:**
```python
for i in range(0, len(rec.t_save), time_subsample):
    X.append([omega, n, rec.t_save[i]])
    y.append(rec.sz_mean[i])
```
The GP treats each `(Omega, N, t)` triple as an **independent observation**. This ignores the strong temporal correlation along a single trajectory. A separable spatiotemporal kernel `k((Omega,N), (Omega',N')) * k_t(t, t')` would respect the physics that points on the same trajectory are highly correlated.

### 7.3 Unfair subsampling
**Line 126:**
```python
parser.add_argument('--time_subsample', type=int, default=5)
```
The GP is trained on only **20% of the time points**. Comparing its MSE to the Transformer (trained on all 400 points) is misleading. If the GP were trained on all points, it would be intractable, but that is a limitation of the baseline, not a fair comparison.

### 7.4 Normalization removes physical bounds
**Line 52:**
```python
normalize_y=True,
```
Normalizing `sz` across the dataset (which includes both fully excited and absorbing trajectories) allows the GP to predict values outside `[-1, 1]`.

---

## 8. `baselines/fno_baseline.py` -- **NEEDS_FIX**

### 8.1 Severe Fourier mode truncation
**Lines 33-39:**
```python
self.fno = FNO(
    n_modes=(16,),
    ...
)
```
For 400 time points, keeping only 16 modes is a **brutal low-pass filter**. The early-time dynamics (rapid drop from `sz=+1`) contain high-frequency content. The FNO will smooth these out, producing inaccurate short-time predictions.

### 8.2 Periodicity assumption is violated
FNO is designed for **spatial domains with periodic or homogeneous boundary conditions** (e.g., PDEs on a torus). The temporal domain `[0, 1000]` has a **fixed initial condition** at `t=0` and a non-periodic steady-state approach. The Fourier basis assumes periodicity, which is **physically wrong** for this problem. The initial condition at `t=0` breaks translation invariance in time.

### 8.3 No physics constraints
Like the Transformer baseline, the FNO has no mechanism to enforce `sz(0)=+1` or `sz in [-1, 1]`.

---

## Cross-Cutting Physics Concerns

### A. Critical slowing down and steady-state extraction
The entire pipeline (`parse_jld2`, `dataset`, `train`, `evaluate`) assumes that `t in [875, 1000]` (the last 50 of 400 points spanning `t_max=1000`) represents the steady state. **This assumption fails catastrophically near the critical point**, where the relaxation time diverges as `tau ~ |Omega - Omega_c|^{-nu_parallel}` (`nu_parallel ~ 1.295` in 2+1D DP). For a system with `N=4900` near `Omega_c`, the system may still be relaxing algebraically at `t=1000`. A rigorous analysis would:
1. Fit the late-time tail to `sz(t) = sz_ss + A t^{-delta} + B exp(-t/tau)`.
2. Extract `sz_ss` by extrapolation.
3. Flag trajectories where the fitted `tau` is comparable to `t_max` as "not converged".

### B. Missing spatial structure
The surrogate predicts only the **global magnetization** `sz_mean(t)`. Rydberg facilitation is a **spatially extended stochastic process**. The DP universality class is defined by spatial correlations, cluster growth, and dynamic percolation. A surrogate that only predicts the mean field misses:
- Spatial correlation length `xi` and its divergence.
- Cluster-size distributions.
- Dynamic structure factor `S(k, t)`.
- Entanglement entropy (in the quantum case).
The project description calls this a "neural surrogate for Rydberg facilitation dynamics," but it is actually a **mean-field surrogate**. This is a massive reduction of the physics.

### C. Hidden universality-class assumptions
The code assumes the transition is in the **directed percolation (DP) universality class**. This is standard for classical facilitation, but quantum fluctuations or long-range interactions can change the universality class (e.g., to the Manna or conserved-DP class). The code does not verify this assumption; it hardcodes DP scaling only implicitly via the `inv_sqrt_n` feature.

### D. Fairness of baseline comparisons
- **GP vs. Transformer:** The GP is trained on 20% of the temporal data (`time_subsample=5`) and treats time points as independent. The Transformer sees the full trajectory and learns temporal correlations. This is an **apples-to-oranges comparison**. A fair comparison would give both methods the same training data format or at least report the GP's performance as a function of subsampling ratio.
- **FNO vs. Transformer:** The FNO uses only 16 Fourier modes and assumes periodicity. The Transformer has no such restriction. Comparing MSE alone obscures the fact that the FNO is structurally handicapped for this problem.

---

## Recommendations (Priority Order)

1. **Remove mixup augmentation immediately** (`data/dataset.py`). It violates the master equation and creates fictitious system sizes.
2. **Fix the `rho_ss` naming bug** across `parse_jld2.py`, `dataset.py`, `train.py`, and `evaluate.py`. Ensure `rho_ss` always means the DP order parameter `(sz + 1)/2`.
3. **Replace the fixed phase threshold** with a size-dependent criterion or a finite-size scaling analysis.
4. **Add physics constraints to the model architecture**: initial-condition residual, output bounds (e.g., `tanh`), and optionally a steady-state bias.
5. **Replace the misleading "physics-informed" smoothness loss** with either a true physics residual (e.g., penalize deviation from a known ODE limit) or admit it is generic regularization.
6. **Validate steady-state convergence** by fitting late-time tails and flagging non-converged trajectories instead of blindly averaging the last 50 points.
7. **Add physics observables to evaluation**: critical point estimator, FSS collapse, relaxation time, initial-condition error, and bounds-violation rate.
8. **Correct the finite-size scaling feature**: replace `1/sqrt(N)` with `N^{-1/(2*nu_perp)}` or simply let the model learn `log(N)` and discover the exponent.
9. **Fix GP baseline**: use a separable spatiotemporal kernel and train on the full temporal grid (or fairly report the subsampling handicap).
10. **Fix or drop FNO baseline**: either increase `n_modes` dramatically or replace the FNO with a temporal convolutional network that respects causality and the initial condition.

---

*End of Audit*
