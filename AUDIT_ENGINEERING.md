# Engineering / ML Audit: Transformer Neural Surrogate for Rydberg Facilitation

**Auditor:** Senior ML Engineer  
**Scope:** 8 files across data pipeline, model architecture, training loop, evaluation, and baselines.  
**Overall Assessment:** The codebase contains **two critical bugs that silently corrupt data and break the training schedule**, several architecture anti-patterns, and unfair baseline comparisons. It is not production-ready.

---

## 1. `data/parse_jld2.py` — CRITICAL

### CRITICAL BUG: Regex silently truncates all floating-point parameters (Line 66–89)
The filename parser uses **lazy** quantifiers `+?` combined with a trailing group that matches `.` or `,`:

```python
# Line 73
m = re.search(r'Ω=([0-9.]+?)(?:\.|,|$)', fname)
omega = float(m.group(1)) if m else None
```

Because `+?` is lazy, it stops at the **first** valid terminator. For a filename like `ρ_ss_2D,Ω=10.0,Δ=2000.0,γ=0.1.jld2`:
- The capture group matches `10`, the literal `.` is consumed by the non-capturing group `(?:\.|,|$)`, and `float("10")` returns `10.0`.
- **Every decimal is truncated to its integer part.** `γ=0.1` becomes `0.0`. `Ω=0.5` becomes `0.0`.

**Impact:** The entire dataset's parameter space is corrupted. Omega and gamma distributions are discretized to integers. Any model trained on this data is learning the wrong physics.

**Fix:** Use a greedy quantifier (remove `?`) or a robust pattern:
```python
r'Ω=([0-9]+(?:\.[0-9]+)?)'
```
Apply the same fix to the delta and gamma parsers (lines 77, 81).

### SEMANTIC BUG: `rho_ss` is not actually `rho` steady-state (Line 154–155)
```python
rho = (sz_mean + 1.0) / 2.0          # DP order parameter, in [0, 1]
rho_ss = float(np.mean(sz_mean[-50:]))  # sz_mean steady state, in [-1, 1]
```
The field `rho_ss` stores the mean of `sz_mean`, not the mean of `rho`. This misnaming propagates through `dataset.py`, `train.py`, and `evaluate.py`. The code is internally consistent (everyone treats `rho_ss` as `sz_mean` scale), but it is semantically wrong and confusing.

**Fix:** Rename to `sz_ss` or compute correctly: `rho_ss = float(np.mean(rho[-50:]))`.

### Other Issues
- **Line 148:** Hardcoded length check `len(sz_mean) != 400`. Magic number; should be a module-level constant or derived from data.
- **Line 158:** `lattice_size = int(np.sqrt(n_atoms))` silently truncates non-square lattices. No assertion or warning.
- **Line 163–164:** `V=dir_params['delta'] or 2000.0` and `Gamma=1.0` are magic numbers with no justification.
- **Line 192:** `pickle.dump(records, f)` uses the default protocol. For large datasets, `protocol=pickle.HIGHEST_PROTOCOL` is preferred.

---

## 2. `data/dataset.py` — NEEDS_FIX

### BUG: Mixup corrupts `n_atoms` and can cause division by zero (Line 86–96)
```python
n_atoms = int(lam * n_atoms + (1.0 - lam) * rec2.n_atoms)
inv_sqrt_n = 1.0 / np.sqrt(n_atoms)
```
Mixing the number of atoms with a continuous interpolation factor is physically meaningless. Worse, `int()` truncates toward zero. For small `lam` and small `n_atoms`, this can yield `n_atoms = 0`, causing `inv_sqrt_n = inf` and downstream NaNs.

**Fix:** Do not mix `n_atoms`. Use the `lam`-interpolated `inv_sqrt_n` directly without re-deriving it from a mixed integer:
```python
inv_sqrt_n = lam * (1.0/np.sqrt(rec.n_atoms)) + (1.0 - lam) * (1.0/np.sqrt(rec2.n_atoms))
```

### BUG: Asymmetric augmentation in mixup (Line 86–96)
`rec2` is drawn from the raw records; jitter and trajectory noise are applied to `rec` but **not** to `rec2`. The mixup is therefore biased toward the cleaner, unaugmented sample.

**Fix:** Apply the same augmentation pipeline to `rec2` before mixing, or draw `j` first and augment both samples independently.

### BUG: Global RNG state is not worker-safe (Line 76, 79, 86, 89)
The dataset uses `random.gauss`, `np.random.normal`, `random.random`, and `np.random.beta`. These rely on the **global** Python and NumPy RNGs. When `DataLoader` uses `num_workers > 0`, each worker forks from the parent process and inherits the **same seeded state**. All workers will generate identical augmentation noise for corresponding samples across epochs.

**Fix:** Implement a `worker_init_fn` that seeds each worker's RNG independently using the worker id and epoch.

### Other Issues
- **Line 113–162:** `create_splits` has a silent failure mode. If `val_omega_indices` is misspelled (e.g., `'evn'`), the `else` branch executes, assigning odd indices to validation and even to test without any warning. Add an assertion.
- **Line 172–195:** `collate_fn` assumes all sequences are length 400 but never asserts it. If a malformed sample sneaks in, the batch will silently succeed but produce wrong shapes downstream.
- **Line 83:** `np.clip(sz_mean, -1.0, 1.0)` after adding noise is a hard floor/ceiling. This can create a pile-up of values at exactly `-1.0` and `1.0`, biasing the gradient at the boundaries.


---

## 3. `models/transformer_surrogate.py` — NEEDS_FIX

### ARCHITECTURE: No residual scaling, suboptimal initialization for depth (Line 91–100)
```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```
The model uses a fixed `std=0.02` for all layers. For a transformer with GELU activations and residual connections, this lacks:
1. **Kaiming/Glorot scaling** for GELU layers.
2. **Residual branch scaling** (e.g., GPT-2's `1/sqrt(N)` scaling or pre-LayerNorm with scaled residuals).

For the default 4-layer model this may train, but for deeper variants the unscaled residual sum can cause activation explosion or vanishing gradients.

**Fix:** Use `nn.init.xavier_uniform_` or `kaiming_uniform_` for linear layers, and consider a per-block residual gain parameter (e.g., `x = x + gain * self.attn(...)`).

### ARCHITECTURE: Parameter conditioning is purely additive (Line 132)
```python
x = param_emb.unsqueeze(1) + time_emb  # (batch, n_time, n_embd)
```
The parameter embedding is broadcast-added to every time step. This is a bias-only conditioning. The transformer cannot perform **multiplicative** gating of time features by parameters (e.g., FiLM, AdaLN, or cross-attention). For a physics surrogate where dynamics speed/amplitude depends strongly on Omega, this limits expressivity.

**Fix:** Consider `x = time_emb * param_emb.unsqueeze(1)` (multiplicative) or use FiLM layers.

### CODE QUALITY: Unused argument (Line 70)
```python
def __init__(self, ..., n_time=400, ...):
    self.n_time = n_time
```
`self.n_time` is stored but never used in `forward`. The model infers `T` from `t.size(1)`. This is fine for flexibility but the argument is misleading if a user expects the model to enforce the length.

### CODE QUALITY: No shape validation in `forward` (Line 102–141)
If `omega` is passed as `(batch, 2)` by mistake, `torch.cat([omega, n_atoms, inv_sqrt_n], dim=-1)` concatenates along the last dimension and produces a `(batch, 6)` tensor. `nn.Linear(3, n_embd)` will then raise a runtime error with a confusing message.

**Fix:** Add assertions:
```python
assert omega.dim() <= 2 and omega.size(-1) == 1, f"omega must be (batch,) or (batch,1), got {omega.shape}"
```

### Other Issues
- **Line 27–43:** `SelfAttention` uses full bidirectional attention. This is correct for direct trajectory regression, but a comment should clarify that future timesteps are visible by design (this is not an autoregressive model).
- **Line 36:** Attention is computed without any memory-efficient implementation (e.g., FlashAttention, `scaled_dot_product_attention`). For `T=400` this is acceptable, but for longer trajectories the O(T²) memory will become a bottleneck.

---

## 4. `train.py` — CRITICAL

### CRITICAL BUG: Scheduler step/epoch mismatch (Line 168, 186)
```python
scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=args.lr * 0.01)
...
for epoch in range(args.max_epochs):
    train_metrics = train_epoch(...)
    scheduler.step()  # Called once per EPOCH
```
`CosineAnnealingLR` is configured with `T_max=args.max_steps` (default 5000) but is stepped **once per epoch** (default max_epochs=1000). The scheduler expects 5000 steps but receives at most 1000 calls. With a typical batch count, the LR will barely decay before early stopping. Conversely, if `max_steps` were smaller than `max_epochs`, the scheduler would restart or behave unpredictably.

**Impact:** The learning rate schedule is broken. The model may never reach the intended minimum LR, hurting convergence.

**Fix:** Either set `T_max=args.max_epochs` (if stepping per epoch) or move `scheduler.step()` inside the batch loop (if stepping per step) and terminate training based on `global_step >= args.max_steps`.

### CRITICAL BUG: `save_path` can be undefined at print time (Line 224, 237)
```python
if val_metrics['mse'] < best_val_mse:
    ...
    save_path = Path(args.output_dir) / 'best_model.pt'
    torch.save(checkpoint, save_path)
...
print(f"Best model saved to: {save_path}")  # Line 237
```
If validation MSE is `NaN` on every epoch (e.g., from gradient explosion, bad initialization, or division by zero), `NaN < inf` is `False`, so `save_path` is never assigned. The final `print` on line 237 raises a `NameError`.

**Fix:** Initialize `save_path = None` before the loop and guard the print:
```python
save_path = None
...
if save_path:
    print(f"Best model saved to: {save_path}")
```

### BUG: Weight decay applied to all parameters (Line 167)
```python
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
```
`AdamW` applies weight decay to **all** parameters, including biases and LayerNorm affine parameters. The transformer standard (BERT/GPT) is to exclude biases and normalization parameters from weight decay. Decaying biases and LayerNorm gammas/betas regularizes the wrong things and can degrade training stability.

**Fix:**
```python
decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
param_groups = [
    {'params': decay_params, 'weight_decay': args.weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]
optimizer = AdamW(param_groups, lr=args.lr)
```

### BUG: Non-deterministic CUDA (Line 31–37)
```python
def set_seed(seed):
    ...
    torch.cuda.manual_seed_all(seed)
```
`torch.backends.cudnn.deterministic` and `torch.backends.cudnn.benchmark` are not set. CUDA convolutions (if any) and some optimized routines will be non-deterministic across runs.

**Fix:**
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### BUG: Training loop ignores `max_steps` (Line 184, 261)
`args.max_steps` (default 5000) is passed but the loop only respects `max_epochs`. The two termination criteria are inconsistent.

**Fix:** Use a step-based loop or add `if global_step >= args.max_steps: break`.

### BUG: `global_step` is not a step counter (Line 234)
```python
global_step = 0
...
    global_step += 1  # Incremented once per EPOCH
```
`global_step` is used as the W&B log step but tracks epochs, not optimization steps. If W&B gradient logging (`log_freq=100`) is interpreted as steps, it will log every 100 epochs. This is misleading.

### BUG: Incomplete batch weighting (Line 66–98)
`train_epoch` computes a simple mean over batches:
```python
total_loss += loss.item()
...
return {'loss': total_loss / n_batches}
```
If the dataset size is not divisible by `batch_size`, the final incomplete batch is weighted equally to a full batch. The reported loss is not the true sample-average loss.

**Fix:** Weight by batch size or set `drop_last=True` in the DataLoader.

### Other Issues
- **Line 152–153:** `DataLoader` has no `num_workers`, `pin_memory`, or `prefetch_factor`. Data loading will bottleneck GPU training.
- **Line 167:** No learning rate warmup. For transformer training, a short linear warmup (e.g., 5% of total steps) is standard and improves stability.
- **Line 217–223:** Checkpoint does not save `scheduler.state_dict()` or RNG states. Resuming training is impossible.
- **Line 178:** `wandb.watch(model, log="all", log_freq=100)` logs all parameter and gradient histograms. For even small models, this is memory-heavy and slows training. Consider `log="gradients"` only.


---

## 5. `scripts/check_data.py` — PASS

This is a visualization and sanity-check script. It is not on the critical path for training or inference, and its issues are cosmetic.

### Issues
- **Line 39:** `if rec.sz_mean[0] < 0.99:` uses an arbitrary threshold. Floating-point initial conditions might legitimately be `0.999999` without being anomalous. A relative tolerance (e.g., `< 1.0 - 1e-4`) would be more robust.
- **Line 64–71:** `plot_representative_trajectories` hardcodes 6 examples. If `n1225` is empty or has fewer than 3 elements, the indexing `n1225[len(n1225)//3]` can be misleading but will not crash (it repeats indices). However, if `n1225` is empty, `n1225[0]` will raise `IndexError`.
- **Line 108:** `rho_ss_vals = [(np.mean(r.sz_mean[-50:]) + 1) / 2 for r in recs]` computes the actual order parameter `rho_ss`, which is **inconsistent** with the `TrajectoryRecord.rho_ss` field (see parse_jld2.py audit). The plotting is correct, but the dataset model is wrong.
- **Line 230:** `print(f"Time range: [{records[0].t_save[0]:.1f}, ...]")` assumes `records` is non-empty.

**Verdict:** Safe to use for exploratory analysis. Fix the empty-list guards before running in an automated pipeline.

---

## 6. `scripts/evaluate.py` — NEEDS_FIX

### BUG: Division by zero in relative L2 error (Line 54)
```python
rel_l2 = np.mean(np.linalg.norm(all_pred - all_true, axis=1) / np.linalg.norm(all_true, axis=1))
```
If any ground-truth trajectory has zero norm (e.g., a flat line at `sz_mean = 0`), this produces `inf` or `NaN`, poisoning the aggregate metric.

**Fix:**
```python
norms = np.linalg.norm(all_true, axis=1)
rel_l2 = np.mean(np.linalg.norm(all_pred - all_true, axis=1) / (norms + 1e-12))
```

### BUG: Model architecture mismatch on load (Line 113–119)
```python
model = RydbergSurrogate(n_layer=args.n_layer, n_head=args.n_head, ...)
checkpoint = torch.load(args.model_path, map_location=device)
model.load_state_dict(checkpoint['model'])
```
The checkpoint stores `args` (line 222 in `train.py`), but `evaluate.py` does not use them. If the user passes the wrong `--n_layer` or `--n_embd`, `load_state_dict` will crash with an opaque size-mismatch error.

**Fix:** Load architecture from checkpoint:
```python
checkpoint = torch.load(args.model_path, map_location=device)
model_args = checkpoint['args']
model = RydbergSurrogate(
    n_layer=model_args['n_layer'],
    n_head=model_args['n_head'],
    n_embd=model_args['n_embd'],
    dropout=model_args['dropout'],
).to(device)
model.load_state_dict(checkpoint['model'])
```

### BUG: Hardcoded phase threshold (Line 60–61)
```python
pred_phase = pred_rho_ss > -0.95
true_phase = true_rho_ss > -0.95
```
The threshold `-0.95` is magic. It should be a named constant shared with `check_data.py` (which uses the same value) or derived from physical reasoning.

### CODE QUALITY: No random seed (Line 105–137)
Evaluation is deterministic in principle, but if any PyTorch operations use non-deterministic algorithms (e.g., `scatter_add_` on CUDA), results may vary slightly across runs. Setting a seed ensures full reproducibility.

### Other Issues
- **Line 118:** `torch.load` without `weights_only=True` is a security risk if loading untrusted checkpoints.
- **Line 82:** `for ax, rec in zip(axes, n1225[::max(1, len(n1225)//6)][:6])` may leave some subplots empty if there are fewer than 6 records. Not a crash, but produces incomplete figures.

---

## 7. `baselines/gp_baseline.py` — NEEDS_FIX

### DESIGN FLAW: Unfair comparison due to time subsampling (Line 37, 126)
```python
def train_gp(train_records, kernel=None, time_subsample=5):
    ...
    for i in range(0, len(rec.t_save), time_subsample):
```
The GP is trained on only **20%** of the time points (every 5th). The transformer baseline sees all 400 points. This is not a like-for-like comparison. The GP is handicapped by design.

**Fix:** Document this explicitly in the paper/README, or train the GP on all data using a sparse approximation (e.g., `sklearn.gaussian_process.kernels.RBF` with `optimizer=None` and inducing points, or switch to `GPyTorch` for scalable variational GP).

### BUG: No input feature normalization (Line 37–55)
The GP inputs are `[omega, n_atoms, t]`. These features have vastly different scales (e.g., `n_atoms` ~ 10³, `t` ~ 10³, `omega` ~ 10⁰). While the RBF kernel has per-dimension length scales, standardizing inputs (`StandardScaler`) improves numerical conditioning of the kernel matrix and optimizer convergence.

**Fix:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

### BUG: `alpha` regularization may be too small (Line 53)
```python
GaussianProcessRegressor(..., alpha=1e-6, normalize_y=True)
```
With `normalize_y=True`, the targets are scaled to unit variance. `alpha=1e-6` is tiny. If the RBF kernel becomes ill-conditioned during optimization, the Cholesky decomposition can fail.

**Fix:** Increase `alpha` to `1e-5` or `1e-4`, or use a WhiteKernel with a learned noise level (already present but bounds may be too tight).

### CODE QUALITY: `y_std` computed but discarded (Line 69)
```python
y_pred, y_std = gp.predict(X_test, return_std=True)
```
`y_std` is never used. If uncertainty quantification is not needed, remove `return_std=True` to save compute.

### Other Issues
- **Line 43–46:** Kernel hyperparameter bounds are magic numbers with no physical justification. `length_scale_bounds=(1e-2, 1e3)` is extremely wide; optimization may wander into pathological regimes.
- **Line 51:** `n_restarts_optimizer=2` is very low. For a 3D hyperparameter space, 5–10 restarts are standard to avoid local optima.
- **Line 97:** `from dataset import create_splits` is imported inside `main()`. This is lazy and makes static analysis harder. Move to the top of the file.


---

## 8. `baselines/fno_baseline.py` — NEEDS_FIX

### ARCHITECTURE: FNO ignores actual time coordinates (Line 41–51)
```python
def forward(self, omega, n_atoms, inv_sqrt_n, t):
    ...
    x = torch.cat([omega_ch, n_ch, inv_sqrt_n_ch], dim=1)
    y = self.fno(x)
    return y.squeeze(1)
```
The argument `t` is passed but **never used**. The FNO learns a mapping on a fixed grid of size `n_time` with `positional_embedding='grid'`. It knows the *relative* position of each sample (index 0, 1, 2, ...) but not the *actual* time value (e.g., `t=0` vs `t=1000`).

**Impact:** If the time grid is non-uniform or if the model needs to generalize to different temporal resolutions, it will fail. For parameter-dependent dynamics where the effective time scale varies with Omega, the FNO cannot adapt because it never sees `t`.

**Fix:** Concatenate `t` as an additional input channel:
```python
t_ch = t.view(batch_size, 1, n_time)
x = torch.cat([omega_ch, n_ch, inv_sqrt_n_ch, t_ch], dim=1)
```
and set `in_channels=4`.

### DESIGN FLAW: Unfair loss function (Line 68)
```python
loss = F.mse_loss(pred, sz_mean)
```
The FNO baseline uses plain MSE. The transformer uses `physics_informed_loss` (MSE + bounds + smoothness). Comparing the two directly is unfair. If the physics-informed penalties help the transformer, the FNO should receive them too, or the transformer should be evaluated with MSE-only for a fair comparison.

**Fix:** Use the same `physics_informed_loss` in the FNO training loop, or train the transformer with MSE-only when benchmarking.

### REPRODUCIBILITY: No random seed (Line 113)
There is no `set_seed` call anywhere. CUDA/cudnn will introduce run-to-run variance.

### Other Issues
- **Line 126–130:** `n_time=400` is hardcoded as the default in `FNOBaseline.__init__`, but the `FNO` model from `neuralop` may raise cryptic errors if the input spatial size does not match what the Fourier modes expect. The code does not validate input length.
- **Line 153:** Checkpoint only saves `model.state_dict()`. It does not save hyperparameters, making it impossible to reload the correct architecture without manually passing CLI args.
- **Line 135:** `AdamW` is used but, like `train.py`, applies weight decay to all parameters including biases and FNO normalization layers.

---

## Cross-Cutting Concerns

### Data Leakage Risk: Mixup across splits?
No. `TrajectoryDataset.mixup` draws `rec2` from `self.records`, and each split has its own `TrajectoryDataset`. So mixup is intra-split. **PASS**.

### Train/Test Leakage in Evaluation?
The evaluation script evaluates on Train, Val, and Test. This is acceptable for post-hoc analysis, but the test metrics must be reported **exactly once** and not used for model selection. The current code does not use test metrics for early stopping. **PASS**, with the caveat that the test set contains interpolation (N=1225 odd omegas) and extrapolation (larger N) tasks that should be reported separately.

### Checkpoint Robustness?
- `train.py` saves model, optimizer, and args but **not scheduler state or RNG state**. Resuming is impossible.
- `fno_baseline.py` saves only `state_dict()`. Architecture metadata is lost.
- Neither uses `torch.save(..., _use_new_zipfile_serialization=True)` explicitly (default in recent PyTorch, so fine).

### Device Placement?
- All `.to(device)` calls are correct.
- No explicit `.cpu()` / `.cuda()` ping-pong in the training loop.
- `evaluate.py` and `gp_baseline.py` move predictions to CPU for numpy aggregation. This is acceptable but can be slow for large-scale evaluation.

### Type Hints & Docstrings?
- Sparse type hints in `parse_jld2.py` and `dataset.py`; almost none in `train.py`, `evaluate.py`, or the baselines.
- Docstrings exist for major functions but are missing in many inner helper functions.
- No `__all__` definitions or module-level docstrings explaining intended usage.

---

## Summary Table

| File | Rating | Primary Issue |
|------|--------|---------------|
| `data/parse_jld2.py` | **CRITICAL** | Regex truncates decimals; `rho_ss` misnamed |
| `data/dataset.py` | **NEEDS_FIX** | Mixup divides by zero; RNG not worker-safe; silent split errors |
| `models/transformer_surrogate.py` | **NEEDS_FIX** | No residual scaling; additive conditioning only; weak init |
| `train.py` | **CRITICAL** | Scheduler step/epoch mismatch; `save_path` NameError; bad WD config |
| `scripts/check_data.py` | **PASS** | Minor plotting guards needed |
| `scripts/evaluate.py` | **NEEDS_FIX** | Division by zero in rel_L2; model reload fragile; no seed |
| `baselines/gp_baseline.py` | **NEEDS_FIX** | Unfair subsampling; no input normalization; low restarts |
| `baselines/fno_baseline.py` | **NEEDS_FIX** | Ignores time `t`; unfair loss; no seed; bad checkpointing |

---

## Priority Action Items

1. **Fix `parse_jld2.py` regex immediately.** Re-parse the entire dataset and invalidate all downstream artifacts. This is a data-corruption bug.
2. **Fix `train.py` scheduler.** Decide whether to step per-epoch or per-batch and make `T_max` consistent.
3. **Fix `train.py` `save_path` scope.** Prevent `NameError` on NaN validation.
4. **Fix `dataset.py` mixup.** Remove `n_atoms` interpolation and seed worker RNGs.
5. **Standardize loss functions across baselines.** Use the same loss (plain MSE) for all models during benchmarking, or apply physics-informed loss to all.
6. **Add input validation and assertions** across all files (shapes, splits, empty lists).
7. **Set proper random seeds** in all training scripts and baseline scripts for reproducibility.
8. **Save full checkpoint metadata** (architecture args, scheduler state, RNG state) in all training scripts.

