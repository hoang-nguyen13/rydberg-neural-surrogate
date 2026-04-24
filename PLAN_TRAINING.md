# Training Plan — 2D Only

## Scope
Train and test exclusively on 2D data. No 1D/3D dimension transfer — model only sees 2D during training, so testing on other dimensions is guaranteed to fail.

## Data

| Split | N | γ | Ω | Trajectories |
|---|---|---|---|---|
| **Train** | 225, 400, 900, 1600, 2500 | 0.1 | 10:0.15:13 | **105** |
| **Val** | 3600 | 0.1 | 10:0.15:13 | **21** |
| **Test (size)** | 4900 | 0.1 | 10:0.15:13 (19 pts) | **19** |
| **Test (γ)** | 3600 | 0.1, 5, 10, 20 | 10:0.15:13 | **84** |

**Excluded:**
- N=100, N=3025 — not in plots
- N=6400, 10000, 21025 — insufficient data
- 1D, 3D — out of scope

## Training Config
- Model: 4 layers, 4 heads, n_embd=96, ~310K params
- Loss: MSE on sz_mean(t)
- Batch: 32
- LR: 1e-3 with cosine decay
- Early stop: patience=50

## Deliverables
1. ρ_ss vs Ω predictions vs data for all N
2. Per-test-set MSE/MAE
3. Ω_c(N) extraction from predicted curves
4. β exponent fit
