# Final Paper Outline: Neural Surrogate for Rydberg Non-Equilibrium Phase Transitions

## Decision Log
- **Section ordering:** Surrogate results first (Section 4), validation after (Section 5). Puts contribution front-and-center.
- **Mean-field baseline:** Skip — not available. Mention in Discussion as standard physics baseline for future work.
- **Data-efficiency ablation:** Skip — no extra compute. Mention as future work.
- **Failure-mode analysis:** Include as supplementary figure (cheap — analyzes existing predictions).
- **Dense-grid validation:** Use fine-grained Ω points already in dataset (10.82, 10.84, etc.) as held-out interpolants.
- **1D/3D:** One sentence in Background only. All tables 2D-only.

---

## 1. Introduction (~2.5 pages)

**The narrative arc:**
1. NEPT in Rydberg lattices is a frontier problem (experimental relevance: Google, Harvard simulators)
2. Classical Monte Carlo fails for low-dephasing quantum regime
3. Exact methods fail for N>50
4. **TWA breakthrough (thesis):** DCTWA handles large N + quantum coherence + open dissipation simultaneously
5. **Thesis validated TWA:** benchmarks (single atom, Dicke, TFIM) + NEPT exponents in 1D/2D/3D
6. **Key finding:** quantum↔classical NEPT maps one-to-one (no d→d+1 shift)
7. **But TWA is expensive:** ~200 trajectories × adaptive SRIW1 × 10⁷ timesteps = hours per phase diagram
8. **This paper:** neural surrogate learns ensemble-averaged TWA dynamics, generalizes across parameters, runs ~10⁴× faster

**Contributions (bullet list):**
- Compact transformer (~310K params) predicting full ⟨S_z(t)⟩ trajectories
- Generalizes to unseen N (4900) and γ (5, 10, 20)
- Recovers effective critical scaling (data collapse, exponents β, δ)
- ~10⁴× speedup with ensemble uncertainty quantification

---

## 2. Background (~1.5 pages)

### 2.1 Rydberg facilitation and NEPT
- Hamiltonian, Lindblad decay/dephasing
- Facilitation regime (V(R_F) = Δ)
- Phase transition: absorbing ↔ active

### 2.2 Truncated Wigner Approximation
- DCTWA maps master equation → SDEs on SU(2) phase space
- Key advance over prior DTWA: treats **both dephasing and decay**
- Scales polynomially with N; validated in thesis against exact solutions
- SRIW1 solver (weak order 2.0) with dt_max = 10⁻⁴
- *Citation to thesis for benchmark validation (single atom, Dicke, TFIM)*

### 2.3 Critical scaling
- ρ_ss ~ |Ω − Ω_c|^β, ρ(t) ~ t^(−δ)
- Data collapse: t^δ ρ vs t|Ω − Ω_c|^(β/δ)
- Dynamic scaling: ρ(t)t^δ vs t/N^(z/d)
- Effective exponents from ensemble-averaged TWA (not true DP)
- Reference values (2D, low γ): β=0.586, δ=0.4577, z=1.86, Ω_c=11.2
- *One sentence on 1D/3D for context:* thesis found β≈0.277 (1D), 0.812 (3D), consistent with DP

---

## 3. Methods (~1.5 pages)

### 3.1 Dataset
- Julia TWA simulations, thesis parameters
- 2D square lattice, open boundaries, Γ=1, Δ=2000
- Train: N∈{225,400,900,1600,2500}, γ=0.1 → **105 trajectories**
- Val: N=3600, γ=0.1 → 21 trajectories
- Test size: N=4900, γ=0.1 → 19 trajectories
- Test γ: N=3600, γ∈{0.1,5,10,20} → 85 trajectories
- Preprocessing: ρ=(sz+1)/2, t≤2000Γ

### 3.2 Surrogate architecture
- Input: (Ω, N, 1/√N, log₁₀γ, d) + time t
- Transformer: 4 layers, 4 heads, n_embd=96, bidirectional attention
- Physics-informed loss: MSE + bounds penalty + smoothness penalty
- Deep ensemble (5 seeds)

### 3.3 Baselines
- Fourier Neural Operator (FNO)
- Gaussian Process (GP)
- Ablations: remove 1/√N, remove split embedding

### 3.4 Evaluation protocol
- Simulation grid: exact Ω points, direct comparison
- Dense grid: step 0.02 (0.005 near Ω_c), labeled "neural predictions"
- Dense-grid validation: against held-out fine-grained TWA points (Ω=10.82, 10.84, etc.)
- Exponent extraction: bootstrap, 95% CIs
- All exponents: *effective exponents* from mean-field TWA

---

## 4. Surrogate Results (~3 pages)

*This is the core contribution. Every figure shows the surrogate.*

### 4.1 Interpolation
- **Fig. 1:** Trajectory overlays — 6 panels spanning sub-critical / near-critical / super-critical
- Train cases (N=225–2500) and val case (N=3600) shown
- Ensemble ±1σ bands

### 4.2 Size extrapolation
- **Fig. 2:** Two panels — (a) dynamics for N=4900 (unseen), (b) phase diagram ρ_ss(Ω)
- TWA ground truth as markers on surrogate dense-grid curve
- Does predicted critical point sharpen correctly?

### 4.3 γ transfer
- **Fig. 3:** Two panels — (a) dynamics for γ=5,10,20, (b) phase diagrams all γ
- TWA markers overlaid
- Critical point shifts left with γ — captured?

### 4.4 Critical scaling from predictions
- **Fig. 4:** Two panels — (a) data collapse t^δ ρ vs t|Ω−Ω_c|^(β/δ), (b) log-log decay at Ω_c
- Extracted from predicted trajectories

---

## 5. Validation & Baselines (~1.5 pages)

### 5.1 TWA ground truth overlay
- Single composite figure (or refer to Figs. 2–3): surrogate curves with TWA markers
- Confirms surrogate reproduces TWA physics, not just interpolates

### 5.2 Metrics and baselines
- **Table 1:** MSE, MAE, Pearson r, ρ_ss MAE, phase accuracy, IC error
  - Rows: train / val / test size / test γ
  - Columns: Transformer / FNO / GP / Ablations
- Speedup row: TWA runtime vs NN inference

### 5.3 Effective exponents
- **Table 2:** β, δ, Ω_c with 95% CIs
  - Columns: Thesis TWA | Transformer (sim grid) | Transformer (dense grid) | FNO | GP
  - Footnote: "True DP exponents shown for reference but not directly comparable to mean-field effective exponents"

### 5.4 Uncertainty quantification
- Ensemble disagreement correlates with absolute error (supplementary)
- Widest uncertainty at largest extrapolation (N=4900, γ=20)

---

## 6. Discussion (~1 page)

### 6.1 What the surrogate learned
- Generalizes across N and γ → learned structure, not memorization
- 1/√N essential (ablation); bidirectional attention captures global time correlations
- Recovers effective exponents → learned scaling physics

### 6.2 Limitations
- Effective exponents only; true DP requires per-trajectory survival probabilities
- TWA itself approximate (misses entanglement)
- Fixed Hamiltonian parameters
- 105 training trajectories — small data regime
- No mean-field baseline comparison (future work)

### 6.3 Outlook
- Interactive phase-diagram exploration for experimentalists
- Bayesian optimization embedding
- Extension to time-dependent driving, disorder, 1D/3D

---

## 7. Conclusion (~0.5 pages)
- TWA is powerful but expensive
- Surrogate makes TWA dynamics interactive
- Reproduces physics, generalizes, 10⁴× faster

---

## Main Text: 4 Figures + 2 Tables

| # | Item | Content |
|---|------|---------|
| **Fig. 1** | Interpolation | Trajectory overlays (6 panels) |
| **Fig. 2** | Size extrapolation | (a) N=4900 dynamics, (b) N=4900 phase diagram |
| **Fig. 3** | γ transfer | (a) Dynamics γ=5,10,20, (b) Phase diagrams all γ |
| **Fig. 4** | Scaling | (a) Data collapse, (b) Log-log decay |
| **Table 1** | Metrics | MSE, MAE, r, ρ_ss MAE, phase acc. across splits and models |
| **Table 2** | Exponents | β, δ, Ω_c: thesis vs transformer vs FNO vs GP |

## Supplementary Material
- TWA benchmark comparisons (single atom, Dicke, TFIM)
- z-tuning collapse plots
- Finite-size scaling overlay
- Residual analysis, bounds/IC histograms
- Error heatmap
- FNO/GP trajectory overlays
- Failure-mode analysis (largest errors, boundary behavior)
- Ensemble calibration
- Out-of-distribution detection
