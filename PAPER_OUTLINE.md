# Paper Outline: From TWA Simulations to Neural Surrogates for Rydberg Non-Equilibrium Phase Transitions

## The Full Narrative Arc

The thesis established that **TWA for spins** is a major methodological advance for open quantum systems. This paper tells the story from that physics through to the surrogate.

---

## 1. Introduction (~2.5 pages)

### 1.1 The physics problem: quantum non-equilibrium phase transitions
- NEPT in driven-dissipative Rydberg lattices
- Directed percolation (DP) universality class: simplest non-equilibrium class, yet unsolved analytically in 1+1D
- The quantum challenge: coherence, entanglement, dephasing — classical MC fails in low-dephasing regime
- Experimental relevance: Rydberg quantum simulators (Google, Harvard, etc.)

### 1.2 Why existing methods fall short
- **Exact diagonalization / density matrix evolution:** Hilbert space ~2^N, impossible for N>50
- **Monte Carlo:** Works in high-dephasing (classical) limit. Fails for low dephasing because it cannot capture quantum coherence, detuning effects, or the full Lindblad dynamics. The thesis explicitly showed this inefficiency.
- **Mean-field:** Misses fluctuations entirely; predicts wrong critical exponents
- **Gap:** No method handles large N *and* quantum coherence *and* open-system dissipation simultaneously

### 1.3 The TWA breakthrough (thesis contribution)
- Hybrid Discrete-Continuous Truncated Wigner Approximation (DCTWA)
- Maps quantum master equation → stochastic differential equations on SU(2) phase space
- Key advances over prior DTWA:
  - Can treat **both dephasing and decay** naturally (prior DTWA could not)
  - Scales polynomially with system size (not exponentially)
  - Captures quantum fluctuations through stochastic noise terms
  - Validated against exact solutions for single atom, Dicke superradiance, equilibrium TFIM
- The thesis used SRIW1 adaptive solver (weak order 2.0) to simulate systems up to N=21,025

### 1.4 Thesis results: TWA validates quantum DP
- 2D square lattice: Ω_c ≈ 11.2, effective exponents β≈0.586, δ≈0.4577
- 1D chain: Ω_c ≈ 29.85, β≈0.277, δ≈0.159
- 3D cube: Ω_c ≈ 7.56, β≈0.812, δ≈0.733
- **Key finding:** Quantum and classical NEPT map one-to-one (no d→d+1 shift unlike equilibrium TFIM)
- Critical point sharper in quantum (low-γ) regime, but exponents unchanged

### 1.5 The remaining bottleneck: TWA is expensive
- Per trajectory: adaptive SRIW1 with dt_max=10⁻⁴, up to 10⁷ time steps
- Ensemble average: ~200 trajectories per parameter point
- A single phase diagram (N=3600, 21 Ω values, 200 trajectories each) ≈ hours of compute
- Dense parameter sweeps, Bayesian optimization, or real-time experimental feedback are intractable

### 1.6 This paper: a neural surrogate for TWA ensemble dynamics
- **Idea:** If TWA produces reliable physics but is expensive, can a neural network learn the *ensemble-averaged* dynamics and generalize across parameters?
- **Contribution:** A compact transformer (~310K params) trained on only 105 trajectories that:
  - Reproduces TWA interpolation (validation N=3600)
  - Generalizes to unseen system sizes (N=4900)
  - Generalizes to unseen dephasing rates (γ=5,10,20)
  - Recovers effective critical scaling (data collapse, exponents β, δ)
  - Runs ~10⁴× faster than direct TWA
- **Distinction from prior neural surrogates:** We target full time trajectories ⟨S_z(t)⟩, test generalization across physical parameters (N, γ), and validate through critical scaling — not just MSE

---

## 2. Background (~2 pages)

### 2.1 Rydberg facilitation and the driven-dissipative lattice
- Hamiltonian in rotating frame
- Lindblad terms: decay (rate Γ), dephasing (rate γ)
- Facilitation regime: van der Waals interaction brings neighbors into resonance
- Phase transition: absorbing (all spins down, ρ=0) ↔ active (sustained excitation, ρ>0)

### 2.2 Truncated Wigner Approximation for spins
- From thesis Chapter 3: Wigner function on SU(2) sphere, flattened Wigner function (FWF)
- Operator-differential mappings: σ_z ρ → drift + diffusion terms
- SDEs in (θ, φ): Itô form with Wiener increments
- Many spins: truncation of mixed derivatives recovers DTWA; dephasing/decay added exactly
- Initial state sampling: discrete delta-peaks on the sphere
- **Why this matters:** TWA is the *only* method that handles large N + quantum coherence + open dissipation

### 2.3 Critical scaling and finite-size effects
- Effective exponents from ensemble-averaged data (mean-field TWA, not true DP)
- Scaling forms: ρ_ss ~ |Ω−Ω_c|^β, ρ(t) ~ t^(−δ)
- Data collapse: t^δ ρ vs t|Ω−Ω_c|^(β/δ)
- Dynamic scaling: ρ(t)t^δ vs t/N^(z/d)
- Thesis reference values (2D, low γ): β=0.586, δ=0.4577, z=1.86, Ω_c=11.2

---

## 3. Methods (~2 pages)

### 3.1 TWA dataset
- Source: Julia simulations with SRIW1, thesis parameters
- 2D square lattice, open boundaries, Γ=1, Δ=2000, V=vdW
- Ω ∈ [10,13] (fine-grained near Ω_c), N ∈ {225,400,900,1600,2500,3600,4900,...}
- Train: N≤2500, γ=0.1 → 105 trajectories
- Val: N=3600, γ=0.1 → 21 trajectories
- Test size: N=4900, γ=0.1 → 19 trajectories
- Test γ: N=3600, γ∈{0.1,5,10,20} → 85 trajectories

### 3.2 Surrogate architecture
- Input: 5 scalars (Ω, N, 1/√N, log₁₀γ, d) + time array t
- Transformer: 4 layers, 4 heads, n_embd=96, bidirectional attention
- Physics-informed loss: MSE + soft bounds penalty + smoothness penalty
- Deep ensemble (5 seeds) for uncertainty quantification

### 3.3 Baselines
- Fourier Neural Operator (FNO)
- Gaussian Process (GP)
- Ablations: remove 1/√N, remove split embedding

### 3.4 Evaluation protocol
- Simulation-grid evaluation: direct comparison to held-out data
- Dense-grid evaluation (step 0.02, 0.005 near Ω_c): labeled as "neural predictions"
- Exponent extraction: bootstrap resampling, 95% CIs
- All exponents labeled as *effective exponents* from mean-field TWA

---

## 4. Results: TWA Physics (~2 pages)

*This section validates the thesis results that the surrogate must reproduce.*

### 4.1 TWA reproduces known benchmarks
- Single driven atom: exact agreement with master equation
- Dicke superradiance: correct collective decay
- Equilibrium TFIM: correct phase transition (though critical point slightly shifted — known TWA limitation)
- **Fig. 1:** Benchmark comparisons (3 panels)

### 4.2 NEPT in 2D: the target physics
- Phase diagram ρ_ss(Ω) for N=3600, γ=0.1
- Critical point Ω_c ≈ 11.2 with finite-size drift
- Finite-size scaling: transition sharpens with N
- **Fig. 2:** Phase transitions for all N values (thesis-exact reproduction)

### 4.3 Critical exponents from TWA
- Log-log dynamics at Ω_c showing algebraic decay
- Data collapse t^δ ρ vs t|Ω−Ω_c|^(β/δ)
- z-tuning collapse: ρ(t)t^δ vs t/N^(z/d)
- **Table 1:** Effective exponents — 1D, 2D, 3D (thesis values)

### 4.4 Quantum vs classical comparison
- Phase diagrams for γ=0.1, 5, 10, 20 at N=3600
- Critical point shifts left with increasing γ
- Transition broadens in classical (high-γ) regime
- Exponents remain similar (one-to-one mapping)
- **Fig. 3:** γ-comparison phase diagrams

---

## 5. Results: Surrogate (~3 pages)

### 5.1 Interpolation quality
- **Fig. 4:** Trajectory overlays (train + val): sub-critical, near-critical, super-critical
- **Table 2:** Metrics: MSE, MAE, Pearson r, ρ_ss MAE, phase accuracy
- Ensemble uncertainty bands shown

### 5.2 Size extrapolation
- **Fig. 5a:** Predicted vs true dynamics for N=4900 (unseen)
- **Fig. 5b:** Predicted phase diagram ρ_ss(Ω) for N=4900; true points as markers, dense-grid curve
- Does predicted Ω_c drift correctly with N?
- **Table 2 (cont.):** Test metrics for N=4900

### 5.3 γ transfer
- **Fig. 6a:** Predicted dynamics for γ=5, 10, 20 (unseen dephasing)
- **Fig. 6b:** Predicted phase diagrams for all γ values
- Critical point shifts left as γ increases — does surrogate capture this?

### 5.4 Critical scaling from predictions
- **Fig. 7a:** Data collapse from predicted trajectories
- **Fig. 7b:** Log-log decay at Ω_c from predictions
- **Table 3:** Effective exponents extracted from predicted data vs thesis TWA
  - Columns: Thesis TWA | Transformer (sim grid) | Transformer (dense grid) | FNO | GP
  - Rows: β, δ, Ω_c with 95% CIs
- Discussion: agreement within error bars → surrogate learned scaling *structure*

### 5.5 Baselines and ablations
- **Table 4:** Full metric comparison across models and test sets
- Ablations: 1/√N essential for size extrapolation; split embedding improves accuracy
- **Fig. 8 (supp):** Trajectory overlays for all three models on N=4900

### 5.6 Speedup and uncertainty
- **Table 5:** Runtime — TWA (thesis-reported, ~minutes/trajectory) vs NN (~ms/trajectory)
- Speedup factor: ~10⁴×
- Ensemble disagreement correlates with absolute error (calibrated uncertainty)

---

## 6. Discussion (~1 page)

### 6.1 What the surrogate learned
- Not memorization: generalizes across N and γ
- 1/√N captures finite-size scaling; bidirectional attention captures global time structure
- Recovers effective exponents → learned the *physics* of the phase transition

### 6.2 Limitations
- Effective exponents only (mean-field TWA data, not true DP)
- TWA itself approximate (misses entanglement, fails for TFIM critical point)
- Fixed Hamiltonian parameters
- Small training set (105 trajectories) — may not transfer to other regimes
- Short t_max may not reach true steady state for largest N near criticality

### 6.3 Outlook
- Surrogate enables interactive phase-diagram exploration for experimentalists
- Embed in Bayesian optimization for experimental parameter tuning
- Extend to time-dependent driving, disorder, 3D lattices
- Higher-order TWA truncations → more accurate training data → better surrogate

---

## 7. Conclusion (~0.5 pages)
- TWA for spins is a powerful method for quantum NEPT (thesis contribution)
- But TWA is expensive — neural surrogate makes it *usable*
- Surrogate reproduces TWA physics, generalizes, and runs 10⁴× faster
- Honest framing: we reproduce the *simulation* physics, not discover new exponents

---

## Figure and Table Summary

### Main Text Figures (7)
| # | Title | Content |
|---|-------|---------|
| Fig. 1 | TWA benchmarks | Single atom, Dicke, TFIM — exact vs TWA |
| Fig. 2 | 2D phase transitions | ρ_ss(Ω) for all N values |
| Fig. 3 | Quantum vs classical | Phase diagrams for γ=0.1,5,10,20 |
| Fig. 4 | Surrogate interpolation | Trajectory overlays (train/val) |
| Fig. 5 | Size extrapolation | (a) N=4900 dynamics, (b) N=4900 phase diagram |
| Fig. 6 | γ transfer | (a) Dynamics γ=5,10,20, (b) Phase diagrams all γ |
| Fig. 7 | Scaling from predictions | (a) Data collapse, (b) Log-log decay |

### Main Text Tables (5)
| # | Title | Content |
|---|-------|---------|
| Table 1 | TWA exponents | β, δ, z, Ω_c for 1D/2D/3D (thesis) |
| Table 2 | Surrogate metrics | MSE, MAE, r, ρ_ss MAE, phase acc. |
| Table 3 | Predicted exponents | β, δ, Ω_c: thesis vs transformer vs FNO vs GP |
| Table 4 | Baseline comparison | Full metrics + ablations |
| Table 5 | Speedup | TWA vs NN runtime |

### Supplementary
- Residuals, bounds/IC histograms, error heatmaps
- z-tuning collapse, FSS overlay
- FNO/GP trajectory overlays
- Ensemble calibration, OOD detection
