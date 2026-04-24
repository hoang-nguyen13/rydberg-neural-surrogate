# MEMO

**To:** Advisor  
**From:** PhD Student  
**Date:** 24 April 2026  
**Re:** Proposed Evaluation Plan for the Rydberg Neural Surrogate

---

## 1. Objective

This memo presents a structured, 16-figure evaluation plan for our transformer-based neural surrogate of 2D Rydberg lattice dynamics. The model (~310K parameters, 4-layer bidirectional transformer) predicts the full time trajectory $\langle S_z(t) \rangle$ given physical parameters $(\Omega, N, 1/\sqrt{N}, \log_{10}\gamma, d)$. After resolving earlier data-parsing and architecture issues, the pipeline is now stable, and I would like your feedback on the scope and prioritization of the proposed validation tests before we commit to final GPU runs and paper writing.

## 2. Model & Data Summary

**Architecture:** 4 layers, 4 heads, $n_{\text{embd}}=96$, full bidirectional attention, direct regression (no causal mask). Physics-informed loss: MSE + bounds penalty + smoothness penalty.

**Dataset (post-cleaning, 2D, $\gamma=0.1$):**

| Split | System Sizes | $\Omega$ range | Trajectories |
|---|---|---|---|
| Train | $N \in \{225, 400, 900, 1600, 2500\}$ | $[10, 13]$, step $0.15$ | **105** |
| Validation | $N = 3600$ | $[10, 13]$, step $0.15$ | 21 |
| Test (size) | $N = 4900$ | $[10, 13]$, step $0.15$ | 19 |
| Test ($\gamma$) | $N = 3600$ | $\gamma \in \{0.1, 5, 10, 20\}$ | 85 |

**Reference physics (thesis):** For 2D low-dephasing, $\Omega_c \approx 11.2$, $\beta \approx 0.586$, $\delta \approx 0.4577$, $z \approx 1.86$.

## 3. Proposed Evaluation Plan: 16 Figures in Four Tiers

I have organized the evaluation into four tiers, progressing from basic sanity checks to deep physics validation.

### Tier 1 — Reconstruction (Figures 1–3)
These verify that the model learns the training manifold without obvious failure modes.

1. **Trajectory overlays.** True vs. predicted $\langle S_z(t) \rangle$ for six representative training cases spanning sub-critical, near-critical, and super-critical $\Omega$.
2. **Residual vs. time.** Mean absolute error $\text{MAE}(t)$ across the training set to detect whether error accumulates at late times or spikes near the phase transition.
3. **Steady-state scatter.** Predicted vs. true $\rho_{\text{ss}}$ (steady-state excitation density), with the $y=x$ line and a tight $R^2$ annotation.

### Tier 2 — Generalization (Figures 4–8)
These test interpolation/extrapolation along the two axes most relevant to the physics: system size $N$ and dephasing rate $\gamma$.

4. **Size extrapolation dynamics.** Full trajectory predictions for $N=4900$ (strictly unseen size).
5. **Size extrapolation phase diagram.** $\rho_{\text{ss}}$ vs. $\Omega$ for $N=4900$; compare against true simulation points.
6. **$\gamma$ transfer dynamics.** Trajectories for $N=3600$ at $\gamma = 5, 10, 20$; only $\gamma=0.1$ was seen during training.
7. **$\gamma$ transfer phase diagram.** $\rho_{\text{ss}}$ vs. $\Omega$ for all tested $\gamma$ values at $N=3600$.
8. **Critical point shift.** Predicted $\rho_{\text{ss}}$ vs. $\Omega$ across *all* available $N$ values to visualize how the transition sharpens with increasing system size.

### Tier 3 — Physics Correctness (Figures 9–13)
These ask whether the network has learned the *structure* of the phase transition, not merely memorized curves.

9. **Finite-size scaling (thesis-exact).** Plot predicted $\rho_{\text{ss}}$ vs. $\Omega$ for all $N$ and overlay the known scaling form from the thesis to see if predicted critical points drift correctly.
10. **Log-log dynamics at $\Omega_c$.** At the thesis-critical drive $\Omega_c = 11.2$, does the predicted $\rho(t)$ exhibit the expected algebraic decay $t^{-\delta}$?
11. **Data collapse.** Plot $t^{\delta} \rho$ vs. $t |\Omega - \Omega_c|^{\beta/\delta}$ using predictions; points from different $\Omega$ should collapse onto a single universal curve.
12. **$z$-tuning collapse.** Plot $\rho(t) \, t^{\delta}$ vs. $t / N^{z/d}$ across all predicted $N$; verify dynamical scaling.
13. **Critical exponent table.** Extract $\beta$ and $\delta$ from predicted data (curve fitting near $\Omega_c$ and algebraic decay, respectively) and tabulate against thesis values. Include error bars from bootstrap resampling where feasible.

### Tier 4 — Physical Sensibility (Figures 14–16)
These are diagnostic checks to expose blind spots and pathologies.

14. **Bounds check.** Histogram of predicted $\langle S_z \rangle$ values outside $[-1, 1]$; count and correlate with parameter regions.
15. **Initial-condition error.** Distribution of $|\text{pred}(t=0) - \text{true}(t=0)|$; all trajectories start at $\langle S_z(0) \rangle = +1$, so large deviations indicate architectural or optimization artifacts.
16. **Error heatmap.** $\text{MAE}(\Omega, N)$ on a grid to reveal parameter "blind spots" (e.g., near the critical point or at the smallest training sizes).

## 4. Key Design Question

> **For phase-transition and scaling plots (Figures 5, 7, 8–12), should the model be evaluated on (a) the exact simulation $\Omega$ grid ($10 : 0.15 : 13$), or (b) a much denser $\Omega$ grid (step $\sim 0.02$) to test continuous interpolation between discrete training/validation points?**

Option (a) is safer—predictions are directly comparable to held-out data. Option (b) is more scientifically compelling because it tests whether the network has learned a smooth physical manifold, allowing us to resolve the critical region with higher precision than the original simulation grid. The risk of (b) is that interpolation artifacts near $\Omega_c$ could produce spurious scaling collapses or incorrect exponent extraction. I lean toward generating both and presenting the dense-grid curves with explicit caveats, but I would appreciate your guidance on how aggressively we should lean on interpolation for the physics figures.

## 5. Risks & Limitations

I want to flag several limitations upfront so we frame the results honestly:

- **Small training set:** Only 105 training trajectories for a 310K-parameter model. While regularization (dropout 0.2, weight decay, augmentation) and the low-dimensional physical manifold help, overfitting remains a real risk.
- **Mean-field data:** The underlying data are ensemble-averaged TWA trajectories. True directed-percolation exponents require per-trajectory survival probabilities, so any extracted $\beta$, $\delta$, or $z$ from mean curves should be labeled as *effective* exponents.
- **Short time horizon:** $t_{\text{max}} = 1000$ may be insufficient for the largest systems ($N = 4900$) to reach true steady state near criticality.
- **Fixed Hamiltonian parameters:** The model sees only $V = \Delta = 2000$, $\Gamma = 1$. Generalization to other Hamiltonians is out of scope.
- **Data collapse is fragile:** With only $\sim 20$ discrete $\Omega$ values, log-log fits and scaling collapses will be under-constrained. I will quote bootstrap confidence intervals, but the exponents should be interpreted cautiously.

## 6. Request for Feedback

1. **Priorities:** If GPU time or page space is limited, which tier should I cut first? My instinct is to keep Tier 1–2 for the paper and move Tier 3 to supplementary material if needed, but I would defer to your view on what will most impress reviewers.

2. **Figure scope:** Are any of Figures 9–13 redundant? For example, if the data collapse in Figure 11 fails, does Figure 12 add independent information, or should we merge them?

3. **Dense-grid evaluation:** Do you endorse evaluating on a dense $\Omega$ grid for the physics plots, or should we stick strictly to the simulation grid?

4. **Missing tests:** Should I include an explicit speedup benchmark (NN inference time vs. Julia TWA runtime)? A table showing $\sim 1\,\text{ms}$ vs. $\sim$ minutes per trajectory would strengthen the surrogate claim.

5. **Baseline comparison:** The FNO and GP baselines are already implemented. Should the final memo include a side-by-side metric table (MSE, MAE, phase accuracy, $\rho_{\text{ss}}$ error) for all three models on the test sets?

Please let me know your thoughts, and I will finalize the evaluation scripts and schedule the GPU runs accordingly.

---

**Prepared by:** PhD Student  
**Project:** Neural Surrogate for Rydberg Facilitation Dynamics  
**Repository:** `nanoGPT/` (working directory)
