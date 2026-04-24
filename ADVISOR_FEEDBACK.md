# Advisor Feedback on Evaluation Plan

**To:** PhD Student  
**From:** Advisor  
**Date:** 24 April 2026  
**Re:** Proposed 16-Figure Evaluation Plan for the Rydberg Neural Surrogate

---

Thank you for this well-structured memo. The tiered organization is logical, and your upfront acknowledgment of limitations (mean-field data, small training set, short $t_{\text{max}}$) signals intellectual maturity. That said, I need to push back on scope, scientific framing, and risk management before you commit GPU cycles. Below are my answers to your five questions, followed by strategic advice on paper architecture.

---

## 1. Priorities: What to Cut, What to Keep, and Where It Goes

**Golden rule for ML-for-Science papers:** The main text must tell a single, coherent story. Everything else is supplementary material. Your current plan would produce a data-dump manuscript that buries the physics under diagnostic noise.

**Main text (5–7 figures max):**

- **Keep Figure 1** (trajectory overlays). This is your handshake with the reader—it must exist.
- **Keep Figure 4** (size extrapolation dynamics) and **Figure 5** (size extrapolation phase diagram). Zero-shot generalization to unseen $N$ is your strongest claim. Lead with it.
- **Keep Figure 6** ($\gamma$ transfer dynamics) and **Figure 7** ($\gamma$ transfer phase diagram). If this works, it distinguishes your transformer from a pure interpolator. It is also your riskiest claim, so if the results are weak, drop them and reframe the paper as a size-extrapolation study.
- **Keep one physics-collapse figure from Tier 3.** My recommendation is **Figure 11** (data collapse) as the centerpiece. It is visually compelling and tests whether the network has learned the *structure* of the transition, not just curve shapes. Put **Figure 13** (critical exponent table) in the main text as a compact summary table, not a figure.
- **Add a baseline comparison figure/table** (see §4 below). Reviewers will demand it.

**Move to supplementary material:**

- **Figures 2, 3, 14, 15, 16.** These are diagnostics. They belong in an appendix titled "Model Diagnostics and Sanity Checks." A single supplementary figure with four sub-panels (residual vs. time, steady-state scatter, bounds histogram, IC-error histogram) is sufficient.
- **Figure 8** (critical point shift across all $N$) is largely redundant with Figure 5 if you annotate the critical points on the phase diagram. If you keep it, merge it with Figure 5 as an inset or a second panel.
- **Figure 9** (finite-size scaling overlay) is dangerous because you are overlaying a scaling form on predictions that are themselves interpolations. I address this in §3.
- **Figure 10** (log-log dynamics at $\Omega_c$) and **Figure 12** ($z$-tuning collapse) are highly correlated with Figure 11. If the data collapse in Figure 11 works, Figures 10 and 12 are redundant. If it fails, they become forensic tools to diagnose *why*—which is valuable, but belongs in supplementary material or a failure-analysis subsection.

**Bottom line:** Your main text should present (i) the model and its task, (ii) interpolation and extrapolation results on held-out data, (iii) one compelling physics test (data collapse), and (iv) baselines + speedup. That is a paper, not a scrapbook.

---

## 2. Dense-Grid Evaluation: Endorsed, But With a Strict Protocol

**Yes, evaluate on a dense $\Omega$ grid.** A surrogate that merely reproduces the simulation grid is a lookup table. The scientific value of your work is precisely that the network learns a smooth underlying physical manifold. However, you must protect yourself—and the reader—from hallucinated structure near the critical point.

**Protocol:**

1. **Hold-out validation on the simulation grid first.** On $N=4900$ and $N=3600$ ($\gamma=0.1$), report all metrics on the exact simulation grid. This is your ground-truth anchor. No caveats needed.
2. **Dense-grid predictions as "model interpolants."** On the dense grid, label these explicitly as *neural network predictions* in all captions and text. Never present them as equivalent to simulation data.
3. **Cross-check density near criticality.** The phase transition region is where interpolation artifacts are most dangerous. I recommend a two-tier dense grid: step $0.02$ away from $\Omega_c$, but step $0.005$ in a narrow window around $\Omega_c \approx 11.2$. If the model predicts non-physical oscillations or a discontinuous derivative in this window, flag it as an interpolation failure.
4. **Do not extract exponents from dense-grid predictions alone.** Use the dense grid for visualization (phase diagrams, S-curves), but extract $\beta$ and $\delta$ from both the simulation grid and the dense grid. If the two estimates disagree outside bootstrap error bars, the dense-grid exponents are untrustworthy and must be discarded.

**Reviewer psychology:** A dense grid that produces a smooth phase diagram looks sophisticated. A dense grid that produces spurious critical scaling looks like curve-fitting overfitting. The difference is transparency. State your protocol explicitly.

---

## 3. Framing Critical Exponent Extraction: Be Honest or Be Embarrassed

This is where I need to be blunt. **You cannot claim to measure directed-percolation critical exponents from ensemble-averaged TWA data.** I know you know this—you mention it in §5—but I need to see it reflected in the *evaluation design*, not just the limitations paragraph.

**What will not survive peer review:**
- Claiming $\beta \approx 0.586$ or $\delta \approx 0.4577$ from your model and treating agreement with the thesis as validation of the network.
- Presenting bootstrap confidence intervals on these exponents without explicitly stating that the underlying data are mean-field ensemble averages.
- Using the phrase "critical exponent" without the modifier "effective" or "apparent."

**What you should do instead:**

1. **Label everything as *effective exponents*.** Your paper should state: "Because the training data consist of ensemble-averaged TWA trajectories rather than per-trajectory survival probabilities, the extracted exponents are effective exponents characterizing the mean-field dynamics. They are not expected to coincide with the exact DP exponents, and deviations do not imply a new universality class."
2. **Compare against the thesis, not against DP literature.** Your ground truth is the thesis simulation, not the DP universality class. If your model reproduces the thesis effective exponents, it has successfully learned the TWA physics. If it deviates, the deviation may reflect TWA breakdown, not model failure. This is a subtle but crucial distinction.
3. **Use the dense grid for visualization, the simulation grid for exponent extraction.** Fit $\rho_{\text{ss}} \sim |\Omega - \Omega_c|^\beta$ and $\rho(t) \sim t^{-\delta}$ using the simulation-grid predictions as the primary result. Use the dense-grid fit as a sensitivity check. Report both, but weight the simulation-grid result higher.
4. **Bootstrap correctly.** Resample trajectories (not time points) with replacement. Report 95% CIs. If your CI on $\beta$ is $\pm 0.15$, say so. Do not pretend precision you do not have.
5. **Consider a null-model check.** Fit the same exponent-extraction procedure to the FNO and GP baselines. If all three models give similar effective exponents, the exponents are properties of the data, not your architecture. If your transformer gives significantly different (and more accurate) exponents, that is a genuine methodological contribution.

**Figure 13 revision:** Do not present a table with one column "DP theory" and one column "Our model." Present three columns: Thesis TWA (your "ground truth"), Our Transformer (simulation grid), Our Transformer (dense grid). Add a footnote: "True DP exponents ($\beta_{\text{DP}} \approx 0.586$, $\delta_{\text{DP}} \approx 0.451$) are shown for reference but are not directly comparable to mean-field effective exponents."

---

## 4. Speedup and Baselines: Non-Negotiable

**Yes to both. Absolutely.**

**Speedup benchmark:** A neural surrogate paper without a speedup claim is like a new compiler paper without benchmark timings. It is not optional. Your memo mentions "$\sim 1$ ms vs. $\sim$ minutes per trajectory" in passing. This must be a formal table in the main text.

- Measure NN inference time on GPU with `torch.no_grad()`, averaged over 1000 forward passes. Report mean and standard deviation.
- Measure TWA runtime on the same machine if possible, or quote the thesis timings with hardware specs. If the Julia code is not available for benchmarking, state the thesis-reported runtime explicitly and label it as a literature value.
- Report the speedup factor ($\times 10^3$, $\times 10^4$, etc.). This is your headline number for the abstract.

**Baseline comparison:** Reviewers will ask "Why a transformer?" if you do not compare against simpler alternatives. Your FNO and GP baselines are already implemented. Use them.

- Include a **metric table** (main text) with MSE, MAE, relative L2, Pearson $r$, $\rho_{\text{ss}}$ MAE, and phase classification accuracy on all test sets: $N=4900$ (size extrapolation), $N=3600$ ($\gamma$ transfer), and held-out $\Omega$ at $N=1225$ (interpolation, if you have it—if not, use your validation split).
- If space permits, include a **single supplementary figure** showing trajectory overlays for all three models on the same $N=4900$ test case. Visual comparison is often more persuasive than a table for qualitative errors (e.g., GP smoothing out the phase transition, FNO violating initial conditions).
- Be honest about baseline failures. If GP extrapolates poorly to $N=4900$, show it. That is evidence that your transformer is not just interpolation.

---

## 5. Missing Evaluation: What You Have Overlooked

Your plan is thorough on reconstruction and physics tests, but it is weak on **model introspection** and **robustness**. Here are three additions I consider essential:

### A. Uncertainty Quantification

You are training a deep ensemble (5 models). Use it. Report the ensemble standard deviation as an uncertainty estimate.

- Plot predicted trajectories with a $\pm 1\sigma$ band from the ensemble on Figure 4 (size extrapolation). If the band widens dramatically at $N=4900$, the model knows it is uncertain.
- Compute the correlation between ensemble disagreement and absolute error. If high-disagreement regions coincide with high-error regions, your uncertainty is calibrated. This is a strong signal for reviewers.
- A single figure or table on ensemble calibration belongs in supplementary material.

### B. Ablation Studies

With only 105 training trajectories, you must justify every architectural choice. I want to see at least two ablations:

1. **Remove $1/\sqrt{N}$ as an input feature.** Train a model with only $(\Omega, N)$. If size extrapolation collapses, $1/\sqrt{N}$ was essential. If it does not, you were lucky and should say so.
2. **Remove the time embedding / parameter conditioning split.** Train a model where time and parameters are concatenated into a single vector before the transformer. If performance drops, your embedding design matters.

Each ablation can be a single line in Table 1 (the baseline comparison table). You do not need separate figures.

### C. Out-of-Distribution Detection

A surrogate that extrapolates confidently to unphysical parameters is dangerous. Test the model on $(\Omega, N)$ far outside the training distribution—e.g., $\Omega = 50$, $N = 10000$—and verify that the ensemble disagreement explodes or the predictions violate physical bounds. This is a one-paragraph result, but it demonstrates scientific maturity.

---

## Strategic Advice: Paper Architecture

Here is how I would structure the final paper, with figure placement:

| Section | Content | Figures/Tables |
|---|---|---|
| Introduction | Physics motivation, surrogate concept, speedup claim | Table 1: Speedup |
| Model | Transformer architecture, physics-informed loss, training protocol | Schematic (not in your 16) |
| Results: Interpolation | $N=1225$ (or validation split) metrics and overlays | Fig. 1: Trajectory overlays |
| Results: Extrapolation | $N=4900$ dynamics and phase diagram | Fig. 2: Size extrapolation dynamics; Fig. 3: Size extrapolation phase diagram |
| Results: Transfer | $\gamma$ dynamics and phase diagram (if strong) | Fig. 4: $\gamma$ transfer (optional—move to supp. if weak) |
| Results: Physics | Data collapse, effective exponents | Fig. 5: Data collapse; Table 2: Effective exponents |
| Results: Baselines | Metric comparison | Table 3: Baseline comparison + ablations |
| Discussion | Limitations, effective vs. true exponents, future work | — |

**Supplementary material:** Residual analysis, steady-state scatter, bounds/IC-error histograms, ensemble uncertainty, FNO/GP trajectory overlays, log-log decay at $\Omega_c$, $z$-tuning collapse, finite-size scaling overlay.

---

## Final Remarks

Your memo shows good instincts, but you are still thinking like a machine-learning engineer producing a benchmark suite. You need to think like a physicist telling a story. The story is: *We built a small transformer that learns the mean-field dynamics of a driven-dissipative phase transition, generalizes to unseen system sizes and dephasing rates, and runs $10^4\times$ faster than TWA. It reproduces the effective critical scaling of the underlying simulations, with honest uncertainty estimates and explicit caveats about mean-field limitations.*

Everything else is decoration.

Cut the 16 figures down to 5 in the main text. Keep the diagnostics in supplementary. Be aggressive about dense-grid evaluation, but be transparent about its risks. Frame the exponents as effective exponents from the outset. Include speedup and baselines—they are not afterthoughts, they are the paper's foundation.

Schedule the GPU runs. I expect to see the streamlined evaluation scripts and a draft of Table 1 (speedup) and Table 3 (baselines) before you generate the full figure suite.

—Advisor
