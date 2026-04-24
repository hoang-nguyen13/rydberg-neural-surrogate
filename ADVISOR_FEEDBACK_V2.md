# Professor Feedback: Revised Paper Outline (V2)

**Overall assessment:** You have found the narrative. The arc from physics problem → TWA breakthrough → TWA bottleneck → surrogate solution is correct. The framing in the Introduction is now genuinely exciting: a reader understands *why* TWA matters and *why* it needs a surrogate. This is no longer a benchmark suite; it is a physics paper with ML as the tool.

That said, structural problems remain. You are still carrying too much thesis baggage into what must be a lean, standalone paper. Below are five specific pushbacks.

---

## 1. Section 4 (TWA Physics Results) Is Too Heavy for the Main Text

**The problem:** Two full pages and three figures (Figs. 1–3) validating TWA itself will confuse readers and reviewers. This is a paper about a *neural surrogate*. If half the main text reproduces your thesis, a referee will ask: "Why is this not just an appendix to the thesis?"

**What to do:**
- **Compress Section 4 to ~0.75 pages and merge it into Background/Methods.** The TWA benchmarks (Fig. 1: single atom, Dicke, TFIM) are standard validation from the thesis. They belong in a short paragraph in Section 2.2 with a single composite inset figure, or better, cite the thesis and move the figure to supplementary. A reader of *this* paper does not need to re-derive trust in TWA; they need to know that TWA is the ground truth you are surrogating.
- **Fig. 2 (phase transitions for all N) is the only TWA figure that earns main-text space**, but only as a *single panel* showing the target physics the surrogate must reproduce. Call it Fig. 1 or 2 in the surrogate-results section, with caption: "TWA ground truth (symbols) and surrogate predictions (curves) for ρ_ss(Ω)." Do not show TWA-only figures separately.
- **Fig. 3 (γ comparison) should be replaced by surrogate predictions.** Show the surrogate's predicted phase diagrams for γ = 0.1, 5, 10, 20 alongside the TWA markers. That is the actual contribution.
- **Table 1 (TWA exponents) should be cut entirely from the main text.** See point 3 below.

**The principle:** Every figure and table in the main text must simultaneously tell a physics story *and* showcase the surrogate. TWA-only content is context, not contribution.

---

## 2. You Still Have Too Many Figures and Tables

Seven main-text figures and five tables is excessive for *Physical Review Research* or *MLST*. At PR Research, you are typically allowed ~6–8 figures *total* (including insets), and tables count against that budget aggressively. At MLST, the figure limit is similarly tight. You need to merge or move roughly 30% of this content.

**Specific recommendations:**

| Current Item | Action | Rationale |
|-------------|--------|-----------|
| Fig. 1 (TWA benchmarks) | **Move to supplementary** or compress to a single small panel in Background | Standard validation, not novel contribution |
| Fig. 2 (TWA phase transitions all N) | **Merge** with surrogate predictions; show only N=3600 or N=4900 ground truth as markers on the surrogate curves | Avoids duplicate physics-only content |
| Fig. 3 (γ comparison, TWA only) | **Replace** with surrogate prediction figure; include TWA markers | Contribution is the *prediction*, not the raw data |
| Fig. 4 (interpolation overlays) | **Keep** | Core surrogate result |
| Fig. 5 (size extrapolation) | **Keep**, but merge (a) and (b) into one two-panel figure | Standard format |
| Fig. 6 (γ transfer) | **Keep**, merge (a) and (b) into one two-panel figure | Core generalization claim |
| Fig. 7 (scaling from predictions) | **Keep**, merge (a) and (b) into one two-panel figure | Critical physics validation |
| Table 1 (TWA exponents 1D/2D/3D) | **Remove** from main text; thesis already published these | See point 3 |
| Table 2 (surrogate metrics) | **Keep**, but slim down | Consider moving ablation rows to Table 4 |
| Table 3 (predicted exponents) | **Keep**, but restructure | See point 3 |
| Table 4 (baseline comparison) | **Merge** with Table 2 | One table for all metrics across models and test sets |
| Table 5 (speedup) | **Keep**, but make it a small inline table or a single row in Table 2/4 | 10⁴× is a headline number, not a full table |

**Target count:** 4–5 main-text figures, 2–3 tables. Everything else (benchmark comparisons, z-tuning collapses, full FSS overlays, residual histograms, FNO/GP overlays, ensemble calibration) goes to supplementary.

---

## 3. The 1D and 3D Results Must Leave the Main Text Entirely

**The problem:** Your surrogate is trained **only on 2D data**. Yet Table 1 prominently displays 1D and 3D exponents. This is a structural trap. A reviewer will immediately ask: "If the model is 2D-only, why are 1D and 3D in Table 1? Does the surrogate generalize across dimensionality?" The answer is no — and you do not want to spend your rebuttal defending why the surrogate cannot do something you never claimed it could.

**What to do:**
- **Remove 1D/3D from Table 1 entirely.** If you feel the background requires dimensional context for the universality-class discussion, mention the values in a single sentence in Section 2.3: "Thesis TWA found effective exponents β ≈ 0.277 (1D), 0.586 (2D), 0.812 (3D), consistent with directed percolation." No table needed.
- **Table 3 (predicted exponents) should be 2D-only.** Columns: Thesis TWA | Transformer (sim) | Transformer (dense) | FNO | GP. Rows: β, δ, z, Ω_c with bootstrap CIs. Clean. Focused. Defensible.
- **Do not mention 1D/3D in the surrogate results.** The Discussion can include one forward-looking sentence: "Extension to 1D and 3D lattices is straightforward in principle, requiring only additional TWA training data." That frames it as future work, not a missing result.

**The principle:** The paper's scope is the 2D surrogate. Do not dilute it with thesis results from other dimensions that the surrogate does not touch.

---

## 4. Section Ordering: Physics-First Is Correct, but Surrogate Results Must Arrive Faster

**The problem:** A reader must wait through 6.5 pages (Introduction + Background + Methods + TWA Physics) before seeing the surrogate results. In modern physics-ML papers, that is too long. Reviewers form an opinion early. If they hit page 6 and still have not seen a neural-network prediction, they may assume the ML component is weak.

**What to do:** Keep the physics-first arc, but compress the TWA-validation overhead so that Section 5 begins by page 4 or 5. Specifically:

1. **Introduction (2.5 pages):** Keep as is — it is excellent.
2. **Background (1.5 pages):** Trim the TWA benchmark discussion. Move single-atom/Dicke/TFIM validation to a paragraph with a citation to the thesis. Keep the Hamiltonian, Lindblad, and scaling-form discussion.
3. **Methods (1.5 pages):** Keep. This is the contract with the reader.
4. **Surrogate Results (3 pages):** This is now Section 4, not Section 5. Start with interpolation (Fig. 4), then size extrapolation (Fig. 5), then γ transfer (Fig. 6), then critical scaling (Fig. 7). This is your core contribution and it should dominate the paper's middle.
5. **Physics Validation & Baselines (1.5 pages):** This becomes Section 5. Show that the surrogate recovers the TWA phase diagram and exponents (merged Fig. 2/3 content), and present the baseline/ablation comparison (merged Table 2/4). This section answers: "Yes, the surrogate is actually correct, and here is how it compares to alternatives."
6. **Discussion / Conclusion:** Unchanged.

**Alternative if you prefer the original order:** Keep Section 4 but compress it to a single page with one figure (TWA phase diagram for N=3600 with surrogate overlay). The original Figs. 1–3 become supplementary. This preserves the "first establish ground truth, then show surrogate" logic without the bloat.

**My recommendation:** Try the reordered version (Surrogate Results as Section 4). It puts your contribution front-and-center while still respecting the physics narrative.

---

## 5. Critical Missing Elements

There are four gaps that a sharp reviewer will notice and that you should address now.

### A. Failure-mode analysis: Where does the surrogate break?

You show successes (interpolation, size extrapolation, γ transfer) but nowhere do you show where the surrogate *fails*. A honest physics-ML paper must include this.

**Add:** A short subsection or supplementary figure showing:
- Trajectories with largest absolute error. Are they near-critical? Early-time? Late-time?
- Error as a function of distance from training data in (Ω, N, γ) space. Does error spike at the boundary?
- A case where the ensemble disagrees significantly — is this a genuine uncertainty or a model failure?

This builds trust more than any accuracy metric.

### B. Data-efficiency ablation

You claim "trained on only 105 trajectories." But you do not show what happens with 50, 25, or 10 trajectories. Is 105 a generous budget or a bare minimum?

**Add:** A supplementary figure showing validation MSE vs. training-set size (log scale). If the curve has not saturated at 105, that is an important finding. If it saturates at 50, you have an even stronger story.

### C. Mean-field comparison

You compare against FNO and GP, but you do not compare against the simplest physical baseline: **mean-field theory**. For Rydberg facilitation, mean-field gives a closed-form (or simple ODE) prediction for ρ(t). It will be wrong near the critical point, but it is the standard physics baseline.

**Add:** A single curve or table row showing mean-field ρ_ss(Ω) alongside TWA and surrogate for N=3600. If the surrogate is closer to TWA than mean-field is, you have a strong physics argument: the NN captures fluctuations that MF misses.

### D. The "dense grid" caveat

You mention evaluating on a dense Ω grid (step 0.02). This is powerful, but you must be explicit about what the surrogate is predicting *between* training points. TWA at Ω = 11.14 was never simulated. The surrogate is hallucinating (interpolating) a physical trajectory. Is this interpolation smooth? Does it introduce non-physical oscillations? A phase diagram built entirely from dense-grid predictions needs a disclaimer or validation against a few held-out TWA points at intermediate Ω.

**Add:** In the dense-grid section, explicitly state: "Dense-grid predictions are validated against TWA at N=3600, Ω = {11.00, 11.10, 11.20} (not in training set), confirming smooth interpolation." Or show a figure with TWA markers on the dense curve.

---

## Summary: Action Items

| Priority | Action |
|----------|--------|
| **Critical** | Remove 1D/3D from main text entirely; make Table 3 strictly 2D |
| **Critical** | Compress or move TWA-only figures (Figs. 1–3) to supplementary; merge TWA data with surrogate predictions where possible |
| **High** | Reduce main-text figure count to 4–5 and table count to 2–3 |
| **High** | Consider reordering: Surrogate Results as Section 4, Physics Validation + Baselines as Section 5 |
| **Medium** | Add failure-mode analysis (largest errors, boundary behavior, ensemble disagreement) |
| **Medium** | Add mean-field baseline comparison |
| **Low** | Add data-efficiency curve in supplementary |
| **Low** | Clarify dense-grid validation with explicit held-out TWA checkpoints |

You are 80% of the way there. The story is right. Now cut the thesis appendix, focus the scope on 2D, and make every figure earn its place by showcasing the surrogate.

—
