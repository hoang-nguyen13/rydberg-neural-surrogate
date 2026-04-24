# Critical Physics Review: Foundation Model for Non-Equilibrium Phase Transitions in Rydberg Facilitation

**Reviewer perspective:** Theoretical physicist, non-equilibrium statistical mechanics, quantum optics, Rydberg arrays, DP universality, TWA methods.  
**Document reviewed:** `/Users/hoangnguyen/PROJECT_PLAN.md` (v1.0, 2026-04-24)

---

## Executive Summary

The project plan is ambitious and technically well-structured for the ML engineering side, but it contains **serious conceptual errors in the treatment of critical phenomena**. The most fundamental problem is that the entire Phases 1–4 rely on **ensemble-averaged trajectories** (`sz_mean[t]`) to extract critical exponents for a phase transition in the **Directed Percolation (DP) universality class**. This is physically unsound: DP transitions are characterized by a **bimodal distribution** of trajectories near criticality (some die, some survive), and ensemble averaging destroys the very fluctuations that define the critical behavior. You cannot reliably extract β, ν, or z from mean trajectories alone. 

The second major issue is the **assumption that TWA data will exhibit DP exponents**. TWA is a semiclassical approximation that often fails to capture the correct universal critical behavior of quantum many-body systems, especially for long-time dynamics where quantum correlations build up non-perturbatively. If the exponents do not match DP, the default assumption should be "TWA breaks down," not "we discovered a new universality class."

The third major issue is the **sigmoid fit to ρ_ss(Ω)** and the associated finite-size scaling. The functional form of a sigmoid has no basis in critical phenomena theory. The proposed finite-size scaling analysis conflates the pseudo-critical point extracted from a sigmoid with the true finite-size critical point, and ignores standard diagnostics like susceptibility peaks or Binder cumulant crossings.

The plan is salvageable, but the physics narrative must shift from "extracting precise critical exponents from mean trajectories" to "learning and predicting non-equilibrium dynamics across parameter space, with critical behavior diagnosed via proper spatial/fluctuation-aware methods." The mean-field model (Phases 1–4) can still yield a useful *prediction engine* for ensemble-averaged dynamics, but the paper’s physics claims must be correspondingly modest.

**Overall verdict:** The engineering roadmap is sound; the physics framing is dangerously overconfident. Significant revisions are needed before this can produce a credible paper in PRX Quantum or similar.

---

## 1. Physics Correctness

### 1.1 Directed Percolation Exponents
The cited DP exponents for 2D are approximately correct:
- β ≈ 0.583 (order parameter)
- ν_⊥ ≈ 0.733 (spatial correlation length)
- ν_∥ ≈ 1.295 (temporal correlation length)
- z = ν_∥/ν_⊥ ≈ 1.766 (dynamic exponent)

The plan writes ν ~ 0.73 and β ~ 0.58, which is fine as shorthand. However, **the plan does not distinguish between ν_⊥ and ν_∥**. In finite-size scaling, the shift of the pseudo-critical point with system size scales as L^{-1/ν_⊥}, but the temporal correlation time diverges as τ ~ L^{z} = L^{ν_∥/ν_⊥}. The plan conflates these and only mentions a single ν. In a 2D spatial system, the relevant finite-size scaling exponent for the spatial correlation length is ν_⊥ ≈ 0.733.

### 1.2 Phase Transition Description for Rydberg Facilitation
The plain-English description of the transition (Section 2.2) is qualitatively correct for a contact-process-like model. However, there are subtle but important issues:

- **Absorbing state definition:** The absorbing state is the all-ground state (sz = -1 for all atoms). In a facilitation model with Δ = V = 2000, an isolated atom is far detuned and cannot be excited. Only atoms adjacent to existing excitations are resonant. This correctly produces an absorbing state at zero excitation density. The plan gets this right.

- **Is it truly DP?** The Rydberg facilitation model with spontaneous emission (Γ) and dephasing (γ) is expected to map to a classical stochastic process in the limit where quantum coherence is rapidly destroyed. With Γ = 1 and γ = 0.1, the ratio γ/Γ = 0.1 is not extremely large. Quantum coherence may still matter, particularly for the *dynamics* and *critical exponents*. TWA captures some quantum fluctuations through stochastic sampling of initial Wigner distributions, but it is a *semiclassical* approximation. Long-range quantum entanglement near criticality is not faithfully represented. **There is a real risk that TWA produces mean-field or non-universal exponents rather than DP.** The plan’s fallback — "could indicate new universality class" — is scientifically unsound without much stronger evidence (e.g., exact diagonalization or quantum trajectory Monte Carlo confirming the exponents).

- **Order parameter:** In DP, the order parameter is the density of active sites, ρ_a, which vanishes continuously at the critical point. Here, the data uses `sz_mean` ∈ [-1, 1]. The mapping to the DP order parameter is:
  ```
  ρ = (sz_mean + 1) / 2
  ```
  In the absorbing phase, ρ → 0. In the active phase, ρ → ρ_active > 0. The plan does not make this rescaling explicit. When fitting for the exponent β, you must use ρ, not sz_mean, because the DP scaling law is ρ ~ (Ω - Ω_c)^β, not sz_mean + 1 ~ (Ω - Ω_c)^β (though the latter is mathematically equivalent, the physical interpretation must be clear).

### 1.3 Finite-Size Scaling Assumptions
The plan proposes (Week 10):
```
Ω_c(L) = Ω_c(∞) + A * L^{-1/ν}
```
This is the standard FSS ansatz for the shift of the pseudo-critical point, where ν = ν_⊥ in 2D. However, there are problems:

1. **What is Ω_c(L)?** The plan defines it as the inflection point of a sigmoid fit to ρ_ss(Ω). The inflection point of a phenomenological sigmoid is not a theoretically justified pseudo-critical point. Different definitions of the finite-size critical point (susceptibility peak, maximum of the derivative dρ/dΩ, crossing of Binder cumulants) can have different correction-to-scaling amplitudes. The plan must pick a physically motivated definition.

2. **Correlation time vs. system size:** The dynamic exponent z ≈ 1.77 means that the relaxation time near criticality scales as τ ~ L^{1.77}. For L = 70 (N = 4900), τ ~ 70^{1.77} ≈ 1700. The simulation only runs to t_max = 1000. **This means the largest systems likely have not reached steady state near criticality.** Computing ρ_ss from the last 50 time points is then not measuring the true steady state, but a transient. This will systematically bias the critical point estimate and round the transition even more than finite-size effects alone.

3. **Data collapse:** The plan does not mention data collapse (finite-size scaling of the entire order parameter curve). A more robust check is to plot ρ * L^{β/ν_⊥} vs. (Ω - Ω_c) * L^{1/ν_⊥} and verify that curves for different L collapse. This is the gold standard for FSS and is completely missing from the plan.

**Recommendation:**
- Explicitly rescale `sz_mean` to the DP order parameter ρ = (sz_mean + 1)/2.
- Acknowledge that TWA may not reproduce DP exponents; benchmark against exact methods for small systems if possible.
- Add a data-collapse analysis to the FSS protocol.
- Check whether t_max = 1000 is sufficient for the largest systems by estimating the relaxation time τ(L) ~ L^z.

---

## 2. Tokenization of Physical Quantities

### 2.1 Uniform 256-Bin Discretization of sz_mean ∈ [-1, 1]
This gives a bin width of Δsz ≈ 0.0078. At first glance, this seems fine for the raw range. However, the **physics risks are significant**:

1. **Coarse-graining near criticality:** Near the critical point, the difference between absorbing (sz ≈ -1.0) and slightly active (sz ≈ -0.94) is only ~0.06, spanning about **8 bins**. The rounding error from uniform binning is ±0.004. When fitting critical exponents, this discretization noise propagates into the fit and can dominate over the true finite-size rounding for large N. The relative error in ρ near criticality can be ~5-10%.

2. **Skewed distribution:** Most trajectories spend most of their time near sz ≈ -1 (absorbing) or near some plateau value (active). Uniform binning wastes resolution in the densely populated regions and provides excessive resolution where no data exists. The plan correctly identifies this risk and proposes k-means as an alternative.

3. **Loss of physical meaning:** The transformer is doing **classification over 256 bins** to predict a continuous physical observable. Cross-entropy loss does not penalize predictions proportionally to physical distance. Predicting the adjacent bin (error ~0.008) incurs the same loss as predicting the opposite extreme (error ~2.0), up to the log-probability difference. The model has no inductive bias that tokens 0 and 1 are closer than tokens 0 and 255.

### 2.2 Adaptive / K-Means Binning
K-means with 256 centroids would allocate more bins to regions with high data density (near -1.0 and near the active plateau). This is better than uniform binning, but it introduces a new problem: **the bin boundaries become data-dependent and non-uniform**. The distance between adjacent tokens is no longer constant, making the cross-entropy loss even more disconnected from the physical error. Moreover, k-means centroids are not ordered (though one can sort them), and the model might struggle with the non-monotonic spacing.

### 2.3 Regression vs. Classification
**The plan should strongly consider regression for the trajectory values.** Instead of discretizing sz_mean into tokens, predict continuous values directly:
- Replace the final softmax layer with a scalar output (or small MLP outputting μ and σ for a Gaussian).
- Use MSE or negative log-likelihood loss.
- Keep tokenization only for the **discrete parameters** (Ω, N, etc.), which are genuinely discrete or categorical.

**Advantages of regression:**
- No discretization error.
- The loss directly penalizes large physical errors more than small ones.
- Critical exponents extracted from continuous predictions are smoother and more reliable.
- The model can interpolate between training values naturally.

**Disadvantages:**
- Slightly harder to implement in a standard nanoGPT framework (requires modifying the output head).
- May require a different generation protocol (sampling from a predicted distribution rather than a categorical distribution).

**Recommendation:**
- **Switch to regression for sz_mean.** Use a hybrid model: token embedding for parameters, but predict continuous values for the time series. If the project is locked into classification for engineering reasons, then at minimum use **finer discretization near critical values** (e.g., adaptive bins with higher resolution around sz ≈ -1.0 and the transition region) or use **K-means binning** with sorted centroids.
- If classification is retained, verify that the binning error (±half bin width) is smaller than the statistical error from the finite trajectory ensemble (std of mean ≈ σ_trajectory/√500).

---

## 3. Critical Point Estimation Strategy

### 3.1 Sigmoid Fit: The Wrong Functional Form
The plan proposes fitting ρ_ss(Ω) to:
```
ρ_ss = a + b / (1 + exp[-k(Ω - Ω_c)])
```
This is **phenomenologically convenient but physically incorrect** for a continuous phase transition in the DP class. Near the critical point, the order parameter behaves as a power law:
```
ρ(Ω) ~ (Ω - Ω_c)^β   for Ω > Ω_c
ρ(Ω) = 0               for Ω ≤ Ω_c
```
A sigmoid:
1. Does not enforce ρ = 0 below Ω_c (it predicts a small but non-zero value).
2. Has an analytic expansion around the inflection point, not a branch-point singularity.
3. Introduces an artificial finite rounding that is unrelated to finite-size scaling.

Fitting a sigmoid to DP data will systematically bias Ω_c, β, and the amplitude. The inflection point of a sigmoid is not the same as the critical point of a power-law singularity.

### 3.2 Alternative Methods
The plan does not mention standard methods from computational statistical mechanics:

**A. Susceptibility peak**
Define the susceptibility (variance of the order parameter):
```
χ = N * (⟨ρ^2⟩ - ⟨ρ⟩^2)
```
In a finite system, χ peaks at the pseudo-critical point Ω_c(L). This is one of the most robust estimators. However, **this requires access to the full distribution of trajectories, not just the mean.** From ensemble-averaged data, you cannot compute χ. This is a fatal limitation of the mean-field dataset.

**B. Derivative method (from mean data)**
For ensemble-averaged data, the derivative dρ_ss/dΩ is maximally steep near the transition. One can define Ω_c(L) as the location of the maximum of dρ_ss/dΩ. In finite systems, this rounds and shifts, but it obeys FSS. This is probably the best you can do with mean trajectories alone, though it is inferior to the susceptibility peak.

**C. Modified Binder cumulant**
For transitions with an absorbing state, the standard Binder cumulant is problematic because the distribution in the absorbing phase is a delta function. Modified cumulants or the ratio of moments (e.g., U = ⟨ρ^2⟩ / ⟨ρ⟩^2) can be used, but again require the full distribution.

**D. Survival probability method**
The true hallmark of DP is the **survival probability** P_surv(t): the probability that the system has not yet reached the absorbing state by time t. At criticality, P_surv(t) ~ t^{-δ} with δ ≈ 0.450 in 2D DP. In the subcritical phase, it decays exponentially; in the supercritical phase, it saturates to a finite value. This is the **defining observable of DP** and is completely absent from the plan.

### 3.3 Finite-Size Effects in 2D DP
The finite-size shift of the pseudo-critical point scales as L^{-1/ν_⊥}, but the **rounding** of the transition (the width over which the order parameter rises) scales as L^{-1/ν_⊥} as well. For 2D DP with ν_⊥ ≈ 0.733, the shift is relatively slow: L^{-1.36}. This means even for L = 70, the shift is ~70^{-1.36} ≈ 0.02 in units of the critical coupling. If the true Ω_c(∞) is around 11.0, the finite-size critical point for L=70 might still be at ~11.3. However, the rounding is also significant, and the sigmoid fit will conflate shift and rounding.

**Recommendation:**
- **Abandon the sigmoid fit.** Instead, define Ω_c(L) as the maximum of dρ_ss/dΩ (from mean data) or, preferably, the peak of the susceptibility (if spatial data is available).
- Fit the power-law form directly: for Ω > Ω_c(L), fit ρ = A * (Ω - Ω_c(L))^β, using only data above the critical point. Use finite-size scaling to extrapolate Ω_c(L) → Ω_c(∞).
- **If the project insists on a smooth fitting function**, use a phenomenological form that explicitly includes the power-law singularity, such as:
  ```
  ρ(Ω) = A * (Ω - Ω_c)^β * [1 + B(Ω - Ω_c) + ...]   for Ω > Ω_c
  ρ(Ω) = 0                                           for Ω ≤ Ω_c
  ```
  with a regularization for finite-size rounding (e.g., replace the step with an error function of width ~L^{-1/ν}).

---

## 4. Phase 6 (Spatial Extension): Compressed Latent Approach

### 4.1 CNN Autoencoder + GPT on Latents
The plan proposes:
- Encode a 35×35 spatial snapshot into a 64-dimensional latent vector.
- Train GPT to predict sequences of 64-dim latent vectors.
- Decode to reconstruct spatial patterns.

**This approach is fundamentally ill-suited for critical systems and is the most risky component of the entire plan.**

### 4.2 Physics Risks of Encoding Spatial Patterns into 64-Dimensional Latents
1. **Loss of critical correlations:** At the DP critical point, spatial correlations decay as a power law: C(r) ~ r^{-(d-2+η)} with η ≈ 0.230 in 2D DP. Correlations exist at all length scales up to the system size. A 64-dimensional latent vector is an extreme compression of a 1225-dimensional spatial configuration. Information-theoretically, it is impossible to faithfully encode long-range power-law correlations in such a small latent space without strong prior assumptions. The autoencoder will necessarily learn to represent only the dominant, short-wavelength features and will smooth over critical fluctuations.

2. **Avalanche and cluster structure:** DP critical dynamics is characterized by fractal clusters of active sites. The cluster size distribution follows a power law. Front propagation involves rough interfaces (KPZ-like fluctuations in some limits). A CNN autoencoder trained with MSE reconstruction loss will prioritize low-frequency content and will fail to reproduce sharp fronts, small isolated clusters, and fractal structure. MSE is particularly bad for sparse, binary-like configurations (active sites are sparse near criticality).

3. **Temporal coherence:** The GPT predicts the next latent vector autoregressively. If the autoencoder loses critical spatial information at each snapshot, the GPT will compound these errors over time. Small reconstruction errors in the latent space can correspond to large physical errors (e.g., missing an active site that nucleates a new avalanche).

4. **System-size generalization:** The autoencoder is trained on fixed-size snapshots (e.g., 35×35). To generalize to L = 50 or L = 70, you either need to train separate autoencoders or use fully convolutional architectures. The plan does not address this.

### 4.3 Alternative Approaches
**Option C in the plan (separate spatial and temporal models) is actually much more promising.** A better architecture would be:

1. **Spatiotemporal CNN or U-Net:** Predict the next spatial snapshot directly from a history of snapshots using 3D convolutions or a recurrent convolutional architecture (ConvLSTM, PredRNN). This preserves spatial locality and critical correlations.

2. **Message-passing neural network (MPNN):** Treat the lattice as a graph. Each node (atom) has a state (ground/Rydberg). The update rule is learned via graph neural networks. This naturally respects the local facilitation constraint (only neighbors matter) and can generalize across system sizes if the message function is size-independent.

3. **Neural operator / Fourier neural operator (FNO):** Learn the mapping in Fourier space, which naturally handles different resolutions and can capture long-range correlations via the low-k modes.

4. **Patch-based tokenization without compression:** The plan mentions patches (e.g., 5×5 → 49 patches). Rather than compressing each patch into a scalar, keep patches as multi-dimensional tokens. Use a vision transformer (ViT) that attends across patches. You don't need a CNN autoencoder if you tokenize patches directly. For a 35×35 lattice with 5×5 patches, you have 49 patches. If each patch is tokenized into a single discrete token (e.g., via VQ-VAE on patches), the sequence length is 49 per snapshot. With temporal subsampling to 40 steps, that's 49 × 40 = 1960 tokens. Still too long for nanoGPT, but manageable with a standard ViT or a hierarchical transformer.

**Recommendation:**
- **Do not use a CNN autoencoder to compress full spatial snapshots into 64-dim vectors for critical dynamics.** The physics risk is too high.
- If you must use a transformer-based approach for spatial data, use **patch-level tokenization** (like ViT) without extreme compression, or switch to a **spatiotemporal CNN/ConvLSTM** architecture.
- Consider a **hybrid approach:** GPT predicts the mean trajectory (Phase 4), and a separate conditional CNN predicts spatial fluctuations given the mean. This is similar to Option C and is physically more interpretable.

---

## 5. Physical Observables to Extract

The plan focuses heavily on ρ_ss and the critical point Ω_c, but misses many key observables that are essential for characterizing a non-equilibrium phase transition, especially in the DP class.

### 5.1 Missing Observables from Mean-Field Data
Even with only `sz_mean[t]` (ensemble-averaged), you can extract:

1. **Relaxation time τ(Ω):** Fit sz_mean(t) to an exponential or stretched exponential relaxation. Near criticality, τ diverges. From mean trajectories, you can define τ as the time to reach, say, 90% of the steady-state value. The divergence τ ~ |Ω - Ω_c|^{-ν_∥} is a key signature.

2. **Stretched exponent:** In the active phase, the approach to steady state may not be purely exponential. A stretched exponential exp[-(t/τ)^α] with α < 1 indicates complex relaxation dynamics.

3. **Early-time curvature / initial slip:** The initial decay from sz=+1 carries information about the facilitation radius and the microscopic dynamics.

### 5.2 Missing Observables Requiring Individual Trajectories
These are the **defining observables of DP** and require per-trajectory data (not ensemble averages):

1. **Survival probability P_surv(t):** Fraction of trajectories that have not reached the absorbing state by time t. At criticality: P_surv ~ t^{-δ} with δ ≈ 0.450 (2D).

2. **Number of active sites N_active(t):** For surviving trajectories, the mean number of active sites grows as t^θ at criticality. The mean over all trajectories (including dead ones) goes as t^{θ - δ}.

3. **Avalanche size distribution:** In the absorbing phase, the probability distribution of avalanche sizes P(S) follows a power law P(S) ~ S^{-τ_s} with a cutoff that diverges at criticality. This is a hallmark of self-organized criticality and DP.

4. **Cluster size distribution and fractal dimension:** At criticality, active clusters are fractal with dimension d_f.

5. **Spatial correlation function C(r):** Requires per-atom spatial data. At criticality: C(r) ~ r^{-(d-2+η)} e^{-r/ξ}, with ξ ~ |Ω - Ω_c|^{-ν_⊥}.

6. **Susceptibility χ:** As discussed above, the variance of the order parameter. Peaks at the pseudo-critical point.

7. **Binder cumulant or modified cumulant:** Requires the full distribution.

8. **Interface roughness / KPZ exponents:** If the excitation front is studied, its width grows as w ~ t^β_KPZ with β_KPZ ≈ 0.24 in 2D.

### 5.3 Missing Observables Requiring Spatial + Temporal Data
The plan mentions the spatial correlation function C(r) and front propagation velocity, which is good. But it misses:
- **Dynamic structure factor S(k, ω)** or its temporal Fourier transform.
- **Equal-time correlation length ξ from Ornstein-Zernike fitting.**
- **Autocorrelation time from temporal correlations.**

**Recommendation:**
- Add extraction of τ(Ω) and its divergence analysis to Phase 4.
- **Strongly consider saving individual trajectories** (or at least a subset) from the TWA simulations, not just ensemble averages. The critical physics of DP is fundamentally in the *fluctuations*, not the mean.
- If storage is an issue, save per-trajectory final states and survival flags (500 booleans per parameter set). This is negligible storage and enables P_surv analysis.

---

## 6. Data Limitations: What Physics Can Realistically Be Claimed?

### 6.1 The Fundamental Limitation of Ensemble-Averaged Data
For Phases 1–4, the training data is `sz_mean[t]` averaged over ~500 trajectories. This is the single most constraining fact about the project.

In a system with an absorbing state, the ensemble-averaged dynamics is **not representative of individual trajectories** near criticality. The distribution of final states is bimodal: some fraction P_surv(∞) end in the active phase, the rest (1 - P_surv) in the absorbing state. The mean is:
```
⟨sz⟩ = P_surv * ⟨sz⟩_active + (1 - P_surv) * (-1)
```
Near criticality, P_surv → 0, and ⟨sz⟩_active may be close to -1 as well. The mean trajectory can look smooth and continuous even though individual trajectories show a discontinuous jump (alive vs. dead). **Critical exponents extracted from mean trajectories are not the DP exponents.** They are some mixture of the survival exponent δ and the active-phase exponent β, convolved with finite-size effects.

### 6.2 What Physics CAN Be Claimed from Mean-Field Data?
Despite the limitations, the mean-field model is not worthless. You can credibly claim:

1. **Interpolation and prediction of ensemble-averaged dynamics:** The model learns the mapping (Ω, N) → ⟨sz⟩(t). This is a genuine generalization task. If it predicts unseen parameter values accurately, it demonstrates that the transformer has learned the effective equations of motion.

2. **Qualitative phase diagram:** The model can predict whether a given parameter set is deep in the absorbing phase, deep in the active phase, or near the transition. This is useful for parameter screening.

3. **Finite-size trends:** The model can predict how the transition shifts and rounds with system size, even if the precise exponents are not DP.

4. **Computational acceleration:** If the transformer is 1000× faster than TWA, it can be used as a surrogate model for mean-field dynamics. This is a legitimate ML-for-physics contribution, even without new critical exponent measurements.

### 6.3 What Physics CANNOT Be Claimed from Mean-Field Data?
You **cannot** credibly claim:
1. Precise extraction of the DP exponent β from mean trajectories.
2. Precise extraction of ν from the shift of sigmoid inflection points.
3. Any claim about avalanche statistics, cluster distributions, or spatial correlations.
4. Any claim about the survival probability exponent δ or the dynamic exponent z.
5. Any claim about a "new universality class" based on deviations from DP in the mean-field data.

### 6.4 What Requires Spatial Data?
Everything related to spatial structure and fluctuations:
- Correlation length ξ and exponent ν_⊥
- Correlation function exponent η
- Cluster size distribution and fractal dimension
- Front roughness and KPZ physics
- Avalanche statistics
- Susceptibility and Binder cumulant
- Dynamic exponent z (requires both spatial and temporal correlation analysis)
- Any claim about directed percolation vs. other universality classes

**Recommendation:**
- **Reframe the paper for Phases 1–4 as "Surrogate modeling of non-equilibrium mean-field dynamics" rather than "extracting critical exponents."** The critical exponent analysis should be explicitly deferred to Phase 6 (or a follow-up paper) where spatial and individual-trajectory data is available.
- If the paper must include critical exponents from mean-field data, frame them as **effective exponents** or **phenomenological fits** with large error bars and clear caveats. Do not claim they test the DP universality class.

---

## 7. Risk Assessment: Underappreciated Physics Risks

### 7.1 Critical Risks (Could Derail Physics Claims)

| Risk | Severity | Why It Matters |
|---|---|---|
| **TWA does not produce DP exponents** | Critical | TWA is semiclassical. For the parameter regime (V=2000, Δ=2000, γ=0.1), quantum correlations may not be fully suppressed. If TWA gives β ≈ 0.5 or 0.7 instead of 0.58, you cannot distinguish "TWA failure" from "new physics" without exact quantum benchmarks. |
| **t_max = 1000 is too short for large systems near criticality** | Critical | With z ≈ 1.77, τ(L=70) ≈ 1700. The largest systems are not in steady state near Ω_c. Extracting "steady-state" values from the last 50 points is measuring a transient, not ρ_ss. |
| **Ensemble averaging destroys critical fluctuations** | Critical | You cannot extract true DP exponents from mean trajectories. The distribution is bimodal. The plan's entire Phase 4 physics analysis rests on a dataset that is structurally incapable of revealing the true critical behavior. |
| **Sigmoid fit produces biased, unphysical exponents** | High | The sigmoid has no power-law singularity. Ω_c and β extracted from it will not obey FSS correctly and will be rejected by reviewers in statistical mechanics. |
| **Compressed latent loses critical spatial structure** | High | A 64-dim autoencoder cannot encode power-law correlations, fractal clusters, or rough fronts. Phase 6 spatial predictions will look artificially smooth. |

### 7.2 Moderate Risks (Can Be Mitigated)

| Risk | Severity | Mitigation |
|---|---|---|
| **Discretization error dominates near criticality** | Moderate | Switch to regression; if not possible, use adaptive binning with finer resolution near sz ≈ -1. |
| **Parameter tokenization too coarse (Ω bins of 0.5)** | Moderate | Near Ω_c ≈ 11.2, a 0.5 bin width is huge. Use finer bins or continuous embeddings for Ω. |
| **Reviewers say "just curve fitting"** | Moderate | Emphasize generalization to unseen parameters and sizes. But without spatial/fluctuation data, this criticism is partially valid. |
| **NOQS authors publish competing work** | Moderate | Move fast, but more importantly, differentiate by focusing on the *surrogate model* aspect rather than competing on critical exponent precision. |

### 7.3 What Could Go Wrong and Still Produce Publishable Results?
1. **If critical exponents from mean-field data deviate from DP:** Publish the work as a **machine learning surrogate model** for non-equilibrium dynamics, not as a critical phenomena paper. Focus on the speedup, interpolation, and generalization. This is still publishable in *Physical Review Research*, *Machine Learning: Science and Technology*, or *npj Computational Materials*.

2. **If spatial extension (Phase 6) fails:** The mean-field surrogate model (Phases 1–4) is already a complete, self-contained project. Write it up as a "foundation model for mean-field non-equilibrium dynamics" and mention spatial extension as future work.

3. **If TWA exponents are wrong:** Use the model to predict dynamics, and compare against exact quantum trajectories for small systems (N=100 or N=225). If the model matches TWA but not exact quantum results, that is actually an interesting finding: it shows the transformer faithfully emulates the TWA but reveals TWA's limitations. This is publishable as a **diagnostic tool**.

4. **If attention analysis is inconclusive:** Null results in interpretability are acceptable if framed honestly. The core contribution is the surrogate model.

---

## 8. Specific Recommendations for Revision

### Immediate Revisions to the Physics Narrative
1. **Change the title and framing.** Instead of "extracting critical exponents," frame the project as:
   > "A Foundation Model for Surrogate Prediction of Non-Equilibrium Rydberg Facilitation Dynamics"
   
   The critical point and effective exponents can be secondary results, not the primary claim.

2. **Explicitly define the order parameter.** Use ρ = (sz_mean + 1)/2. All fits for β should use ρ, not sz_mean.

3. **Remove the sigmoid fit.** Replace with:
   - For mean-field data: maximum of dρ/dΩ as the pseudo-critical point, plus direct power-law fit ρ = A(Ω - Ω_c)^β above threshold.
   - For spatial data (future): susceptibility peak and Binder cumulant crossing.

4. **Add a check for steady-state convergence.** For each (N, Ω), fit the last 100 points to a constant + exponential decay. If the exponential decay constant is comparable to the window size, the system has not converged. Flag these points and exclude them from steady-state analysis.

### Revisions to Data Engineering (Phase 1)
5. **Switch from classification to regression for sz_mean.** This is the single highest-impact change for physics correctness. The output head should predict a continuous value. If classification is retained for engineering reasons, add a **physics-aware loss** (e.g., weighted MSE in token space where weights depend on bin distance).

6. **Finer parameter resolution for Ω near criticality.** The 0.5-wide bins for Ω are too coarse. Use continuous embeddings or much finer bins (width ~0.1) in the 10–13 range.

7. **Save trajectory metadata:** Even if you only train on ensemble averages, store per-trajectory survival flags and final active-site counts. This costs negligible storage and enables proper DP analysis later.

### Revisions to Evaluation (Phase 4)
8. **Add relaxation time analysis.** Extract τ(Ω, N) from the mean trajectories. Plot τ vs. |Ω - Ω_c| and check for divergence. This is a standard physics diagnostic.

9. **Add data-collapse analysis.** For different system sizes, plot ρ * L^{β/ν} vs. (Ω - Ω_c) * L^{1/ν} using literature values of β and ν. If the curves collapse, it supports the DP hypothesis. If they don't, it suggests TWA deviations or insufficient data quality.

10. **Be explicit about the difference between pseudo-critical and true critical points.** Define exactly how Ω_c(N) is extracted, and discuss correction-to-scaling terms.

### Revisions to Spatial Extension (Phase 5–6)
11. **Do not compress spatial snapshots into 64-dim latents for critical dynamics.** Replace with one of:
    - Patch-based ViT tokenization (5×5 patches → 49 tokens per snapshot).
    - Spatiotemporal CNN/ConvLSTM.
    - Graph neural network on the lattice.
    - Hybrid: GPT for mean trajectory + conditional CNN for spatial fluctuations.

12. **Generate per-atom data for at least the critical region across all sizes.** 50–100 trajectories per parameter set is sufficient. This is essential for any claims about universality.

### Revisions to the Risk Register
13. **Add the following risks:**
    - "TWA data may not exhibit DP universality due to semiclassical approximation failure." (Probability: High; Impact: Critical)
    - "Ensemble-averaged data structurally prevents extraction of true critical exponents." (Probability: Certain; Impact: Critical)
    - "t_max = 1000 insufficient for largest systems near criticality." (Probability: High; Impact: High)
    - "64-dim autoencoder destroys critical spatial correlations." (Probability: High; Impact: High)

---

## 9. Revised Priorities & Suggested Timeline Adjustments

Given the physics limitations identified, I recommend the following reprioritization:

### Option A: Minimal Change (Focus on Surrogate Modeling)
- **Keep Phases 1–4 largely as-is** but reframe the physics claims.
- Remove critical exponent extraction from mean-field data or relegate it to an appendix with heavy caveats.
- Focus the paper on: (1) accurate prediction of mean-field trajectories, (2) generalization to unseen parameters and sizes, (3) speedup over TWA.
- **Defer Phase 6** to a follow-up paper where spatial data and proper critical analysis are done correctly.
- **Timeline:** 4–5 months to a solid, lower-risk paper.

### Option B: Maximal Physics (Fix the Critical Analysis)
- **Phase 0 (revised):** Re-run simulations (or re-analyze existing data) to extract **per-trajectory survival probabilities** and, for small systems, full spatial snapshots.
- **Phase 1 (revised):** Switch to **regression** for sz_mean. Add survival probability as a second output channel.
- **Phase 4 (revised):** Replace sigmoid fit with susceptibility peak / derivative method. Add P_surv(t) analysis.
- **Phase 6 (revised):** Replace CNN autoencoder with **ConvLSTM** or **patch-based ViT**.
- **Timeline:** 8–10 months to a high-impact paper in *Physical Review Letters* or *PRX Quantum*.

### My Recommendation
Go with **Option A for the first paper**, treating it as a proof-of-concept for ML surrogate modeling of open quantum system dynamics. The results will be publishable and will build confidence in the methodology. Then pursue **Option B as a second paper** with proper spatial data and critical analysis. Trying to do both in one paper with the current limitations is likely to produce a manuscript that is rejected by statistical mechanics reviewers for fundamental physics flaws.

---

## Appendix: Quick Reference — 2D Directed Percolation Exponents

| Exponent | Symbol | Value | Relevant Observable |
|---|---|---|---|
| Order parameter | β | 0.583 | ρ ~ (Ω - Ω_c)^β |
| Spatial correlation | ν_⊥ | 0.733 | ξ ~ \|Ω - Ω_c\|^{-ν_⊥} |
| Temporal correlation | ν_∥ | 1.295 | τ ~ \|Ω - Ω_c\|^{-ν_∥} |
| Dynamic | z | 1.766 | z = ν_∥/ν_⊥ |
| Survival probability | δ | 0.450 | P_surv(t) ~ t^{-δ} |
| Active sites (surviving) | θ (or η) | ~0.68 | N_active(t) ~ t^θ |
| Spatial correlation func. | η | 0.230 | C(r) ~ r^{-(d-2+η)} |
| Avalanche size | τ_s | 1.268 | P(S) ~ S^{-τ_s} |

*Source: Hinrichsen, Adv. Phys. 2000; Ódor, Rev. Mod. Phys. 2004; Jensen, *Self-Organized Criticality* (1998).*

---

*Review completed: 2026-04-24*  
*Recommendation: Revise physics framing before proceeding to implementation.*
