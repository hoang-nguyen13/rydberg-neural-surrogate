# Foundation Model for Non-Equilibrium Phase Transitions in Rydberg Facilitation Dynamics

## Complete Implementation Plan (A to Z)

**Approved approach:** Hybrid — Mean-Field First, Spatial Extension Later
**Timeline:** ~6 months (26 weeks)
**Status:** Phase 0 complete, Phase 1–4 are mean-field, Phase 5–6 are spatial extension

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Physics in Plain English](#2-the-physics-in-plain-english)
3. [Data Foundation (Phase 0 — COMPLETE)](#3-data-foundation-phase-0--complete)
4. [Phase 1: Data Engineering (Weeks 1–3)](#4-phase-1-data-engineering-weeks-13)
5. [Phase 2: Model Architecture (Weeks 4–5)](#5-phase-2-model-architecture-weeks-45)
6. [Phase 3: Training (Weeks 6–8)](#6-phase-3-training-weeks-68)
7. [Phase 4: Evaluation & Physics (Weeks 9–12)](#7-phase-4-evaluation--physics-weeks-912)
8. [Phase 5: Spatial Data Generation (Weeks 13–15)](#8-phase-5-spatial-data-generation-weeks-1315)
9. [Phase 6: Spatiotemporal Model (Weeks 16–20)](#9-phase-6-spatiotemporal-model-weeks-1620)
10. [Phase 7: Interpretability (Weeks 21–22)](#10-phase-7-interpretability-weeks-2122)
11. [Phase 8: Paper Writing (Weeks 23–26)](#11-phase-8-paper-writing-weeks-2326)
12. [Hardware & Dependencies](#12-hardware--dependencies)
13. [Risk Register](#13-risk-register)

---

## 1. Executive Summary

We train a decoder-only transformer (nanoGPT) to predict the time evolution of Rydberg atom facilitation dynamics. The model is **conditioned on Hamiltonian parameters** (Omega, N, dimension) and autoregressively generates the mean excitation density sz_mean(t). From generated trajectories, we extract physical observables: steady-state density rho_ss, critical point Omega_c(N), and critical exponents.

**Why this works:** Your master thesis already generated 290 trajectory ensembles across 10 system sizes (N=100 to N=4900) and 43 Rabi frequencies (Omega=0 to 30). The data shows a clean non-equilibrium phase transition — excitations either die out (absorbing phase) or sustain (active phase). A transformer can learn this mapping from parameters to dynamics.

**End product:** A single neural network that, given a new Hamiltonian it has never seen, predicts the full time evolution and critical properties without running a single line of Julia simulation.

---

## 2. The Physics in Plain English

### 2.1 What is Rydberg facilitation?

You have a 2D lattice of atoms. Each atom can be in the ground state or a Rydberg excited state. An excited atom **facilitates** (helps) its neighbors to also become excited. This creates a chain reaction — an avalanche of excitations spreading outward from an initial seed.

### 2.2 The phase transition

Depending on how hard you drive the system (Rabi frequency Omega):
- **Low Omega** (weak driving): The avalanche dies out. All atoms eventually return to ground state. This is the **absorbing phase**.
- **High Omega** (strong driving): The avalanche sustains itself. A finite fraction of atoms stays excited forever. This is the **active phase**.
- **At some critical Omega_c**: The system is exactly at the boundary. This is a **non-equilibrium phase transition** in the "directed percolation" universality class.

### 2.3 What your data looks like

For each parameter set, you have ~500 stochastic trajectories. We average them to get `sz_mean[t]` — the mean excitation density over time. It always starts at +1.0 (one excited atom) and either decays to -1.0 (absorbing) or plateaus at some higher value (active).

### 2.4 What the GPT will do

Instead of simulating 500 trajectories for a new parameter set, the GPT will:
1. Read the Hamiltonian parameters as a "prompt"
2. Generate a synthetic `sz_mean[t]` trajectory
3. From this trajectory, we read off whether the system is absorbing or active
4. We sweep Omega to find the critical point

---

## 3. Data Foundation (Phase 0 — COMPLETE)

### 3.1 What we discovered

| Parameter | Value | How we know |
|---|---|---|
| Interaction V | 2000 | From `run_job.slurm`: V=$DELTA |
| Spontaneous emission Gamma | 1 | From `run_job.slurm`: GAMMA=1 |
| Detuning Delta | 2000 | Filename + run script |
| Dephasing gamma | 0.1 | Filename |
| Final time t_max | 1000 | TF=1000 in run script |
| Number of time steps | 400 | Measured from JLD2 files |
| Time step Delta t | ~2.5 | tSave[1] - tSave[0] |
| System dimension | 2D | All filenames say rho_ss_2D |
| Trajectories per point | 500 | NTRAJ=500 in run script |

### 3.2 Data coverage

| System Size | Lattice | Omega Coverage | # Points | Role |
|---|---|---|---|---|
| N=225 | 15x15 | 0–30 | 43 | **Training** (diverse) |
| N=400 | 20x20 | 0–27 | 42 | **Training** (diverse) |
| N=900 | 30x30 | 10–26 | 25 | **Training** (mixed) |
| N=100 | 10x10 | 10–13 | 21 | **Training** (critical) |
| N=1225 | 35x35 | 10–13 (fine) | 21 | **Test** (critical, fine) |
| N=1600 | 40x40 | 10–13 (fine) | 21 | **Validation** (critical) |
| N=2500 | 50x50 | 10–13 (fine) | 21 | **OOD Test** (size extrapolation) |
| N=3025 | 55x55 | 10–11.8 | 33 | **OOD Test** (size extrapolation) |
| N=3600 | 60x60 | 0–29 | 42 | **OOD Test** (size extrapolation) |
| N=4900 | 70x70 | 10–13 (fine) | 21 | **OOD Test** (size extrapolation) |

**Unreadable files:** 12 files (N=1089 all 6, N=3481 all 3, N=9 all 3) have JLD2 encoding issues. **Decision:** Skip them. We have 290 usable files.

### 3.3 The phase transition in your data

**N=1225 (35x35) — the critical region:**
- Omega = 10.0, 10.9: sz_mean ends at -1.0 → **absorbing phase**
- Omega = 11.5: sz_mean ends at -0.939 → **near critical**
- Omega = 12.7, 13.0: sz_mean ends at -0.8 → **active phase**

**Critical point estimate:** Omega_c ~ 11.2 for N=1225

**N=225 (15x15) — wide range:**
- Omega = 0–10: ends at -1.0 → **absorbing**
- Omega = 20, 30: ends at -0.65 → **active**

**Finite-size effect:** Smaller systems have lower Omega_c. This is real physics.

---

## 4. Phase 1: Data Engineering (Weeks 1–3)

### Week 1: JLD2 Parsing & Dataset Construction

**Goal:** Convert 290 JLD2 files into a clean, queryable dataset.

**Detailed tasks:**

1. **Write `data/parse_jld2.py`**
   - Use `h5py` to read all 290 JLD2 files
   - Extract `sz_mean` (shape 400) and `tSave` (shape 400)
   - Parse parameters from directory/filenames
   - Build a Pandas DataFrame with columns:
     ```
     omega, delta, gamma, V, Gamma, n_atoms, lattice_size, dimension,
     sz_mean (np.array shape 400), tSave (np.array shape 400)
     ```

2. **Write `data/dataset.py`**
   - PyTorch `Dataset` class
   - Returns a single sequence per parameter set
   - Sequence format: `[params_tokens..., sz_token_0, sz_token_1, ..., sz_token_399]`

3. **Discretization strategy**
   
   We must convert continuous `sz_mean` values into discrete tokens.
   
   **Option A: Uniform bins** (simpler)
   - Range: sz_mean spans [-1.0, +1.0]
   - 256 uniform bins -> token IDs 0–255
   - Bin width: ~0.008
   
   **Option B: K-means bins** (better distribution)
   - Run k-means on all sz_mean values across all files
   - 256 centroids -> token IDs 0–255
   - Better handles clustering near -1.0 (most values end up here)
   
   **Decision:** Start with Option A. If model struggles with the skewed distribution, switch to B.

4. **Parameter tokenization**

   Each parameter gets its own token(s):
   
   | Parameter | Range | Bins | Token IDs |
   |---|---|---|---|
   | Omega | 0–30 | 60 bins of width 0.5 | 10–69 |
   | N (system size) | 100–4900 | 10 discrete bins | 70–79 |
   | Lattice size L | 10–70 | derived from N | 80–89 |
   | Delta | fixed 2000 | 1 token | 90 |
   | gamma | fixed 0.1 | 1 token | 91 |
   | V | fixed 2000 | 1 token | 92 |
   | Gamma | fixed 1 | 1 token | 93 |
   | Dimension | 2D only | 1 token | 94 |
   | Special tokens | START, END, PAD | 3 tokens | 0–2 |
   | sz_mean bins | 256 bins | 256 tokens | 100–355 |
   
   **Total vocabulary: ~360 tokens**

5. **Sequence format**

   ```
   [START] [DIM=2D] [N=1225] [L=35] [Delta=2000] [gamma=0.1] [V=2000] [Gamma=1] [Omega=10.5]
   [sz_0] [sz_1] [sz_2] ... [sz_399] [END]
   ```
   
   Context length: 1 (START) + 8 (params) + 400 (time steps) + 1 (END) = **410 tokens**

6. **Train/Val/Test split (parameter-based, NOT random)**

   | Split | System Sizes | Omega Range | Purpose |
   |---|---|---|---|
   | **Train** | N=225, 400, 900, 100 | All available | Learn diverse dynamics |
   | **Validation** | N=1600 | All available (10–13) | Hyperparameter tuning |
   | **Test (interpolation)** | N=1225 | Even-indexed Omega (10.0, 10.3, ...) | Interpolation accuracy |
   | **Test (extrapolation)** | N=1225 | Odd-indexed Omega (10.15, 10.45, ...) | Extrapolation accuracy |
   | **OOD (size)** | N=2500, 3025, 3600, 4900 | All available | Zero-shot size generalization |

   **Why parameter-based split?** Random split would leak information. We must test on *unseen parameters* to claim generalization.

### Week 2: Data Quality Checks

**Goal:** Ensure the dataset is clean and ready for training.

1. **Visualize sample trajectories**
   - Plot sz_mean(t) for N=1225 at Omega = 10.0, 11.0, 12.0, 13.0
   - Confirm phase transition is visible
   - Save as reference figures

2. **Check for anomalies**
   - Any trajectories that don't start at +1.0?
   - Any trajectories with NaN or Inf?
   - Any trajectories that oscillate wildly?
   - Flag outliers

3. **Verify time grid consistency**
   - All files have tSave[0]=0, tSave[-1]=1000, nT=400
   - Confirm no exceptions

4. **Compute steady-state values**
   - rho_ss = mean of last 50 time points
   - Plot rho_ss vs Omega for each N
   - This is our "ground truth" phase diagram

5. **Build data loader**
   - Batch size: 64–128
   - Shuffle: True (within train split)
   - Collate: pad sequences to max length (410)

### Week 3: Baseline Models

**Goal:** Establish simple baselines before training the transformer.

1. **Interpolation baseline**
   - For a test point (N=1225, Omega=10.45), interpolate between nearest training points
   - Linear interpolation in Omega
   - Metric: MSE on sz_mean trajectory

2. **ODE baseline**
   - Fit a simple relaxation ODE: ds/dt = -(s - s_infty)/tau
   - Learn s_infty and tau per parameter set
   - This is a deterministic, memoryless baseline

3. **LSTM baseline**
   - Train a small LSTM on the same data
   - Same parameter conditioning
   - Compare to transformer later

**Why baselines?** Reviewers will ask "why not just interpolate?" We need numbers showing the transformer does better.

---

## 5. Phase 2: Model Architecture (Weeks 4–5)

### Week 4: Modify nanoGPT

**Goal:** Adapt Karpathy's nanoGPT for Hamiltonian-conditioned trajectory generation.

**What nanoGPT does now:**
- Input: token IDs (integers)
- Embedding: lookup table
- Transformer blocks: attention + MLP
- Output: next-token probability distribution
- Loss: cross-entropy

**What we need to change:**

1. **Vocabulary size:** Change from 65 (Shakespeare chars) to ~360 (our tokens)

2. **Context length:** Change from 1024 to 512 (our sequences are 410 tokens)

3. **Parameter conditioning:** 
   
   **Approach A (token-based, RECOMMENDED for Phase 1):**
   - Treat parameters as regular tokens
   - Prepend them to the sequence
   - The causal mask naturally prevents the model from "looking ahead" at future sz values
   - The model learns to attend to parameter tokens when predicting sz values
   
   **Approach B (continuous embedding):**
   - Add a small MLP: embed_params(Omega, N, ...) -> vector
   - Add this vector to every token embedding in the sequence
   - More flexible but harder to debug
   
   **Decision:** Use Approach A. Simpler, no architectural changes needed, and we can ablate parameter tokens to verify they matter.

4. **Model size**

   | Hyperparameter | Value | Rationale |
   |---|---|---|
   | n_layer | 6 | Deeper than LSTM, shallow enough to train fast |
   | n_head | 6 | Standard ratio |
   | n_embd | 384 | ~6M parameters, fits on single GPU |
   | dropout | 0.1 | Regularization |
   | block_size | 512 | Covers our 410-token sequences |
   | vocab_size | 360 | Our token space |

5. **Output head**
   - Linear layer: n_embd -> vocab_size
   - No change needed — same as language modeling
   - The model predicts the *next token*, which is either the next sz bin or [END]

### Week 5: Model Verification

**Goal:** Make sure the model can overfit a single example before training on full data.

1. **Overfitting test**
   - Train on a single trajectory (N=1225, Omega=11.5) for 10,000 steps
   - Loss should go to near-zero
   - Generated trajectory should match training trajectory exactly
   - If this fails, debug tokenization or architecture

2. **Parameter ablation**
   - Train two models:
     - Model A: Full sequence with parameters
     - Model B: Same data but parameter tokens replaced with random tokens
   - Model A should outperform Model B
   - This proves the model is actually using the Hamiltonian information

3. **Memory check**
   - Verify model fits in GPU memory
   - Batch size 64 should use ~4GB on A100
   - If OOM, reduce batch size or model size

---

## 6. Phase 3: Training (Weeks 6–8)

### Week 6: Full Training Run

**Training configuration:**

```python
# Optimizer
optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

# Learning rate schedule
warmup_steps = 2000
max_steps = 50000
lr = min_lr + (max_lr - min_lr) * cos((step - warmup) / (max_steps - warmup) * pi/2)

# Loss
loss = CrossEntropyLoss(logits, target_tokens)  # standard next-token prediction

# Mixed precision
Use torch.cuda.amp for faster training
```

**Monitoring:**
- Log train/val loss every 100 steps
- Log learning rate every 100 steps
- Save checkpoint every 5000 steps
- Track best validation loss

**Expected behavior:**
- Train loss should decrease smoothly
- Val loss should decrease then plateau
- If val loss increases -> overfitting, stop early
- Target final val loss: < 1.0 nats/token (cross-entropy)

### Week 7: Training Diagnostics

**Goal:** Understand what the model is learning.

1. **Loss curves**
   - Plot train/val loss vs steps
   - Identify if model is underfitting (both high) or overfitting (train low, val high)

2. **Per-token loss analysis**
   - Compute loss at each position in the sequence
   - Is loss higher at the beginning (harder to predict initial dynamics) or end (harder to predict steady-state)?

3. **Attention visualization**
   - Extract attention weights from a validation example
   - Do attention heads attend to parameter tokens?
   - Do they attend to recent time steps (short memory) or distant ones (long memory)?

4. **Checkpoint selection**
   - Choose the checkpoint with lowest validation loss
   - Not the last checkpoint — usually overfitted

### Week 8: Generation & Validation

**Goal:** Generate synthetic trajectories and compare to ground truth.

1. **Trajectory generation protocol**
   - For a test parameter set, feed parameter tokens + [START]
   - Sample next token from model distribution (temperature=1.0)
   - Append sampled token, feed back into model
   - Repeat until [END] token or max length
   - Convert token IDs back to continuous sz values (take bin centers)

2. **Generate 100 trajectories per test point**
   - This gives us an ensemble
   - We can compute mean and variance

3. **Visual comparison**
   - Plot true vs. generated trajectories side by side
   - Show 5 examples: absorbing, near-critical, active, plus interpolation/extrapolation

4. **Quantitative validation**
   - MSE between true and generated trajectories
   - MAE on steady-state value
   - Correlation coefficient between true and generated time series

---

## 7. Phase 4: Evaluation & Physics (Weeks 9–12)

### Week 9: Steady-State Prediction

**Goal:** Can the GPT predict whether a system is absorbing or active?

1. **Extract rho_ss from generated trajectories**
   - rho_ss_pred = mean of last 50 generated time steps
   - Compare to rho_ss_true from data

2. **Phase diagram plot**
   - Plot rho_ss vs Omega for each system size
   - True values (solid lines) vs predicted (dashed lines)
   - Should show the characteristic "knee" at Omega_c

3. **Classification accuracy**
   - Absorbing: rho_ss < -0.95
   - Active: rho_ss > -0.95
   - Compute precision/recall for phase classification

### Week 10: Critical Point Estimation

**Goal:** Find Omega_c(N) from GPT-generated data.

1. **Fitting procedure**
   - For each N, generate trajectories at 20–30 Omega values
   - Fit rho_ss(Omega) to a sigmoid: rho_ss = a + b / (1 + e^{-k(Omega - Omega_c)})
   - Extract Omega_c from fit

2. **Finite-size scaling**
   - Plot Omega_c(N) vs 1/L where L = sqrt(N)
   - Fit: Omega_c(L) = Omega_c_infty + A * L^{-1/nu}
   - Extract critical exponent nu
   - Compare to directed percolation: nu ~ 0.73 (2D)

3. **Critical exponent beta**
   - Near criticality: rho_ss - rho_ss_absorbing ~ (Omega - Omega_c)^beta
   - Fit log-log plot
   - Compare to DP: beta ~ 0.58 (2D)

### Week 11: Generalization Tests

**Goal:** Prove the model generalizes, not memorizes.

1. **Interpolation test**
   - Train on Omega = {10.0, 10.3, 10.6, ...}
   - Test on Omega = {10.15, 10.45, 10.75, ...}
   - Metric: MSE should be low

2. **Extrapolation test**
   - Train on Omega = 0–20
   - Test on Omega = 25, 30
   - Metric: MSE may be higher, but should capture trend

3. **Size extrapolation**
   - Train on N = 100, 225, 400, 900
   - Test on N = 1225, 1600, 2500, 4900
   - This is the **killer test**. If the model trained on small systems predicts large-system dynamics, it learned scale-invariant physics.

4. **Cross-dimension (if 1D/3D data exists)**
   - Train on 2D, test on 1D or 3D
   - (Note: your current data is all 2D, so this may not be possible without new data)

### Week 12: Baseline Comparison

**Goal:** Show the transformer beats simpler methods.

| Method | What it does | Expected MSE |
|---|---|---|
| **Constant prediction** | Predict rho_ss = training mean | High |
| **Linear interpolation** | Interpolate between nearest training points | Medium |
| **LSTM** | Recurrent network, same data | Medium-Low |
| **nanoGPT (ours)** | Transformer with attention | **Lowest** |

**Comparison metrics:**
- Trajectory MSE
- Steady-state MAE
- Critical point error |Omega_c_pred - Omega_c_true|

---

## 8. Phase 5: Spatial Data Generation (Weeks 13–15)

**Prerequisite:** Mean-field results must be strong enough to justify the extension.

**Goal:** Generate per-atom spatial trajectory data for selected parameters.

### Week 13: Simulation Setup

1. **Install Julia** (if not already available on your cluster)
2. **Select parameter subset:**
   - 2D systems: N = 225 (15x15), 900 (30x30), 1225 (35x35)
   - Omega values: 5 points spanning critical region
     - Absorbing: Omega = 10.0
     - Near-critical: Omega = 10.75, 11.5
     - Active: Omega = 12.25, 13.0
   - Trajectories: 50–100 per parameter set
   - Total: 3 sizes x 5 Omega x 100 traj = 1500 trajectories

3. **Modify `dp_perc.jl`**
   - Ensure output format matches what we need
   - Save `sz_atoms[k, t]` as main output
   - Add progress logging

### Week 14: Run Simulations

1. **Execute simulations**
   - Run on cluster or local machine
   - Parallelize across CPU cores
   - Estimated time: ~1–2 hours per (N, Omega) point on single CPU
   - Total: ~15–30 CPU-hours

2. **Monitor progress**
   - Check for failures or NaN outputs
   - Verify spatial patterns look physical

### Week 15: Data Verification

1. **Visual spot checks**
   - Plot 2D heatmaps of sz at t=0, t=100, t=200, t=400
   - Confirm excitation spreads from center
   - Confirm absorbing vs. active phases look different

2. **Compare to mean data**
   - Compute mean of spatial data -> should match existing `sz_mean`
   - This validates the new simulations

---

## 9. Phase 6: Spatiotemporal Model (Weeks 16–20)

### Week 16: Spatial Tokenization

**Goal:** Convert 2D spatial snapshots into token sequences.

1. **Patch encoding**
   - Divide LxL lattice into PxP patches
   - Each patch = average sz over PxP atoms
   - Example: N=1225 (35x35), P=5 -> 7x7 = 49 patches

2. **Hierarchical encoding (for large systems)**
   - N=2500 (50x50): P=10 -> 5x5 = 25 patches
   - Or multi-scale: coarse (10x10) + fine (5x5)

3. **Sequence format**
   ```
   [START] [DIM=2D] [N=1225] [L=35] [PATCH=5] [Omega=10.5] ...
   [TIME=0] [p00] [p01] ... [p66]
   [TIME=1] [p00] [p01] ... [p66]
   ...
   ```

4. **Context length estimate**
   - Params: ~10 tokens
   - Per snapshot: 1 (TIME) + 49 (patches) = 50 tokens
   - 400 snapshots: 20,000 tokens
   - **Problem:** 20K >> nanoGPT's 512 context limit

5. **Solutions for long context:**
   
   **Option A: Temporal subsampling**
   - Don't predict every time step
   - Predict every 10th step: 400 -> 40 snapshots
   - Context: 10 + 40 x 50 = 2010 tokens
   - Still too long
   
   **Option B: Compressed latent per snapshot**
   - Train a small CNN autoencoder
   - Encoder: 35x35 -> 64-dim latent vector
   - GPT predicts latent vectors autoregressively
   - Decoder: latent -> 35x35 reconstruction
   - Context: 10 + 400 x 1 = 410 tokens (OK)
   
   **Option C: Separate spatial and temporal models**
   - Model 1: Predict mean time series (already done in Phase 4)
   - Model 2: Predict spatial fluctuations conditioned on mean
   - More complex but tractable
   
   **Decision:** Option B (compressed latent) for full spatiotemporal, or skip to Option C if Option B fails.

### Week 17: Model Adaptation

1. **Add CNN encoder/decoder**
   - Encoder: Conv2d -> flatten -> linear -> latent (64 dims)
   - Decoder: latent -> linear -> reshape -> ConvTranspose2d
   - Pre-train autoencoder on spatial data (MSE reconstruction loss)

2. **Modify GPT**
   - Input: parameter tokens + latent vectors
   - Output: next latent vector
   - Keep same transformer architecture, just change token meaning

### Week 18: Training

1. **Pre-train autoencoder**
   - Train CNN to reconstruct spatial snapshots
   - Target MSE: < 0.01 per pixel

2. **Train GPT on latent sequences**
   - Same training protocol as Phase 3
   - Loss: MSE on latent vectors (regression, not classification)

### Week 19: Spatial Evaluation

1. **Pattern quality**
   - Decode predicted latent vectors -> 2D snapshots
   - Visual comparison: true vs. predicted at t=0, 100, 200, 400

2. **Spatial correlation function**
   - C(r) = <s_z^(i) s_z^(i+r)> - <s_z>^2
   - Compare true vs. predicted

3. **Front propagation**
   - Measure excitation front radius vs. time
   - Compare true vs. predicted velocity

### Week 20: Critical Exponents from Spatial Data

1. **Correlation length xi**
   - Fit C(r) ~ e^{-r/xi} at critical Omega
   - Compute xi for multiple system sizes

2. **Critical exponent nu**
   - xi ~ |Omega - Omega_c|^{-nu}
   - Fit log-log plot
   - Compare to DP: nu ~ 0.73

3. **Dynamic exponent z**
   - tau ~ xi^z where tau is relaxation time
   - Requires careful analysis

---

## 10. Phase 7: Interpretability (Weeks 21–22)

### Week 21: Attention Analysis

**Goal:** Understand what the transformer learned physically.

1. **Extract attention maps**
   - For a validation example, compute attention weights A^(l,h)_{ij}
   - l = layer, h = head, i = query position, j = key position

2. **Hypothesis 1: Parameter attention**
   - Do attention heads attend to parameter tokens when predicting dynamics?
   - Measure attention weight from sz tokens to parameter tokens
   - Expect: strong attention early in sequence

3. **Hypothesis 2: Temporal locality**
   - Do attention heads attend to recent time steps?
   - Measure attention decay with temporal distance
   - Expect: exponential decay (Markovian memory)

4. **Hypothesis 3: Critical slowing down**
   - Near criticality, correlation time diverges
   - Does attention range increase near Omega_c?
   - Compute effective attention range vs. Omega

### Week 22: Ablation Studies

1. **Head ablation**
   - Remove each attention head individually
   - Measure degradation in prediction accuracy
   - Identify "physical" heads vs. "redundant" heads

2. **Parameter ablation**
   - Replace parameter tokens with random values
   - Measure degradation
   - Confirms model uses parameters, not just memorizes

3. **Position ablation**
   - Randomize position embeddings
   - If performance drops, model learned temporal ordering

---

## 11. Phase 8: Paper Writing (Weeks 23–26)

### Week 23: Results Compilation

1. **Collect all figures:**
   - Fig 1: Phase diagram rho_ss(Omega, N) — true vs. predicted
   - Fig 2: Finite-size scaling Omega_c vs. 1/L
   - Fig 3: Trajectory comparisons (5 examples)
   - Fig 4: Generalization tests (interpolation, extrapolation, size)
   - Fig 5: Baseline comparison bar chart
   - Fig 6: Attention maps (if interpretability is strong)
   - Fig 7: Spatial patterns (if Phase 6 completed)

2. **Collect all tables:**
   - Table 1: Critical exponents comparison (ours vs. DP theory)
   - Table 2: Model architecture and hyperparameters
   - Table 3: Baseline comparison metrics

### Week 24: Draft Writing

**Target structure:**
1. **Introduction:** Motivation — non-equilibrium phase transitions are hard to simulate, foundation models can accelerate discovery
2. **Model:** Architecture, tokenization, training
3. **Data:** Description of Rydberg facilitation dynamics, simulation parameters
4. **Results:**
   - Mean-field predictions (Phase 4)
   - Critical point estimation
   - Finite-size scaling
   - Generalization tests
   - Spatial predictions (if Phase 6)
5. **Interpretability:** What attention heads learned
6. **Discussion:** Limitations, future work
7. **Methods:** Detailed training protocol, baselines

### Week 25: Internal Review

1. **Co-author feedback**
2. **Fix figures:** Improve aesthetics, ensure fonts are readable
3. **Supplementary material:** Additional trajectories, hyperparameter sweeps

### Week 26: Submission

1. **Target venue:** PRX Quantum or Physical Review Research
2. **Preprint:** arXiv simultaneous submission
3. **Code release:** GitHub repo with model weights and inference script

---

## 12. Hardware & Dependencies

### Hardware

| Task | Resource | Time |
|---|---|---|
| Data preprocessing | Laptop CPU | Hours |
| Mean-field training | 1x A100 (or equivalent) | 2–3 days |
| Spatial simulation | 20 CPU cores | 1–2 days |
| Spatial training | 1x A100 | 3–5 days |
| **Total GPU** | 1x A100 | ~1 week |

### Software Dependencies

```
Python 3.10+
PyTorch 2.0+ (with CUDA)
numpy, scipy, pandas, matplotlib
h5py (for reading JLD2)
torch.utils.tensorboard (for logging)
```

Optional:
```
Julia 1.9+ (for Phase 5 spatial simulation)
DifferentialEquations.jl, JLD2.jl (existing dependencies)
```

---

## 13. Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| GPT fails to generalize across system sizes | Medium | **Critical** | Use relative coordinates, scale-invariant features; if fails, scope paper to single-size prediction |
| Critical exponents don't match DP theory | Medium | High | Could indicate new universality class or finite-size effects; still publishable as methodology |
| Spatial simulation too slow/buggy | Medium | Medium | Skip Phase 5–6; mean-field paper is still standalone |
| Attention analysis inconclusive | Medium | Medium | Frame as exploratory; null results are informative |
| Reviewers say "just curve fitting" | Medium | High | Emphasize generalization (unseen sizes/parameters) and physical interpretability |
| NOQS authors publish dissipative extension | Medium | **Critical** | Move fast; our facilitation physics is distinct from their driven dynamics |
| JLD2 reader issues for some files | Already happened | Low | Skip 12 problematic files; 290 files is sufficient |

---

## Glossary

| Term | Meaning |
|---|---|
| **TWA** | Truncated Wigner Approximation — semiclassical stochastic method |
| **sz_mean** | Mean expectation value of sigma_z over all atoms |
| **Absorbing phase** | System decays to all-ground state; excitations die out |
| **Active phase** | System sustains finite excitation density |
| **Directed percolation** | Universality class for non-equilibrium phase transitions with absorbing states |
| **OOD** | Out-of-distribution (unseen during training) |
| **Token** | Discrete integer representing a value or parameter bin |
| **Context length** | Maximum sequence length the transformer can process |
| **Patch encoding** | Dividing spatial lattice into smaller regions for tokenization |

---

## Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-24 | Approved Hybrid approach | De-risked, incremental, builds on existing data |
| 2026-04-24 | Skip 12 problematic JLD2 files | 96% of data is readable; sufficient for training |
| 2026-04-24 | Use token-based parameter conditioning | Simpler than continuous embedding; easier to debug |
| 2026-04-24 | Start with uniform discretization | Simplest; switch to k-means if needed |
| 2026-04-24 | Context length 512 | Accommodates 410-token sequences with headroom |

---

*Plan version 1.0 — 2026-04-24*
*Next review: End of Phase 1 (Week 3)*
