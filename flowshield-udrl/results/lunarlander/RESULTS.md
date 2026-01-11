# Results Analysis Report

**FlowShield-UDRL - LunarLander-v3 Experiments**  
**Date**: January 10, 2026  
**Version**: 1.1.0

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Dataset Analysis](#2-dataset-analysis)
3. [Training Results](#3-training-results)
4. [Performance Comparison](#4-performance-comparison)
5. [Visualization Guide](#5-visualization-guide)
6. [Technical Insights](#6-technical-insights)
7. [Recommendations](#7-recommendations)

---

## 1. Executive Summary

### Key Achievements

| Metric                | Before (v1.0) | After (v1.1) | Improvement |
| --------------------- | ------------- | ------------ | ----------- |
| Training Data Quality | R = -40       | R = 242.7    | **+700%**   |
| OOD Performance       | ~50           | 235.0        | **+370%**   |
| Variance (OOD)        | ~150          | 26.0         | **-83%**    |
| OOD Detection Rate    | 0%            | 77%          | **+77pp**   |

### Main Finding

**Flow Matching Shield** significantly outperforms baseline and Diffusion Shield on OOD commands:

- **+11.2%** mean return improvement
- **-69%** variance reduction
- **77%** OOD detection accuracy

---

## 2. Dataset Analysis

### 2.1 Expert Data Distribution

**File**: `results/lunarlander/figures/expert_data_distribution.png`

![Expert Data Distribution](figures/expert_data_distribution.png)

**Interpretation:**

- **Distribution Shape**: Near-Gaussian centered at R≈240
- **Tight Clustering**: 95% of episodes within [189, 296]
- **Quality Confirmation**: No failed episodes (all R > 150)

**Observed Trends:**

- Expert achieves consistent high returns
- Low variance indicates reliable policy
- All trajectories represent successful landings

**Implications:**

- [POSITIVE] High-quality training data for UDRL
- [NOTE] Limited diversity may affect command-following

---

### 2.2 Expert Trajectories

**File**: `results/lunarlander/figures/expert_trajectories.png`

**Interpretation:**

- Trajectories show controlled descent patterns
- Consistent landing at pad center
- Low velocity at touchdown

**Strengths:**

- Stable hovering behavior
- Efficient fuel usage
- Precise landing

---

### 2.3 UDRL Limitation Analysis

**File**: `results/lunarlander/figures/udrl_limitation_analysis.png`

![UDRL Limitation](figures/udrl_limitation_analysis.png)

**Key Insight:**

This visualization explains why UDRL doesn't perfectly follow commands:

1. **Data Homogeneity**: σ = 26.8 (very low variance)
2. **Behavioral Uniformity**: All trajectories perform the SAME action (land correctly)
3. **Command Ambiguity**: Commands R=200, R=220, R=250 all lead to similar behavior

**Technical Explanation:**

```
If D = {(s,a,g) : g ∈ [220, 260]}  (narrow range)
Then π(a|s, g) ≈ π(a|s)  (ignores command!)
Because all g values map to same optimal behavior
```

**Recommendation:**
To achieve true command-following, train on DIVERSE data:

- R ∈ [50-100]: Controlled crashes
- R ∈ [150-200]: Suboptimal landings
- R ∈ [250+]: Perfect landings

---

## 3. Training Results

### 3.1 Policy Training

**File**: `results/lunarlander/figures/policy_training.png`

![Policy Training](figures/policy_training.png)

**Metrics:**
| Epoch | Train Loss | Val Loss | Status |
|-------|------------|----------|--------|
| 0 | 2.5 | 2.4 | Starting |
| 25 | 1.2 | 1.1 | Rapid descent |
| 50 | 0.85 | 0.82 | Converging |
| 75 | 0.78 | 0.80 | Early stopped |

**Analysis:**

- **Convergence**: Smooth, monotonic decrease
- **No Overfitting**: Train/val gap < 0.05
- **Early Stopping**: Triggered at epoch ~75 (patience=15)

**Strengths:**

- Efficient training (<100 epochs)
- Good generalization

---

### 3.2 Flow Shield Training

**File**: `results/lunarlander/figures/flow_training.png`

![Flow Training](figures/flow_training.png)

**Metrics:**
| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
| 0 | 0.8 | 0.75 |
| 30 | 0.25 | 0.22 |
| 60 | 0.15 | 0.14 |

**Analysis:**

- **Fast Convergence**: ~30 epochs to reach plateau
- **Stable Training**: No oscillations
- **Low Final Loss**: 0.14-0.15 (excellent velocity field learning)

**Technical Note:**
Flow Matching loss measures velocity field prediction error:

```
L = E[||v_θ(g_t, s, t) - v_target||²]
```

---

### 3.3 Quantile Shield Training

**File**: `results/lunarlander/figures/quantile_training.png`

![Quantile Training](figures/quantile_training.png)

**Metrics:**

- Final pinball loss: ~15 (on return scale [0-300])
- τ = 0.1 (10th percentile, conservative)

**Analysis:**

- Learns lower bound of achievable returns
- Useful for rejection, limited for projection

---

### 3.4 Diffusion Shield Training

**File**: `results/lunarlander/figures/diffusion_training.png`

![Diffusion Training](figures/diffusion_training.png)

**Metrics:**

- Denoising loss: ~0.3
- T = 100 diffusion steps

**Analysis:**

- Higher loss than Flow (0.3 vs 0.15)
- Requires iterative denoising (slower inference)

---

## 4. Performance Comparison

### 4.1 Final Comparison

**File**: `results/lunarlander/figures/final_comparison.png`

![Final Comparison](figures/final_comparison.png)

**Results Table:**

| Method    | ID Return (R=220) | OOD Return (R=350) | OOD Detection |
| --------- | ----------------- | ------------------ | ------------- |
| No Shield | 230.5 ± 18.2      | 211.4 ± 84.7       | 0%            |
| Quantile  | 228.1 ± 19.5      | 215.3 ± 75.2       | 23%           |
| Diffusion | 229.8 ± 17.8      | 209.9 ± 84.1       | 14%           |
| **Flow**  | **228.9 ± 15.4**  | **235.0 ± 26.0**   | **77%**       |

**Key Observations:**

1. **In-Distribution (ID):**
   - All methods perform similarly (~228-230)
   - Shields don't degrade ID performance
2. **Out-of-Distribution (OOD):**

   - **Flow Shield dominates**: +11.2% vs no shield
   - **Variance dramatically reduced**: 26.0 vs 84.7
   - Other shields provide minimal improvement

3. **OOD Detection:**
   - Flow: 77% detection rate
   - Quantile: 23% (only catches extreme OOD)
   - Diffusion: 14% (poor calibration)

**Statistical Significance:**

- Flow vs No Shield: p < 0.01 (significant)
- Flow vs Diffusion: p < 0.05 (significant)

---

### 4.2 Shield Comparison Projection

**File**: `results/lunarlander/figures/shield_comparison_projection.png`

![Shield Comparison](figures/shield_comparison_projection.png)

**Interpretation:**

- Visualizes how each shield projects OOD commands
- Flow Shield projects to manifold boundary
- Quantile clips to conservative estimate

---

### 4.3 OOD Detection Visualizations

**Files**:

- `results/lunarlander/figures/ood_quantile.png`
- `results/lunarlander/figures/ood_diffusion.png`

**Quantile OOD Detection:**

- Uses threshold on predicted return bound
- Simple but effective for extreme OOD
- Limited for near-boundary commands

**Diffusion OOD Detection:**

- Uses reconstruction error
- Noisy signal (high variance)
- Poor discrimination power

---

## 5. Visualization Guide

### Figures Inventory

| Filename                           | Description              | Key Insight            |
| ---------------------------------- | ------------------------ | ---------------------- |
| `final_comparison.png`             | Main results comparison  | Flow Shield wins       |
| `policy_training.png`              | UDRL training curves     | Clean convergence      |
| `flow_training.png`                | Flow shield training     | Fast convergence       |
| `quantile_training.png`            | Quantile training        | Stable learning        |
| `diffusion_training.png`           | Diffusion training       | Slower convergence     |
| `expert_data_distribution.png`     | Data quality             | High R, low variance   |
| `expert_trajectories.png`          | Trajectory visualization | Consistent behavior    |
| `expert_policy_episode.gif`        | Animation of expert      | Smooth landing         |
| `expert_policy_keyframes.png`      | Key moments              | Landing sequence       |
| `shield_comparison_projection.png` | Projection comparison    | Flow is best           |
| `ood_quantile.png`                 | Quantile OOD analysis    | Conservative bounds    |
| `ood_diffusion.png`                | Diffusion OOD analysis   | Noisy detection        |
| `udrl_limitation_analysis.png`     | UDRL limitation          | Data homogeneity issue |

---

## 6. Technical Insights

### 6.1 Why Flow Matching Works

1. **Continuous Mapping**: Learns smooth transport from noise to commands
2. **Density Estimation**: Provides actual log p(g|s) values
3. **Efficient Projection**: Single ODE solve vs iterative denoising
4. **Better Calibration**: Threshold naturally separates ID/OOD

### 6.2 Why Diffusion Underperforms

1. **Discrete Steps**: T=100 steps adds noise
2. **Reconstruction Focus**: Not optimized for density estimation
3. **Slow Inference**: 100x more forward passes than Flow

### 6.3 Quantile Limitations

1. **No Projection**: Only rejection, no safe alternative
2. **Single Bound**: τ=0.1 gives one value, not distribution
3. **Conservative Bias**: May reject valid commands

### 6.4 Architecture Choices

| Decision                     | Rationale                      |
| ---------------------------- | ------------------------------ |
| ResidualBlock in Policy      | Better gradient flow           |
| LayerNorm (not BatchNorm)    | Works with small batches       |
| InputNormalizer (online)     | Handles unseen states          |
| CosineScheduler + Warmup     | Stable training, better optima |
| Early Stopping (patience=15) | Prevents overfitting           |

---

## 7. Recommendations

### 7.1 Immediate Improvements

1. **Diverse Training Data**

   ```bash
   # Train agents with different reward scales
   python scripts/collect_expert_data.py --reward-scale 0.5 --output data/medium.npz
   python scripts/collect_expert_data.py --reward-scale 0.2 --output data/low.npz
   # Combine with expert data for diverse dataset
   ```

2. **Ensemble Shields**

   - Combine Flow + Quantile for robust detection
   - Use Quantile as backup when Flow is uncertain

3. **Threshold Tuning**
   - Current: calibrated at 95th percentile
   - Consider: adaptive threshold based on state

### 7.2 Future Directions

1. **Multi-Environment Testing**

   - Highway-Env (more complex state space)
   - Humanoid locomotion (high-dimensional action)

2. **Online Adaptation**

   - Fine-tune shield during deployment
   - Update threshold based on observed failures

3. **Theoretical Analysis**
   - Prove OOD detection guarantees
   - Characterize projection optimality

### 7.3 Publication Readiness

| Component      | Status      | Action Needed              |
| -------------- | ----------- | -------------------------- |
| Core Algorithm | Complete    | -                          |
| Experiments    | Complete    | Add Highway-Env            |
| Baselines      | Complete    | Add VAE baseline           |
| Ablations      | Partial     | Add architecture ablations |
| Writing        | Not started | Draft paper                |

---

## Appendix

### A. Command Reference

```bash
# Full training pipeline
python scripts/train_policy.py --data data/lunarlander_expert.npz --epochs 100
python scripts/train_flow.py --data data/lunarlander_expert.npz --epochs 100
python scripts/train_quantile.py --data data/lunarlander_expert.npz --epochs 100
python scripts/train_diffusion.py --data data/lunarlander_expert.npz --epochs 100

# Evaluation
python scripts/evaluate_models.py --env lunarlander --data data/lunarlander_expert.npz

# Quick experiment
python scripts/run_experiments.py --env lunarlander --quick
```

### B. Model Checkpoints

| Model       | File                  | Size   | Performance |
| ----------- | --------------------- | ------ | ----------- |
| UDRL Policy | `policy.pt`           | 0.8 MB | R=230 (ID)  |
| Flow Shield | `flow_shield.pt`      | 1.2 MB | 77% OOD det |
| Quantile    | `quantile_shield.pt`  | 0.6 MB | 23% OOD det |
| Diffusion   | `diffusion_shield.pt` | 1.0 MB | 14% OOD det |
| PPO Expert  | `ppo_lunarlander.zip` | 2.1 MB | R=242.7     |

### C. Hardware Used

- **GPU**: NVIDIA RTX (CUDA enabled)
- **RAM**: 16 GB
- **Training Time**: ~2 hours (all models)

---

_Report generated: January 10, 2026_
_FlowShield-UDRL v1.1.0_
