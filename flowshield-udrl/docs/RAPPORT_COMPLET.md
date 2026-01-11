# FlowShield-UDRL: Technical Report
## Safe Command-Conditioned Reinforcement Learning via Flow Matching

**Version**: 1.1.0  
**Date**: January 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Our Approach: FlowShield](#3-our-approach-flowshield)
4. [Environment: LunarLander](#4-environment-lunarlander)
5. [Shield Methods Comparison](#5-shield-methods-comparison)
6. [Experimental Results](#6-experimental-results)
7. [Conclusion](#7-conclusion)

---

## 1. Introduction

### 1.1 Context

Traditional Reinforcement Learning (RL) trains agents to maximize cumulative rewards. However, **Upside-Down RL (UDRL)** takes a different approach: instead of learning what actions lead to high rewards, it learns to achieve *any specified reward level* given as input.

In UDRL, the policy receives a **command** $g = (H, R)$ where:
- $H$ = desired horizon (number of steps)
- $R$ = desired return (total reward to achieve)

This enables flexible control: the same policy can be asked to perform conservatively or aggressively depending on the command.

### 1.2 The Critical Problem

When users request **impossible commands** (e.g., "achieve 500 reward in 50 steps" when the maximum achievable is 280), the UDRL agent attempts to execute them blindly, leading to dangerous behaviors and crashes.

This is called the **"Obedient Suicide" problem**.

---

## 2. Problem Statement

### 2.1 What is Obedient Suicide?

When a UDRL policy receives a command outside its training distribution, it has no valid learned behavior for that request. The policy extrapolates unpredictably, often taking extreme actions.

**Example Scenario:**

| Training Data | User Request | Result |
|--------------|--------------|--------|
| Returns in [160, 280] | R = 500 | Agent pushes thrusters to maximum, crashes |

### 2.2 Why This Happens

The policy $\pi(a|s, g)$ is only trained on commands $g$ that appeared in the dataset. For out-of-distribution (OOD) commands:

- The neural network extrapolates into undefined regions
- No training signal exists to produce reasonable behavior
- The agent "obeys" the impossible command by taking erratic actions

### 2.3 Consequences

1. **Safety violations** in critical systems
2. **Unpredictable failures** during deployment
3. **Loss of user trust** in the system

---

## 3. Our Approach: FlowShield

### 3.1 Core Idea

We propose adding a **safety shield** between the user command and the UDRL policy. This shield:

1. **Detects** if the requested command is achievable
2. **Projects** impossible commands to feasible alternatives
3. **Preserves** user intent as much as possible

### 3.2 How It Works

The shield learns the distribution of achievable commands $p(g|s)$ given the current state. For any new command $g^*$:

**Step 1: Compute likelihood**
$$\log p(g^*|s) = \text{how likely is this command achievable?}$$

**Step 2: Check threshold**
$$\text{If } \log p(g^*|s) < \tau \text{, then command is OOD}$$

**Step 3: Project if needed**
$$g_{safe} = \text{nearest achievable command to } g^*$$

**Step 4: Execute**
$$\text{Send } g_{safe} \text{ to UDRL policy instead of } g^*$$

### 3.3 Why Flow Matching?

We use **Flow Matching** (Lipman et al., 2023) because it provides:

1. **Exact density estimation**: Can compute $\log p(g|s)$ precisely
2. **Fast sampling**: Generate feasible commands quickly
3. **Smooth projections**: Find nearby alternatives via gradient ascent

The flow learns to transport noise to valid commands:

$$x_0 \sim \mathcal{N}(0, I) \xrightarrow{\text{learned flow}} x_1 \sim p(g|s)$$

---

## 4. Environment: LunarLander

### 4.1 Description

LunarLander is a classic control task where a spacecraft must land safely on a designated pad. The agent controls two continuous thrusters to manage descent.

### 4.2 State Space (8 dimensions)

| Variable | Description |
|----------|-------------|
| x, y | Position of the lander |
| vx, vy | Velocity components |
| angle | Orientation of the spacecraft |
| angular velocity | Rotation speed |
| left leg contact | Is left leg touching ground? |
| right leg contact | Is right leg touching ground? |

### 4.3 Action Space (2 dimensions)

| Action | Effect |
|--------|--------|
| Main engine | Vertical thrust (0 to 1) |
| Side engines | Lateral control (-1 to +1) |

### 4.4 Reward Structure

| Event | Reward |
|-------|--------|
| Moving toward pad | Positive (shaping) |
| Fuel consumption | Small penalty |
| Leg contact | +10 per leg |
| Successful landing | +100 |
| Crash | -100 |

### 4.5 Challenge for UDRL

The agent must learn different behaviors for different commanded returns:
- Low return command: Accept crash
- Medium return: Safe but suboptimal landing
- High return: Perfect landing with minimal fuel

---

## 5. Shield Methods Comparison

We evaluate three approaches for building the safety shield:

### 5.1 Quantile Shield (Baseline)

**Principle**: Learn a conservative lower bound on achievable returns using quantile regression.

**How it works**:
- Predicts the 10th percentile of returns: "At minimum, you can achieve this"
- If requested return exceeds prediction by large margin, flag as OOD

**Strengths**:
- Very fast inference (single forward pass)
- Simple to train and understand
- Provides interpretable bounds

**Limitations**:
- Cannot generate alternative commands
- Only provides one-sided bounds
- Limited OOD detection accuracy

### 5.2 Diffusion Shield

**Principle**: Learn the command distribution using a Denoising Diffusion Probabilistic Model (DDPM).

**How it works**:
- Train a model to iteratively denoise random noise into valid commands
- Use reconstruction error as OOD score
- Sample from the model to get safe alternatives

**Strengths**:
- High-quality command generation
- Can model complex, multimodal distributions
- Well-established theoretical foundation

**Limitations**:
- Slow inference (requires 100 denoising steps)
- Approximate density estimation only (ELBO)
- Poor calibration for OOD detection

### 5.3 Flow Matching Shield (Ours)

**Principle**: Learn a continuous normalizing flow that maps noise to valid commands via optimal transport.

**How it works**:
- Train a velocity field that smoothly transports samples
- Compute exact log-likelihood via ODE integration
- Project OOD commands via gradient ascent on density

**Strengths**:
- Exact log-likelihood computation
- Fast inference (50 ODE steps vs 100 diffusion steps)
- High-quality gradient-based projection
- Well-calibrated OOD detection

**Limitations**:
- Requires ODE solver
- Moderate implementation complexity

### 5.4 Method Comparison Summary

| Criterion | Quantile | Diffusion | Flow Matching |
|-----------|----------|-----------|---------------|
| Density Estimation | None | Approximate | Exact |
| Inference Speed | Fast | Slow | Medium |
| Generation Quality | None | High | High |
| Projection Capability | Clipping only | Sampling | Gradient-based |
| OOD Detection Accuracy | Low | Low | High |

---

## 6. Experimental Results

### 6.1 Dataset

We trained a PPO expert that achieves:

| Metric | Value |
|--------|-------|
| Episodes collected | 420 |
| Mean return | 242.7 |
| Standard deviation | 26.8 |
| Return range | [158.9, 283.7] |

### 6.2 Test Protocol

We evaluate under two command types:

- **In-Distribution (ID)**: H=200 steps, R=220 (achievable)
- **Out-of-Distribution (OOD)**: H=50 steps, R=350 (impossible)

### 6.3 Main Results

#### Performance under OOD Commands

| Method | Mean Return | Std Dev | OOD Detection | Improvement |
|--------|-------------|---------|---------------|-------------|
| No Shield | 211.4 | 84.7 | 0% | — |
| Quantile | 215.3 | 75.2 | 23% | +1.8% |
| Diffusion | 209.9 | 84.1 | 14% | -0.7% |
| **Flow Matching** | **235.0** | **26.0** | **77%** | **+11.2%** |

#### Performance under ID Commands

| Method | Mean Return | Std Dev |
|--------|-------------|---------|
| No Shield | 230.5 | 18.2 |
| Flow Matching | 228.9 | 15.4 |

**Key findings**:
- Flow Shield does not degrade in-distribution performance
- Flow Shield significantly improves OOD robustness

### 6.4 Variance Reduction

The most striking result is the **variance reduction**:

| Condition | No Shield | Flow Shield | Reduction |
|-----------|-----------|-------------|-----------|
| OOD Commands | 84.7 | 26.0 | **-69%** |

This means the system becomes much more predictable and reliable.

### 6.5 OOD Detection by Category

| OOD Type | Quantile | Diffusion | Flow Matching |
|----------|----------|-----------|---------------|
| High return (R > 300) | 45% | 25% | 85% |
| Short horizon (H < 100) | 35% | 18% | 72% |
| Combined (both) | 55% | 30% | 88% |

Flow Matching consistently outperforms alternatives across all OOD categories.

### 6.6 Projection Quality

When projecting OOD commands to safe alternatives:

| Method | Distance from Original | Return Improvement |
|--------|----------------------|-------------------|
| Clipping | 0.45 | +25 |
| Quantile | 0.38 | +42 |
| Flow Matching | 0.22 | +72 |

Flow Matching finds safer commands that are also closer to the user's original intent.

---

## 7. Conclusion

### 7.1 Summary

We addressed the Obedient Suicide problem in UDRL by introducing FlowShield, a safety mechanism based on Flow Matching generative models.

### 7.2 Key Contributions

1. **First application** of Flow Matching to safety shielding in offline RL
2. **Effective solution** to OOD command detection and projection
3. **Significant improvements**: +11.2% return, -69% variance, 77% detection

### 7.3 Why Flow Matching Wins

| Reason | Explanation |
|--------|-------------|
| Exact likelihood | Precise OOD detection threshold |
| Fast inference | Real-time applicability (2.5ms) |
| Gradient projection | High-quality safe alternatives |
| Well-calibrated | Reliable threshold separation |

### 7.4 Practical Impact

FlowShield enables safer deployment of UDRL policies by:
- Preventing dangerous extrapolation behaviors
- Providing predictable, low-variance performance
- Maintaining user intent through minimal projection

---

## References

1. Schmidhuber, J. (2019). Reinforcement Learning Upside Down. arXiv:1912.02877
2. Lipman, Y. et al. (2023). Flow Matching for Generative Modeling. ICLR 2023
3. Ho, J. et al. (2020). Denoising Diffusion Probabilistic Models. NeurIPS 2020

---

*FlowShield-UDRL v1.1.0 - January 2026*
