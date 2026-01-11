<div align="center">

# 🛡️ FlowShield-UDRL

### Safe Command-Conditioned Reinforcement Learning via Flow Matching

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-enabled-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

---

### 🎯 Preventing Catastrophic Failures in Command-Conditioned RL

<img src="results/final/gifs/shield_demo/OOD_crash_demo.gif" alt="Shield Demo" width="900"/>

<sub>**Without Shield**: Agent crashes (-106) | **With Shield**: Safe landing (+289) ✅</sub>

[📊 Results](#-key-results) •
[🎬 Demos](#-visual-demonstrations) •
[🚀 Install](#-installation) •
[⚙️ Quick Start](#-quick-start) •
[🧠 Methodology](#-methodology)

</div>

---

## 📋 Table of Contents

- [🔬 Overview](#-overview)
- [📊 Key Results](#-key-results)
- [🎬 Visual Demonstrations](#-visual-demonstrations)
- [🚀 Installation](#-installation)
- [⚙️ Quick Start](#-quick-start)
- [📁 Project Structure](#-project-structure)
- [🧠 Methodology](#-methodology)
- [🧪 Experiments](#-experiments)
- [📈 Results Analysis](#-results-analysis)

---

## 🔬 Overview

### The Problem: "Obedient Suicide" in UDRL

**Upside-Down Reinforcement Learning (UDRL)** trains policies conditioned on desired outcomes (horizon, return). However, when users request **impossible or out-of-distribution (OOD) commands**, the agent blindly attempts execution, leading to:

- ⚠️ **Erratic, dangerous behavior**
- 💥 **System crashes and safety violations**  
- ❌ **Failure to achieve reasonable performance**

**Real-world Impact**: In safety-critical applications (robotics, autonomous systems), blindly following impossible commands can cause physical damage or mission failure.

### Our Solution: Flow Matching Safety Shield

We leverage **Flow Matching** (Lipman et al., 2023) — a state-of-the-art generative modeling technique — to create a safety shield that:

1. 📊 **Models** the distribution `p(g|s)` of achievable commands given state
2. 🔍 **Detects** OOD commands via log-likelihood estimation
3. 🛡️ **Projects** unsafe commands onto the manifold of achievable commands

### Key Innovation

```
Flow Matching + UDRL Safety = First application to Safe Offline RL
```

This work addresses a **critical research gap** in command-conditioned reinforcement learning by providing a principled, density-based approach to safety.

---

## 📊 Key Results

### LunarLander-v3 (Continuous Control) — Benchmark Results

| Method               | Mean Return | Std Dev  | OOD Detection | Improvement |
| -------------------- | ----------- | -------- | ------------- | ----------- |
| No Shield (Baseline) | 211.4       | 84.7     | 0%            | -           |
| Diffusion Shield     | 209.9       | 84.1     | 14%           | -0.7%       |
| Quantile Shield      | 218.3       | 62.4     | 45%           | +3.3%       |
| **Flow Shield** ⭐    | **235.0**   | **26.0** | **77%**       | **+11.2%**  |

#### 🏆 Highlights

- ✅ **+11.2%** improvement in mean return vs. unprotected baseline
- ✅ **-69%** reduction in variance (26.0 vs 84.7) — dramatically more reliable
- ✅ **77%** OOD command detection rate — catches most impossible requests
- ✅ **Seamless integration** — works with any pre-trained UDRL policy

#### Visual Comparison

| Scenario                       | No Shield           | With Flow Shield    |
| ------------------------------ | ------------------- | ------------------- |
| **OOD Extreme** (H=5, R=500)   | Crash (return -106) | Safe landing (+289) |
| **Moderate OOD** (H=50, R=350) | Crash (return -165) | Safe landing (+267) |
| **In-Distribution** (H=200, R=220) | Landing (+230)  | Landing (+231)      |

---

## 🎬 Visual Demonstrations

### Extreme OOD Command: H=5, R=500

<div align="center">
<img src="results/final/gifs/shield_demo/OOD_extreme_demo.gif" alt="OOD Extreme Demo" width="800"/>
<br/>
<sub><b>Left</b>: Crash -1199 | <b>Right</b>: Safe landing +267 ✅</sub>
</div>

### Moderate OOD Command: H=50, R=350

<div align="center">
<img src="results/final/gifs/shield_demo/OOD_moderate_demo.gif" alt="OOD Moderate Demo" width="800"/>
<br/>
<sub><b>Left</b>: Crash -165 | <b>Right</b>: Safe landing +267 ✅</sub>
</div>

### In-Distribution Command: H=200, R=220

<div align="center">
<img src="results/final/gifs/shield_demo/ID_demo.gif" alt="ID Demo" width="800"/>
<br/>
<sub><b>Left</b>: Landing +231 | <b>Right</b>: Landing +231 ✅ (Shield maintains performance)</sub>
</div>

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup

```bash
# Clone repository
git clone https://github.com/MerlinMaven/FlowShield_UDRL.git
cd FlowShield_UDRL/flowshield-udrl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Dependencies

```
torch>=2.0.0
gymnasium>=0.29.0
stable-baselines3>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
torchdyn>=1.0.0
tqdm>=4.65.0
```

---

## Quick Start

### Option 1: Evaluate Pre-trained Models (Recommended)

```bash
# Evaluate existing models (no training needed)
python scripts/evaluate_models.py --env lunarlander --data data/lunarlander_expert.npz
```

### Option 2: Train from Scratch

```bash
# Step 1: Train UDRL Policy
python scripts/train_policy.py --data data/lunarlander_expert.npz --epochs 100

# Step 2: Train Safety Shields
python scripts/train_flow.py --data data/lunarlander_expert.npz --epochs 100
python scripts/train_quantile.py --data data/lunarlander_expert.npz --epochs 100
python scripts/train_diffusion.py --data data/lunarlander_expert.npz --epochs 100

# Step 3: Evaluate
python scripts/evaluate_models.py --env lunarlander --data data/lunarlander_expert.npz
```

### Option 3: Generate New Expert Data

```bash
# Train PPO expert (500k timesteps)
python scripts/collect_expert_data.py --train-expert --timesteps 500000

# Generate trajectories
python scripts/collect_expert_data.py --n-episodes 500 --output data/my_expert.npz
```

### Training Options

| Option           | Description             | Default |
| ---------------- | ----------------------- | ------- |
| `--epochs N`     | Training epochs         | 100     |
| `--patience N`   | Early stopping patience | 15      |
| `--seed N`       | Random seed             | 42      |
| `--device`       | Force cpu/cuda          | auto    |
| `--no-scheduler` | Disable LR scheduler    | False   |

---

## Project Structure

```
flowshield-udrl/
├── data/                             # Datasets
│   └── lunarlander_expert.npz        # PPO expert (420 ep, R=242.7)
│
├── results/lunarlander/              # Experiment outputs
│   ├── figures/                      # Visualizations
│   ├── models/                       # Trained models
│   └── metrics/                      # Evaluation results
│
├── scripts/                          # Executable scripts
│   ├── models.py                     # Core model definitions
│   ├── train_policy.py               # UDRL training
│   ├── train_flow.py                 # Flow shield training
│   ├── evaluate_models.py            # Evaluation pipeline
│   └── collect_expert_data.py        # Data generation
│
├── src/                              # Source library
│   ├── models/                       # Neural networks
│   ├── data/                         # Dataset handling
│   ├── envs/                         # Environment wrappers
│   ├── training/                     # Training loops
│   └── evaluation/                   # Metrics & visualization
│
├── tests/                            # Unit tests
├── notebooks/                        # Jupyter tutorials
├── docs/                             # Sphinx documentation
└── requirements.txt                  # Dependencies
```

---

## Methodology

### 1. UDRL Policy

Supervised learning on command-conditioned trajectories:

```
L_BC = -E[(s,a,g) ~ D][log π_θ(a|s, g)]
```

where `g = (H, R)` represents horizon and return-to-go.

### 2. Flow Matching Shield

Learn optimal transport from noise to command distribution:

```
dg_t/dt = v_θ(g_t, s, t),  t ∈ [0, 1]
```

**Capabilities:**

- **Density estimation**: `log p(g|s)` via ODE integration
- **Sampling**: Draw from `p(g|s)` for command generation
- **Projection**: Map OOD commands to manifold boundary

### 3. OOD Detection and Projection

```python
# Pseudocode
log_prob = flow_shield.log_prob(command, state)
if log_prob < threshold:
    safe_command = flow_shield.project(command, state)
    action = policy(state, safe_command)
else:
    action = policy(state, command)
```

### 4. Architecture Details

| Component        | Architecture           | Parameters |
| ---------------- | ---------------------- | ---------- |
| Policy           | MLP 256-256 + ResBlock | ~200K      |
| Flow Shield      | MLP 256-256-256        | ~300K      |
| Quantile Shield  | MLP 256-256            | ~150K      |
| Diffusion Shield | DDPM 256-256           | ~250K      |

---

## Experiments

### Dataset Statistics

| Metric       | Value          |
| ------------ | -------------- |
| Episodes     | 420            |
| Transitions  | 129,217        |
| Mean Return  | 242.7 ± 26.8   |
| Return Range | [158.9, 283.7] |
| Mean Horizon | 307.7          |

### Evaluation Protocol

1. **In-Distribution (ID)**: Command `g = (200, 220)`
2. **Out-of-Distribution (OOD)**: Command `g = (50, 350)`
3. **Metrics**: Episode return, variance, OOD detection rate

### Hyperparameters

| Parameter      | Value                    |
| -------------- | ------------------------ |
| Learning Rate  | 3e-4                     |
| Batch Size     | 256                      |
| Hidden Dim     | 256                      |
| Epochs         | 100                      |
| Early Stopping | patience=15              |
| Optimizer      | AdamW                    |
| Scheduler      | CosineAnnealing + Warmup |

---

## Results Analysis

### Training Convergence

All models converge efficiently with early stopping:

- **Policy**: Negative log-likelihood ~0.8 (50-60 epochs)
- **Flow Shield**: ODE loss ~0.15 (70-80 epochs)
- **Quantile Shield**: Pinball loss ~15.0 (60-70 epochs)
- **Diffusion Shield**: Denoising loss ~0.3 (80-90 epochs)

### OOD Performance Breakdown

| Scenario            | No Shield    | Flow Shield  | Improvement |
| ------------------- | ------------ | ------------ | ----------- |
| ID Command (R=220)  | 230.5 ± 18.2 | 228.9 ± 15.4 | -0.7%       |
| OOD Command (R=350) | 211.4 ± 84.7 | 235.0 ± 26.0 | **+11.2%**  |

**Critical Insight**: Flow Shield dramatically reduces variance on OOD commands (69% reduction) while maintaining near-identical performance on in-distribution commands.

### Limitations and Future Work

1. **Data Homogeneity**: Current expert dataset has relatively low variance, limiting command-following precision
2. **UDRL Fundamental Challenge**: When training data lacks diversity, the policy learns average behavior
3. **Recommendation**: For precise command-following, collect expert data with diverse return profiles

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

This project builds upon excellent open-source work:

- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) — PPO implementation for expert training
- [torchdyn](https://github.com/DiffEqML/torchdyn) — Neural ODE solvers for flow matching
- [Gymnasium](https://gymnasium.farama.org/) — LunarLander-v3 environment

**Key References:**
- Schmidhuber (2019): *Reinforcement Learning Upside Down* — Original UDRL formulation
- Lipman et al. (2023): *Flow Matching for Generative Modeling* — Flow matching framework

---

<div align="center">

**FlowShield-UDRL** — Making command-conditioned RL safe and reliable

</div>
