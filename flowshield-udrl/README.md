# FlowShield-UDRL

**Safe Command-Conditioned Reinforcement Learning via Flow Matching**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-enabled-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Experiments](#experiments)
- [Results Analysis](#results-analysis)
- [Changelog](#changelog)
- [Citation](#citation)

---

## Overview

### The Problem: "Obedient Suicide" in UDRL

**Upside-Down Reinforcement Learning (UDRL)** trains policies conditioned on desired outcomes (horizon, return). However, when users request **impossible/out-of-distribution (OOD) commands**, the agent blindly attempts execution, causing:

- Erratic, dangerous behavior
- System crashes and safety violations
- Failure to achieve reasonable performance

### Our Solution: Flow Matching Safety Shield

We leverage **Flow Matching** (Lipman et al., 2023) to:

1. **Model** the distribution `p(g|s)` of achievable commands given state
2. **Detect** OOD commands via log-likelihood estimation
3. **Project** unsafe commands onto the manifold of achievable commands

### Key Innovation

```
Flow Matching + UDRL Safety = First application to Safe Offline RL
```

This addresses a critical gap in command-conditioned RL research.

---

## Key Results

### LunarLander-v3 (Continuous) - January 2026

| Method               | Mean Return | Std Dev  | OOD Detection | Improvement |
| -------------------- | ----------- | -------- | ------------- | ----------- |
| No Shield (Baseline) | 211.4       | 84.7     | 0%            | -           |
| Diffusion Shield     | 209.9       | 84.1     | 14%           | -0.7%       |
| **Flow Shield**      | **235.0**   | **26.0** | **77%**       | **+11.2%**  |

**Key Findings:**

- **+11.2%** improvement in mean return with Flow Shield
- **-69%** reduction in variance (26.0 vs 84.7)
- **77%** OOD command detection rate

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup

```bash
# Clone repository
git clone https://github.com/your-username/flowshield-udrl.git
cd flowshield-udrl

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

### Option 1: Use Pre-trained Models

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

All scripts support:

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
│   │   ├── final_comparison.png      # Main results
│   │   ├── policy_training.png       # Training curves
│   │   ├── flow_training.png
│   │   ├── quantile_training.png
│   │   ├── diffusion_training.png
│   │   ├── expert_data_distribution.png
│   │   ├── expert_trajectories.png
│   │   ├── expert_policy_episode.gif
│   │   └── udrl_limitation_analysis.png
│   ├── models/                       # Trained models
│   │   ├── policy.pt                 # UDRL policy
│   │   ├── flow_shield.pt            # Flow Matching shield
│   │   ├── quantile_shield.pt        # Quantile shield
│   │   └── diffusion_shield.pt       # Diffusion shield
│   └── metrics/                      # Evaluation results
│       └── comparison_results.json
│
├── scripts/                          # Executable scripts
│   ├── models.py                     # Core model definitions
│   ├── train_policy.py               # UDRL training
│   ├── train_flow.py                 # Flow shield training
│   ├── train_quantile.py             # Quantile shield training
│   ├── train_diffusion.py            # Diffusion shield training
│   ├── evaluate_models.py            # Evaluation pipeline
│   ├── collect_expert_data.py        # Data generation
│   ├── run_experiments.py            # Full experiment runner
│   └── logger.py                     # Logging utilities
│
├── src/                              # Source library
│   ├── models/                       # Neural networks
│   │   ├── agent/                    # UDRL policy
│   │   ├── safety/                   # Safety shields
│   │   └── components/               # Building blocks
│   ├── data/                         # Dataset handling
│   ├── envs/                         # Environment wrappers
│   ├── training/                     # Training loops
│   ├── evaluation/                   # Metrics & visualization
│   └── utils/                        # Utilities
│
├── tests/                            # Unit tests
├── notebooks/                        # Jupyter tutorials
├── docs/                             # Sphinx documentation
├── configs/                          # YAML configurations
├── models/                           # Saved RL models
│   └── ppo_lunarlander.zip           # Trained PPO expert
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

### 3. OOD Detection & Projection

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

### Training Curves

All models converge within 50-80 epochs with early stopping:

- **Policy**: Converges to loss ~0.8 (negative log-likelihood)
- **Flow Shield**: ODE loss ~0.15 after convergence
- **Quantile Shield**: Pinball loss ~15 (return scale)
- **Diffusion Shield**: Denoising loss ~0.3

### OOD Performance Comparison

| Scenario            | No Shield    | Flow Shield  | Improvement |
| ------------------- | ------------ | ------------ | ----------- |
| ID Command (R=220)  | 230.5 ± 18.2 | 228.9 ± 15.4 | -0.7%       |
| OOD Command (R=350) | 211.4 ± 84.7 | 235.0 ± 26.0 | **+11.2%**  |

**Key Insight**: Flow Shield dramatically reduces variance on OOD commands while maintaining ID performance.

### Limitations

1. **Data Homogeneity**: Expert data has low variance (σ=26.8), limiting command-following precision
2. **UDRL Fundamental**: Policy learns average behavior when training data lacks diversity
3. **Recommendation**: For better command-following, train with diverse return profiles

---

## Changelog

### v1.1.0 (January 2026) - Current

**Major Changes:**

- Trained PPO expert achieving R=242.7 (vs previous R=-40)
- Generated high-quality expert dataset (420 episodes)
- Retrained all models on GPU with expert data
- Fixed 8 major code issues (ResBlock, normalization, etc.)
- Removed Hydra dependency (pure argparse)
- Added early stopping, cosine scheduler, gradient clipping
- Calibrated OOD thresholds automatically

**Performance Improvements:**

- Flow Shield: +11.2% return improvement on OOD
- Variance reduction: 69% lower on OOD commands

**Cleaned Up:**

- Removed 5 obsolete datasets (65+ MB freed)
- Removed 7 redundant scripts
- Removed 4 old result directories
- Removed 9 duplicate figures

### v1.0.0 (December 2025)

- Initial implementation
- Basic UDRL + Flow Matching pipeline

---

## Citation

```bibtex
@article{flowshield2026,
  title={FlowShield-UDRL: Safe Command-Conditioned RL via Flow Matching},
  author={Your Name},
  journal={arXiv preprint},
  year={2026}
}
```

### References

```bibtex
@inproceedings{schmidhuber2019udrl,
  title={Reinforcement Learning Upside Down},
  author={Schmidhuber, Jürgen},
  booktitle={arXiv preprint arXiv:1912.02877},
  year={2019}
}

@inproceedings{lipman2023flow,
  title={Flow Matching for Generative Modeling},
  author={Lipman, Yaron and Chen, Ricky TQ and others},
  booktitle={ICLR},
  year={2023}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) for PPO implementation
- [torchdyn](https://github.com/DiffEqML/torchdyn) for ODE solvers
- [Gymnasium](https://gymnasium.farama.org/) for LunarLander environment

---

<p align="center">
  <b>FlowShield-UDRL</b> — Making command-conditioned RL safe and reliable
</p>
