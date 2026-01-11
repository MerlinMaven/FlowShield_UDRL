======================
Experimental Protocol
======================

This document describes the experimental methodology used to evaluate 
FlowShield-UDRL on LunarLander-v3.

Overview
--------

Our experiments answer three key questions:

1. **Does the shield prevent Obedient Suicide?**
2. **Which shield method is most effective?**
3. **What are the computational trade-offs?**

Training Pipeline
-----------------

The complete pipeline consists of four stages:

.. code-block:: text

   ┌─────────────────┐
   │  1. Data        │
   │  Collection     │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  2. UDRL        │
   │  Training       │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  3. Shield      │
   │  Training       │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  4. Shielded    │
   │  Evaluation     │
   └─────────────────┘

Stage 1: Data Collection
^^^^^^^^^^^^^^^^^^^^^^^^

Generate expert trajectories using trained PPO:

.. code-block:: bash

   # Option A: Train PPO from scratch (500k timesteps)
   python scripts/collect_expert_data.py --train-expert --timesteps 500000

   # Option B: Collect from existing expert
   python scripts/collect_expert_data.py --n-episodes 500 --output data/expert.npz

**Dataset statistics:**

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Metric
     - Value
     - Notes
   * - Episodes
     - 420
     - After filtering
   * - Transitions
     - 129,217
     - All timesteps
   * - Mean Return
     - 242.7 ± 26.8
     - High-quality expert
   * - Mean Horizon
     - 307.7
     - Steps per episode

Stage 2: UDRL Training
^^^^^^^^^^^^^^^^^^^^^^

Train goal-conditioned policy:

.. code-block:: bash

   python scripts/train_policy.py \
       --data data/lunarlander_expert.npz \
       --epochs 100 \
       --patience 15

**Training features:**

- Early stopping (patience=15)
- Cosine annealing LR with warmup
- Gradient clipping (max_norm=1.0)
- 10% validation split

Stage 3: Shield Training
^^^^^^^^^^^^^^^^^^^^^^^^

Train density estimator on command distribution:

.. code-block:: bash

   # Flow Matching Shield (recommended)
   python scripts/train_flow.py \
       --data data/lunarlander_expert.npz \
       --epochs 100

   # Quantile Shield (baseline)
   python scripts/train_quantile.py \
       --data data/lunarlander_expert.npz \
       --epochs 100

   # Diffusion Shield
   python scripts/train_diffusion.py \
       --data data/lunarlander_expert.npz \
       --epochs 100

Stage 4: Evaluation
^^^^^^^^^^^^^^^^^^^

Test with OOD commands:

.. code-block:: bash

   python scripts/evaluate_models.py \
       --env lunarlander \
       --data data/lunarlander_expert.npz

Evaluation Metrics
------------------

Primary Metrics
^^^^^^^^^^^^^^^

1. **Mean Episode Return** :math:`\bar{R}`

   .. math::
      \bar{R} = \frac{1}{N} \sum_{i=1}^{N} R_i

2. **Success Rate** :math:`\text{SR}`

   .. math::
      \text{SR} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[R_i > R_{\text{threshold}}]

3. **Crash Rate** :math:`\text{CR}`

   .. math::
      \text{CR} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{episode } i \text{ ends in failure}]

Safety Metrics
^^^^^^^^^^^^^^

4. **OOD Detection Rate** :math:`\text{DR}`

   .. math::
      \text{DR} = \frac{\# \text{correctly flagged OOD}}{\# \text{true OOD}}

5. **False Positive Rate** :math:`\text{FPR}`

   .. math::
      \text{FPR} = \frac{\# \text{incorrectly flagged as OOD}}{\# \text{in-distribution}}

6. **Shield Activation Rate** :math:`\text{SAR}`

   .. math::
      \text{SAR} = \frac{\# \text{projected commands}}{\# \text{total commands}}

LunarLander-Specific Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

7. **Landing Quality Score** :math:`Q`

   Composite score measuring precision of successful landings:

   .. math::
      Q = 1 - \text{clip}\left(0.5 \cdot \frac{|x|}{0.5} + 0.3 \cdot \frac{|\theta|}{0.3} + 0.2 \cdot \frac{v}{0.5}, 0, 1\right)

   Where :math:`x` = horizontal error, :math:`\theta` = angle, :math:`v` = velocity.

8. **Obedient Suicide Rate** :math:`\text{OSR}`

   Core metric for the problem we solve:

   .. math::
      \text{OSR} = P(\text{crash} | \text{ambitious command})

   An "ambitious command" is one in the top 25th percentile of return/horizon ratio.

9. **Command Achievement Ratio** :math:`\text{CAR}`

   How well the agent tracks target returns:

   .. math::
      \text{CAR} = \frac{R_{\text{actual}}}{R_{\text{target}}}

Command Generation Protocol
---------------------------

We evaluate under two command regimes:

1. **In-Distribution (ID)**

   Commands within training distribution:
   
   - Horizon: H = 200 steps
   - Return: R = 220 (near mean)

2. **Out-of-Distribution (OOD)**

   Commands outside training distribution:
   
   - Horizon: H = 50 steps (too short)
   - Return: R = 350 (above max achievable)

Baseline Comparisons
--------------------

We compare against:

1. **No Shield (Baseline)**
   
   - UDRL policy executes raw commands
   - Shows failure rate under OOD

2. **Quantile Shield**
   
   - Pinball loss regression
   - Conservative lower bound estimation

3. **Diffusion Shield**
   
   - DDPM-based density estimation
   - Reconstruction-based OOD detection

4. **Flow Matching Shield (Ours)**
   
   - OT-CFM with exact likelihood
   - Gradient-based projection

Hyperparameters
---------------

**Training hyperparameters:**

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Parameter
     - Value
     - Notes
   * - Learning Rate
     - 1e-3
     - With cosine annealing
   * - Batch Size
     - 256
     - Standard
   * - Hidden Dim
     - 256
     - All networks
   * - Epochs
     - 100
     - With early stopping
   * - Patience
     - 15
     - Early stopping
   * - Optimizer
     - AdamW
     - Weight decay 1e-5
   * - Gradient Clip
     - 1.0
     - Max norm

**Shield-specific:**

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 25

   * - Shield
     - Layers
     - Special
     - Threshold
   * - Flow
     - 4
     - ODE steps=50
     - 95th percentile
   * - Quantile
     - 3
     - τ=0.1
     - Fixed bound
   * - Diffusion
     - 4
     - T=100 steps
     - Reconstruction

Reproducibility
---------------

To ensure reproducibility:

1. **Fixed seeds:** ``--seed 42`` for all experiments
2. **Version control:** Requirements locked in ``requirements.txt``
3. **Checkpoints:** Models saved with training metadata
4. **Automatic calibration:** OOD threshold set at 95th percentile

**Reproduce main experiment:**

.. code-block:: bash

   # Full pipeline with fixed seed
   python scripts/train_policy.py --data data/lunarlander_expert.npz --seed 42
   python scripts/train_flow.py --data data/lunarlander_expert.npz --seed 42
   python scripts/evaluate_models.py --env lunarlander --data data/lunarlander_expert.npz

Computational Resources
-----------------------

**Hardware used:**

- GPU: NVIDIA RTX (CUDA enabled)
- RAM: 16 GB
- Training time: ~2 hours (all models)

**Time estimates:**

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Stage
     - Time
     - Notes
   * - Data collection
     - 30 min
     - 500k PPO timesteps
   * - UDRL training
     - 10 min
     - 75 epochs (early stop)
   * - Shield training
     - 15 min each
     - 3 shields
   * - Evaluation
     - 5 min
     - 8 test episodes

References
----------

Optimization: Adam optimizer [Kingma2014]_, cosine annealing [Loshchilov2017]_.

