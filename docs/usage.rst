=====
Usage
=====

This section provides detailed usage instructions for FlowShield-UDRL.

Project Structure
-----------------

.. code-block:: text

    flowshield-udrl/
    ├── data/                             # Datasets
    │   └── lunarlander_expert.npz        # PPO expert (420 ep, R=242.7)
    │
    ├── results/lunarlander/              # Experiment outputs
    │   ├── figures/                      # Visualizations
    │   │   ├── final_comparison.png      # Main results
    │   │   ├── policy_training.png       # Training curves
    │   │   ├── flow_training.png
    │   │   ├── expert_policy_episode.gif # Trained agent animation
    │   │   └── ...
    │   ├── models/                       # Trained models
    │   │   ├── policy.pt                 # UDRL policy
    │   │   ├── flow_shield.pt            # Flow Matching shield
    │   │   ├── quantile_shield.pt        # Quantile shield
    │   │   └── diffusion_shield.pt       # Diffusion shield
    │   └── metrics/                      # Evaluation results
    │       └── comparison_results.json
    │
    ├── scripts/                          # Executable scripts
    │   ├── models.py                     # Shared model definitions
    │   ├── train_policy.py               # UDRL training
    │   ├── train_flow.py                 # Flow shield training
    │   ├── train_quantile.py             # Quantile shield training
    │   ├── train_diffusion.py            # Diffusion shield training
    │   ├── evaluate_models.py            # Evaluation pipeline
    │   ├── collect_expert_data.py        # Data generation
    │   └── run_experiments.py            # Full experiment runner
    │
    ├── src/                              # Source library
    │   ├── models/                       # Neural networks
    │   │   ├── agent/                    # UDRL policy
    │   │   └── safety/                   # Safety shields
    │   ├── envs/                         # Environment wrappers
    │   ├── training/                     # Training utilities
    │   └── evaluation/                   # Metrics and visualization
    │
    └── docs/                             # This documentation

Training Pipeline
-----------------

Phase 1: Data Collection
^^^^^^^^^^^^^^^^^^^^^^^^

Generate expert trajectories using PPO:

.. code-block:: bash

    # Train PPO expert from scratch
    python scripts/collect_expert_data.py --train-expert --timesteps 500000

    # Collect trajectories from trained expert
    python scripts/collect_expert_data.py --n-episodes 500 --output data/my_expert.npz

Or use the provided expert dataset:

.. code-block:: bash

    # Dataset already included: data/lunarlander_expert.npz
    # 420 episodes, mean return = 242.7

Phase 2: Train UDRL Policy
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python scripts/train_policy.py --data data/lunarlander_expert.npz --epochs 100

**Training features:**

- Early stopping with patience=15
- Cosine annealing LR scheduler with warmup
- Gradient clipping (max_norm=1.0)
- Automatic validation split (10%)

Phase 3: Train Safety Shields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Flow Matching Shield (recommended)
    python scripts/train_flow.py --data data/lunarlander_expert.npz --epochs 100

    # Quantile Shield (fast baseline)
    python scripts/train_quantile.py --data data/lunarlander_expert.npz --epochs 100

    # Diffusion Shield
    python scripts/train_diffusion.py --data data/lunarlander_expert.npz --epochs 100

Phase 4: Evaluation
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Full evaluation with environment
    python scripts/evaluate_models.py --env lunarlander --data data/lunarlander_expert.npz

    # Offline analysis (no environment needed)
    python scripts/evaluate_models.py --offline --data data/lunarlander_expert.npz

Evaluation Metrics
------------------

OOD Detection
^^^^^^^^^^^^^

Test commands are categorized as:

.. list-table::
   :header-rows: 1

   * - Command Type
     - Horizon
     - Return
     - Expected Detection
   * - In-Distribution
     - 200
     - 220
     - No (should not trigger)
   * - Out-of-Distribution
     - 50
     - 350
     - Yes (should trigger)
   * - Extreme OOD
     - 10
     - 500
     - Yes (must trigger)

Configuration Reference
-----------------------

All scripts use argparse with consistent options:

Training Scripts
^^^^^^^^^^^^^^^^

.. code-block:: bash

    python scripts/train_flow.py \
        --data data/lunarlander_expert.npz \
        --epochs 100 \
        --lr 1e-3 \
        --batch-size 256 \
        --hidden-dim 256 \
        --patience 15 \
        --seed 42

Evaluation Scripts
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python scripts/evaluate_models.py \
        --env lunarlander \
        --data data/lunarlander_expert.npz \
        --n-episodes 10 \
        --device cuda

Full Experiment Runner
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Run complete pipeline
    python scripts/run_experiments.py --env lunarlander

    # Quick test mode
    python scripts/run_experiments.py --env lunarlander --quick

Command Line Reference
----------------------

Available Scripts
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Script
     - Purpose
   * - ``train_policy.py``
     - Train UDRL policy
   * - ``train_flow.py``
     - Train Flow Matching shield
   * - ``train_quantile.py``
     - Train Quantile shield
   * - ``train_diffusion.py``
     - Train Diffusion shield
   * - ``evaluate_models.py``
     - Evaluate and compare models
   * - ``collect_expert_data.py``
     - Generate training data
   * - ``run_experiments.py``
     - Full experiment pipeline

Using Make
^^^^^^^^^^

.. code-block:: bash

    make train              # Train Flow shield on LunarLander
    make evaluate           # Run evaluation
    make test               # Run unit tests
    make docs               # Build documentation
    make clean              # Remove generated files
