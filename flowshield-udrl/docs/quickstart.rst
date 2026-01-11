==========
Quickstart
==========

This guide will get you running FlowShield-UDRL in under 5 minutes.

Option 1: Use Pre-trained Models
--------------------------------

Evaluate existing models without any training:

.. code-block:: bash

    # Evaluate all shields on LunarLander
    python scripts/evaluate_models.py --env lunarlander --data data/lunarlander_expert.npz

Option 2: Train from Scratch
----------------------------

Step 1: Train UDRL Policy
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python scripts/train_policy.py --data data/lunarlander_expert.npz --epochs 100

Step 2: Train Safety Shields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Flow Matching Shield (recommended)
    python scripts/train_flow.py --data data/lunarlander_expert.npz --epochs 100

    # Quantile Shield (baseline)
    python scripts/train_quantile.py --data data/lunarlander_expert.npz --epochs 100

    # Diffusion Shield
    python scripts/train_diffusion.py --data data/lunarlander_expert.npz --epochs 100

Step 3: Evaluate
^^^^^^^^^^^^^^^^

.. code-block:: bash

    python scripts/evaluate_models.py --env lunarlander --data data/lunarlander_expert.npz

Option 3: Generate New Expert Data
----------------------------------

.. code-block:: bash

    # Train PPO expert (500k timesteps)
    python scripts/collect_expert_data.py --train-expert --timesteps 500000

    # Generate expert trajectories
    python scripts/collect_expert_data.py --n-episodes 500 --output data/my_expert.npz

Use in Your Code
----------------

.. code-block:: python

    import torch
    import numpy as np
    
    # Load trained models
    policy = torch.load("results/lunarlander/models/policy.pt")
    flow_shield = torch.load("results/lunarlander/models/flow_shield.pt")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    flow_shield = flow_shield.to(device)
    
    # Example state from LunarLander
    state = np.array([0.1, 0.5, -0.1, -0.2, 0.05, 0.0, 0., 0.])
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    # User requests potentially dangerous command
    user_command = torch.tensor([[50, 500.0]]).to(device)  # H=50, R=500 (OOD!)
    
    # Shield checks if command is achievable
    flow_shield.eval()
    is_ood, log_prob = flow_shield.is_ood(state_tensor, user_command)
    
    if is_ood:
        print(f"OOD DETECTED: Command rejected (log_prob={log_prob.item():.2f})")
        safe_command = flow_shield.project(state_tensor, user_command)
        print(f"PROJECTION: Safe command H={safe_command[0,0]:.0f}, R={safe_command[0,1]:.0f}")
    else:
        safe_command = user_command
        print("PASSED: Command is within distribution")
    
    # Execute with safe command
    policy.eval()
    action = policy.sample(state_tensor, safe_command)
    print(f"Action: {action.detach().cpu().numpy()}")

Training Options
----------------

All training scripts support these common options:

.. list-table::
   :header-rows: 1
   :widths: 25 50 15

   * - Option
     - Description
     - Default
   * - ``--epochs N``
     - Number of training epochs
     - 100
   * - ``--patience N``
     - Early stopping patience
     - 15
   * - ``--seed N``
     - Random seed for reproducibility
     - 42
   * - ``--device``
     - Force cpu or cuda
     - auto
   * - ``--no-scheduler``
     - Disable learning rate scheduler
     - False
   * - ``--lr``
     - Learning rate
     - 1e-3
   * - ``--batch-size``
     - Training batch size
     - 256

Next Steps
----------

- :doc:`concepts/udrl` - Understand Upside-Down RL
- :doc:`concepts/ood_problem` - The Obedient Suicide problem
- :doc:`methods/flow_matching` - How Flow Matching works
- :doc:`experiments/results` - View experimental results
