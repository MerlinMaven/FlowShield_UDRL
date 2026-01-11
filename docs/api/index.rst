=============
API Reference
=============

Complete API documentation for FlowShield-UDRL.

.. toctree::
   :maxdepth: 2
   :caption: Modules

   envs
   models
   training
   evaluation

Overview
--------

The FlowShield-UDRL codebase is organized into the following modules:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - ``flowshield.envs``
     - Environment wrappers and factory
   * - ``flowshield.models``
     - Neural network models (UDRL, Shields)
   * - ``flowshield.training``
     - Training loops and utilities
   * - ``flowshield.evaluation``
     - Evaluation and metrics

Quick Import
------------

.. code-block:: python

   # Core components
   from flowshield.envs import make_env
   from flowshield.models import UDRLAgent, FlowMatchingShield
   from flowshield.training import Trainer
   from flowshield.evaluation import evaluate_agent
   
   # Create environment
   env = make_env("lunarlander")
   
   # Load trained models
   agent = UDRLAgent.load("checkpoints/udrl_lunarlander.pt")
   shield = FlowMatchingShield.load("checkpoints/shield_lunarlander.pt")
   
   # Evaluate
   metrics = evaluate_agent(env, agent, shield)

Module Dependencies
-------------------

.. code-block:: text

   flowshield/
   ├── envs/           # Environment wrappers
   │   ├── __init__.py
   │   ├── factory.py
   │   └── wrappers.py
   │
   ├── models/         # Neural networks
   │   ├── __init__.py
   │   ├── udrl.py
   │   ├── shields/
   │   │   ├── base.py
   │   │   ├── quantile.py
   │   │   ├── diffusion.py
   │   │   └── flow_matching.py
   │   └── networks.py
   │
   ├── training/       # Training utilities
   │   ├── __init__.py
   │   ├── trainer.py
   │   ├── dataset.py
   │   └── utils.py
   │
   └── evaluation/     # Evaluation
       ├── __init__.py
       ├── evaluate.py
       └── metrics.py

Type Hints
----------

All public APIs use type hints for clarity:

.. code-block:: python

   from typing import Tuple, Optional, Dict
   import torch
   import numpy as np
   
   def evaluate_episode(
       env: gym.Env,
       agent: UDRLAgent,
       shield: Optional[BaseShield] = None,
       command: Tuple[float, float] = (100, 100),
       render: bool = False
   ) -> Dict[str, float]:
       """
       Run single episode and return metrics.
       
       Args:
           env: Gymnasium environment
           agent: Trained UDRL agent
           shield: Optional safety shield
           command: (horizon, return) target
           render: Whether to render
           
       Returns:
           Dictionary with 'return', 'length', 'success', 'crash'
       """
       ...
