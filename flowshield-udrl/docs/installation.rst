============
Installation
============

Requirements
------------

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU acceleration)

System Dependencies
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Package
     - Purpose
   * - ``torch``
     - Deep learning framework
   * - ``gymnasium``
     - RL environments (LunarLander)
   * - ``stable-baselines3``
     - PPO expert training
   * - ``torchdyn``
     - ODE solvers for Flow Matching
   * - ``numpy``
     - Numerical computing
   * - ``matplotlib``
     - Visualization

Installation Steps
------------------

From Source (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/your-username/flowshield-udrl.git
    cd flowshield-udrl

    # Create virtual environment
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # or: venv\Scripts\activate  # Windows

    # Install in development mode
    pip install -e .

Using pip
^^^^^^^^^

.. code-block:: bash

    pip install -r requirements.txt
    pip install -e .

Using Conda
^^^^^^^^^^^

.. code-block:: bash

    conda create -n flowshield python=3.10
    conda activate flowshield
    conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install -e .

Verify Installation
-------------------

.. code-block:: python

    # Test imports
    import torch
    import gymnasium as gym
    
    # Test environment
    env = gym.make("LunarLander-v3", continuous=True)
    print(f"State dim: {env.observation_space.shape}")
    print(f"Action dim: {env.action_space.shape}")
    
    # Test model loading
    from pathlib import Path
    models_dir = Path("results/lunarlander/models")
    if models_dir.exists():
        policy = torch.load(models_dir / "policy.pt", map_location="cpu")
        print(f"Policy loaded: {sum(p.numel() for p in policy.parameters()):,} parameters")

Expected output:

.. code-block:: text

    State dim: (8,)
    Action dim: (2,)
    Policy loaded: 200,000 parameters

GPU Support
-----------

To verify CUDA is available:

.. code-block:: python

    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")

Troubleshooting
---------------

Gymnasium Version Issues
^^^^^^^^^^^^^^^^^^^^^^^^

If ``LunarLander-v3`` is not found:

.. code-block:: bash

    pip install gymnasium[box2d]

PyTorch CUDA Mismatch
^^^^^^^^^^^^^^^^^^^^^

If you encounter CUDA errors:

.. code-block:: bash

    # Check your CUDA version
    nvidia-smi
    
    # Install matching PyTorch
    pip install torch --index-url https://download.pytorch.org/whl/cu118

TorchDyn Installation
^^^^^^^^^^^^^^^^^^^^^

If ``torchdyn`` fails:

.. code-block:: bash

    pip install torchdyn --no-deps
    pip install torchdiffeq
