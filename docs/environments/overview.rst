======================
Environments Overview
======================

FlowShield-UDRL supports continuous control environments where goal-conditioned 
policies can exhibit **Obedient Suicide** when given OOD commands.

Supported Environments
----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 15 30

   * - Environment
     - State dim
     - Action dim
     - Action space
     - Use case
   * - LunarLander-v2
     - 8
     - 2
     - Continuous [-1,1]²
     - Simple control with landing
   * - Highway-Env
     - 25
     - 2
     - Continuous [0,1]×[-1,1]
     - Autonomous driving

Environment Requirements
------------------------

For FlowShield to be effective, environments should have:

1. **Continuous State Space**
   
   - Dense observations that reflect environment dynamics
   - Sufficient information for decision-making

2. **Continuous Action Space**
   
   - Smooth control signals
   - Bounded action ranges

3. **Clear Success/Failure**
   
   - Measurable reward signals
   - Identifiable failure states

4. **Variable Episode Lengths**
   
   - Different horizons achievable based on strategy
   - Natural termination conditions

State Normalization
-------------------

All states are normalized before processing:

.. math::
   :label: state_normalization

   s_{\text{norm}} = \frac{s - \mu_s}{\sigma_s + \epsilon}

where :math:`\mu_s` and :math:`\sigma_s` are computed from the training dataset.

Command Normalization
---------------------

Commands :math:`g = (H, R)` are also normalized:

.. math::
   :label: command_normalization

   \begin{aligned}
   H_{\text{norm}} &= \frac{H - H_{\text{min}}}{H_{\text{max}} - H_{\text{min}}} \\
   R_{\text{norm}} &= \frac{R - R_{\text{min}}}{R_{\text{max}} - R_{\text{min}}}
   \end{aligned}

This ensures both dimensions contribute equally to density estimation.

Adding New Environments
-----------------------

To add a new environment:

1. **Create wrapper** that inherits from ``ContinuousEnvWrapper``

2. **Register** in ``envs/factory.py``

3. **Create config** in ``configs/env/``

4. **Collect dataset** with behavioral policy

Example wrapper:

.. code-block:: python

   from flowshield.envs.wrappers import ContinuousEnvWrapper
   
   class MyEnvWrapper(ContinuousEnvWrapper):
       def __init__(self):
           env = gym.make("MyEnv-v0")
           super().__init__(env)
       
       @property
       def max_episode_steps(self):
           return 500
       
       @property
       def state_dim(self):
           return self.observation_space.shape[0]
       
       @property
       def action_dim(self):
           return self.action_space.shape[0]

Common Interface
----------------

All environments expose:

.. code-block:: python

   class BaseEnv:
       def reset(self) -> np.ndarray:
           """Reset environment and return initial state."""
           
       def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
           """Take action and return (next_state, reward, done, info)."""
       
       @property
       def state_dim(self) -> int:
           """Dimension of observation space."""
       
       @property
       def action_dim(self) -> int:
           """Dimension of action space."""
       
       @property
       def max_episode_steps(self) -> int:
           """Maximum steps per episode."""

Visualization
-------------

Each environment provides rendering:

.. code-block:: python

   env = make_env("lunarlander")
   env.render_mode = "human"
   
   state = env.reset()
   for _ in range(100):
       action = policy(state, command)
       state, reward, done, info = env.step(action)
       env.render()
       if done:
           break

Comparison
----------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - LunarLander
     - Highway-Env
   * - Difficulty
     - Medium
     - Hard
   * - OOD risk
     - Crash on surface
     - Collision with vehicles
   * - Typical reward
     - [-200, +200]
     - [-1, +1] per step
   * - Typical horizon
     - 100-300 steps
     - 50-200 steps
   * - Visual
     - 2D physics
     - Top-down driving
