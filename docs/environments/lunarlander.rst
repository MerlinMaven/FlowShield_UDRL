================
LunarLander-v3
================

LunarLander-v3 is a classic control environment from Gymnasium where an agent 
must safely land a spacecraft on a landing pad.

.. image:: /_static/expert_policy_episode.gif
   :alt: Trained UDRL agent landing on LunarLander
   :align: center
   :width: 80%

*Animation: Trained UDRL policy performing a successful landing.*

Overview
--------

**Goal:** Safely land on the pad between the two flags.

**Challenge:** Balance fuel usage, velocity, and position control.

**OOD Risk:** Aggressive commands can cause crashes or fuel depletion.

.. code-block:: python

   import gymnasium as gym
   env = gym.make("LunarLander-v3", continuous=True)

Expert Data Statistics
----------------------

Our trained PPO expert achieves excellent performance:

.. image:: /_static/expert_data_distribution.png
   :alt: Expert data return distribution
   :align: center
   :width: 80%

*Distribution of episode returns from the trained PPO expert.*

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Metric
     - Value
     - Notes
   * - Episodes
     - 420
     - High-quality trajectories
   * - Mean Return
     - **242.7 ± 26.8**
     - Near-optimal performance
   * - Return Range
     - [158.9, 283.7]
     - All successful landings
   * - Mean Horizon
     - 307.7 steps
     - Full episode length

Expert Trajectories
^^^^^^^^^^^^^^^^^^^

.. image:: /_static/expert_trajectories.png
   :alt: Visualization of expert landing trajectories
   :align: center
   :width: 80%

*Typical expert trajectories showing controlled descent patterns.*

State Space
-----------

8-dimensional continuous observation:

.. list-table::
   :header-rows: 1
   :widths: 10 25 15 15 35

   * - Dim
     - Name
     - Min
     - Max
     - Description
   * - 0
     - :math:`x`
     - -∞
     - +∞
     - Horizontal position
   * - 1
     - :math:`y`
     - -∞
     - +∞
     - Vertical position
   * - 2
     - :math:`v_x`
     - -∞
     - +∞
     - Horizontal velocity
   * - 3
     - :math:`v_y`
     - -∞
     - +∞
     - Vertical velocity
   * - 4
     - :math:`\theta`
     - -π
     - +π
     - Angle
   * - 5
     - :math:`\omega`
     - -∞
     - +∞
     - Angular velocity
   * - 6
     - :math:`c_L`
     - 0
     - 1
     - Left leg contact
   * - 7
     - :math:`c_R`
     - 0
     - 1
     - Right leg contact

**State vector:**

.. math::
   :label: lunar_state

   s = (x, y, v_x, v_y, \theta, \omega, c_L, c_R) \in \mathbb{R}^8

Action Space
------------

2-dimensional continuous:

.. list-table::
   :header-rows: 1
   :widths: 15 20 15 15 35

   * - Dim
     - Name
     - Min
     - Max
     - Effect
   * - 0
     - Main engine
     - -1.0
     - +1.0
     - 0 = off, +1 = full thrust
   * - 1
     - Side engine
     - -1.0
     - +1.0
     - -1 = left, +1 = right

**Action vector:**

.. math::
   :label: lunar_action

   a = (a_{\text{main}}, a_{\text{side}}) \in [-1, 1]^2

Reward Structure
----------------

The reward function encourages soft landings:

.. math::
   :label: lunar_reward

   r = r_{\text{shaping}} + r_{\text{legs}} + r_{\text{fuel}} + r_{\text{terminal}}

Components:

1. **Shaping reward** (dense):

   .. math::
      r_{\text{shaping}} = -100 \cdot \sqrt{x^2 + y^2} - 100 \cdot \sqrt{v_x^2 + v_y^2} - 100 \cdot |\theta|

2. **Leg contact bonus:**

   .. math::
      r_{\text{legs}} = 10 \cdot c_L + 10 \cdot c_R

3. **Fuel penalty:**

   .. math::
      r_{\text{fuel}} = -0.3 \cdot |a_{\text{main}}| - 0.03 \cdot |a_{\text{side}}|

4. **Terminal reward:**

   .. math::
      r_{\text{terminal}} = \begin{cases}
      +100 & \text{successful landing} \\
      -100 & \text{crash}
      \end{cases}

Episode Termination
-------------------

An episode terminates when:

1. **Success:** Both legs contact ground, low velocity, upright angle
2. **Crash:** Body touches ground (not legs first)
3. **Out of bounds:** :math:`|x| > 1` or :math:`y > 2`
4. **Timeout:** 1000 steps reached

OOD Commands
------------

Examples of dangerous commands :math:`g = (H, R)`:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Command
     - Why OOD
     - Expected behavior
   * - :math:`H=10, R=200`
     - Impossible to land in 10 steps
     - Reckless dive, crash
   * - :math:`H=50, R=500`
     - Reward exceeds maximum possible
     - Thruster abuse, fuel waste
   * - :math:`H=500, R=-100`
     - Negative reward with long horizon
     - Intentional crash attempt

Observed Command Distribution
-----------------------------

From our expert dataset:

.. math::
   \begin{aligned}
   H &\sim [200, 450] \text{ steps} \\
   R &\sim [160, 280] \text{ total reward}
   \end{aligned}

OOD detection threshold is calibrated to reject commands outside the 95th percentile.

Training the Expert
-------------------

.. code-block:: bash

   # Train PPO expert (500k timesteps)
   python scripts/collect_expert_data.py --train-expert --timesteps 500000

   # Collect trajectories
   python scripts/collect_expert_data.py --n-episodes 500 --output data/my_expert.npz

Visualization
-------------

.. code-block:: python

   import gymnasium as gym
   import torch
   
   # Load trained models
   policy = torch.load("results/lunarlander/models/policy.pt")
   policy.eval()
   
   env = gym.make("LunarLander-v3", 
                  continuous=True, 
                  render_mode="human")
   
   state, _ = env.reset()
   done = False
   total_reward = 0
   
   # Use achievable command
   command = torch.tensor([[200, 220.0]])  # H=200, R=220
   
   while not done:
       state_tensor = torch.FloatTensor(state).unsqueeze(0)
       action = policy.sample(state_tensor, command).detach().numpy()[0]
       state, reward, terminated, truncated, info = env.step(action)
       total_reward += reward
       done = terminated or truncated
   
   print(f"Episode return: {total_reward:.1f}")
   env.close()

OOD Comparison
--------------

.. image:: /_static/comparison_ood.gif
   :alt: Comparison of agent behavior with and without shield under OOD commands
   :align: center
   :width: 100%

*Comparison: Agent behavior under OOD command (H=50, R=350) with and without Flow Shield.*

Safety Shield Application
-------------------------

Shield workflow:

1. **Receive command:** :math:`g^* = (H^*, R^*)` from external source
2. **Compute density:** :math:`\log p(g^* | s)` using FlowShield
3. **Check threshold:** If :math:`\log p < \tau`, command is OOD
4. **Project:** :math:`g_{\text{safe}} = \text{project}(g^*, s)` via gradient ascent
5. **Execute:** Feed :math:`g_{\text{safe}}` to UDRL policy

Failure Mode Examples
---------------------

**Without shield:**

- **Command:** "Land in 20 steps with +500 reward"
- **Policy action:** Maximum thrust + aggressive rotation
- **Outcome:** Crash at high velocity = -100 reward

**With shield:**

- **Command:** "Land in 20 steps with +500 reward" 
- **Detection:** :math:`\log p(g|s) = -15 < \tau = -5` → OOD!
- **Projection:** :math:`g_{\text{safe}} = (200, 220)`
- **Policy action:** Controlled descent
- **Outcome:** Safe landing = +220 reward

Results on LunarLander
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Method
     - OOD Return
     - Variance
     - Detection
   * - No Shield
     - 211.4
     - 84.7
     - 0%
   * - **Flow Shield**
     - **235.0**
     - **26.0**
     - **77%**

**Flow Shield provides +11.2% improvement with 69% lower variance.**

References
----------

OpenAI Gym environment: [Brockman2016]_.

