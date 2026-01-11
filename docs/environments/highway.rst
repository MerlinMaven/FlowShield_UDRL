============
Highway-Env
============

Highway-Env is a realistic driving simulation for autonomous vehicle control,
featuring multi-lane highways with traffic.

Overview
--------

.. note::
   Experiments on Highway-Env are in progress. Results will be added upon completion.

**Goal:** Drive safely while maintaining speed and avoiding collisions.

**Challenge:** Navigate through traffic with continuous control.

**OOD Risk:** Aggressive commands lead to collisions or unsafe maneuvers.

.. code-block:: python

   import gymnasium as gym
   import highway_env
   
   env = gym.make("highway-v0")
   env.configure({"action": {"type": "ContinuousAction"}})

State Space
-----------

25-dimensional observation (5 vehicles × 5 features):

**Kinematics observation** for ego vehicle + 4 nearest vehicles:

.. list-table::
   :header-rows: 1
   :widths: 10 20 15 15 40

   * - Dim
     - Feature
     - Min
     - Max
     - Description
   * - 0-4
     - Presence
     - 0
     - 1
     - Vehicle detected
   * - 5-9
     - :math:`x`
     - 0
     - 1
     - Longitudinal position (normalized)
   * - 10-14
     - :math:`y`
     - -1
     - 1
     - Lateral position (normalized)
   * - 15-19
     - :math:`v_x`
     - 0
     - 1
     - Longitudinal velocity (normalized)
   * - 20-24
     - :math:`v_y`
     - -1
     - 1
     - Lateral velocity (normalized)

**State structure:**

.. math::
   :label: highway_state

   s = \begin{bmatrix}
   p_1 & x_1 & y_1 & v_{x,1} & v_{y,1} \\
   p_2 & x_2 & y_2 & v_{x,2} & v_{y,2} \\
   \vdots & \vdots & \vdots & \vdots & \vdots \\
   p_5 & x_5 & y_5 & v_{x,5} & v_{y,5}
   \end{bmatrix} \in \mathbb{R}^{25}

where vehicle 1 is the ego vehicle.

Action Space
------------

2-dimensional continuous control:

.. list-table::
   :header-rows: 1
   :widths: 15 25 15 15 30

   * - Dim
     - Name
     - Min
     - Max
     - Effect
   * - 0
     - Acceleration
     - 0
     - 1
     - Speed control
   * - 1
     - Steering
     - -1
     - +1
     - Lane change

**Action vector:**

.. math::
   :label: highway_action

   a = (a_{\text{accel}}, a_{\text{steer}}) \in [0,1] \times [-1,1]

Vehicle Dynamics
----------------

The ego vehicle follows kinematic bicycle model:

.. math::
   :label: bicycle_model

   \begin{aligned}
   \dot{x} &= v \cos(\theta + \beta) \\
   \dot{y} &= v \sin(\theta + \beta) \\
   \dot{\theta} &= \frac{v}{l_r} \sin(\beta) \\
   \dot{v} &= a_{\text{accel}}
   \end{aligned}

where :math:`\beta = \arctan\left(\frac{l_r}{l_f + l_r} \tan(\delta)\right)` is the slip angle.

Reward Structure
----------------

Per-step reward:

.. math::
   :label: highway_reward

   r = r_{\text{speed}} + r_{\text{collision}} + r_{\text{lane}}

Components:

1. **Speed reward:**

   .. math::
      r_{\text{speed}} = \alpha \cdot \frac{v}{v_{\max}} \in [0, \alpha]

   where :math:`\alpha = 0.4` typically.

2. **Collision penalty:**

   .. math::
      r_{\text{collision}} = \begin{cases}
      -1 & \text{if collision} \\
      0 & \text{otherwise}
      \end{cases}

3. **Lane reward:**

   .. math::
      r_{\text{lane}} = \beta \cdot \mathbb{1}[\text{on rightmost lane}]

**Typical per-step reward:** :math:`r \in [-1, +1]`

**Episode total:** :math:`R \in [-50, +200]` depending on duration and speed.

Episode Termination
-------------------

Episodes end when:

1. **Collision:** Ego vehicle hits another vehicle
2. **Off-road:** Ego vehicle leaves the highway
3. **Timeout:** Maximum duration reached (varies by config)

OOD Commands
------------

Dangerous commands for Highway-Env:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Command
     - Why OOD
     - Likely outcome
   * - :math:`H=10, R=100`
     - Cannot achieve 100 reward in 10 steps
     - Aggressive acceleration, collision
   * - :math:`H=200, R=200`
     - Maximum speed for 200 steps
     - No braking, inevitable crash
   * - :math:`H=50, R=-20`
     - Negative reward expectation
     - Intentional collision

Collision Detection
-------------------

The environment uses axis-aligned bounding boxes (AABB):

.. math::
   :label: collision

   \text{collision} = \bigcap_{i \in \{x,y\}} (|p_{\text{ego},i} - p_{\text{other},i}| < s_i)

where :math:`s_i` is the safety margin in dimension :math:`i`.

Configuration
-------------

.. code-block:: yaml

   # configs/env/highway.yaml
   name: "highway-v0"
   
   # Action type
   action:
     type: "ContinuousAction"
   
   # Observation
   observation:
     type: "Kinematics"
     vehicles_count: 5
     features:
       - "presence"
       - "x"
       - "y"
       - "vx"
       - "vy"
     absolute: false
     normalize: true
   
   # Simulation
   simulation_frequency: 15
   policy_frequency: 1
   duration: 40  # seconds
   
   # Traffic
   vehicles_count: 50
   lanes_count: 4
   initial_lane_id: 1
   
   # Rewards
   reward_speed_range: [20, 30]
   collision_reward: -1
   high_speed_reward: 0.4
   right_lane_reward: 0.1
   
   # Command ranges
   horizon_min: 20
   horizon_max: 300
   return_min: -50
   return_max: 150

Traffic Modeling
----------------

Other vehicles follow IDM (Intelligent Driver Model):

.. math::
   :label: idm

   \dot{v} = a_{\max} \left[ 1 - \left(\frac{v}{v_0}\right)^\delta 
   - \left(\frac{s^*(v, \Delta v)}{s}\right)^2 \right]

where:

- :math:`v_0` = desired velocity
- :math:`s` = gap to leading vehicle
- :math:`s^*` = desired gap
- :math:`a_{\max}` = maximum acceleration
- :math:`\delta` = acceleration exponent

Lane changes follow MOBIL model:

.. math::
   :label: mobil

   \text{change if: } \tilde{a}_c - a_c + p(\tilde{a}_n - a_n + \tilde{a}_o - a_o) > \Delta a_{\text{th}}

Safety Shield Application
-------------------------

In Highway-Env, the shield is critical:

1. **High stakes:** Collisions are catastrophic
2. **Dense traffic:** Small errors propagate
3. **Speed variance:** OOD commands cause instability

**Shield workflow:**

.. code-block:: python

   # Receive external command
   g_star = (horizon=50, return=150)
   
   # FlowShield evaluation
   log_prob = shield.log_prob(state, g_star)
   
   if log_prob < threshold:
       # OOD detected - project to safe region
       g_safe = shield.project(state, g_star)
       print(f"Projected: {g_star} -> {g_safe}")
   else:
       g_safe = g_star
   
   # Execute with safe command
   action = policy(state, g_safe)

Multi-Agent Considerations
--------------------------

The observation only includes **nearest 4 vehicles**. This means:

1. **Partial observability:** Cannot see all traffic
2. **Dynamic selection:** Observed vehicles change
3. **Safety margins:** Must account for unseen vehicles

FlowShield learns the achievable distribution **given partial observations**,
naturally accounting for this uncertainty.

Visualization
-------------

.. code-block:: python

   import gymnasium as gym
   import highway_env
   
   env = gym.make("highway-v0", render_mode="human")
   env.configure({
       "action": {"type": "ContinuousAction"},
       "observation": {
           "type": "Kinematics",
           "vehicles_count": 5
       }
   })
   
   state, _ = env.reset()
   done = False
   
   while not done:
       action = policy(state, command)
       state, reward, terminated, truncated, info = env.step(action)
       done = terminated or truncated
       env.render()
   
   env.close()

Failure Mode Examples
---------------------

**Without shield:**

- Command: "Drive 200 steps, achieve 180 reward"
- Policy: Maximum acceleration, no lane awareness
- Outcome: Collision at step 35 → -35 total reward

**With shield:**

- Command: "Drive 200 steps, achieve 180 reward"
- Detection: :math:`\log p(g|s) = -12 < \tau = -6` → OOD
- Projection: :math:`g_{\text{safe}} = (200, 120)`
- Policy: Moderate speed, careful lane changes
- Outcome: Complete 200 steps → +125 total reward

References
----------

Highway-Env simulation: [Leurent2018]_.

Safe model-based RL: [Berkenkamp2017]_.

