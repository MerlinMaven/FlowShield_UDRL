=================================
Upside-Down Reinforcement Learning
=================================

Upside-Down Reinforcement Learning (UDRL) is a paradigm that transforms RL into 
a supervised learning problem by conditioning the policy on desired outcomes.

Core Idea
---------

Traditional RL learns a policy :math:`\pi(a|s)` that maximizes expected return.
UDRL instead learns a **goal-conditioned policy**:

.. math::
   :label: udrl_policy

   \pi_\theta(a|s, g)

where :math:`g = (H, R)` is a **command** specifying:

- :math:`H` = **Horizon**: number of remaining steps
- :math:`R` = **Target Return**: desired cumulative reward

Mathematical Formulation
------------------------

Training Objective
^^^^^^^^^^^^^^^^^^

UDRL is trained via **behavioral cloning** on a dataset :math:`\mathcal{D}` of 
trajectories collected by any policy:

.. math::
   :label: udrl_loss

   \mathcal{L}_{UDRL}(\theta) = \mathbb{E}_{(s, a, g) \sim \mathcal{D}} 
   \left[ -\log \pi_\theta(a | s, g) \right]

For continuous actions, this becomes:

.. math::
   :label: udrl_loss_continuous

   \mathcal{L}_{UDRL}(\theta) = \mathbb{E}_{(s, a, g) \sim \mathcal{D}} 
   \left[ \| a - \mu_\theta(s, g) \|^2 \right]

Command Computation
^^^^^^^^^^^^^^^^^^^

For each trajectory :math:`\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T)`,
we compute **hindsight commands** at each timestep :math:`t`:

.. math::
   :label: hindsight_command

   g_t = \left( H_t, R_t \right) = \left( T - t, \sum_{k=t}^{T} r_k \right)

This is the key insight: we label each state with the **actual** horizon and 
return that followed.

Inference Protocol
^^^^^^^^^^^^^^^^^^

At test time:

1. User specifies desired command :math:`g^* = (H^*, R^*)`
2. At each step :math:`t`:
   
   - Query :math:`a_t = \pi_\theta(s_t, g_t)`
   - Update command: :math:`g_{t+1} = (H_t - 1, R_t - r_t)`

3. Terminate when :math:`H_t = 0` or episode ends

Algorithm
---------

.. topic:: Algorithm: UDRL Training

   **Input:** Dataset :math:`\mathcal{D}` of trajectories
   
   **Output:** Trained policy :math:`\pi_\theta`
   
   1. **For each** trajectory :math:`\tau \in \mathcal{D}`:
   
      a. Compute hindsight commands :math:`\{g_t\}` using Eq. :eq:`hindsight_command`
      
      b. Add :math:`\{(s_t, a_t, g_t)\}` to training set
   
   2. **Train** :math:`\pi_\theta` via supervised learning on Eq. :eq:`udrl_loss`

Advantages of UDRL
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Advantage
     - Description
   * - **Simplicity**
     - Pure supervised learning, no value functions
   * - **Flexibility**
     - Works with any trajectory data
   * - **Multi-goal**
     - Single policy for many return levels
   * - **Offline-friendly**
     - No need for environment interaction

Limitations
-----------

The critical limitation that motivates FlowShield:

.. warning::
   
   **UDRL blindly trusts user commands.**
   
   If you request :math:`g = (50, 500)` on LunarLander (where max return ≈ 300),
   the agent will **try** to achieve this impossible goal, leading to erratic
   and dangerous behavior.

This is the **Obedient Suicide Problem** discussed in the next section.

Network Architecture
--------------------

Our UDRL policy uses:

.. code-block:: text

   Input: state s ∈ ℝ^8, command g ∈ ℝ^2
   
   ┌─────────────┐     ┌─────────────┐
   │   State s   │     │  Command g  │
   └──────┬──────┘     └──────┬──────┘
          │                   │
          ▼                   ▼
   ┌─────────────┐     ┌─────────────┐
   │   MLP (2)   │     │Fourier Enc. │
   └──────┬──────┘     └──────┬──────┘
          │                   │
          └─────────┬─────────┘
                    │
                    ▼
            ┌───────────────┐
            │  Concat + MLP │
            │   (4 layers)  │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │    μ, σ       │
            │ (action dist) │
            └───────────────┘

**Fourier Features** for command embedding improve extrapolation:

.. math::
   :label: fourier_features

   \gamma(g) = \left[ \sin(2\pi \mathbf{B} g), \cos(2\pi \mathbf{B} g) \right]

where :math:`\mathbf{B} \sim \mathcal{N}(0, \sigma^2)` is a fixed random matrix.

References
----------

This section is based on [Schmidhuber2019]_ and [Srivastava2019]_.

Fourier features for improved extrapolation are inspired by [Rahimi2007]_.

