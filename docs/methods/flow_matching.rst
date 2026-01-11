===================
Flow Matching Shield
===================

The Flow Matching Shield uses **Optimal Transport Conditional Flow Matching (OT-CFM)** 
to learn the conditional distribution :math:`p(g|s)`. This is our **recommended method**.

Overview
--------

Flow Matching learns a continuous-time flow that transforms a simple prior 
(Gaussian) into the data distribution. Key advantages:

- **Exact log-likelihood** via change of variables
- **Fast inference** with ODE solvers (no iterative denoising)
- **Stable training** with simple regression loss

Mathematical Foundation
-----------------------

Continuous Normalizing Flows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A flow defines a time-dependent transformation:

.. math::
   :label: flow_ode

   \frac{dz}{dt} = v_\theta(z, t, s), \quad z(0) = z_0 \sim p_0(z), \quad z(1) = g \sim p_1(g|s)

where :math:`v_\theta` is the **velocity field** we learn.

Change of Variables
^^^^^^^^^^^^^^^^^^^

The density at time :math:`t` follows:

.. math::
   :label: change_of_variables

   \log p_t(z|s) = \log p_0(z_0) - \int_0^t \text{div}(v_\theta(z(\tau), \tau, s)) \, d\tau

At :math:`t = 1`, this gives the **exact log-likelihood** of the data.

Conditional Flow Matching
-------------------------

Training Objective
^^^^^^^^^^^^^^^^^^

CFM uses a simple regression loss:

.. math::
   :label: cfm_loss

   \mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, g_0, g_1} 
   \left[ \| v_\theta(g_t, t, s) - u_t(g_t | g_0, g_1) \|^2 \right]

where:

- :math:`t \sim \mathcal{U}(0, 1)`
- :math:`g_0 \sim p_0 = \mathcal{N}(0, \mathbf{I})` (prior)
- :math:`g_1 \sim p_{\text{data}}(g|s)` (data)
- :math:`g_t` is the interpolation between :math:`g_0` and :math:`g_1`
- :math:`u_t` is the target velocity

Optimal Transport Interpolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use **straight-line interpolation** (OT-CFM):

.. math::
   :label: ot_interpolation

   g_t = (1 - t) g_0 + t g_1

The target velocity is simply:

.. math::
   :label: ot_velocity

   u_t(g_t | g_0, g_1) = g_1 - g_0

This gives **optimal transport** paths that don't cross, leading to stable training.

Gaussian Conditional Path
^^^^^^^^^^^^^^^^^^^^^^^^^

With noise :math:`\sigma_{\min}`:

.. math::
   :label: gaussian_path

   p_t(g | g_1) = \mathcal{N}(g; t g_1, (1-t+\sigma_{\min})^2 \mathbf{I})

At :math:`t=0`: broad Gaussian prior

At :math:`t=1`: concentrated at data point :math:`g_1`

Network Architecture
--------------------

Similar to Diffusion, with time conditioning:

.. code-block:: text

   Inputs: noisy g_t, time t, state s
   
   ┌──────────┐  ┌──────────┐  ┌──────────┐
   │   g_t    │  │    t     │  │    s     │
   └────┬─────┘  └────┬─────┘  └────┬─────┘
        │             │             │
        │        ┌────▼────┐        │
        │        │Sinusoidal│        │
        │        │Embedding │        │
        │        └────┬────┘        │
        │             │             │
        └─────────────┼─────────────┘
                      │
               ┌──────▼──────┐
               │   Concat    │
               │   + MLP     │
               │ (4 layers)  │
               │  SiLU act.  │
               └──────┬──────┘
                      │
                      ▼
               ┌─────────────┐
               │ v_θ output  │
               │   (dim=2)   │
               └─────────────┘

Sampling (Inference)
--------------------

Euler Method
^^^^^^^^^^^^

Simple and effective:

.. topic:: Algorithm: Euler Sampling

   **Input:** State :math:`s`, number of steps :math:`N`
   
   **Output:** Sample :math:`g \sim p(g|s)`
   
   1. Initialize :math:`z_0 \sim \mathcal{N}(0, \mathbf{I})`
   
   2. Set :math:`\Delta t = 1/N`
   
   3. **For** :math:`k = 0, \ldots, N-1`:
   
      a. Compute :math:`t_k = k \cdot \Delta t`
      
      b. Predict velocity: :math:`v = v_\theta(z_k, t_k, s)`
      
      c. Update: :math:`z_{k+1} = z_k + \Delta t \cdot v`
   
   4. **Return** :math:`z_N`

Midpoint Method
^^^^^^^^^^^^^^^

Second-order accuracy:

.. math::
   :label: midpoint

   \begin{aligned}
   v_{\text{mid}} &= v_\theta(z_k + \frac{\Delta t}{2} v_\theta(z_k, t_k, s), t_k + \frac{\Delta t}{2}, s) \\
   z_{k+1} &= z_k + \Delta t \cdot v_{\text{mid}}
   \end{aligned}

RK4 (Runge-Kutta 4)
^^^^^^^^^^^^^^^^^^^

Fourth-order accuracy for smooth flows:

.. math::
   :label: rk4

   \begin{aligned}
   k_1 &= v_\theta(z, t, s) \\
   k_2 &= v_\theta(z + \frac{h}{2}k_1, t + \frac{h}{2}, s) \\
   k_3 &= v_\theta(z + \frac{h}{2}k_2, t + \frac{h}{2}, s) \\
   k_4 &= v_\theta(z + h k_3, t + h, s) \\
   z_{t+h} &= z_t + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)
   \end{aligned}

Log-Likelihood Computation
--------------------------

Exact via Divergence
^^^^^^^^^^^^^^^^^^^^

Using the instantaneous change of variables:

.. math::
   :label: exact_logprob

   \log p(g|s) = \log p_0(g_0) - \int_0^1 \text{div}(v_\theta(g_t, t, s)) \, dt

Computing the divergence exactly requires :math:`O(d)` evaluations.

Hutchinson Trace Estimator
^^^^^^^^^^^^^^^^^^^^^^^^^^

For efficiency, we estimate the divergence stochastically:

.. math::
   :label: hutchinson

   \text{div}(v) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,\mathbf{I})} 
   \left[ \epsilon^\top \frac{\partial v}{\partial g} \epsilon \right]

This requires only **one** Jacobian-vector product per sample.

.. topic:: Algorithm: Log-Probability Computation

   **Input:** Command :math:`g`, state :math:`s`, steps :math:`N`
   
   **Output:** :math:`\log p(g|s)`
   
   1. Set :math:`z_1 = g`, :math:`\log p = 0`
   
   2. **For** :math:`k = N, \ldots, 1` (reverse):
   
      a. :math:`t_k = k / N`
      
      b. Sample :math:`\epsilon \sim \mathcal{N}(0, \mathbf{I})`
      
      c. Compute :math:`v, \frac{\partial v}{\partial g} \epsilon` via autodiff
      
      d. :math:`\text{div}_k = \epsilon^\top \frac{\partial v}{\partial g} \epsilon`
      
      e. :math:`z_{k-1} = z_k - \Delta t \cdot v`
      
      f. :math:`\log p \mathrel{+}= \Delta t \cdot \text{div}_k`
   
   3. :math:`\log p \mathrel{+}= \log p_0(z_0)`
   
   4. **Return** :math:`\log p`

OOD Detection
-------------

Direct Log-Likelihood
^^^^^^^^^^^^^^^^^^^^^

OOD detection uses the computed log-probability:

.. math::
   :label: ood_detection_flow

   \text{is\_OOD}(g, s) = \mathbb{1}[\log p(g|s) < \tau]

This is **exact** (up to numerical precision), unlike Diffusion's ELBO.

Command Projection
------------------

Gradient-Based Projection
^^^^^^^^^^^^^^^^^^^^^^^^^

The key advantage of Flow Matching: **exact gradients** of log-likelihood.

.. math::
   :label: flow_projection

   g_{k+1} = g_k + \eta \nabla_g \log p_\theta(g_k | s)

This performs gradient ascent to find higher-density commands.

.. topic:: Algorithm: Gradient Projection

   **Input:** OOD command :math:`g^*`, state :math:`s`, steps :math:`K`, lr :math:`\eta`
   
   **Output:** Projected command :math:`g_{\text{safe}}`
   
   1. :math:`g_0 = g^*`
   
   2. **For** :math:`k = 0, \ldots, K-1`:
   
      a. Compute :math:`\log p(g_k | s)` and :math:`\nabla_g \log p(g_k | s)`
      
      b. :math:`g_{k+1} = g_k + \eta \nabla_g \log p(g_k | s)`
   
   3. **Return** :math:`g_K`

With momentum (Adam-like):

.. math::
   :label: momentum_projection

   \begin{aligned}
   m_{k+1} &= \beta m_k + (1-\beta) \nabla_g \log p(g_k | s) \\
   g_{k+1} &= g_k + \eta m_{k+1}
   \end{aligned}

Configuration
-------------

.. code-block:: yaml

   # configs/shield/flow_matching.yaml
   method: "flow_matching"
   
   # Architecture
   hidden_dim: 256
   n_layers: 4
   time_embed_dim: 128
   
   # Flow Matching
   sigma_min: 1e-4           # Minimum noise
   use_ot_plan: true         # OT interpolation
   
   # ODE Solver
   solver: "euler"           # euler, midpoint, rk4
   n_integration_steps: 50
   
   # Density estimation
   hutchinson_samples: 10
   
   # Training
   learning_rate: 1e-4
   n_epochs: 200
   use_ema: true
   ema_decay: 0.999
   
   # Projection
   projection_method: "gradient"
   projection_steps: 50
   projection_lr: 0.1

Advantages
----------

.. list-table::
   :widths: 30 70

   * - **Exact likelihood**
     - True :math:`\log p(g|s)` via change of variables
   * - **Fast inference**
     - 10-50 ODE steps (vs 100-1000 for DDPM)
   * - **Stable training**
     - Simple regression loss, no noise schedule tuning
   * - **Gradient projection**
     - Exact gradients for high-quality projection

Limitations
-----------

.. list-table::
   :widths: 30 70

   * - **Divergence cost**
     - Computing log-prob requires Jacobian traces
   * - **Solver choice**
     - Higher-order solvers need more function evaluations

Comparison Summary
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Feature
     - Quantile
     - Diffusion
     - **Flow Matching**
   * - Log-likelihood
     - None
     - Approximate
     - **Exact**
   * - Training
     - Pinball loss
     - Denoising
     - Velocity regression
   * - Inference steps
     - 1
     - 20-1000
     - 10-50
   * - Projection
     - Clip
     - Sampling
     - **Gradient ascent**
   * - Complexity
     - Low
     - High
     - Medium

Why Flow Matching is Recommended
--------------------------------

For FlowShield's use case, Flow Matching provides:

1. **OOD Detection:** Exact log-likelihood is critical for reliable detection
2. **Projection:** Gradient-based projection finds optimal safe commands
3. **Speed:** Few ODE steps enable real-time safety checking
4. **Simplicity:** No noise schedule tuning required

References
----------

This section is based on [Lipman2022]_, [Tong2023]_, and [Liu2022]_.

Neural ODEs for continuous normalizing flows were introduced in [Chen2018]_.

