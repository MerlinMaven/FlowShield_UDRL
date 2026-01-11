=============
Safety Shields
=============

Safety shields are modules that intercept user commands before they reach the 
UDRL policy, detecting and correcting out-of-distribution (OOD) requests.

Overview
--------

A safety shield implements two core functions:

.. math::
   :label: shield_functions

   \begin{aligned}
   \text{is\_safe}(s, g) &: \mathcal{S} \times \mathcal{G} \to \{0, 1\} \\
   \text{project}(s, g) &: \mathcal{S} \times \mathcal{G} \to \mathcal{G}
   \end{aligned}

The shield pipeline:

.. code-block:: text

   User Command g*        Safety Shield           UDRL Policy
        │                      │                      │
        ▼                      ▼                      ▼
   ┌─────────┐          ┌───────────┐          ┌───────────┐
   │ H=50    │    ──▶   │ is_safe?  │    ──▶   │ π(a|s,g)  │
   │ R=500   │          │           │          │           │
   └─────────┘          │ If OOD:   │          └───────────┘
                        │ project() │
                        └───────────┘
                              │
                              ▼
                        ┌───────────┐
                        │ g_safe:   │
                        │ H=50      │
                        │ R=200     │
                        └───────────┘

Core Principle: Density Estimation
----------------------------------

All our shield methods are based on learning the conditional density:

.. math::
   :label: conditional_density

   p_\theta(g | s)

Given this density, we can:

1. **Detect OOD:** A command is OOD if :math:`\log p_\theta(g|s) < \tau`

2. **Project:** Find the nearest in-distribution command:

   .. math::
      :label: projection

      g_{\text{safe}} = \arg\max_{g' \in \mathcal{G}} \log p_\theta(g' | s)
      \quad \text{s.t.} \quad \|g' - g\| \leq \epsilon

Three Shield Methods
--------------------

We implement three approaches to learning :math:`p_\theta(g|s)`:

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Method
     - Training
     - Inference
     - Log-likelihood
   * - **Quantile**
     - Pinball loss
     - Direct prediction
     - Implicit (threshold)
   * - **Diffusion**
     - Denoising score matching
     - Iterative denoising
     - Approximate
   * - **Flow Matching**
     - Velocity field regression
     - ODE integration
     - Exact (change of variables)

OOD Detection
-------------

Threshold Selection
^^^^^^^^^^^^^^^^^^^

The threshold :math:`\tau` controls the trade-off:

- **Low τ:** Permissive, may allow dangerous commands
- **High τ:** Conservative, may reject valid commands

.. math::
   :label: threshold_tradeoff

   \begin{aligned}
   \text{FPR}(\tau) &= P(\log p(g|s) < \tau | g \text{ is ID}) \\
   \text{TPR}(\tau) &= P(\log p(g|s) < \tau | g \text{ is OOD})
   \end{aligned}

We select :math:`\tau` to achieve a target false positive rate (e.g., 5%).

State-Dependent Thresholds
^^^^^^^^^^^^^^^^^^^^^^^^^^

Optionally, use percentile-based thresholds:

.. math::
   :label: percentile_threshold

   \tau(s) = \text{percentile}_\alpha \left( \{ \log p(g_i | s) : g_i \in \mathcal{D} \} \right)

This adapts to the local density around each state.

Projection Methods
------------------

Gradient-Based Projection
^^^^^^^^^^^^^^^^^^^^^^^^^

The most effective method for Flow Matching:

.. math::
   :label: gradient_projection

   g_{t+1} = g_t + \eta \nabla_g \log p_\theta(g_t | s)

This performs gradient ascent to find higher-density commands.

.. topic:: Algorithm: Gradient Projection

   **Input:** OOD command :math:`g^*`, state :math:`s`, steps :math:`N`, learning rate :math:`\eta`
   
   **Output:** Projected command :math:`g_{\text{safe}}`
   
   1. Initialize :math:`g_0 = g^*`
   
   2. **For** :math:`t = 0, \ldots, N-1`:
   
      a. Compute :math:`\nabla_g \log p_\theta(g_t | s)`
      
      b. Update :math:`g_{t+1} = g_t + \eta \nabla_g \log p_\theta(g_t | s)`
   
   3. **Return** :math:`g_N`

Sampling-Based Projection
^^^^^^^^^^^^^^^^^^^^^^^^^

Alternative: sample from :math:`p(g|s)` and select closest to :math:`g^*`:

.. math::
   :label: sampling_projection

   g_{\text{safe}} = \arg\min_{g_i \sim p(g|s)} \| g_i - g^* \|

Clip Projection (Baseline)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Simple but state-independent:

.. math::
   :label: clip_projection

   g_{\text{safe}} = \text{clip}(g^*, g_{\min}, g_{\max})

where bounds are computed from training data.

Mathematical Guarantees
-----------------------

Projection Quality
^^^^^^^^^^^^^^^^^^

For gradient projection with step size :math:`\eta` and :math:`L`-smooth 
log-density:

.. math::
   :label: projection_convergence

   \log p(g_N | s) - \log p(g^* | s) \geq \frac{1}{2\eta L} \sum_{t=0}^{N-1} 
   \| \nabla_g \log p(g_t | s) \|^2

Detection Consistency
^^^^^^^^^^^^^^^^^^^^^

Under standard assumptions, as :math:`|\mathcal{D}| \to \infty`:

.. math::
   :label: detection_consistency

   P(\text{is\_safe}(s, g) = \mathbb{1}[g \text{ is truly ID}]) \to 1

Implementation
--------------

All shields inherit from ``BaseShield``:

.. code-block:: python

   class BaseShield(nn.Module):
       def compute_loss(self, state, command) -> Tensor:
           """Training objective."""
           ...
       
       def get_safety_score(self, state, command) -> Tensor:
           """Higher = more likely in-distribution."""
           ...
       
       def is_safe(self, state, command, threshold) -> Tensor:
           """Boolean mask: True = safe command."""
           ...
       
       def project(self, state, command) -> Tensor:
           """Project OOD command to safe region."""
           ...
       
       def log_prob(self, state, command) -> Tensor:
           """Log probability (exact for Flow, approximate for others)."""
           ...

See the method-specific pages for implementation details:

- :doc:`../methods/quantile`
- :doc:`../methods/diffusion`
- :doc:`../methods/flow_matching`

References
----------

Safe RL foundations: [Amodei2016]_ and [Garcia2015]_.

FFJORD for continuous normalizing flows: [Grathwohl2018]_.

