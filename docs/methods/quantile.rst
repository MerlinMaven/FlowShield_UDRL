==============
Quantile Shield
==============

The Quantile Shield is our **baseline method**, using quantile regression to 
estimate conservative return bounds.

Overview
--------

Instead of modeling the full density :math:`p(g|s)`, we estimate quantiles:

.. math::
   :label: quantile_definition

   Q_\tau(R|s) = \inf \{ r : P(R \leq r | s) \geq \tau \}

For :math:`\tau = 0.9`, this gives the 90th percentile of achievable returns.

Mathematical Foundation
-----------------------

Pinball Loss
^^^^^^^^^^^^

Quantile regression minimizes the **pinball loss** (asymmetric L1):

.. math::
   :label: pinball_loss

   \mathcal{L}_\tau(\hat{R}, R) = \begin{cases}
   \tau (R - \hat{R}) & \text{if } R \geq \hat{R} \\
   (1-\tau) (\hat{R} - R) & \text{if } R < \hat{R}
   \end{cases}

Equivalently:

.. math::
   :label: pinball_compact

   \mathcal{L}_\tau(\hat{R}, R) = (R - \hat{R}) \cdot (\tau - \mathbb{1}[R < \hat{R}])

Training Objective
^^^^^^^^^^^^^^^^^^

We train a network :math:`f_\theta(s)` to predict :math:`Q_\tau(R|s)`:

.. math::
   :label: quantile_objective

   \min_\theta \mathbb{E}_{(s, R) \sim \mathcal{D}} 
   \left[ \mathcal{L}_\tau(f_\theta(s), R) \right]

The network predicts both horizon and return quantiles:

.. math::
   :label: quantile_output

   f_\theta(s) = (Q_\tau(H|s), Q_\tau(R|s))

Network Architecture
--------------------

Simple MLP architecture:

.. code-block:: text

   Input: state s ∈ ℝ^8
        │
        ▼
   ┌───────────────┐
   │    MLP        │
   │  (4 layers)   │
   │  hidden=256   │
   │  activation=  │
   │    SiLU       │
   └───────┬───────┘
           │
           ▼
   ┌───────────────┐
   │  Output: 2    │
   │ (H_τ, R_τ)    │
   └───────────────┘

OOD Detection
-------------

A command :math:`g = (H, R)` is flagged as OOD if:

.. math::
   :label: quantile_ood

   R > Q_\tau(R|s) + \epsilon

where :math:`\epsilon` is a tolerance margin.

For two-sided bounds, we train two networks:

- :math:`f_{\theta}^{\text{high}}` with :math:`\tau = 0.95` (upper bound)
- :math:`f_{\theta}^{\text{low}}` with :math:`\tau = 0.05` (lower bound)

Command Projection
------------------

Clip-Based Projection
^^^^^^^^^^^^^^^^^^^^^

Simple and fast:

.. math::
   :label: clip_projection_quantile

   g_{\text{safe}} = \left( 
     \text{clip}(H, H_{\min}, H_\tau), 
     \text{clip}(R, R_\tau^{\text{low}}, R_\tau^{\text{high}}) 
   \right)

Limitations
^^^^^^^^^^^

This projection is **state-independent** once the quantiles are computed.
It cannot adapt to local geometry of the achievable region.

Advantages
----------

.. list-table::
   :widths: 30 70

   * - **Speed**
     - Single forward pass for detection and projection
   * - **Simplicity**
     - Easy to implement and debug
   * - **Interpretability**
     - Quantiles have clear probabilistic meaning
   * - **Stability**
     - No iterative inference required

Limitations
-----------

.. list-table::
   :widths: 30 70

   * - **No density**
     - Cannot compute :math:`p(g|s)` directly
   * - **Unimodal**
     - Assumes single-peaked return distribution
   * - **Independence**
     - Treats H and R independently

Configuration
-------------

.. code-block:: yaml

   # configs/shield/quantile.yaml
   method: "quantile"
   
   # Architecture
   hidden_dim: 256
   n_layers: 4
   activation: "silu"
   command_dim: 2
   
   # Quantile regression
   tau: 0.9              # 90th percentile
   n_quantiles: 1        # Single quantile
   pinball_loss: true
   
   # Training
   learning_rate: 1e-3
   n_epochs: 100

Implementation
--------------

Key methods in ``QuantileShield``:

.. code-block:: python

   class QuantileShield(BaseShield):
       def compute_loss(self, state, command):
           """Pinball loss for quantile regression."""
           pred = self.network(state)
           target = command
           
           # Asymmetric loss
           error = target - pred
           loss = torch.where(
               error >= 0,
               self.tau * error,
               (self.tau - 1) * error
           )
           return loss.mean()
       
       def get_safety_score(self, state, command):
           """Negative distance to quantile boundary."""
           pred = self.network(state)  # (H_tau, R_tau)
           # Score: how much command exceeds quantile
           return pred[:, 1] - command[:, 1]  # R_tau - R_requested
       
       def project(self, state, command):
           """Clip to quantile bounds."""
           bounds = self.network(state)
           return torch.clamp(command, max=bounds)

When to Use
-----------

.. tip::
   **Recommended for** the following scenarios:
   
   - You need a **fast baseline** for comparison
   - Your return distribution is **unimodal**
   - You only need **one-sided bounds** (upper limit)
   - Interpretability is important

.. warning::
   **Not recommended when:**
   
   - You need **exact density estimation**
   - The command space has **complex geometry**
   - **Gradient-based projection** is required

Comparison with Other Methods
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Criterion
     - Quantile
     - Diffusion
     - Flow Matching
   * - Training speed
     - **High**
     - Medium
     - Medium
   * - Inference speed
     - **High**
     - Low
     - **High**
   * - Log-likelihood
     - None
     - Approximate
     - **Exact**
   * - Projection quality
     - Low
     - Medium
     - **High**

References
----------

Quantile regression fundamentals from [Koenker2001]_.

Distributional RL with quantile regression from [Dabney2018]_.
