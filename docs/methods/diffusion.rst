================
Diffusion Shield
================

The Diffusion Shield uses **Denoising Diffusion Probabilistic Models (DDPM)** 
to learn the conditional distribution :math:`p(g|s)`.

Overview
--------

Diffusion models learn a data distribution by:

1. **Forward process:** Gradually add noise to data
2. **Reverse process:** Learn to denoise step-by-step

For our shield, we learn :math:`p(g|s)` by conditioning the reverse process 
on the state :math:`s`.

Mathematical Foundation
-----------------------

Forward Diffusion Process
^^^^^^^^^^^^^^^^^^^^^^^^^

Starting from data :math:`g_0 \sim p_{\text{data}}(g|s)`, we define a 
Markov chain that gradually adds Gaussian noise:

.. math::
   :label: forward_diffusion

   q(g_t | g_{t-1}) = \mathcal{N}(g_t; \sqrt{1-\beta_t} g_{t-1}, \beta_t \mathbf{I})

where :math:`\beta_t \in (0, 1)` is the noise schedule.

The marginal at time :math:`t`:

.. math::
   :label: forward_marginal

   q(g_t | g_0) = \mathcal{N}(g_t; \sqrt{\bar{\alpha}_t} g_0, (1-\bar{\alpha}_t) \mathbf{I})

where :math:`\alpha_t = 1 - \beta_t` and :math:`\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s`.

**Reparameterization:** We can sample :math:`g_t` directly:

.. math::
   :label: reparameterization

   g_t = \sqrt{\bar{\alpha}_t} g_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, 
   \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})

Reverse Process
^^^^^^^^^^^^^^^

The reverse process learns to denoise:

.. math::
   :label: reverse_diffusion

   p_\theta(g_{t-1} | g_t, s) = \mathcal{N}(g_{t-1}; \mu_\theta(g_t, t, s), \sigma_t^2 \mathbf{I})

Training Objective
^^^^^^^^^^^^^^^^^^

We train a noise prediction network :math:`\epsilon_\theta(g_t, t, s)`:

.. math::
   :label: ddpm_loss

   \mathcal{L}_{\text{DDPM}} = \mathbb{E}_{t, g_0, \epsilon} 
   \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} g_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, t, s) \|^2 \right]

This is **denoising score matching**—equivalent to learning the score function:

.. math::
   :label: score_relation

   \nabla_g \log q(g_t | g_0) = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}

Noise Schedules
---------------

Linear Schedule
^^^^^^^^^^^^^^^

.. math::
   :label: linear_schedule

   \beta_t = \beta_{\min} + \frac{t}{T}(\beta_{\max} - \beta_{\min})

Typical values: :math:`\beta_{\min} = 10^{-4}`, :math:`\beta_{\max} = 0.02`.

Cosine Schedule
^^^^^^^^^^^^^^^

Smoother noise addition:

.. math::
   :label: cosine_schedule

   \bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2

where :math:`s = 0.008` is a small offset.

Quadratic Schedule
^^^^^^^^^^^^^^^^^^

More noise early in training:

.. math::
   :label: quadratic_schedule

   \beta_t = \beta_{\min} + \left(\frac{t}{T}\right)^2 (\beta_{\max} - \beta_{\min})

Network Architecture
--------------------

The noise prediction network uses time and state conditioning:

.. code-block:: text

   Inputs: noisy g_t, timestep t, state s
   
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
               └──────┬──────┘
                      │
                      ▼
               ┌─────────────┐
               │  ε_θ output │
               │   (dim=2)   │
               └─────────────┘

**Sinusoidal Time Embedding:**

.. math::
   :label: time_embedding

   \text{PE}(t, 2i) &= \sin(t / 10000^{2i/d}) \\
   \text{PE}(t, 2i+1) &= \cos(t / 10000^{2i/d})

Sampling (Inference)
--------------------

DDPM Sampling
^^^^^^^^^^^^^

Starting from :math:`g_T \sim \mathcal{N}(0, \mathbf{I})`:

.. topic:: Algorithm: DDPM Sampling

   **For** :math:`t = T, T-1, \ldots, 1`:
   
   1. Predict noise: :math:`\hat{\epsilon} = \epsilon_\theta(g_t, t, s)`
   
   2. Compute mean:
   
      .. math::
         \mu_t = \frac{1}{\sqrt{\alpha_t}} \left( g_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \hat{\epsilon} \right)
   
   3. Sample: :math:`g_{t-1} = \mu_t + \sigma_t z`, where :math:`z \sim \mathcal{N}(0, \mathbf{I})`
   
   **Return** :math:`g_0`

DDIM Sampling (Faster)
^^^^^^^^^^^^^^^^^^^^^^

Deterministic sampling with fewer steps:

.. math::
   :label: ddim

   g_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{g}_0 + \sqrt{1-\bar{\alpha}_{t-1}} \hat{\epsilon}

where :math:`\hat{g}_0 = \frac{g_t - \sqrt{1-\bar{\alpha}_t} \hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}`.

OOD Detection
-------------

Reconstruction Error
^^^^^^^^^^^^^^^^^^^^

For OOD detection, we use reconstruction error:

1. Add noise to command: :math:`g_t = \sqrt{\bar{\alpha}_t} g + \sqrt{1-\bar{\alpha}_t} \epsilon`
2. Denoise: :math:`\hat{g}_0 = \text{Denoise}(g_t, s)`
3. Score: :math:`\text{score} = -\|g - \hat{g}_0\|^2`

High reconstruction error indicates OOD.

ELBO-Based
^^^^^^^^^^

The Evidence Lower Bound provides an approximate log-likelihood:

.. math::
   :label: elbo

   \log p(g|s) \geq -\mathcal{L}_{\text{ELBO}}(g, s)

Command Projection
------------------

Conditional Sampling
^^^^^^^^^^^^^^^^^^^^

For projection, we sample from :math:`p(g|s)` and select the closest sample 
to the original command:

.. code-block:: python

   def project(self, state, command, n_samples=100):
       # Sample from p(g|s)
       samples = self.sample(state, n_samples)
       
       # Find closest to original command
       distances = torch.cdist(samples, command.unsqueeze(1))
       closest_idx = distances.argmin(dim=0)
       
       return samples[closest_idx]

Guided Sampling
^^^^^^^^^^^^^^^

Alternatively, use classifier-free guidance toward the target:

.. math::
   :label: guided_sampling

   \hat{\epsilon}_{\text{guided}} = \hat{\epsilon} - w \cdot \nabla_g \|g - g^*\|^2

Configuration
-------------

.. code-block:: yaml

   # configs/shield/diffusion.yaml
   method: "diffusion"
   
   # Architecture
   hidden_dim: 256
   n_layers: 4
   time_embed_dim: 128
   
   # Diffusion process
   n_timesteps: 1000
   beta_start: 1e-4
   beta_end: 0.02
   beta_schedule: "linear"  # linear, cosine, quadratic
   
   # Inference
   n_inference_steps: 20    # DDIM-style fast sampling
   clip_denoised: true
   
   # Training
   learning_rate: 1e-4
   n_epochs: 200
   use_ema: true
   ema_decay: 0.999

Advantages
----------

.. list-table::
   :widths: 30 70

   * - **Quality**
     - High-fidelity generation of command distribution
   * - **Flexibility**
     - Can model complex, multimodal distributions
   * - **Conditioning**
     - Natural mechanism for state conditioning

Limitations
-----------

.. list-table::
   :widths: 30 70

   * - **Speed**
     - Requires many denoising steps (20-1000)
   * - **Likelihood**
     - No exact log-likelihood (only ELBO)
   * - **Complexity**
     - More hyperparameters than Quantile

When to Use
-----------

.. tip::
   **Recommended for** the following scenarios:
   
   - You need **high-quality** samples from :math:`p(g|s)`
   - The command distribution is **multimodal**
   - Training stability is more important than inference speed

.. warning::
   **Not recommended when:**
   
   - You need **exact log-likelihood**
   - **Real-time** inference is critical
   - You prefer simpler methods

References
----------

This section is based on [Ho2020]_ and [Song2020]_.

DDIM sampling for faster inference was introduced in [Song2021]_.
