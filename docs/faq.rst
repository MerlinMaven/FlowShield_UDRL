===
FAQ
===

Frequently Asked Questions about FlowShield-UDRL.

General Questions
-----------------

What is FlowShield-UDRL?
^^^^^^^^^^^^^^^^^^^^^^^^

FlowShield-UDRL is a safety framework for goal-conditioned reinforcement learning.
It combines:

- **UDRL (Upside-Down RL):** Policy conditioned on target horizon and return
- **FlowShield:** Density-based detection and correction of dangerous commands

The system prevents "Obedient Suicide" where an agent follows impossible commands
leading to failure.

Why "Obedient Suicide"?
^^^^^^^^^^^^^^^^^^^^^^^

The term describes a failure mode where:

1. Agent receives an out-of-distribution command (e.g., "achieve 500 reward in 10 steps")
2. Agent obediently attempts the impossible task
3. This leads to dangerous behavior and failure

The agent "commits suicide" by following orders it should have refused.

How is this different from safe RL?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Traditional safe RL methods:

- Constrain the **policy** to avoid unsafe actions
- Use reward shaping or constraints during training

FlowShield:

- Constrains the **commands** given to the policy
- Works at inference time, no retraining needed
- Detects when commands are outside training distribution

Technical Questions
-------------------

Which shield method should I use?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Use case
     - Recommendation
   * - Fast baseline
     - Quantile Shield
   * - High quality, slow inference OK
     - Diffusion Shield
   * - Best overall (recommended)
     - **Flow Matching Shield**

Flow Matching provides the best balance of accuracy, speed, and theoretical guarantees.

How do I choose the threshold?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the **5th percentile** of log-probabilities on validation data:

.. code-block:: python

   # Compute log-probs on validation set
   log_probs = shield.log_prob(val_states, val_commands)
   
   # 5th percentile as threshold
   threshold = np.percentile(log_probs.numpy(), 5)

This ensures 95% of in-distribution commands pass.

How many ODE steps for Flow Matching?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recommended: **50 Euler steps** or **20 RK4 steps**.

More steps = higher accuracy but slower inference.

For real-time applications, 20 Euler steps may be acceptable.

Why not just clip commands?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simple clipping:

.. list-table::
   :widths: 70 30
   :header-rows: 0
   
   * - Ignores state-dependence
     - **Limitation**
   * - Cannot handle correlated H and R
     - **Limitation**
   * - No probability estimation
     - **Limitation**

FlowShield:

.. list-table::
   :widths: 70 30
   :header-rows: 0
   
   * - State-conditioned density
     - **Advantage**
   * - Models joint (H, R) distribution
     - **Advantage**
   * - Provides calibrated confidence
     - **Advantage**

Training Questions
------------------

How much data do I need?
^^^^^^^^^^^^^^^^^^^^^^^^

Recommended: **10,000 episodes** per environment.

You can start with less for initial experiments:

- 1,000 episodes: Basic functionality
- 5,000 episodes: Reasonable performance
- 10,000 episodes: Near-optimal
- 25,000+ episodes: Diminishing returns

What behavioral policy should I use?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Mixed policies** work best:

- Random exploration (30%)
- Noisy expert (50%)
- Pure expert (20%)

This provides diverse commands covering the achievable distribution.

How long does training take?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On a single RTX 3090:

.. list-table::
   :widths: 40 30 30

   * - Component
     - LunarLander
     - Highway
   * - Data collection
     - 1 hour
     - 2 hours
   * - UDRL training
     - 2 hours
     - 4 hours
   * - Shield training
     - 1 hour
     - 2 hours

Debugging
---------

My shield rejects everything
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Possible causes:

1. **Threshold too high:** Lower the threshold (more negative)
2. **Insufficient training:** Train shield longer
3. **Data mismatch:** Ensure test commands match training distribution range

.. code-block:: python

   # Check log-prob distribution
   log_probs = shield.log_prob(states, commands)
   print(f"Log-prob range: [{log_probs.min():.1f}, {log_probs.max():.1f}]")
   print(f"Threshold: {threshold}")

My shield accepts everything
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Possible causes:

1. **Threshold too low:** Raise the threshold (less negative)
2. **Undertrained shield:** More epochs or larger model
3. **Commands not actually OOD:** Verify test commands are outside training

Projection doesn't improve commands
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Check:

1. **Number of steps:** Increase projection steps (20-50)
2. **Learning rate:** Try different values (0.05-0.2)
3. **Gradient magnitude:** Verify gradients are non-zero

.. code-block:: python

   # Debug projection
   for step in range(20):
       log_p = shield.log_prob(state, command)
       grad = torch.autograd.grad(log_p, command)[0]
       print(f"Step {step}: log_p={log_p.item():.2f}, |grad|={grad.norm():.4f}")
       command = command + 0.1 * grad

Performance Issues
------------------

Inference is too slow
^^^^^^^^^^^^^^^^^^^^^

Options:

1. **Reduce ODE steps:** 50 → 20
2. **Use simpler solver:** RK4 → Euler
3. **Batch processing:** Evaluate multiple states together
4. **GPU acceleration:** Ensure CUDA is enabled

Training loss not decreasing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Try:

1. **Lower learning rate:** 1e-4 → 1e-5
2. **Increase model capacity:** More layers or hidden dims
3. **Check data:** Ensure (state, command) pairs are correct
4. **Gradient clipping:** Add gradient norm clipping

Extending FlowShield
--------------------

How do I add a new environment?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Create wrapper inheriting from ``ContinuousEnvWrapper``
2. Register in ``envs/factory.py``
3. Create config in ``configs/env/``
4. Collect training data
5. Train and evaluate

See :doc:`environments/overview` for details.

How do I add a new shield method?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Create class inheriting from ``BaseShield``
2. Implement required methods:
   - ``compute_loss``
   - ``log_prob``
   - ``is_ood``
   - ``project``
3. Register in config
4. Test on existing environments

Can I use FlowShield with other RL algorithms?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes! FlowShield works with any goal-conditioned policy:

- Hindsight Experience Replay (HER)
- Goal-Conditioned Behavioral Cloning
- Other command-conditioned methods

The shield only needs (state, command) pairs for training.

Citations
---------

How should I cite this work?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bibtex

   @article{flowshield2024,
     title={FlowShield: Safe Goal-Conditioned RL via Density-Based Command Shielding},
     author={...},
     journal={...},
     year={2024}
   }

Further Reading
^^^^^^^^^^^^^^^^

- Goal-conditioned RL: [Kaelbling1993]_, [Schaul2015]_
- Hindsight experience replay for command learning: [Andrychowicz2017]_
- Attention mechanisms for feature encoding: [Vaswani2017]_

