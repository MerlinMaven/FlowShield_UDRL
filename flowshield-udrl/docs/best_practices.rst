===================
Best Practices Guide
===================

This guide covers best practices for training, evaluation, and deployment 
of FlowShield-UDRL systems.

.. contents:: Table of Contents
   :local:
   :depth: 2

Data Collection
---------------

Expert Data Quality
^^^^^^^^^^^^^^^^^^^

The quality of expert data directly impacts all downstream performance:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Criterion
     - Recommendation
   * - **Minimum episodes**
     - ≥400 episodes for LunarLander
   * - **Mean return**
     - Should be ≥90% of optimal (>220 for LunarLander)
   * - **Return variance**
     - Low variance indicates consistent expert (std < 30)
   * - **State coverage**
     - Full task space should be explored

.. code-block:: python

   # Check data quality before training
   data = np.load("expert_data.npz")
   returns = data["episode_returns"]
   
   assert len(returns) >= 400, "Need more episodes"
   assert np.mean(returns) >= 220, "Expert quality too low"
   assert np.std(returns) < 50, "Expert too inconsistent"
   
   print(f"Data quality: OK (R={np.mean(returns):.1f}±{np.std(returns):.1f})")

Command Distribution
^^^^^^^^^^^^^^^^^^^^

Ensure commands cover the intended operation range:

.. code-block:: python

   horizons = data["commands"][:, 0]
   returns = data["commands"][:, 1]
   
   # Check coverage
   print(f"Horizon range: [{horizons.min():.0f}, {horizons.max():.0f}]")
   print(f"Return range: [{returns.min():.0f}, {returns.max():.0f}]")
   
   # Visualize
   plt.scatter(horizons, returns, alpha=0.3)
   plt.xlabel("Horizon")
   plt.ylabel("Target Return")
   plt.title("Command Distribution")

Training Guidelines
-------------------

UDRL Policy
^^^^^^^^^^^

**Hyperparameters:**

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Parameter
     - Recommended
     - Notes
   * - ``hidden_dim``
     - 256
     - Larger for complex envs
   * - ``n_layers``
     - 4
     - With residual connections
   * - ``learning_rate``
     - 3e-4
     - Use scheduler for long training
   * - ``batch_size``
     - 256
     - Larger batches = more stable
   * - ``epochs``
     - 100-200
     - Until validation plateaus

**Early stopping:**

.. code-block:: python

   best_val_loss = float("inf")
   patience = 10
   patience_counter = 0
   
   for epoch in range(max_epochs):
       train_loss = train_epoch(...)
       val_loss = validate(...)
       
       if val_loss < best_val_loss:
           best_val_loss = val_loss
           patience_counter = 0
           save_checkpoint(model, "best.pt")
       else:
           patience_counter += 1
           
       if patience_counter >= patience:
           print(f"Early stopping at epoch {epoch}")
           break

Flow Shield
^^^^^^^^^^^

**Critical hyperparameters:**

- ``n_steps``: 10-20 ODE integration steps
- ``t_min``: 0.001 (avoid t=0 instability)
- ``log_prob_scale``: Tune for your threshold

**Training tips:**

1. Match data distribution exactly to UDRL training data
2. Use gradient clipping (max_norm=1.0)
3. Monitor log-probabilities during training
4. Validate OOD detection on held-out test set

Evaluation Protocol
-------------------

Command Selection
^^^^^^^^^^^^^^^^^

Test on a range of commands:

.. code-block:: python

   # In-distribution commands (from training data range)
   id_commands = [
       (200, 220),  # Typical
       (150, 200),  # Moderate
       (250, 250),  # High horizon
   ]
   
   # Out-of-distribution commands
   ood_commands = [
       (50, 350),   # Impossible: too high return
       (20, 200),   # Impossible: too short horizon
       (100, 500),  # Extreme: way above max
       (10, 100),   # Borderline: achievable but rare
   ]

Metrics to Report
^^^^^^^^^^^^^^^^^

**Always report:**

1. ``mean_return`` ± ``std_return``
2. ``success_rate`` and ``crash_rate``
3. ``shield_activation_rate`` (for shield methods)
4. ``auroc`` (OOD detection quality)

**For safety claims:**

1. ``obedient_suicide_rate``
2. ``crash_reduction_relative``
3. ``worst_5pct_mean`` (tail risk)

Statistical Significance
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from scipy import stats
   
   # Compare two methods with bootstrap CI
   n_bootstrap = 1000
   diffs = []
   
   for _ in range(n_bootstrap):
       idx = np.random.choice(len(returns_a), len(returns_a), replace=True)
       diff = np.mean(returns_a[idx]) - np.mean(returns_b[idx])
       diffs.append(diff)
   
   ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
   
   if ci_low > 0:
       print("Method A significantly better")
   elif ci_high < 0:
       print("Method B significantly better")
   else:
       print("No significant difference")

Deployment Considerations
-------------------------

Threshold Tuning
^^^^^^^^^^^^^^^^

The OOD detection threshold balances:

- **Low threshold**: More activations, safer but may hurt ID performance
- **High threshold**: Fewer activations, better ID but may miss OOD

.. code-block:: python

   # Find optimal threshold via validation
   thresholds = np.linspace(-10, -2, 50)
   best_threshold = -6.0
   best_score = 0
   
   for thresh in thresholds:
       shield.threshold = thresh
       metrics = evaluate(policy, shield, val_commands)
       
       # Optimize: good ID performance + good OOD detection
       score = metrics["id_return"] + metrics["ood_detection"] * 100
       
       if score > best_score:
           best_score = score
           best_threshold = thresh

Latency Requirements
^^^^^^^^^^^^^^^^^^^^

Shield inference adds latency:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Method
     - CPU (ms)
     - GPU (ms)
     - Suitable for
   * - Quantile
     - 0.5
     - 0.1
     - Real-time (>100 Hz)
   * - Flow
     - 2.0
     - 0.3
     - Fast control (>30 Hz)
   * - Diffusion
     - 10.0
     - 1.5
     - Slow control (<10 Hz)

Fallback Behavior
^^^^^^^^^^^^^^^^^

Define what happens when OOD is detected:

.. code-block:: python

   class SafeAgent:
       def __init__(self, policy, shield, fallback_command):
           self.policy = policy
           self.shield = shield
           self.fallback = fallback_command
       
       def act(self, state, command):
           # Check if command is safe
           log_prob = self.shield.log_prob(state, command)
           
           if log_prob < self.threshold:
               # Option 1: Project to nearest safe command
               safe_command = self.shield.project(state, command)
               
               # Option 2: Use conservative fallback
               # safe_command = self.fallback
               
               command = safe_command
           
           return self.policy.act(state, command)

Common Pitfalls
---------------

Training Issues
^^^^^^^^^^^^^^^

1. **Model trained on bad data**
   
   - Symptom: High loss but low return
   - Fix: Verify expert data quality first

2. **Architecture mismatch**
   
   - Symptom: Cannot load checkpoint
   - Fix: Match model architecture exactly

3. **Overfitting**
   
   - Symptom: Training loss drops, validation rises
   - Fix: Early stopping, dropout, weight decay

Evaluation Issues
^^^^^^^^^^^^^^^^^

1. **Evaluating with training commands only**
   
   - Symptom: Great results, but fails in deployment
   - Fix: Always test OOD commands

2. **Not enough episodes**
   
   - Symptom: Highly variable metrics
   - Fix: ≥50 episodes per command type

3. **Ignoring tail risk**
   
   - Symptom: Good mean, bad worst-case
   - Fix: Report worst_5pct_mean

Shield Issues
^^^^^^^^^^^^^

1. **Threshold too aggressive**
   
   - Symptom: Shield activates on ID commands
   - Fix: Tune threshold on validation set

2. **Shield trained on different data**
   
   - Symptom: Poor OOD detection
   - Fix: Shield must see same data distribution as policy
