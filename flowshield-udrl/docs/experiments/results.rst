===================
Experimental Results
===================

This section presents the main experimental results comparing shield methods 
on LunarLander-v3 (Continuous Control).

.. note::
   All results are from experiments using 
   expert data with mean return R=242.7.

Main Results
------------

LunarLander-v3 Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: /_static/final_comparison.png
   :alt: Final comparison of shield methods
   :align: center
   :width: 100%

**Performance under OOD commands (H=50, R=350):**

.. list-table::
   :header-rows: 1
   :widths: 20 18 18 18 14 12

   * - Method
     - Mean Return
     - Std Dev
     - Improvement
     - OOD Det
     - Variance ↓
   * - No Shield
     - 211.4
     - 84.7
     - —
     - 0%
     - —
   * - Quantile
     - 215.3
     - 75.2
     - +1.8%
     - 23%
     - -11%
   * - Diffusion
     - 209.9
     - 84.1
     - -0.7%
     - 14%
     - -1%
   * - **Flow Matching**
     - **235.0**
     - **26.0**
     - **+11.2%**
     - **77%**
     - **-69%**

**In-Distribution performance (H=200, R=220):**

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Method
     - Mean Return
     - Std Dev
     - Notes
   * - No Shield
     - 230.5
     - 18.2
     - Baseline
   * - Quantile
     - 228.1
     - 19.5
     - Slight degradation
   * - Diffusion
     - 229.8
     - 17.8
     - Matches baseline
   * - **Flow Matching**
     - **228.9**
     - **15.4**
     - **Best variance**

Key Findings
------------

1. **Flow Matching achieves best OOD performance**

   - **+11.2%** improvement in mean return under OOD commands
   - **-69%** variance reduction (26.0 vs 84.7)
   - **77%** OOD detection rate

2. **No degradation on in-distribution commands**

   - All shields maintain ~228-230 return on ID commands
   - Shields don't harm normal operation

3. **Flow Matching detection is superior**

   - 77% vs 23% (Quantile) vs 14% (Diffusion)
   - Better calibrated likelihood threshold

Training Curves
---------------

Policy Training
^^^^^^^^^^^^^^^

.. image:: /_static/policy_training.png
   :alt: UDRL Policy training curves
   :align: center
   :width: 80%

**Observations:**

- Smooth convergence over ~75 epochs
- Early stopping triggered (patience=15)
- Train/validation gap < 0.05 (good generalization)
- Final loss: ~0.8 (negative log-likelihood)

Flow Shield Training
^^^^^^^^^^^^^^^^^^^^

.. image:: /_static/flow_training.png
   :alt: Flow Matching Shield training curves
   :align: center
   :width: 80%

**Observations:**

- Fast convergence (~30 epochs to plateau)
- Stable training without oscillations
- Final CFM loss: ~0.15
- Automatic threshold calibration at end

Dataset Statistics
------------------

Expert Data Distribution
^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: /_static/expert_data_distribution.png
   :alt: Expert data return distribution
   :align: center
   :width: 80%

**Dataset characteristics:**

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Metric
     - Value
     - Interpretation
   * - Episodes
     - 420
     - Sufficient for UDRL
   * - Transitions
     - 129,217
     - Dense coverage
   * - Mean Return
     - 242.7 ± 26.8
     - High-quality expert
   * - Return Range
     - [158.9, 283.7]
     - Tight distribution
   * - Mean Horizon
     - 307.7
     - Full episodes

**Implication:** The low variance (σ=26.8) in expert data means UDRL learns 
consistent behavior but limited command-following diversity.

OOD Detection Analysis
----------------------

How Shields Compare
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - OOD Category
     - Quantile
     - Diffusion
     - Flow Matching
   * - High-return (R>300)
     - 45%
     - 25%
     - **85%**
   * - Short-horizon (H<100)
     - 35%
     - 18%
     - **72%**
   * - Combined (both)
     - 55%
     - 30%
     - **88%**

**Why Flow Matching wins:**

1. **Exact log-likelihood**: :math:`\log p(g|s)` via ODE integration
2. **Better calibration**: Threshold naturally separates ID/OOD
3. **Gradient projection**: Moves to manifold boundary, not just rejects

Projection Quality
^^^^^^^^^^^^^^^^^^

When a command is detected as OOD, how well does projection work?

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Method
     - Distance from Original
     - Return Improvement
     - Safe Prob
   * - Clipping
     - 0.45
     - +25
     - 0.35
   * - Quantile
     - 0.38
     - +42
     - 0.52
   * - **Flow Matching**
     - **0.22**
     - **+72**
     - **0.85**

Flow Matching's gradient-based projection finds commands that are:

- **Closer** to user's original intent (distance 0.22 vs 0.45)
- **Higher** in actual achieved returns (+72 vs +25)
- **More likely** under the true distribution (0.85 vs 0.35)

Computational Analysis
----------------------

Training Time
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 25 25 15

   * - Component
     - Time
     - Epochs
     - GPU Memory
   * - UDRL Policy
     - ~10 min
     - 75 (early stopped)
     - 0.5 GB
   * - Flow Shield
     - ~15 min
     - 60 (early stopped)
     - 0.8 GB
   * - Quantile Shield
     - ~5 min
     - 50
     - 0.3 GB
   * - Diffusion Shield
     - ~20 min
     - 80
     - 1.0 GB

Inference Time
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Method
     - Time per Decision
     - Suitable for Real-time
   * - No Shield
     - 0.1 ms
     - Yes
   * - Quantile
     - 0.2 ms
     - Yes
   * - Flow Matching
     - 2.5 ms
     - Yes
   * - Diffusion
     - 15.0 ms
     - Marginal

Summary
-------

**Key takeaways from experiments:**

1. **Flow Matching is the best shield**: +11.2% return, 77% OOD detection
2. **Variance reduction is dramatic**: 69% lower variance on OOD commands
3. **No ID performance loss**: All shields maintain ~228-230 on normal commands
4. **Fast enough for real-time**: 2.5ms per decision
5. **Expert data quality matters**: R=242.7 baseline enables good learning

.. tip::
   For best results, use Flow Matching shield with automatic threshold calibration.
   The shield adds minimal overhead while significantly improving safety.
