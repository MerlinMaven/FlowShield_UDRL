=================
Ablation Studies
=================

This section presents ablation experiments to understand the contribution of 
each component in FlowShield.

Overview
--------

We ablate:

1. **Threshold selection** for OOD detection
2. **ODE integration steps** for Flow Matching
3. **Network architecture** depth and width
4. **Projection method** comparison
5. **Training data size** requirements
6. **Command normalization** impact

1. Threshold Selection
----------------------

How to choose the detection threshold :math:`\tau`?

Validation-Based Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^

We use the 5th percentile of log-probabilities on validation data:

.. math::
   :label: threshold_selection

   \tau = Q_{0.05}(\log p(g|s)), \quad (s, g) \sim \mathcal{D}_{\text{val}}

This ensures 95% of in-distribution commands pass.

Results by Percentile
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Percentile
     - Threshold
     - FPR
     - TPR
     - F1
   * - 1st
     - -8.5
     - 1%
     - 65%
     - 0.78
   * - 5th
     - -6.0
     - 5%
     - 92%
     - **0.93**
   * - 10th
     - -4.5
     - 10%
     - 95%
     - 0.92
   * - 20th
     - -3.0
     - 20%
     - 98%
     - 0.88

**Recommendation:** 5th percentile provides best F1 score.

2. ODE Integration Steps
------------------------

Number of Euler steps vs. accuracy and speed.

Accuracy Analysis
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Steps
     - :math:`\log p` error
     - Detection F1
     - Return
     - Time (ms)
   * - 10
     - 0.85
     - 0.82
     - 98
     - 0.5
   * - 20
     - 0.42
     - 0.88
     - 108
     - 1.0
   * - 50
     - 0.15
     - 0.93
     - 118
     - 2.5
   * - 100
     - 0.05
     - 0.94
     - 120
     - 5.0

**Error metric:** :math:`|\log p_{\text{approx}} - \log p_{\text{exact}}|`

Solver Comparison
^^^^^^^^^^^^^^^^^

With 20 function evaluations:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Solver
     - :math:`\log p` error
     - Detection F1
     - Time (ms)
   * - Euler
     - 0.42
     - 0.88
     - 1.0
   * - Midpoint
     - 0.18
     - 0.91
     - 1.0
   * - RK4
     - 0.08
     - 0.93
     - 2.0
   * - Dopri5
     - 0.03
     - 0.94
     - 3.5

**Recommendation:** Euler with 50 steps or RK4 with 20 steps.

3. Network Architecture
-----------------------

Width Analysis
^^^^^^^^^^^^^^

Fixed depth (4 layers), varying width:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Hidden dim
     - Parameters
     - Val loss
     - Detection F1
     - Return
   * - 64
     - 25K
     - 2.45
     - 0.82
     - 95
   * - 128
     - 80K
     - 1.85
     - 0.88
     - 108
   * - 256
     - 280K
     - 1.42
     - 0.93
     - 118
   * - 512
     - 1M
     - 1.38
     - 0.93
     - 119

**Observation:** 256 achieves near-optimal with reasonable parameters.

Depth Analysis
^^^^^^^^^^^^^^

Fixed width (256), varying depth:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Layers
     - Parameters
     - Val loss
     - Detection F1
     - Return
   * - 2
     - 140K
     - 1.95
     - 0.85
     - 102
   * - 3
     - 210K
     - 1.62
     - 0.90
     - 112
   * - 4
     - 280K
     - 1.42
     - 0.93
     - 118
   * - 6
     - 420K
     - 1.40
     - 0.93
     - 118

**Recommendation:** 4 layers with 256 hidden dimensions.

Time Embedding
^^^^^^^^^^^^^^

Effect of time embedding dimension:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Embed dim
     - Val loss
     - Detection F1
     - Return
   * - 32
     - 1.65
     - 0.89
     - 108
   * - 64
     - 1.52
     - 0.91
     - 112
   * - 128
     - 1.42
     - 0.93
     - 118
   * - 256
     - 1.40
     - 0.93
     - 118

4. Projection Methods
---------------------

Comparing different projection strategies for Flow Matching:

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Method
     - :math:`d_{\text{orig}}` ↓
     - :math:`\Delta R` ↑
     - Time (ms)
   * - Clip (baseline)
     - 0.45
     - +25
     - 0.1
   * - Random sampling
     - 0.35
     - +48
     - 5.0
   * - Nearest sample
     - 0.28
     - +58
     - 8.0
   * - Gradient (5 steps)
     - 0.25
     - +65
     - 1.0
   * - **Gradient (20 steps)**
     - **0.22**
     - **+72**
     - 2.0
   * - Gradient (50 steps)
     - 0.21
     - +74
     - 5.0

Projection Step Size
^^^^^^^^^^^^^^^^^^^^

Learning rate for gradient projection:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Learning rate
     - Convergence
     - :math:`\Delta R`
     - Stability
   * - 0.01
     - Slow
     - +55
     - High
   * - 0.05
     - Medium
     - +68
     - High
   * - **0.1**
     - **Fast**
     - **+72**
     - **Medium**
   * - 0.5
     - Very fast
     - +70
     - Low

5. Training Data Size
---------------------

How much data is needed?

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Episodes
     - Val loss
     - Detection F1
     - Return
     - Crash %
   * - 500
     - 2.85
     - 0.62
     - 65
     - 35%
   * - 1,000
     - 2.15
     - 0.72
     - 85
     - 25%
   * - 2,500
     - 1.75
     - 0.82
     - 98
     - 18%
   * - 5,000
     - 1.55
     - 0.88
     - 108
     - 12%
   * - 10,000
     - 1.42
     - 0.93
     - 118
     - 8%
   * - 25,000
     - 1.38
     - 0.94
     - 120
     - 7%

**Observation:** Diminishing returns after 10,000 episodes.

Data Quality Analysis
^^^^^^^^^^^^^^^^^^^^^

Effect of behavioral policy quality:

.. list-table::
   :header-rows: 1
   :widths: 35 25 25 25

   * - Behavioral policy
     - Command coverage
     - Detection F1
     - Return
   * - Random
     - Low (poor episodes)
     - 0.75
     - 85
   * - ε-greedy (ε=0.3)
     - Medium
     - 0.88
     - 108
   * - **Mixed policies**
     - **High**
     - **0.93**
     - **118**
   * - Expert only
     - Low (only good)
     - 0.82
     - 95

**Key insight:** Diverse command distribution > expert-only data.

6. Command Normalization
------------------------

Impact of normalizing commands before density estimation:

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Normalization
     - Val loss
     - Detection F1
     - Return
   * - None (raw)
     - 3.25
     - 0.72
     - 85
   * - Min-max [0,1]
     - 1.65
     - 0.88
     - 108
   * - **Standardization**
     - **1.42**
     - **0.93**
     - **118**
   * - Log transform
     - 1.58
     - 0.90
     - 112

**Recommendation:** Standardize to zero mean, unit variance.

7. Additional Ablations
-----------------------

EMA (Exponential Moving Average)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 25

   * - EMA decay
     - Val loss
     - Detection F1
     - Return
   * - None
     - 1.55
     - 0.90
     - 112
   * - 0.99
     - 1.48
     - 0.92
     - 115
   * - **0.999**
     - **1.42**
     - **0.93**
     - **118**
   * - 0.9999
     - 1.45
     - 0.92
     - 116

Activation Functions
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 25

   * - Activation
     - Val loss
     - Detection F1
     - Return
   * - ReLU
     - 1.62
     - 0.89
     - 108
   * - GELU
     - 1.48
     - 0.91
     - 114
   * - **SiLU (Swish)**
     - **1.42**
     - **0.93**
     - **118**
   * - Mish
     - 1.45
     - 0.92
     - 116

Summary of Recommendations
--------------------------

Based on ablation studies, we recommend:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Component
     - Recommendation
   * - Threshold
     - 5th percentile of validation log-probs
   * - ODE steps
     - 50 (Euler) or 20 (RK4)
   * - Architecture
     - 4 layers, 256 hidden, 128 time embed
   * - Projection
     - Gradient with 20 steps, lr=0.1
   * - Training data
     - 10,000 episodes, mixed policies
   * - Normalization
     - Standardization (μ=0, σ=1)
   * - EMA
     - 0.999 decay
   * - Activation
     - SiLU (Swish)
