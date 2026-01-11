=======================
The Obedient Suicide Problem
=======================

The **Obedient Suicide Problem** is the central safety issue that FlowShield addresses.

Problem Statement
-----------------

Consider a UDRL agent trained on LunarLander trajectories with returns in 
:math:`[-500, 300]`. What happens when we request:

.. math::
   g^* = (H=50, R=500)

The agent has **never seen** a trajectory achieving return 500. Yet it will 
blindly attempt to satisfy this impossible command.

Formal Definition
-----------------

.. admonition:: Definition: Out-of-Distribution (OOD) Command

   A command :math:`g^*` is **out-of-distribution** with respect to the training 
   data :math:`\mathcal{D}` if:
   
   .. math::
      :label: ood_definition
      
      p_{\mathcal{D}}(g^* | s) \approx 0
   
   That is, no trajectory in :math:`\mathcal{D}` achieved command :math:`g^*` 
   from state :math:`s`.

Observed Behaviors
------------------

When given OOD commands, we observe three failure modes:

1. Erratic Actions
^^^^^^^^^^^^^^^^^^

The policy outputs are essentially random in OOD regions:

.. math::
   \pi(a | s, g^*_{\text{OOD}}) \approx \text{Uniform}(\mathcal{A})

This leads to:

- Oscillating controls
- Fuel waste
- Crashes

2. Aggressive Strategies
^^^^^^^^^^^^^^^^^^^^^^^^

The agent may interpret "high return in few steps" as requiring extreme actions:

- Maximum thrust in arbitrary directions
- Ignoring landing pad location
- Treating fuel as infinite

3. Obedient Suicide
^^^^^^^^^^^^^^^^^^^

The most dangerous behavior: the agent **rationally** concludes that the 
fastest way to end an impossible episode is to crash.

.. topic:: Example: Obedient Suicide on LunarLander

   Command: :math:`g = (H=10, R=500)`
   
   Agent reasoning (implicit):
   
   1. "I need 500 return in 10 steps"
   2. "Maximum possible per step ≈ 10 points"
   3. "This is impossible"
   4. "Best strategy: end episode quickly"
   5. "Crash into ground → episode ends"
   
   Result: Agent deliberately crashes.

Mathematical Analysis
---------------------

Extrapolation Error
^^^^^^^^^^^^^^^^^^^

UDRL learns via supervised learning. Outside the training distribution, 
extrapolation error grows:

.. math::
   :label: extrapolation_error

   \mathbb{E}\left[ \| \pi_\theta(s, g) - \pi^*(s, g) \|^2 \right] 
   \propto d(g, \text{supp}(\mathcal{D}))

where :math:`d(\cdot, \cdot)` is distance to the support of training data.

Value Mismatch
^^^^^^^^^^^^^^

The agent implicitly learns a Q-function. For OOD commands:

.. math::
   :label: value_mismatch

   Q_\theta(s, a, g_{\text{OOD}}) \neq Q^*(s, a, g_{\text{OOD}})

Since :math:`Q_\theta` was never trained on such commands, its predictions 
are unreliable.

Visualization
-------------

The command space can be visualized as:

.. code-block:: text

   Return (R)
       │
   500 │  ┌─────────────────────┐
       │  │    OOD REGION       │
       │  │   (Dangerous!)      │
   300 │  ├─────────────────────┤
       │  │                     │
       │  │  ACHIEVABLE REGION  │
       │  │   (Training Data)   │
       │  │                     │
  -500 │  └─────────────────────┘
       └──────────────────────────▶ Horizon (H)
           10    50   100   200

The **boundary** between achievable and OOD regions is what our shields learn.

Why Standard Solutions Fail
---------------------------

Approach 1: Clamping
^^^^^^^^^^^^^^^^^^^^

Simple idea: clamp requested return to :math:`[R_{\min}, R_{\max}]`.

.. math::
   R_{\text{safe}} = \text{clip}(R^*, R_{\min}, R_{\max})

**Problem:** This ignores state dependency! The achievable return depends on 
the current state :math:`s`.

.. code-block:: text

   State A (near landing pad): achievable R ∈ [100, 280]
   State B (upside-down, falling): achievable R ∈ [-400, -100]
   
   Global clamp would allow R=200 for State B → Still OOD!

Approach 2: Ensemble Uncertainty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train ensemble of policies, use disagreement as uncertainty.

**Problem:** Ensembles are expensive and uncertainty is not calibrated.

Approach 3: Conservative Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use pessimistic value estimates.

**Problem:** Doesn't provide a mechanism to **project** OOD commands.

Our Solution: Safety Shields
----------------------------

FlowShield addresses all these issues:

1. **State-dependent**: Models :math:`p(g|s)` explicitly
2. **Principled detection**: Uses log-likelihood for OOD score
3. **Projection mechanism**: Maps OOD commands to safe alternatives

.. math::
   :label: shield_operation

   g_{\text{safe}} = \begin{cases}
   g^* & \text{if } \log p(g^* | s) > \tau \\
   \text{Project}(g^*, s) & \text{otherwise}
   \end{cases}

See :doc:`safety_shields` for the detailed methodology.

References
----------

OOD detection foundations: [Hendrycks2017]_, [Nalisnick2019]_, and [Ren2019]_.

