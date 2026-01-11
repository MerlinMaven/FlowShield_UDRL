.. FlowShield-UDRL documentation master file

=========================================
FlowShield-UDRL Documentation
=========================================

**Flow Matching for Safe Command-Conditioned Reinforcement Learning**

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/pytorch-2.0+-red.svg
   :target: https://pytorch.org/

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

----

Overview
--------

FlowShield-UDRL addresses a critical safety flaw in **Upside-Down Reinforcement Learning (UDRL)**: 
the **"Obedient Suicide" problem**.

The Problem
^^^^^^^^^^^

UDRL trains a policy :math:`\pi(a|s,g)` conditioned on a command :math:`g = (H, R)` where:

- :math:`H` = horizon (number of steps)
- :math:`R` = target return (cumulative reward)

When users request **impossible (out-of-distribution) commands**, the agent attempts to 
execute them blindly, resulting in:

- Erratic, dangerous behavior
- System crashes
- Safety violations in critical applications

Our Solution
^^^^^^^^^^^^

We use **Flow Matching** [Lipman2022]_ to:

1. Model the distribution :math:`p(g|s)` of achievable commands
2. Compute :math:`\log p(g|s)` to detect OOD commands
3. Project OOD commands onto the manifold of safe, achievable commands

Key Results
^^^^^^^^^^^

.. figure:: /_static/final_comparison.png
   :alt: Performance comparison of shield methods
   :align: center
   :width: 90%

   Comparison of shield methods under OOD commands on LunarLander-v3.

**LunarLander-v3 Performance Summary:**

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 20 15

   * - Method
     - Mean Return
     - Std Dev
     - Improvement
     - OOD Det
   * - No Shield
     - 211.4
     - 84.7
     - —
     - 0%
   * - Diffusion
     - 209.9
     - 84.1
     - -0.7%
     - 14%
   * - **Flow Matching**
     - **235.0**
     - **26.0**
     - **+11.2%**
     - **77%**

**Key findings:**

- **+11.2%** improvement in mean return with Flow Shield
- **-69%** reduction in variance (26.0 vs 84.7)
- **77%** OOD command detection rate

Three Shield Methods
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Method
     - Principle
     - Characteristics
   * - **Quantile Shield**
     - Pinball loss regression
     - Fast, simple baseline
   * - **Diffusion Shield**
     - DDPM denoising
     - High quality, slow inference
   * - **Flow Matching Shield**
     - OT-CFM with ODE
     - Fast inference, exact likelihood

----

Quick Start
-----------

.. code-block:: bash

   # Train Flow Matching shield
   python scripts/train_flow.py --data data/lunarlander_expert.npz --epochs 100

   # Evaluate all shields
   python scripts/evaluate_models.py --env lunarlander --data data/lunarlander_expert.npz

See :doc:`quickstart` for detailed instructions.

----

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   usage

.. toctree::
   :maxdepth: 2
   :caption: Theory & Concepts

   concepts/udrl
   concepts/ood_problem
   concepts/safety_shields

.. toctree::
   :maxdepth: 2
   :caption: Methods

   methods/quantile
   methods/diffusion
   methods/flow_matching

.. toctree::
   :maxdepth: 2
   :caption: Environments

   environments/overview
   environments/lunarlander
   environments/highway

.. toctree::
   :maxdepth: 2
   :caption: Experiments

   experiments/protocol
   experiments/results
   experiments/ablations
   experiments/visualizations

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/envs
   api/models
   api/training
   api/evaluation

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   best_practices
   faq
   bibliography

----

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

----

Citation
--------

If you use FlowShield-UDRL in your research, please cite:

.. code-block:: bibtex

    @article{flowshield2026,
        title={FlowShield: Safe Command-Conditioned RL via Flow Matching},
        author={FlowShield Team},
        journal={arXiv preprint},
        year={2026}
    }
