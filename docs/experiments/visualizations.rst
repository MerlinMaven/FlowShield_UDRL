========================
Advanced Visualizations
========================

This page presents the advanced analysis visualizations generated for 
FlowShield-UDRL.

Project Dashboard
-----------------

.. image:: /_static/project_dashboard.png
   :alt: Project Dashboard Overview
   :align: center
   :width: 100%

The dashboard provides a comprehensive overview of:

- **Dataset metrics**: 420 expert episodes, 129K transitions
- **Performance comparison**: Shield methods on ID/OOD commands
- **Variance analysis**: Flow Matching achieves 69% lower variance
- **Key achievements**: +11.2% improvement, 77% OOD detection

Comprehensive Data Analysis
---------------------------

.. image:: /_static/comprehensive_data_analysis.png
   :alt: Comprehensive Data Analysis
   :align: center
   :width: 100%

This 9-panel visualization includes:

1. **Return distribution**: Expert episodes achieve mean R=242.7 ± 26.8
2. **Episode length distribution**: Mean ~307 steps
3. **Return vs Length correlation**: Positive correlation (r=0.45)
4. **Action 2D histogram**: Coverage of main/side engine commands
5. **State dimension statistics**: Mean ± std for each state dimension
6. **State-Action correlation heatmap**: Shows control patterns
7. **Horizon vs Return-to-go**: Command space coverage
8. **Action magnitude trajectory**: Sample temporal profile
9. **X-Y trajectory sample**: Landing approach visualization

State-RTG Analysis
------------------

.. image:: /_static/state_rtg_analysis.png
   :alt: State vs Return-to-Go Analysis
   :align: center
   :width: 100%

This analysis shows how each state dimension affects achievable returns:

- **Position X**: Centered positions achieve higher returns
- **Position Y**: Higher altitude correlated with higher RTG (longer horizon)
- **Velocity**: Lower velocities near landing → higher returns
- **Angle**: Near-zero angles are optimal
- **Leg contacts**: Successful landings show leg contact patterns

Action-State Analysis
---------------------

.. image:: /_static/action_by_state_analysis.png
   :alt: Actions by State Phase Analysis
   :align: center
   :width: 100%

This analysis reveals the expert's control strategy:

**Top row:**

- Main engine usage vs height, colored by vertical velocity
- Side engine usage vs X position, colored by angle
- Total action magnitude vs velocity magnitude

**Bottom row:**

- Action distribution at high altitude (y > 0.5)
- Action distribution at low altitude (y < 0.2)
- Action distribution during landing phase

Key insights:

- More main engine thrust at low altitude with high downward velocity
- Side engines correct horizontal position and angle
- Landing phase shows controlled, symmetric thrust

UDRL Limitation Analysis
------------------------

.. image:: /_static/udrl_limitation_analysis.png
   :alt: UDRL Limitation Analysis
   :align: center
   :width: 100%

This visualization demonstrates the "Obedient Suicide" problem that motivates 
our safety shields.
