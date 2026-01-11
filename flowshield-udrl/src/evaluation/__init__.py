"""
Evaluation module for FlowShield-UDRL.

- evaluator: Main evaluation class
- metrics: Safety and performance metrics
- visualization: Plotting and analysis tools
"""

from .evaluator import Evaluator, SafetyEvaluator
from .metrics import (
    compute_ood_metrics,
    compute_performance_metrics,
    compute_safety_metrics,
    obedient_suicide_rate,
)
from .visualization import (
    plot_command_distribution,
    plot_flow_trajectories,
    plot_safety_boundaries,
    plot_training_curves,
)

__all__ = [
    "Evaluator",
    "SafetyEvaluator",
    "compute_safety_metrics",
    "compute_performance_metrics",
    "compute_ood_metrics",
    "obedient_suicide_rate",
    "plot_command_distribution",
    "plot_flow_trajectories",
    "plot_safety_boundaries",
    "plot_training_curves",
]
