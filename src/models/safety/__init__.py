"""
Safety modules for FlowShield-UDRL.

Provides three shield implementations:
1. QuantileShield: Baseline with pinball loss (fast, conservative)
2. DiffusionShield: DDPM-based generative model
3. FlowMatchShield: Optimal Transport Flow Matching (recommended)

Also includes legacy modules for backward compatibility.
"""

# New unified shields (recommended)
from .shields import (
    BaseShield,
    QuantileShield,
    DiffusionShield,
    FlowMatchShield,
    create_shield,
)

# Legacy modules (for backward compatibility)
from .base import BaseSafetyModule
from .flow_matching import FlowMatchingSafetyModule, FlowMatchingModel
from .projector import CommandProjector
from .quantile import QuantileSafetyModule

__all__ = [
    # New unified API
    "BaseShield",
    "QuantileShield",
    "DiffusionShield",
    "FlowMatchShield",
    "create_shield",
    # Legacy API
    "BaseSafetyModule",
    "FlowMatchingSafetyModule",
    "FlowMatchingModel",
    "QuantileSafetyModule",
    "CommandProjector",
]

