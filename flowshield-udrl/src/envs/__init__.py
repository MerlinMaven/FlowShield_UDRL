"""
Environment module for FlowShield-UDRL.

Provides:
- Continuous Wrappers: LunarLander, Highway-Env with standardized interfaces
- Wrappers: Command conditioning, normalization, trajectory recording
- Factory: Easy environment creation

For scientific experiments, use the continuous environment wrappers.
"""

# Standard wrappers
from .wrappers import (
    CommandWrapper,
    FrameStack,
    NormalizeObservation,
    NormalizeReward,
    RecordTrajectory,
)

# Continuous environment wrappers (recommended for experiments)
from .continuous_wrappers import (
    ContinuousEnvWrapper,
    LunarLanderWrapper,
    HighwayWrapper,
    MixedQualityCollector,
    TrajectoryData,
    TransitionData,
    make_env as make_continuous_env,
    trajectories_to_dataset,
)

# Factory
from .factory import make_env, make_udrl_env, register_env, get_env_info

__all__ = [
    # Standard wrappers
    "CommandWrapper",
    "NormalizeObservation",
    "NormalizeReward",
    "RecordTrajectory",
    "FrameStack",
    # Continuous environments (recommended)
    "ContinuousEnvWrapper",
    "LunarLanderWrapper",
    "HighwayWrapper",
    "MixedQualityCollector",
    "TrajectoryData",
    "TransitionData",
    "make_continuous_env",
    "trajectories_to_dataset",
    # Factory
    "make_env",
    "make_udrl_env",
    "register_env",
    "get_env_info",
]

