"""
Environment factory and registration.

For scientific experiments, use LunarLander-v2 or Highway-Env.
"""

from typing import Any, Callable, Dict, Optional

import gymnasium as gym

from .wrappers import CommandWrapper, NormalizeObservation, RecordTrajectory
from .continuous_wrappers import LunarLanderWrapper, HighwayWrapper


# Environment registry
ENV_REGISTRY: Dict[str, Callable] = {
    "lunarlander": lambda **kwargs: LunarLanderWrapper(gym.make("LunarLander-v2", continuous=True, **kwargs)),
    "highway": lambda **kwargs: HighwayWrapper(gym.make("highway-fast-v0", **kwargs)),
}


def register_env(name: str, env_fn: Callable) -> None:
    """
    Register a new environment.
    
    Args:
        name: Environment name
        env_fn: Factory function that returns environment
    """
    ENV_REGISTRY[name] = env_fn


def make_env(
    env_name: str,
    wrappers: Optional[list] = None,
    **kwargs,
) -> gym.Env:
    """
    Create an environment by name.
    
    Args:
        env_name: Name of environment
        wrappers: Optional list of wrapper classes to apply
        **kwargs: Additional arguments for environment
        
    Returns:
        Gymnasium environment
        
    Example:
        >>> env = make_env("lunarlander")
        >>> env = make_env("LunarLander-v2")
    """
    # Check custom registry first
    if env_name in ENV_REGISTRY:
        env = ENV_REGISTRY[env_name](**kwargs)
    else:
        # Try gymnasium
        try:
            env = gym.make(env_name, **kwargs)
        except gym.error.Error:
            raise ValueError(f"Unknown environment: {env_name}")
    
    # Apply wrappers
    if wrappers:
        for wrapper_cls in wrappers:
            env = wrapper_cls(env)
    
    return env


def make_udrl_env(
    env_name: str,
    command_dim: int = 2,
    normalize_obs: bool = True,
    record_trajectory: bool = False,
    **kwargs,
) -> gym.Env:
    """
    Create environment with UDRL-specific wrappers.
    
    Args:
        env_name: Environment name
        command_dim: Dimension of command vector
        normalize_obs: Whether to normalize observations
        record_trajectory: Whether to record trajectories
        **kwargs: Additional environment arguments
        
    Returns:
        Wrapped environment ready for UDRL
    """
    env = make_env(env_name, **kwargs)
    
    if record_trajectory:
        env = RecordTrajectory(env)
    
    if normalize_obs:
        env = NormalizeObservation(env)
    
    env = CommandWrapper(env, command_dim=command_dim)
    
    return env


def get_env_info(env_name: str, **kwargs) -> Dict[str, Any]:
    """
    Get information about an environment.
    
    Args:
        env_name: Environment name
        
    Returns:
        Dictionary with environment information
    """
    env = make_env(env_name, **kwargs)
    
    info = {
        "name": env_name,
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "state_dim": env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else None,
    }
    
    if hasattr(env.action_space, 'n'):
        info["action_dim"] = env.action_space.n
        info["action_type"] = "discrete"
    elif hasattr(env.action_space, 'shape'):
        info["action_dim"] = env.action_space.shape[0]
        info["action_type"] = "continuous"
    
    env.close()
    
    return info
