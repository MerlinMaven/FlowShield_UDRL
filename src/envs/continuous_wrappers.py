"""
Standardized Wrappers for Continuous Control Environments.

This module provides unified wrappers for:
- LunarLander-v2 (Continuous)
- Highway-Env

Key features:
1. Standardized state/action spaces
2. Automatic command (horizon, return-to-go) computation
3. Mixed-quality data collection (expert/random/medium)
4. Normalization and preprocessing
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import highway_env
    HIGHWAY_AVAILABLE = True
except ImportError:
    HIGHWAY_AVAILABLE = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TransitionData:
    """Single transition for UDRL training."""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    horizon: int           # Steps remaining in episode
    return_to_go: float    # Sum of future rewards


@dataclass
class TrajectoryData:
    """Complete trajectory with computed UDRL commands."""
    states: np.ndarray           # (T+1, state_dim)
    actions: np.ndarray          # (T, action_dim)
    rewards: np.ndarray          # (T,)
    dones: np.ndarray            # (T,)
    horizons: np.ndarray         # (T,) - steps remaining
    returns_to_go: np.ndarray    # (T,) - future cumulative reward
    total_return: float          # Episode return
    length: int                  # Episode length
    
    def to_transitions(self) -> List[TransitionData]:
        """Convert to list of transitions."""
        transitions = []
        for t in range(self.length):
            transitions.append(TransitionData(
                state=self.states[t],
                action=self.actions[t],
                reward=self.rewards[t],
                next_state=self.states[t + 1] if t + 1 < len(self.states) else self.states[t],
                done=self.dones[t],
                horizon=self.horizons[t],
                return_to_go=self.returns_to_go[t],
            ))
        return transitions


# =============================================================================
# Base Continuous Environment Wrapper
# =============================================================================

class ContinuousEnvWrapper(gym.Wrapper):
    """
    Unified wrapper for continuous control environments.
    
    Handles:
    - State normalization (optional, with running statistics)
    - Reward scaling
    - Command tracking (horizon, return-to-go)
    - Trajectory recording
    
    Args:
        env: Base Gymnasium environment
        normalize_obs: Whether to normalize observations
        normalize_reward: Whether to scale rewards
        reward_scale: Reward scaling factor
        clip_obs: Observation clipping range
        clip_reward: Reward clipping range
    """
    
    def __init__(
        self,
        env: gym.Env,
        normalize_obs: bool = True,
        normalize_reward: bool = False,
        reward_scale: float = 1.0,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
    ):
        super().__init__(env)
        
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward
        self.reward_scale = reward_scale
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        
        # Observation normalization statistics
        self._obs_mean = np.zeros(self.observation_space.shape, dtype=np.float64)
        self._obs_var = np.ones(self.observation_space.shape, dtype=np.float64)
        self._obs_count = 0
        
        # Episode tracking
        self._step_count = 0
        self._episode_return = 0.0
        self._episode_states: List[np.ndarray] = []
        self._episode_actions: List[np.ndarray] = []
        self._episode_rewards: List[float] = []
        
        # Command tracking (for inference)
        self._target_horizon = 0
        self._target_return = 0.0
        
    @property
    def state_dim(self) -> int:
        return int(np.prod(self.observation_space.shape))
    
    @property
    def action_dim(self) -> int:
        if isinstance(self.action_space, spaces.Box):
            return int(np.prod(self.action_space.shape))
        elif isinstance(self.action_space, spaces.Discrete):
            return self.action_space.n
        else:
            raise ValueError(f"Unsupported action space: {type(self.action_space)}")
    
    @property
    def is_continuous_action(self) -> bool:
        return isinstance(self.action_space, spaces.Box)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and episode tracking."""
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Reset episode tracking
        self._step_count = 0
        self._episode_return = 0.0
        self._episode_states = [obs.copy()]
        self._episode_actions = []
        self._episode_rewards = []
        
        # Set target command from options
        if options:
            self._target_horizon = options.get("horizon", 1000)
            self._target_return = options.get("target_return", 0.0)
        
        # Normalize if enabled
        if self.normalize_obs:
            obs = self._normalize_obs(obs, update=True)
        
        info["command"] = self.get_command()
        return obs.astype(np.float32), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take step and track episode data."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Record
        self._episode_actions.append(np.array(action).copy())
        self._episode_rewards.append(reward)
        self._step_count += 1
        self._episode_return += reward
        
        if not (terminated or truncated):
            self._episode_states.append(obs.copy())
        
        # Update command
        self._target_horizon = max(0, self._target_horizon - 1)
        self._target_return -= reward
        
        # Normalize
        if self.normalize_obs:
            obs = self._normalize_obs(obs, update=True)
        
        if self.normalize_reward:
            reward = np.clip(reward * self.reward_scale, -self.clip_reward, self.clip_reward)
        
        info["command"] = self.get_command()
        info["episode_return"] = self._episode_return
        info["episode_length"] = self._step_count
        
        return obs.astype(np.float32), float(reward), terminated, truncated, info
    
    def get_command(self) -> Tuple[int, float]:
        """Get current (horizon, return_to_go) command."""
        return (self._target_horizon, self._target_return)
    
    def set_command(self, horizon: int, target_return: float) -> None:
        """Set target command for episode."""
        self._target_horizon = horizon
        self._target_return = target_return
    
    def get_trajectory(self) -> Optional[TrajectoryData]:
        """Get completed trajectory with computed commands."""
        if len(self._episode_actions) == 0:
            return None
        
        rewards = np.array(self._episode_rewards)
        T = len(rewards)
        
        # Compute returns-to-go (future cumulative reward)
        returns_to_go = np.zeros(T, dtype=np.float32)
        running_return = 0.0
        for t in range(T - 1, -1, -1):
            running_return += rewards[t]
            returns_to_go[t] = running_return
        
        # Compute horizons (steps remaining)
        horizons = np.arange(T, 0, -1, dtype=np.int32)
        
        return TrajectoryData(
            states=np.array(self._episode_states, dtype=np.float32),
            actions=np.array(self._episode_actions, dtype=np.float32),
            rewards=rewards.astype(np.float32),
            dones=np.zeros(T, dtype=bool),  # Internal dones
            horizons=horizons,
            returns_to_go=returns_to_go,
            total_return=float(self._episode_return),
            length=T,
        )
    
    def _normalize_obs(self, obs: np.ndarray, update: bool = False) -> np.ndarray:
        """Normalize observation with running statistics."""
        if update:
            self._obs_count += 1
            delta = obs - self._obs_mean
            self._obs_mean += delta / self._obs_count
            delta2 = obs - self._obs_mean
            self._obs_var += (delta * delta2 - self._obs_var) / self._obs_count
        
        normalized = (obs - self._obs_mean) / np.sqrt(self._obs_var + 1e-8)
        return np.clip(normalized, -self.clip_obs, self.clip_obs)
    
    def set_obs_stats(self, mean: np.ndarray, var: np.ndarray) -> None:
        """Set normalization statistics."""
        self._obs_mean = mean.copy()
        self._obs_var = var.copy()
        self._obs_count = 1  # Mark as initialized
    
    def get_obs_stats(self) -> Dict[str, np.ndarray]:
        """Get normalization statistics."""
        return {"mean": self._obs_mean.copy(), "var": self._obs_var.copy()}


# =============================================================================
# LunarLander Wrapper
# =============================================================================

class LunarLanderWrapper(ContinuousEnvWrapper):
    """
    Wrapper for LunarLander-v2 (Continuous).
    
    State space (8 dims):
        [x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact]
    
    Action space (2 dims, continuous):
        [main_engine, side_engine] in [-1, 1]
    
    Reward structure:
        - Moving toward landing pad: positive
        - Crash: -100
        - Successful landing: +100 to +140
        - Each leg contact: +10
        - Fuel usage: -0.03 * main_engine, -0.03 * |side_engine|
    
    Typical returns:
        - Random policy: -200 to -100
        - Medium policy: 0 to 100
        - Expert policy: 200 to 280
    """
    
    def __init__(
        self,
        normalize_obs: bool = True,
        normalize_reward: bool = False,
        reward_scale: float = 1.0,
        render_mode: Optional[str] = None,
    ):
        # Create base environment
        env = gym.make(
            "LunarLander-v2",
            continuous=True,
            render_mode=render_mode,
        )
        
        super().__init__(
            env=env,
            normalize_obs=normalize_obs,
            normalize_reward=normalize_reward,
            reward_scale=reward_scale,
        )
        
        # LunarLander-specific bounds
        self.horizon_max = 1000
        self.return_min = -500
        self.return_max = 300
    
    def get_landing_success(self, info: Dict) -> bool:
        """Check if landing was successful."""
        # Successful landing: both legs touching, low velocity
        if hasattr(self.env.unwrapped, 'lander'):
            lander = self.env.unwrapped.lander
            # Check legs contact and velocity
            left_leg = self.env.unwrapped.legs[0].ground_contact
            right_leg = self.env.unwrapped.legs[1].ground_contact
            # Velocity check would require accessing lander velocity
            return left_leg and right_leg
        return False
    
    def is_crash(self, reward: float) -> bool:
        """Check if episode ended in crash."""
        return reward <= -100


# =============================================================================
# Highway-Env Wrapper
# =============================================================================

class HighwayWrapper(ContinuousEnvWrapper):
    """
    Wrapper for Highway-Env with continuous actions.
    
    State space (25 dims for 5 vehicles):
        Each vehicle: [presence, x, y, vx, vy]
        Flattened: 5 * 5 = 25
    
    Action space (2 dims):
        [acceleration, steering] continuous
    
    Reward:
        - High speed: positive
        - Right lane: positive
        - Collision: negative (-1)
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        normalize_obs: bool = True,
        normalize_reward: bool = True,
        reward_scale: float = 1.0,
        render_mode: Optional[str] = None,
    ):
        if not HIGHWAY_AVAILABLE:
            raise ImportError("highway-env not installed. Run: pip install highway-env")
        
        # Default highway config
        default_config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": False,
                "normalize": True,
            },
            "action": {
                "type": "ContinuousAction",
                "acceleration_range": [-5.0, 5.0],
                "steering_range": [-0.7, 0.7],
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "duration": 40,
            "collision_reward": -1.0,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.4,
            "reward_speed_range": [20, 30],
        }
        
        if config:
            default_config.update(config)
        
        env = gym.make("highway-v0", render_mode=render_mode)
        env.configure(default_config)
        
        super().__init__(
            env=env,
            normalize_obs=normalize_obs,
            normalize_reward=normalize_reward,
            reward_scale=reward_scale,
        )
        
        # Highway-specific bounds
        self.horizon_max = 200
        self.return_min = -50
        self.return_max = 50
    
    def is_collision(self, info: Dict) -> bool:
        """Check if episode ended in collision."""
        return info.get("crashed", False)


# =============================================================================
# Mixed Quality Data Collector
# =============================================================================

class MixedQualityCollector:
    """
    Collect trajectories with mixed policy quality.
    
    Essential for training generative models (Flow/Diffusion) to learn
    the full distribution of achievable returns, not just expert behavior.
    
    Args:
        env_wrapper: Wrapped environment
        expert_policy: High-quality policy (e.g., trained SAC/PPO)
        medium_policy: Medium-quality policy (e.g., partially trained)
        expert_ratio: Proportion of expert trajectories
        random_ratio: Proportion of random trajectories
        medium_ratio: Proportion of medium trajectories
    """
    
    def __init__(
        self,
        env_wrapper: ContinuousEnvWrapper,
        expert_policy: Optional[Any] = None,
        medium_policy: Optional[Any] = None,
        expert_ratio: float = 0.3,
        random_ratio: float = 0.4,
        medium_ratio: float = 0.3,
    ):
        self.env = env_wrapper
        self.expert_policy = expert_policy
        self.medium_policy = medium_policy
        self.expert_ratio = expert_ratio
        self.random_ratio = random_ratio
        self.medium_ratio = medium_ratio
        
        assert abs(expert_ratio + random_ratio + medium_ratio - 1.0) < 1e-6
        
    def collect(
        self,
        n_episodes: int,
        max_steps: int = 1000,
        verbose: bool = True,
    ) -> List[TrajectoryData]:
        """
        Collect mixed-quality trajectories.
        
        Returns:
            List of TrajectoryData objects
        """
        trajectories = []
        
        n_expert = int(n_episodes * self.expert_ratio)
        n_random = int(n_episodes * self.random_ratio)
        n_medium = n_episodes - n_expert - n_random
        
        # Collect expert trajectories
        if n_expert > 0 and self.expert_policy is not None:
            if verbose:
                print(f"Collecting {n_expert} expert trajectories...")
            expert_trajs = self._collect_with_policy(
                self.expert_policy, n_expert, max_steps
            )
            trajectories.extend(expert_trajs)
        
        # Collect random trajectories
        if verbose:
            print(f"Collecting {n_random} random trajectories...")
        random_trajs = self._collect_random(n_random, max_steps)
        trajectories.extend(random_trajs)
        
        # Collect medium trajectories
        if n_medium > 0 and self.medium_policy is not None:
            if verbose:
                print(f"Collecting {n_medium} medium trajectories...")
            medium_trajs = self._collect_with_policy(
                self.medium_policy, n_medium, max_steps
            )
            trajectories.extend(medium_trajs)
        elif n_medium > 0:
            # Use noisy random if no medium policy
            if verbose:
                print(f"Collecting {n_medium} noisy trajectories...")
            noisy_trajs = self._collect_random(n_medium, max_steps, noise_scale=0.3)
            trajectories.extend(noisy_trajs)
        
        if verbose:
            returns = [t.total_return for t in trajectories]
            print(f"Collected {len(trajectories)} trajectories")
            print(f"  Return range: [{min(returns):.1f}, {max(returns):.1f}]")
            print(f"  Mean return: {np.mean(returns):.1f} ± {np.std(returns):.1f}")
        
        return trajectories
    
    def _collect_with_policy(
        self,
        policy: Any,
        n_episodes: int,
        max_steps: int,
    ) -> List[TrajectoryData]:
        """Collect trajectories using a policy."""
        trajectories = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            step = 0
            
            while not done and step < max_steps:
                # Get action from policy
                if hasattr(policy, 'get_action'):
                    action = policy.get_action(obs)
                elif callable(policy):
                    action = policy(obs)
                else:
                    action = self.env.action_space.sample()
                
                obs, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                step += 1
            
            traj = self.env.get_trajectory()
            if traj is not None:
                trajectories.append(traj)
        
        return trajectories
    
    def _collect_random(
        self,
        n_episodes: int,
        max_steps: int,
        noise_scale: float = 1.0,
    ) -> List[TrajectoryData]:
        """Collect trajectories with random policy."""
        trajectories = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            step = 0
            
            while not done and step < max_steps:
                action = self.env.action_space.sample()
                
                if noise_scale != 1.0 and isinstance(self.env.action_space, spaces.Box):
                    # Scale random actions for "medium" behavior
                    action = action * noise_scale
                    action = np.clip(
                        action,
                        self.env.action_space.low,
                        self.env.action_space.high
                    )
                
                obs, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                step += 1
            
            traj = self.env.get_trajectory()
            if traj is not None:
                trajectories.append(traj)
        
        return trajectories


# =============================================================================
# Factory Functions
# =============================================================================

def make_env(
    env_name: str,
    continuous: bool = True,
    normalize_obs: bool = True,
    normalize_reward: bool = False,
    reward_scale: float = 1.0,
    render_mode: Optional[str] = None,
    **kwargs,
) -> ContinuousEnvWrapper:
    """
    Factory function to create wrapped environments.
    
    Args:
        env_name: Environment name ("LunarLander-v2", "highway-v0", etc.)
        continuous: Use continuous action space
        normalize_obs: Normalize observations
        normalize_reward: Normalize rewards
        reward_scale: Reward scaling factor
        render_mode: Gymnasium render mode
        **kwargs: Additional environment-specific arguments
    
    Returns:
        Wrapped environment
    """
    env_name_lower = env_name.lower()
    
    if "lunarlander" in env_name_lower or "lunar" in env_name_lower:
        return LunarLanderWrapper(
            normalize_obs=normalize_obs,
            normalize_reward=normalize_reward,
            reward_scale=reward_scale,
            render_mode=render_mode,
        )
    
    elif "highway" in env_name_lower:
        return HighwayWrapper(
            config=kwargs.get("highway_config"),
            normalize_obs=normalize_obs,
            normalize_reward=normalize_reward,
            reward_scale=reward_scale,
            render_mode=render_mode,
        )
    
    else:
        # Generic wrapper for other Gymnasium environments
        env = gym.make(env_name, render_mode=render_mode, **kwargs)
        return ContinuousEnvWrapper(
            env=env,
            normalize_obs=normalize_obs,
            normalize_reward=normalize_reward,
            reward_scale=reward_scale,
        )


def trajectories_to_dataset(
    trajectories: List[TrajectoryData],
    normalize_commands: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Convert trajectories to flat dataset for training.
    
    Args:
        trajectories: List of TrajectoryData
        normalize_commands: Whether to normalize horizon/return-to-go
    
    Returns:
        Dictionary with arrays:
            - states: (N, state_dim)
            - actions: (N, action_dim)
            - rewards: (N,)
            - commands: (N, 2) - [horizon, return_to_go]
    """
    all_states = []
    all_actions = []
    all_rewards = []
    all_horizons = []
    all_returns_to_go = []
    
    for traj in trajectories:
        # Exclude last state (no corresponding action)
        states = traj.states[:-1] if len(traj.states) > traj.length else traj.states
        all_states.append(states)
        all_actions.append(traj.actions)
        all_rewards.append(traj.rewards)
        all_horizons.append(traj.horizons)
        all_returns_to_go.append(traj.returns_to_go)
    
    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)
    horizons = np.concatenate(all_horizons, axis=0).astype(np.float32)
    returns_to_go = np.concatenate(all_returns_to_go, axis=0)
    
    # Stack commands
    commands = np.stack([horizons, returns_to_go], axis=-1)
    
    # Normalize commands
    command_mean = None
    command_std = None
    
    if normalize_commands:
        command_mean = commands.mean(axis=0)
        command_std = commands.std(axis=0) + 1e-8
        commands = (commands - command_mean) / command_std
    
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "commands": commands,
        "command_mean": command_mean,
        "command_std": command_std,
        "n_transitions": len(states),
        "n_trajectories": len(trajectories),
    }
