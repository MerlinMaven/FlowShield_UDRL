"""
Environment Wrappers for FlowShield-UDRL.

Provides utility wrappers for Gymnasium environments:
- CommandWrapper: Add command (horizon, return) to observation
- NormalizeObservation: Running mean/std normalization
- NormalizeReward: Reward scaling and normalization
- RecordTrajectory: Record full trajectories for offline RL
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CommandWrapper(gym.Wrapper):
    """
    Wrapper that adds command (horizon, return-to-go) tracking.
    
    This wrapper tracks the remaining horizon and return-to-go during
    an episode, useful for UDRL-style evaluation.
    
    Args:
        env: Environment to wrap
        initial_horizon: Initial horizon for the episode
        initial_return: Initial target return
        
    Example:
        >>> env = CommandWrapper(make_env("lunarlander"), initial_horizon=50, initial_return=10.0)
        >>> obs, info = env.reset()
        >>> print(info["command"])  # (horizon, return_to_go)
    """
    
    def __init__(
        self,
        env: gym.Env,
        initial_horizon: int = 50,
        initial_return: float = 0.0,
    ):
        super().__init__(env)
        
        self.initial_horizon = initial_horizon
        self.initial_return = initial_return
        
        self._horizon = initial_horizon
        self._return_to_go = initial_return
        self._cumulative_reward = 0.0
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and command tracking."""
        # Check for custom command in options
        if options:
            self._horizon = options.get("horizon", self.initial_horizon)
            self._return_to_go = options.get("target_return", self.initial_return)
        else:
            self._horizon = self.initial_horizon
            self._return_to_go = self.initial_return
        
        self._cumulative_reward = 0.0
        
        obs, info = self.env.reset(seed=seed, options=options)
        info["command"] = self.get_command()
        info["cumulative_reward"] = self._cumulative_reward
        
        return obs, info
    
    def step(
        self,
        action: Any,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step and update command tracking."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update tracking
        self._horizon = max(0, self._horizon - 1)
        self._return_to_go = self._return_to_go - reward
        self._cumulative_reward += reward
        
        # Add to info
        info["command"] = self.get_command()
        info["cumulative_reward"] = self._cumulative_reward
        
        return obs, reward, terminated, truncated, info
    
    def get_command(self) -> Tuple[int, float]:
        """Get current command (horizon, return_to_go)."""
        return (self._horizon, self._return_to_go)
    
    def set_command(self, horizon: int, target_return: float) -> None:
        """Set command for next episode."""
        self.initial_horizon = horizon
        self.initial_return = target_return


class NormalizeObservation(gym.Wrapper):
    """
    Normalize observations using running mean and standard deviation.
    
    Args:
        env: Environment to wrap
        epsilon: Small constant for numerical stability
        clip: Clip normalized observations to [-clip, clip]
        
    Example:
        >>> env = NormalizeObservation(make_env("lunarlander"))
        >>> obs, _ = env.reset()
        >>> # Observations are now normalized
    """
    
    def __init__(
        self,
        env: gym.Env,
        epsilon: float = 1e-8,
        clip: float = 10.0,
    ):
        super().__init__(env)
        
        self.epsilon = epsilon
        self.clip = clip
        
        # Running statistics
        obs_shape = env.observation_space.shape
        self.running_mean = np.zeros(obs_shape, dtype=np.float64)
        self.running_var = np.ones(obs_shape, dtype=np.float64)
        self.count = 0
        
        # Update observation space
        self.observation_space = spaces.Box(
            low=-clip,
            high=clip,
            shape=obs_shape,
            dtype=np.float32,
        )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset and normalize initial observation."""
        obs, info = self.env.reset(seed=seed, options=options)
        return self._normalize(obs), info
    
    def step(
        self,
        action: Any,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step and normalize observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize(obs), reward, terminated, truncated, info
    
    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation and update statistics."""
        # Update running statistics
        self._update_statistics(obs)
        
        # Normalize
        normalized = (obs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # Clip
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)
    
    def _update_statistics(self, obs: np.ndarray) -> None:
        """Update running mean and variance using Welford's algorithm."""
        self.count += 1
        delta = obs - self.running_mean
        self.running_mean += delta / self.count
        delta2 = obs - self.running_mean
        self.running_var += (delta * delta2 - self.running_var) / self.count
    
    def set_statistics(self, mean: np.ndarray, var: np.ndarray, count: int = 1) -> None:
        """Set normalization statistics from external source."""
        self.running_mean = mean.copy()
        self.running_var = var.copy()
        self.count = count
    
    def get_statistics(self) -> Dict[str, np.ndarray]:
        """Get current normalization statistics."""
        return {
            "mean": self.running_mean.copy(),
            "var": self.running_var.copy(),
            "count": self.count,
        }


class NormalizeReward(gym.Wrapper):
    """
    Normalize rewards using running statistics.
    
    Args:
        env: Environment to wrap
        gamma: Discount factor for return-based normalization
        epsilon: Small constant for numerical stability
        clip: Clip normalized rewards
        
    Example:
        >>> env = NormalizeReward(make_env("lunarlander"), gamma=0.99)
        >>> _, reward, _, _, _ = env.step(action)
        >>> # Reward is now normalized
    """
    
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        clip: float = 10.0,
    ):
        super().__init__(env)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip
        
        # Running return statistics
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0
        
        # For return tracking
        self._return = 0.0
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset return tracking."""
        self._return = 0.0
        return self.env.reset(seed=seed, options=options)
    
    def step(
        self,
        action: Any,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step and normalize reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update return
        self._return = self._return * self.gamma + reward
        
        # Update statistics
        self._update_statistics(self._return)
        
        # Normalize
        normalized_reward = reward / np.sqrt(self.running_var + self.epsilon)
        normalized_reward = np.clip(normalized_reward, -self.clip, self.clip)
        
        # Store original in info
        info["original_reward"] = reward
        
        # Reset return on episode end
        if terminated or truncated:
            self._return = 0.0
        
        return obs, normalized_reward, terminated, truncated, info
    
    def _update_statistics(self, ret: float) -> None:
        """Update running variance of returns."""
        self.count += 1
        delta = ret - self.running_mean
        self.running_mean += delta / self.count
        delta2 = ret - self.running_mean
        self.running_var += (delta * delta2 - self.running_var) / self.count


class RecordTrajectory(gym.Wrapper):
    """
    Wrapper that records complete trajectories for offline RL.
    
    Records states, actions, rewards, and computes returns-to-go
    and horizons after episode completion.
    
    Args:
        env: Environment to wrap
        max_trajectories: Maximum number of trajectories to store
        
    Example:
        >>> env = RecordTrajectory(make_env("lunarlander"))
        >>> obs, _ = env.reset()
        >>> for _ in range(100):
        ...     action = env.action_space.sample()
        ...     obs, _, done, _, _ = env.step(action)
        ...     if done:
        ...         break
        >>> trajectory = env.get_current_trajectory()
    """
    
    def __init__(
        self,
        env: gym.Env,
        max_trajectories: int = 1000,
    ):
        super().__init__(env)
        
        self.max_trajectories = max_trajectories
        
        # Storage for completed trajectories
        self.trajectories: deque = deque(maxlen=max_trajectories)
        
        # Current trajectory being recorded
        self._current_states: List[np.ndarray] = []
        self._current_actions: List[Any] = []
        self._current_rewards: List[float] = []
        self._current_dones: List[bool] = []
        self._current_infos: List[Dict] = []
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset and start new trajectory recording."""
        # Save previous trajectory if exists
        if len(self._current_states) > 0:
            self._save_trajectory()
        
        # Clear current trajectory
        self._current_states = []
        self._current_actions = []
        self._current_rewards = []
        self._current_dones = []
        self._current_infos = []
        
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Record initial state
        self._current_states.append(obs.copy())
        
        return obs, info
    
    def step(
        self,
        action: Any,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step and record transition."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Record
        self._current_actions.append(action)
        self._current_rewards.append(reward)
        self._current_dones.append(terminated or truncated)
        self._current_infos.append(info.copy())
        
        if not (terminated or truncated):
            self._current_states.append(obs.copy())
        
        # Save on episode end
        if terminated or truncated:
            self._save_trajectory()
        
        return obs, reward, terminated, truncated, info
    
    def _save_trajectory(self) -> None:
        """Save current trajectory to storage."""
        if len(self._current_actions) == 0:
            return
        
        trajectory = {
            "states": np.array(self._current_states),
            "actions": np.array(self._current_actions),
            "rewards": np.array(self._current_rewards),
            "dones": np.array(self._current_dones),
            "infos": self._current_infos.copy(),
        }
        
        # Compute returns-to-go
        rewards = trajectory["rewards"]
        returns_to_go = np.zeros_like(rewards)
        running_return = 0.0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + running_return
            returns_to_go[t] = running_return
        trajectory["returns_to_go"] = returns_to_go
        
        # Compute horizons
        horizons = np.arange(len(rewards), 0, -1)
        trajectory["horizons"] = horizons
        
        self.trajectories.append(trajectory)
    
    def get_current_trajectory(self) -> Optional[Dict[str, np.ndarray]]:
        """Get the current trajectory being recorded."""
        if len(self._current_actions) == 0:
            return None
        
        trajectory = {
            "states": np.array(self._current_states),
            "actions": np.array(self._current_actions),
            "rewards": np.array(self._current_rewards),
            "dones": np.array(self._current_dones),
        }
        return trajectory
    
    def get_all_trajectories(self) -> List[Dict[str, np.ndarray]]:
        """Get all recorded trajectories."""
        return list(self.trajectories)
    
    def clear_trajectories(self) -> None:
        """Clear all stored trajectories."""
        self.trajectories.clear()
    
    def get_dataset(self) -> Dict[str, np.ndarray]:
        """
        Convert all trajectories to a flat dataset.
        
        Returns:
            Dictionary with concatenated arrays for all transitions
        """
        if len(self.trajectories) == 0:
            return {}
        
        all_states = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_returns_to_go = []
        all_horizons = []
        
        for traj in self.trajectories:
            # States (excluding final state for alignment)
            all_states.append(traj["states"][:-1] if len(traj["states"]) > len(traj["actions"]) 
                             else traj["states"])
            all_actions.append(traj["actions"])
            all_rewards.append(traj["rewards"])
            all_dones.append(traj["dones"])
            all_returns_to_go.append(traj["returns_to_go"])
            all_horizons.append(traj["horizons"])
        
        return {
            "states": np.concatenate(all_states),
            "actions": np.concatenate(all_actions),
            "rewards": np.concatenate(all_rewards),
            "dones": np.concatenate(all_dones),
            "returns_to_go": np.concatenate(all_returns_to_go),
            "horizons": np.concatenate(all_horizons),
        }


class FrameStack(gym.Wrapper):
    """
    Stack observations from last N frames.
    
    Useful for environments where history matters.
    
    Args:
        env: Environment to wrap
        n_frames: Number of frames to stack
    """
    
    def __init__(self, env: gym.Env, n_frames: int = 4):
        super().__init__(env)
        
        self.n_frames = n_frames
        self._frames: deque = deque(maxlen=n_frames)
        
        # Update observation space
        low = np.repeat(env.observation_space.low[np.newaxis, ...], n_frames, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], n_frames, axis=0)
        self.observation_space = spaces.Box(
            low=low.flatten(),
            high=high.flatten(),
            dtype=env.observation_space.dtype,
        )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset and initialize frame stack."""
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Initialize with copies of first frame
        for _ in range(self.n_frames):
            self._frames.append(obs.copy())
        
        return self._get_stacked_obs(), info
    
    def step(
        self,
        action: Any,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step and update frame stack."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs.copy())
        return self._get_stacked_obs(), reward, terminated, truncated, info
    
    def _get_stacked_obs(self) -> np.ndarray:
        """Get stacked observation."""
        return np.concatenate(list(self._frames))
