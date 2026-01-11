"""
Trajectory handling for Offline RL.

A Trajectory contains:
- states: List of states visited
- actions: List of actions taken
- rewards: List of rewards received
- returns_to_go: Cumulative return from each step
- horizons: Remaining steps from each position
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class Trajectory:
    """
    Single trajectory from an RL episode.
    
    Attributes:
        states: Array of states, shape (T+1, state_dim) or (T, state_dim)
        actions: Array of actions, shape (T,) or (T, action_dim)
        rewards: Array of rewards, shape (T,)
        dones: Array of done flags, shape (T,)
        returns_to_go: Computed return-to-go at each step
        horizons: Remaining steps at each position
        infos: Optional list of info dicts
        
    Example:
        >>> traj = Trajectory(
        ...     states=np.random.randn(11, 2),  # T+1 states
        ...     actions=np.random.randint(0, 4, 10),  # T actions
        ...     rewards=np.random.randn(10),  # T rewards
        ...     dones=np.zeros(10, dtype=bool)
        ... )
        >>> print(traj.returns_to_go)  # Automatically computed
    """
    
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    
    # Computed automatically
    returns_to_go: np.ndarray = field(default=None, repr=False)
    horizons: np.ndarray = field(default=None, repr=False)
    
    # Optional
    infos: Optional[List[Dict]] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Compute returns-to-go and horizons after initialization."""
        # Ensure arrays
        self.states = np.asarray(self.states)
        self.actions = np.asarray(self.actions)
        self.rewards = np.asarray(self.rewards)
        self.dones = np.asarray(self.dones)
        
        # Compute if not provided
        if self.returns_to_go is None:
            self._compute_returns_to_go()
        if self.horizons is None:
            self._compute_horizons()
    
    def _compute_returns_to_go(self, gamma: float = 1.0) -> None:
        """
        Compute return-to-go at each timestep.
        
        Return-to-go at step t = sum of rewards from t to end.
        For UDRL, we typically use gamma=1.0 (undiscounted).
        
        Args:
            gamma: Discount factor (default 1.0 for undiscounted)
        """
        T = len(self.rewards)
        self.returns_to_go = np.zeros(T, dtype=np.float32)
        
        running_return = 0.0
        for t in reversed(range(T)):
            running_return = self.rewards[t] + gamma * running_return
            self.returns_to_go[t] = running_return
    
    def _compute_horizons(self) -> None:
        """
        Compute remaining horizon at each timestep.
        
        Horizon at step t = number of steps remaining (T - t).
        """
        T = len(self.rewards)
        self.horizons = np.arange(T, 0, -1, dtype=np.int32)
    
    def __len__(self) -> int:
        """Return number of transitions."""
        return len(self.actions)
    
    @property
    def total_return(self) -> float:
        """Total episode return."""
        return float(self.rewards.sum())
    
    @property
    def length(self) -> int:
        """Episode length."""
        return len(self.actions)
    
    @property
    def success(self) -> bool:
        """Whether episode was successful (positive return)."""
        return self.total_return > 0
    
    def get_transition(self, idx: int) -> Dict[str, Any]:
        """
        Get a single transition.
        
        Args:
            idx: Transition index
            
        Returns:
            Dictionary with state, action, reward, next_state, done, etc.
        """
        return {
            "state": self.states[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_state": self.states[idx + 1] if idx + 1 < len(self.states) else self.states[idx],
            "done": self.dones[idx],
            "return_to_go": self.returns_to_go[idx],
            "horizon": self.horizons[idx],
        }
    
    def get_udrl_sample(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get a UDRL training sample (state, command, action).
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with state, command (horizon, return), action
        """
        return {
            "state": self.states[idx].astype(np.float32),
            "command": np.array([self.horizons[idx], self.returns_to_go[idx]], dtype=np.float32),
            "action": self.actions[idx],
        }
    
    def truncate(self, max_length: int) -> "Trajectory":
        """
        Truncate trajectory to maximum length.
        
        Args:
            max_length: Maximum number of transitions
            
        Returns:
            Truncated trajectory
        """
        if len(self) <= max_length:
            return self
        
        return Trajectory(
            states=self.states[:max_length + 1],
            actions=self.actions[:max_length],
            rewards=self.rewards[:max_length],
            dones=self.dones[:max_length],
            infos=self.infos[:max_length] if self.infos else None,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "returns_to_go": self.returns_to_go,
            "horizons": self.horizons,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        """Create from dictionary."""
        return cls(
            states=data["states"],
            actions=data["actions"],
            rewards=data["rewards"],
            dones=data["dones"],
            returns_to_go=data.get("returns_to_go"),
            horizons=data.get("horizons"),
        )


class TrajectoryBuffer:
    """
    Buffer for storing and managing multiple trajectories.
    
    Provides utilities for:
    - Adding trajectories
    - Computing statistics
    - Sampling
    - Serialization
    
    Example:
        >>> buffer = TrajectoryBuffer()
        >>> buffer.add(trajectory)
        >>> print(buffer.statistics)
    """
    
    def __init__(self, max_trajectories: Optional[int] = None):
        """
        Initialize buffer.
        
        Args:
            max_trajectories: Maximum number of trajectories to store
        """
        self.max_trajectories = max_trajectories
        self.trajectories: List[Trajectory] = []
        
        # Cached statistics
        self._stats_cache: Optional[Dict[str, float]] = None
    
    def add(self, trajectory: Trajectory) -> None:
        """Add a trajectory to the buffer."""
        self.trajectories.append(trajectory)
        self._stats_cache = None  # Invalidate cache
        
        # Remove oldest if over capacity
        if self.max_trajectories and len(self.trajectories) > self.max_trajectories:
            self.trajectories.pop(0)
    
    def add_batch(self, trajectories: List[Trajectory]) -> None:
        """Add multiple trajectories."""
        for traj in trajectories:
            self.add(traj)
    
    def __len__(self) -> int:
        """Number of trajectories."""
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Trajectory:
        """Get trajectory by index."""
        return self.trajectories[idx]
    
    @property
    def total_transitions(self) -> int:
        """Total number of transitions across all trajectories."""
        return sum(len(t) for t in self.trajectories)
    
    @property
    def statistics(self) -> Dict[str, float]:
        """
        Compute buffer statistics.
        
        Returns:
            Dictionary with mean, std, min, max for returns and lengths
        """
        if self._stats_cache is not None:
            return self._stats_cache
        
        if len(self.trajectories) == 0:
            return {}
        
        returns = np.array([t.total_return for t in self.trajectories])
        lengths = np.array([t.length for t in self.trajectories])
        
        self._stats_cache = {
            "n_trajectories": len(self.trajectories),
            "total_transitions": self.total_transitions,
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std()),
            "return_min": float(returns.min()),
            "return_max": float(returns.max()),
            "length_mean": float(lengths.mean()),
            "length_std": float(lengths.std()),
            "length_min": float(lengths.min()),
            "length_max": float(lengths.max()),
            "success_rate": float((returns > 0).mean()),
        }
        
        return self._stats_cache
    
    def get_command_statistics(self) -> Dict[str, np.ndarray]:
        """
        Get statistics for commands (horizon, return) across all trajectories.
        
        Returns:
            Dictionary with mean, std, min, max for each command dimension
        """
        all_horizons = []
        all_returns = []
        
        for traj in self.trajectories:
            all_horizons.extend(traj.horizons.tolist())
            all_returns.extend(traj.returns_to_go.tolist())
        
        horizons = np.array(all_horizons)
        returns = np.array(all_returns)
        
        return {
            "horizon_mean": horizons.mean(),
            "horizon_std": horizons.std(),
            "horizon_min": horizons.min(),
            "horizon_max": horizons.max(),
            "return_mean": returns.mean(),
            "return_std": returns.std(),
            "return_min": returns.min(),
            "return_max": returns.max(),
            "command_mean": np.array([horizons.mean(), returns.mean()]),
            "command_std": np.array([horizons.std(), returns.std()]),
        }
    
    def sample_trajectories(self, n: int, replace: bool = True) -> List[Trajectory]:
        """
        Sample trajectories from buffer.
        
        Args:
            n: Number of trajectories to sample
            replace: Whether to sample with replacement
            
        Returns:
            List of sampled trajectories
        """
        indices = np.random.choice(len(self.trajectories), size=n, replace=replace)
        return [self.trajectories[i] for i in indices]
    
    def filter_by_return(
        self,
        min_return: Optional[float] = None,
        max_return: Optional[float] = None,
    ) -> "TrajectoryBuffer":
        """
        Filter trajectories by return.
        
        Args:
            min_return: Minimum return threshold
            max_return: Maximum return threshold
            
        Returns:
            New buffer with filtered trajectories
        """
        filtered = TrajectoryBuffer(self.max_trajectories)
        
        for traj in self.trajectories:
            ret = traj.total_return
            if min_return is not None and ret < min_return:
                continue
            if max_return is not None and ret > max_return:
                continue
            filtered.add(traj)
        
        return filtered
    
    def split(self, val_ratio: float = 0.1) -> tuple:
        """
        Split buffer into train and validation sets.
        
        Args:
            val_ratio: Fraction for validation
            
        Returns:
            Tuple of (train_buffer, val_buffer)
        """
        n_val = int(len(self.trajectories) * val_ratio)
        n_train = len(self.trajectories) - n_val
        
        # Shuffle indices
        indices = np.random.permutation(len(self.trajectories))
        
        train_buffer = TrajectoryBuffer()
        val_buffer = TrajectoryBuffer()
        
        for i in indices[:n_train]:
            train_buffer.add(self.trajectories[i])
        for i in indices[n_train:]:
            val_buffer.add(self.trajectories[i])
        
        return train_buffer, val_buffer
    
    def save(self, path: Union[str, Path]) -> None:
        """Save buffer to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "trajectories": [t.to_dict() for t in self.trajectories],
            "statistics": self.statistics,
        }
        
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrajectoryBuffer":
        """Load buffer from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        buffer = cls()
        for traj_dict in data["trajectories"]:
            buffer.add(Trajectory.from_dict(traj_dict))
        
        return buffer
    
    def to_flat_dataset(self) -> Dict[str, np.ndarray]:
        """
        Convert to flat arrays for dataset creation.
        
        Returns:
            Dictionary with concatenated arrays
        """
        if len(self.trajectories) == 0:
            return {}
        
        all_states = []
        all_actions = []
        all_rewards = []
        all_returns_to_go = []
        all_horizons = []
        
        for traj in self.trajectories:
            # Use states[:-1] if we have T+1 states for T transitions
            n_transitions = len(traj.actions)
            states = traj.states[:n_transitions]
            
            all_states.append(states)
            all_actions.append(traj.actions)
            all_rewards.append(traj.rewards)
            all_returns_to_go.append(traj.returns_to_go)
            all_horizons.append(traj.horizons)
        
        return {
            "states": np.concatenate(all_states).astype(np.float32),
            "actions": np.concatenate(all_actions),
            "rewards": np.concatenate(all_rewards).astype(np.float32),
            "returns_to_go": np.concatenate(all_returns_to_go).astype(np.float32),
            "horizons": np.concatenate(all_horizons).astype(np.int32),
        }
