"""
PyTorch Datasets for UDRL Training.

Provides:
- UDRLDataset: Dataset for training UDRL policy π(a|s,g)
- CommandDataset: Dataset for training safety modules on command distribution
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .trajectory import Trajectory, TrajectoryBuffer


class UDRLDataset(Dataset):
    """
    PyTorch Dataset for UDRL training.
    
    Each sample contains:
    - state: Observation from the environment
    - command: (horizon, return_to_go) tuple
    - action: Action executed
    
    Args:
        trajectories: List of Trajectory objects or TrajectoryBuffer
        normalize_states: Whether to normalize states
        normalize_commands: Whether to normalize commands
        state_mean: Pre-computed state mean (optional)
        state_std: Pre-computed state std (optional)
        command_mean: Pre-computed command mean (optional)
        command_std: Pre-computed command std (optional)
        
    Example:
        >>> dataset = UDRLDataset(trajectory_buffer)
        >>> state, command, action = dataset[0]
        >>> print(state.shape, command.shape)
    """
    
    def __init__(
        self,
        trajectories: Union[List[Trajectory], TrajectoryBuffer],
        normalize_states: bool = True,
        normalize_commands: bool = True,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        command_mean: Optional[np.ndarray] = None,
        command_std: Optional[np.ndarray] = None,
    ):
        # Convert to list if buffer
        if isinstance(trajectories, TrajectoryBuffer):
            self.trajectories = trajectories.trajectories
            self.buffer = trajectories
        else:
            self.trajectories = trajectories
            self.buffer = TrajectoryBuffer()
            for t in trajectories:
                self.buffer.add(t)
        
        self.normalize_states = normalize_states
        self.normalize_commands = normalize_commands
        
        # Flatten trajectories to transitions
        self._build_dataset()
        
        # Compute or use provided statistics
        if normalize_states:
            if state_mean is not None and state_std is not None:
                self.state_mean = state_mean
                self.state_std = state_std
            else:
                self.state_mean = self.states.mean(axis=0)
                self.state_std = self.states.std(axis=0) + 1e-8
        
        if normalize_commands:
            if command_mean is not None and command_std is not None:
                self.command_mean = command_mean
                self.command_std = command_std
            else:
                self.command_mean = self.commands.mean(axis=0)
                self.command_std = self.commands.std(axis=0) + 1e-8
    
    def _build_dataset(self) -> None:
        """Flatten trajectories into arrays."""
        all_states = []
        all_commands = []
        all_actions = []
        
        for traj in self.trajectories:
            n = len(traj.actions)
            
            # States (handle T+1 vs T case)
            states = traj.states[:n] if len(traj.states) > n else traj.states
            
            # Commands: (horizon, return_to_go)
            commands = np.stack([
                traj.horizons.astype(np.float32),
                traj.returns_to_go.astype(np.float32)
            ], axis=1)
            
            all_states.append(states.astype(np.float32))
            all_commands.append(commands)
            all_actions.append(traj.actions)
        
        self.states = np.concatenate(all_states)
        self.commands = np.concatenate(all_commands)
        self.actions = np.concatenate(all_actions)
    
    def __len__(self) -> int:
        """Number of samples."""
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            Tuple of (state, command, action) tensors
        """
        state = self.states[idx].copy()
        command = self.commands[idx].copy()
        action = self.actions[idx]
        
        # Normalize
        if self.normalize_states:
            state = (state - self.state_mean) / self.state_std
        
        if self.normalize_commands:
            command = (command - self.command_mean) / self.command_std
        
        return (
            torch.from_numpy(state).float(),
            torch.from_numpy(command).float(),
            torch.tensor(action).long() if isinstance(action, (int, np.integer)) 
            else torch.from_numpy(action.astype(np.float32)),
        )
    
    def get_statistics(self) -> Dict[str, np.ndarray]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with mean, std, min, max for states and commands
        """
        stats = {
            "n_samples": len(self),
            "state_dim": self.states.shape[1] if len(self.states.shape) > 1 else 1,
            "command_dim": self.commands.shape[1],
        }
        
        if self.normalize_states:
            stats["state_mean"] = self.state_mean
            stats["state_std"] = self.state_std
        
        if self.normalize_commands:
            stats["command_mean"] = self.command_mean
            stats["command_std"] = self.command_std
        
        # Raw statistics
        stats["state_min"] = self.states.min(axis=0)
        stats["state_max"] = self.states.max(axis=0)
        stats["command_min"] = self.commands.min(axis=0)
        stats["command_max"] = self.commands.max(axis=0)
        
        return stats
    
    def denormalize_command(self, command: np.ndarray) -> np.ndarray:
        """Convert normalized command back to original scale."""
        if self.normalize_commands:
            return command * self.command_std + self.command_mean
        return command
    
    def normalize_command(self, command: np.ndarray) -> np.ndarray:
        """Normalize a command to dataset scale."""
        if self.normalize_commands:
            return (command - self.command_mean) / self.command_std
        return command
    
    def split(
        self,
        val_ratio: float = 0.1,
        shuffle: bool = True,
    ) -> Tuple["UDRLDataset", "UDRLDataset"]:
        """
        Split into train and validation sets.
        
        Args:
            val_ratio: Fraction for validation
            shuffle: Whether to shuffle before splitting
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        n = len(self)
        indices = np.arange(n)
        
        if shuffle:
            np.random.shuffle(indices)
        
        n_val = int(n * val_ratio)
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        # Create new datasets with shared statistics
        train_dataset = _SubsetUDRLDataset(
            self, train_indices,
            state_mean=self.state_mean if self.normalize_states else None,
            state_std=self.state_std if self.normalize_states else None,
            command_mean=self.command_mean if self.normalize_commands else None,
            command_std=self.command_std if self.normalize_commands else None,
        )
        
        val_dataset = _SubsetUDRLDataset(
            self, val_indices,
            state_mean=self.state_mean if self.normalize_states else None,
            state_std=self.state_std if self.normalize_states else None,
            command_mean=self.command_mean if self.normalize_commands else None,
            command_std=self.command_std if self.normalize_commands else None,
        )
        
        return train_dataset, val_dataset


class _SubsetUDRLDataset(Dataset):
    """Subset of UDRLDataset with shared statistics."""
    
    def __init__(
        self,
        parent: UDRLDataset,
        indices: np.ndarray,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        command_mean: Optional[np.ndarray] = None,
        command_std: Optional[np.ndarray] = None,
    ):
        self.parent = parent
        self.indices = indices
        self.state_mean = state_mean
        self.state_std = state_std
        self.command_mean = command_mean
        self.command_std = command_std
        self.normalize_states = state_mean is not None
        self.normalize_commands = command_mean is not None
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.parent[self.indices[idx]]


class CommandDataset(Dataset):
    """
    Dataset for training safety modules on command distribution p(g|s).
    
    Each sample contains:
    - state: Observation
    - command: (horizon, return_to_go) that was achieved from that state
    
    This is used to train Flow Matching or other density models
    on the distribution of achievable commands.
    
    Args:
        trajectories: List of Trajectory objects or TrajectoryBuffer
        normalize_states: Whether to normalize states
        normalize_commands: Whether to normalize commands
        
    Example:
        >>> dataset = CommandDataset(trajectory_buffer)
        >>> state, command = dataset[0]
    """
    
    def __init__(
        self,
        trajectories: Union[List[Trajectory], TrajectoryBuffer],
        normalize_states: bool = True,
        normalize_commands: bool = True,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        command_mean: Optional[np.ndarray] = None,
        command_std: Optional[np.ndarray] = None,
    ):
        if isinstance(trajectories, TrajectoryBuffer):
            self.trajectories = trajectories.trajectories
        else:
            self.trajectories = trajectories
        
        self.normalize_states = normalize_states
        self.normalize_commands = normalize_commands
        
        # Build dataset
        self._build_dataset()
        
        # Statistics
        if normalize_states:
            self.state_mean = state_mean if state_mean is not None else self.states.mean(axis=0)
            self.state_std = state_std if state_std is not None else self.states.std(axis=0) + 1e-8
        
        if normalize_commands:
            self.command_mean = command_mean if command_mean is not None else self.commands.mean(axis=0)
            self.command_std = command_std if command_std is not None else self.commands.std(axis=0) + 1e-8
    
    def _build_dataset(self) -> None:
        """Build arrays from trajectories."""
        all_states = []
        all_commands = []
        
        for traj in self.trajectories:
            n = len(traj.actions)
            states = traj.states[:n] if len(traj.states) > n else traj.states
            
            commands = np.stack([
                traj.horizons.astype(np.float32),
                traj.returns_to_go.astype(np.float32)
            ], axis=1)
            
            all_states.append(states.astype(np.float32))
            all_commands.append(commands)
        
        self.states = np.concatenate(all_states)
        self.commands = np.concatenate(all_commands)
    
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            Tuple of (state, command) tensors
        """
        state = self.states[idx].copy()
        command = self.commands[idx].copy()
        
        if self.normalize_states:
            state = (state - self.state_mean) / self.state_std
        
        if self.normalize_commands:
            command = (command - self.command_mean) / self.command_std
        
        return (
            torch.from_numpy(state).float(),
            torch.from_numpy(command).float(),
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "n_samples": len(self),
            "state_mean": self.state_mean if self.normalize_states else None,
            "state_std": self.state_std if self.normalize_states else None,
            "command_mean": self.command_mean if self.normalize_commands else None,
            "command_std": self.command_std if self.normalize_commands else None,
            "command_min": self.commands.min(axis=0),
            "command_max": self.commands.max(axis=0),
        }
    
    def sample_states(self, n: int) -> torch.Tensor:
        """Sample n random states from the dataset."""
        indices = np.random.choice(len(self), size=n, replace=True)
        states = self.states[indices].copy()
        
        if self.normalize_states:
            states = (states - self.state_mean) / self.state_std
        
        return torch.from_numpy(states).float()
    
    def get_commands_for_state(
        self,
        state: np.ndarray,
        tolerance: float = 0.1,
    ) -> np.ndarray:
        """
        Get all commands that were achieved from similar states.
        
        Useful for understanding the command distribution at a state.
        
        Args:
            state: Query state
            tolerance: Maximum L2 distance for state matching
            
        Returns:
            Array of commands achieved from similar states
        """
        # Normalize state if needed
        if self.normalize_states:
            state = (state - self.state_mean) / self.state_std
        
        # Find similar states
        distances = np.linalg.norm(self.states - state, axis=1)
        mask = distances < tolerance
        
        return self.commands[mask]
