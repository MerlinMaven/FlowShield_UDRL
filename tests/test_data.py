"""
Tests for data handling.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from src.data import (
    Trajectory,
    TrajectoryBuffer,
    UDRLDataset,
    CommandDataset,
)


class TestTrajectory:
    """Tests for Trajectory dataclass."""
    
    def test_trajectory_creation(self):
        """Test creating a trajectory."""
        traj = Trajectory(
            states=np.array([[0, 0], [0, 1], [1, 1]]),
            actions=np.array([0, 1]),
            rewards=np.array([0.0, 1.0]),
        )
        
        assert len(traj) == 2
        assert traj.states.shape == (3, 2)
        assert traj.actions.shape == (2,)
        assert traj.rewards.shape == (2,)
    
    def test_trajectory_total_return(self):
        """Test trajectory total return."""
        traj = Trajectory(
            states=np.array([[0, 0], [0, 1], [1, 1], [1, 0]]),
            actions=np.array([0, 1, 2]),
            rewards=np.array([1.0, 2.0, 3.0]),
        )
        
        assert traj.total_return == 6.0
    
    def test_trajectory_returns_to_go(self):
        """Test returns-to-go computation."""
        traj = Trajectory(
            states=np.array([[0, 0], [0, 1], [1, 1], [1, 0]]),
            actions=np.array([0, 1, 2]),
            rewards=np.array([1.0, 2.0, 3.0]),
        )
        
        rtg = traj.compute_returns_to_go(gamma=1.0)
        
        assert rtg.shape == (3,)
        assert rtg[0] == 6.0  # 1 + 2 + 3
        assert rtg[1] == 5.0  # 2 + 3
        assert rtg[2] == 3.0  # 3
    
    def test_trajectory_discounted_returns(self):
        """Test discounted returns-to-go."""
        traj = Trajectory(
            states=np.array([[0, 0], [0, 1], [1, 1]]),
            actions=np.array([0, 1]),
            rewards=np.array([1.0, 1.0]),
        )
        
        rtg = traj.compute_returns_to_go(gamma=0.5)
        
        assert rtg[0] == 1.0 + 0.5 * 1.0
        assert rtg[1] == 1.0


class TestTrajectoryBuffer:
    """Tests for TrajectoryBuffer."""
    
    def test_buffer_creation(self):
        """Test creating a buffer."""
        buffer = TrajectoryBuffer()
        assert len(buffer) == 0
    
    def test_add_trajectory(self):
        """Test adding trajectories."""
        buffer = TrajectoryBuffer()
        
        traj = Trajectory(
            states=np.array([[0, 0], [0, 1]]),
            actions=np.array([0]),
            rewards=np.array([1.0]),
        )
        
        buffer.add(traj)
        
        assert len(buffer) == 1
    
    def test_buffer_statistics(self):
        """Test computing buffer statistics."""
        buffer = TrajectoryBuffer()
        
        for i in range(5):
            traj = Trajectory(
                states=np.random.randn(10, 2),
                actions=np.random.randint(0, 4, 9),
                rewards=np.random.randn(9),
            )
            buffer.add(traj)
        
        stats = buffer.compute_statistics()
        
        assert "n_trajectories" in stats
        assert "n_transitions" in stats
        assert "mean_return" in stats
        assert stats["n_trajectories"] == 5
    
    def test_buffer_save_load(self):
        """Test saving and loading buffer."""
        buffer = TrajectoryBuffer()
        
        for i in range(3):
            traj = Trajectory(
                states=np.random.randn(5, 2).astype(np.float32),
                actions=np.random.randint(0, 4, 4),
                rewards=np.random.randn(4).astype(np.float32),
            )
            buffer.add(traj)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer.pt"
            buffer.save(path)
            
            loaded = TrajectoryBuffer.load(path)
            
            assert len(loaded) == len(buffer)
            assert np.allclose(
                loaded.trajectories[0].states,
                buffer.trajectories[0].states
            )


class TestUDRLDataset:
    """Tests for UDRL dataset."""
    
    def test_dataset_creation(self):
        """Test creating dataset from tensors."""
        states = torch.randn(100, 8)
        actions = torch.randint(0, 4, (100,))
        commands = torch.randn(100, 2)
        
        dataset = UDRLDataset(
            states=states,
            actions=actions,
            commands=commands,
        )
        
        assert len(dataset) == 100
    
    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        states = torch.randn(100, 8)
        actions = torch.randint(0, 4, (100,))
        commands = torch.randn(100, 2)
        
        dataset = UDRLDataset(
            states=states,
            actions=actions,
            commands=commands,
        )
        
        item = dataset[0]
        
        assert "state" in item
        assert "action" in item
        assert "command" in item
        assert item["state"].shape == (8,)
    
    def test_dataset_from_buffer(self):
        """Test creating dataset from trajectory buffer."""
        buffer = TrajectoryBuffer()
        
        for i in range(10):
            traj = Trajectory(
                states=np.random.randn(20, 4).astype(np.float32),
                actions=np.random.randint(0, 4, 19),
                rewards=np.random.randn(19).astype(np.float32),
            )
            buffer.add(traj)
        
        dataset = UDRLDataset.from_buffer(buffer)
        
        assert len(dataset) > 0
        
        item = dataset[0]
        assert "state" in item
        assert "command" in item
    
    def test_dataset_normalization(self):
        """Test dataset normalization."""
        states = torch.randn(100, 8) * 10 + 5  # Non-zero mean and scale
        actions = torch.randint(0, 4, (100,))
        commands = torch.randn(100, 2)
        
        dataset = UDRLDataset(
            states=states,
            actions=actions,
            commands=commands,
            normalize=True,
        )
        
        # Check normalized states have ~0 mean and ~1 std
        all_states = torch.stack([dataset[i]["state"] for i in range(len(dataset))])
        
        assert abs(all_states.mean().item()) < 0.5
        assert abs(all_states.std().item() - 1.0) < 0.5


class TestCommandDataset:
    """Tests for command-only dataset (for Flow Matching)."""
    
    def test_dataset_creation(self):
        """Test creating command dataset."""
        states = torch.randn(100, 8)
        commands = torch.randn(100, 2)
        
        dataset = CommandDataset(
            states=states,
            commands=commands,
        )
        
        assert len(dataset) == 100
    
    def test_dataset_getitem(self):
        """Test getting items."""
        states = torch.randn(100, 8)
        commands = torch.randn(100, 2)
        
        dataset = CommandDataset(
            states=states,
            commands=commands,
        )
        
        item = dataset[0]
        
        assert "state" in item
        assert "command" in item
        assert item["state"].shape == (8,)
        assert item["command"].shape == (2,)


class TestDataLoading:
    """Tests for data loading utilities."""
    
    def test_dataloader_iteration(self):
        """Test iterating through dataloader."""
        from torch.utils.data import DataLoader
        
        states = torch.randn(100, 8)
        actions = torch.randint(0, 4, (100,))
        commands = torch.randn(100, 2)
        
        dataset = UDRLDataset(
            states=states,
            actions=actions,
            commands=commands,
        )
        
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        for batch in loader:
            assert "state" in batch
            assert "action" in batch
            assert "command" in batch
            assert batch["state"].shape[0] <= 16
            break
    
    def test_train_val_split(self):
        """Test splitting dataset."""
        from torch.utils.data import random_split
        
        states = torch.randn(100, 8)
        actions = torch.randint(0, 4, (100,))
        commands = torch.randn(100, 2)
        
        dataset = UDRLDataset(
            states=states,
            actions=actions,
            commands=commands,
        )
        
        train_size = 80
        val_size = 20
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )
        
        assert len(train_dataset) == 80
        assert len(val_dataset) == 20
