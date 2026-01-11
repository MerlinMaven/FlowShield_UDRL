"""
Tests for Flow Matching implementation.
"""

import pytest
import torch
import torch.nn as nn

from src.models.safety.flow_matching import (
    FlowMatchingModel,
    FlowMatchingSafetyModule,
    VectorFieldNetwork,
)


class TestVectorFieldNetwork:
    """Tests for the vector field network."""
    
    def test_forward(self):
        """Test forward pass."""
        net = VectorFieldNetwork(
            command_dim=2,
            state_dim=8,
            hidden_dim=64,
            n_layers=3,
        )
        
        g_t = torch.randn(32, 2)
        state = torch.randn(32, 8)
        t = torch.rand(32)
        
        velocity = net(g_t, state, t)
        
        assert velocity.shape == (32, 2)
    
    def test_time_conditioning(self):
        """Test that output depends on time."""
        net = VectorFieldNetwork(
            command_dim=2,
            state_dim=8,
            hidden_dim=64,
            n_layers=3,
        )
        
        g_t = torch.randn(1, 2)
        state = torch.randn(1, 8)
        
        v_early = net(g_t, state, torch.tensor([0.1]))
        v_late = net(g_t, state, torch.tensor([0.9]))
        
        # Velocities should be different at different times
        assert not torch.allclose(v_early, v_late, atol=1e-3)
    
    def test_state_conditioning(self):
        """Test that output depends on state."""
        net = VectorFieldNetwork(
            command_dim=2,
            state_dim=8,
            hidden_dim=64,
            n_layers=3,
        )
        
        g_t = torch.randn(1, 2)
        t = torch.tensor([0.5])
        
        state1 = torch.randn(1, 8)
        state2 = torch.randn(1, 8)
        
        v1 = net(g_t, state1, t)
        v2 = net(g_t, state2, t)
        
        assert not torch.allclose(v1, v2, atol=1e-3)


class TestFlowMatchingModel:
    """Tests for Flow Matching model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = FlowMatchingModel(
            command_dim=2,
            state_dim=8,
            hidden_dim=64,
            n_layers=3,
        )
        
        assert model.command_dim == 2
        assert model.state_dim == 8
    
    def test_compute_loss(self):
        """Test loss computation."""
        model = FlowMatchingModel(
            command_dim=2,
            state_dim=8,
            hidden_dim=64,
            n_layers=3,
        )
        
        state = torch.randn(32, 8)
        command = torch.randn(32, 2)
        
        loss_dict = model.compute_loss(state, command)
        
        assert "loss" in loss_dict
        assert loss_dict["loss"].ndim == 0  # Scalar
        assert not torch.isnan(loss_dict["loss"])
        assert loss_dict["loss"] >= 0
    
    def test_sample(self):
        """Test sampling."""
        model = FlowMatchingModel(
            command_dim=2,
            state_dim=8,
            hidden_dim=64,
            n_layers=3,
            n_steps=10,  # Fewer steps for testing
        )
        
        state = torch.randn(4, 8)
        
        samples = model.sample(state, n_samples=5)
        
        assert samples.shape == (4, 5, 2)
        assert not torch.isnan(samples).any()
    
    def test_sample_with_trajectory(self):
        """Test sampling with trajectory output."""
        model = FlowMatchingModel(
            command_dim=2,
            state_dim=8,
            hidden_dim=64,
            n_layers=3,
            n_steps=10,
        )
        
        state = torch.randn(2, 8)
        
        samples, trajectory = model.sample(
            state, n_samples=3, return_trajectory=True
        )
        
        assert samples.shape == (2, 3, 2)
        assert trajectory.shape == (2, 3, 11, 2)  # n_steps + 1
    
    def test_log_prob(self):
        """Test log probability computation."""
        model = FlowMatchingModel(
            command_dim=2,
            state_dim=8,
            hidden_dim=64,
            n_layers=3,
            n_steps=10,
        )
        
        state = torch.randn(8, 8)
        command = torch.randn(8, 2)
        
        log_prob = model.log_prob(state, command)
        
        assert log_prob.shape == (8,)
        assert not torch.isnan(log_prob).any()
    
    def test_training_convergence(self):
        """Test that training reduces loss."""
        torch.manual_seed(42)
        
        model = FlowMatchingModel(
            command_dim=2,
            state_dim=8,
            hidden_dim=64,
            n_layers=3,
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Generate training data from a simple distribution
        n_samples = 1000
        states = torch.randn(n_samples, 8)
        # Commands depend on states (simple linear relationship for testing)
        commands = states[:, :2] + 0.1 * torch.randn(n_samples, 2)
        
        initial_loss = None
        final_loss = None
        
        for epoch in range(50):
            idx = torch.randperm(n_samples)[:64]
            
            loss_dict = model.compute_loss(states[idx], commands[idx])
            loss = loss_dict["loss"]
            
            if epoch == 0:
                initial_loss = loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            final_loss = loss.item()
        
        # Loss should decrease
        assert final_loss < initial_loss
    
    def test_sample_quality_after_training(self):
        """Test that samples match training distribution after training."""
        torch.manual_seed(42)
        
        model = FlowMatchingModel(
            command_dim=2,
            state_dim=4,
            hidden_dim=64,
            n_layers=3,
            n_steps=50,
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Training data: commands are states[:2] + noise
        n_samples = 1000
        states = torch.randn(n_samples, 4)
        commands = states[:, :2] * 2 + torch.randn(n_samples, 2) * 0.1
        
        # Train
        for epoch in range(100):
            idx = torch.randperm(n_samples)[:64]
            loss_dict = model.compute_loss(states[idx], commands[idx])
            
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()
        
        # Sample and check distribution
        test_states = torch.randn(100, 4)
        expected_commands = test_states[:, :2] * 2
        
        with torch.no_grad():
            samples = model.sample(test_states, n_samples=50)
        
        # Mean of samples should be close to expected
        sample_means = samples.mean(dim=1)
        error = (sample_means - expected_commands).abs().mean()
        
        # Allow for some error due to limited training
        assert error < 1.0  # This is a loose bound


class TestFlowMatchingSafetyModuleAdvanced:
    """Advanced tests for Flow Matching safety module."""
    
    def test_ood_detection_separation(self):
        """Test that ID and OOD commands have different scores after training."""
        torch.manual_seed(42)
        
        module = FlowMatchingSafetyModule(
            state_dim=4,
            command_dim=2,
            hidden_dim=64,
            n_layers=3,
            n_integration_steps=20,
        )
        
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
        
        # Training data: commands in [0, 1]^2
        n_samples = 500
        states = torch.randn(n_samples, 4)
        commands = torch.rand(n_samples, 2)  # Uniform in [0, 1]
        
        # Train
        for epoch in range(50):
            idx = torch.randperm(n_samples)[:32]
            loss_dict = module.compute_loss(states[idx], commands[idx])
            
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()
        
        # Test: ID commands should have higher scores than OOD
        test_states = torch.randn(50, 4)
        id_commands = torch.rand(50, 2)  # In distribution
        ood_commands = torch.rand(50, 2) * 5 + 5  # Out of distribution [5, 10]
        
        with torch.no_grad():
            id_scores = module.get_safety_score(test_states, id_commands)
            ood_scores = module.get_safety_score(test_states, ood_commands)
        
        # ID should generally have higher scores
        assert id_scores.mean() > ood_scores.mean()
    
    def test_projection_moves_towards_data(self):
        """Test that projection moves commands towards training distribution."""
        torch.manual_seed(42)
        
        module = FlowMatchingSafetyModule(
            state_dim=4,
            command_dim=2,
            hidden_dim=64,
            n_layers=3,
            n_integration_steps=20,
            ood_threshold=-10.0,
            projection_steps=10,
            projection_lr=0.5,
        )
        
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
        
        # Training data centered at (0, 0)
        n_samples = 500
        states = torch.randn(n_samples, 4)
        commands = torch.randn(n_samples, 2) * 0.5  # Centered at origin
        
        for epoch in range(50):
            idx = torch.randperm(n_samples)[:32]
            loss_dict = module.compute_loss(states[idx], commands[idx])
            
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()
        
        # OOD command far from origin
        test_states = torch.randn(10, 4)
        ood_commands = torch.ones(10, 2) * 10  # Far from training data
        
        projected = module.project(test_states, ood_commands)
        
        # Projected should be closer to origin than original
        original_dist = ood_commands.norm(dim=-1).mean()
        projected_dist = projected.norm(dim=-1).mean()
        
        assert projected_dist < original_dist
    
    def test_sample_consistency(self):
        """Test that samples are consistent with learned distribution."""
        torch.manual_seed(42)
        
        module = FlowMatchingSafetyModule(
            state_dim=4,
            command_dim=2,
            hidden_dim=64,
            n_layers=3,
            n_integration_steps=30,
        )
        
        # Training distribution: bimodal
        n_samples = 500
        states = torch.randn(n_samples, 4)
        mode = (torch.rand(n_samples) > 0.5).float().unsqueeze(-1)
        commands = mode * torch.randn(n_samples, 2) + (1 - mode) * (torch.randn(n_samples, 2) + 5)
        
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
        
        for epoch in range(100):
            idx = torch.randperm(n_samples)[:32]
            loss_dict = module.compute_loss(states[idx], commands[idx])
            
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()
        
        # Samples should cover both modes
        test_states = torch.randn(20, 4)
        with torch.no_grad():
            samples = module.sample_commands(test_states, n_samples=50)
        
        # Check variance covers range
        sample_variance = samples.reshape(-1, 2).var(dim=0)
        assert sample_variance.min() > 0.5  # Should have reasonable spread


class TestNumericalStability:
    """Tests for numerical stability."""
    
    def test_large_time_values(self):
        """Test stability with edge time values."""
        net = VectorFieldNetwork(
            command_dim=2,
            state_dim=8,
            hidden_dim=64,
            n_layers=3,
        )
        
        g_t = torch.randn(32, 2)
        state = torch.randn(32, 8)
        
        # Test edge times
        for t_val in [0.0, 0.001, 0.999, 1.0]:
            t = torch.full((32,), t_val)
            velocity = net(g_t, state, t)
            
            assert not torch.isnan(velocity).any()
            assert not torch.isinf(velocity).any()
    
    def test_large_inputs(self):
        """Test stability with large input values."""
        module = FlowMatchingSafetyModule(
            state_dim=8,
            command_dim=2,
            hidden_dim=64,
            n_layers=3,
            n_integration_steps=10,
        )
        
        state = torch.randn(16, 8) * 100
        command = torch.randn(16, 2) * 100
        
        score = module.get_safety_score(state, command)
        
        assert not torch.isnan(score).any()
        assert not torch.isinf(score).any()
    
    def test_gradient_stability(self):
        """Test gradient stability during training."""
        model = FlowMatchingModel(
            command_dim=2,
            state_dim=8,
            hidden_dim=64,
            n_layers=4,
        )
        
        state = torch.randn(32, 8)
        command = torch.randn(32, 2)
        
        for _ in range(10):
            loss_dict = model.compute_loss(state, command)
            loss_dict["loss"].backward()
            
            for param in model.parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any()
                    assert not torch.isinf(param.grad).any()
            
            model.zero_grad()
