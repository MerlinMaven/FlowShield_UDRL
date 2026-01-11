"""
Tests for model implementations.
"""

import pytest
import torch
import torch.nn as nn

from src.models.agent import UDRLPolicy, UDRLPolicyWithValue
from src.models.components import MLP, ResidualMLP, SinusoidalTimeEmbedding
from src.models.safety import (
    BaseSafetyModule,
    FlowMatchingSafetyModule,
    QuantileSafetyModule,
)


class TestMLP:
    """Tests for MLP components."""
    
    def test_mlp_forward(self):
        """Test MLP forward pass."""
        mlp = MLP(input_dim=8, output_dim=4, hidden_dims=64, n_layers=3)
        
        x = torch.randn(32, 8)
        y = mlp(x)
        
        assert y.shape == (32, 4)
    
    def test_mlp_gradients(self):
        """Test that gradients flow through MLP."""
        mlp = MLP(input_dim=8, output_dim=4, hidden_dims=64, n_layers=3)
        
        x = torch.randn(32, 8, requires_grad=True)
        y = mlp(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_residual_mlp(self):
        """Test ResidualMLP."""
        mlp = ResidualMLP(input_dim=64, output_dim=64, hidden_dim=64, n_layers=4)
        
        x = torch.randn(32, 64)
        y = mlp(x)
        
        assert y.shape == (32, 64)


class TestEmbeddings:
    """Tests for embedding layers."""
    
    def test_sinusoidal_time_embedding(self):
        """Test sinusoidal time embedding."""
        embed = SinusoidalTimeEmbedding(dim=64)
        
        t = torch.linspace(0, 1, 100)
        emb = embed(t)
        
        assert emb.shape == (100, 64)
        
        # Different times should have different embeddings
        assert not torch.allclose(emb[0], emb[50])
    
    def test_time_embedding_deterministic(self):
        """Test that time embedding is deterministic."""
        embed = SinusoidalTimeEmbedding(dim=64)
        
        t = torch.tensor([0.5])
        emb1 = embed(t)
        emb2 = embed(t)
        
        assert torch.allclose(emb1, emb2)


class TestUDRLPolicy:
    """Tests for UDRL policy."""
    
    def test_discrete_policy(self):
        """Test discrete action policy."""
        policy = UDRLPolicy(
            state_dim=8,
            action_dim=4,
            hidden_dim=64,
            action_type="discrete",
        )
        
        state = torch.randn(32, 8)
        command = torch.randn(32, 2)
        
        action = policy.get_action(state, command)
        
        assert action.shape == (32,)
        assert (action >= 0).all() and (action < 4).all()
    
    def test_continuous_policy(self):
        """Test continuous action policy."""
        policy = UDRLPolicy(
            state_dim=8,
            action_dim=2,
            hidden_dim=64,
            action_type="continuous",
        )
        
        state = torch.randn(32, 8)
        command = torch.randn(32, 2)
        
        action = policy.get_action(state, command)
        
        assert action.shape == (32, 2)
    
    def test_policy_log_prob(self):
        """Test policy log probability computation."""
        policy = UDRLPolicy(
            state_dim=8,
            action_dim=4,
            hidden_dim=64,
            action_type="discrete",
        )
        
        state = torch.randn(32, 8)
        command = torch.randn(32, 2)
        action = torch.randint(0, 4, (32,))
        
        log_prob = policy.log_prob(state, command, action)
        
        assert log_prob.shape == (32,)
        assert not torch.isnan(log_prob).any()
        assert (log_prob <= 0).all()  # Log probs are non-positive
    
    def test_policy_loss(self):
        """Test policy loss computation."""
        policy = UDRLPolicy(
            state_dim=8,
            action_dim=4,
            hidden_dim=64,
            action_type="discrete",
        )
        
        state = torch.randn(32, 8)
        command = torch.randn(32, 2)
        action = torch.randint(0, 4, (32,))
        
        loss_dict = policy.get_loss(state, command, action)
        
        assert "loss" in loss_dict
        assert not torch.isnan(loss_dict["loss"])
        assert loss_dict["loss"] >= 0
    
    def test_deterministic_action(self):
        """Test deterministic action selection."""
        policy = UDRLPolicy(
            state_dim=8,
            action_dim=4,
            hidden_dim=64,
            action_type="discrete",
        )
        
        state = torch.randn(1, 8)
        command = torch.randn(1, 2)
        
        policy.eval()
        action1 = policy.get_action(state, command, deterministic=True)
        action2 = policy.get_action(state, command, deterministic=True)
        
        assert torch.equal(action1, action2)
    
    def test_policy_with_value(self):
        """Test policy with value head."""
        policy = UDRLPolicyWithValue(
            state_dim=8,
            action_dim=4,
            hidden_dim=64,
            action_type="discrete",
        )
        
        state = torch.randn(32, 8)
        command = torch.randn(32, 2)
        
        action = policy.get_action(state, command)
        value = policy.get_value(state, command)
        
        assert action.shape == (32,)
        assert value.shape == (32, 1)


class TestFlowMatchingSafetyModule:
    """Tests for Flow Matching safety module."""
    
    def test_initialization(self):
        """Test module initialization."""
        module = FlowMatchingSafetyModule(
            state_dim=8,
            command_dim=2,
            hidden_dim=64,
            n_layers=3,
        )
        
        # Check parameters exist
        params = list(module.parameters())
        assert len(params) > 0
    
    def test_safety_score(self):
        """Test safety score computation."""
        module = FlowMatchingSafetyModule(
            state_dim=8,
            command_dim=2,
            hidden_dim=64,
            n_layers=3,
            n_integration_steps=10,  # Fewer steps for testing
        )
        
        state = torch.randn(16, 8)
        command = torch.randn(16, 2)
        
        score = module.get_safety_score(state, command)
        
        assert score.shape == (16,)
        assert not torch.isnan(score).any()
    
    def test_is_safe(self):
        """Test safety classification."""
        module = FlowMatchingSafetyModule(
            state_dim=8,
            command_dim=2,
            hidden_dim=64,
            n_layers=3,
            ood_threshold=-10.0,
            n_integration_steps=10,
        )
        
        state = torch.randn(16, 8)
        command = torch.randn(16, 2)
        
        is_safe = module.is_safe(state, command)
        
        assert is_safe.shape == (16,)
        assert is_safe.dtype == torch.bool
    
    def test_sample_commands(self):
        """Test command sampling."""
        module = FlowMatchingSafetyModule(
            state_dim=8,
            command_dim=2,
            hidden_dim=64,
            n_layers=3,
            n_integration_steps=10,
        )
        
        state = torch.randn(4, 8)
        
        samples = module.sample_commands(state, n_samples=10)
        
        assert samples.shape == (4, 10, 2)
    
    def test_project(self):
        """Test command projection."""
        module = FlowMatchingSafetyModule(
            state_dim=8,
            command_dim=2,
            hidden_dim=64,
            n_layers=3,
            n_integration_steps=10,
            projection_steps=5,  # Fewer steps for testing
        )
        
        state = torch.randn(8, 8)
        command = torch.randn(8, 2) * 10  # Large commands (likely OOD)
        
        projected = module.project(state, command)
        
        assert projected.shape == (8, 2)
    
    def test_compute_loss(self):
        """Test flow matching loss computation."""
        module = FlowMatchingSafetyModule(
            state_dim=8,
            command_dim=2,
            hidden_dim=64,
            n_layers=3,
        )
        
        state = torch.randn(32, 8)
        command = torch.randn(32, 2)
        
        loss_dict = module.compute_loss(state, command)
        
        assert "loss" in loss_dict
        assert not torch.isnan(loss_dict["loss"])
        assert loss_dict["loss"] >= 0
    
    def test_training_step(self):
        """Test that training step works."""
        module = FlowMatchingSafetyModule(
            state_dim=8,
            command_dim=2,
            hidden_dim=64,
            n_layers=3,
        )
        
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
        
        state = torch.randn(32, 8)
        command = torch.randn(32, 2)
        
        # Forward pass
        loss_dict = module.compute_loss(state, command)
        loss = loss_dict["loss"]
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        for param in module.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()
        
        optimizer.step()


class TestQuantileSafetyModule:
    """Tests for Quantile safety module."""
    
    def test_initialization(self):
        """Test module initialization."""
        module = QuantileSafetyModule(
            state_dim=8,
            command_dim=2,
            hidden_dim=64,
            tau=0.9,
        )
        
        params = list(module.parameters())
        assert len(params) > 0
    
    def test_return_bounds(self):
        """Test return bounds prediction."""
        module = QuantileSafetyModule(
            state_dim=8,
            command_dim=2,
            hidden_dim=64,
            tau=0.9,
        )
        
        state = torch.randn(16, 8)
        
        lower, upper = module.get_return_bounds(state)
        
        assert lower.shape == (16,)
        assert upper.shape == (16,)
    
    def test_is_safe(self):
        """Test safety classification."""
        module = QuantileSafetyModule(
            state_dim=8,
            command_dim=2,
            hidden_dim=64,
            tau=0.9,
        )
        
        state = torch.randn(16, 8)
        command = torch.randn(16, 2)
        
        is_safe = module.is_safe(state, command)
        
        assert is_safe.shape == (16,)
        assert is_safe.dtype == torch.bool
    
    def test_compute_loss(self):
        """Test quantile loss computation."""
        module = QuantileSafetyModule(
            state_dim=8,
            command_dim=2,
            hidden_dim=64,
            tau=0.9,
        )
        
        state = torch.randn(32, 8)
        target_return = torch.randn(32)
        
        loss_dict = module.compute_loss(state, target_return)
        
        assert "loss" in loss_dict
        assert "loss_upper" in loss_dict
        assert "loss_lower" in loss_dict
        assert not torch.isnan(loss_dict["loss"])


class TestModelIntegration:
    """Integration tests for models."""
    
    def test_policy_with_safety(self):
        """Test policy combined with safety module."""
        policy = UDRLPolicy(
            state_dim=8,
            action_dim=4,
            hidden_dim=64,
            action_type="discrete",
        )
        
        safety = FlowMatchingSafetyModule(
            state_dim=8,
            command_dim=2,
            hidden_dim=64,
            n_integration_steps=10,
        )
        
        state = torch.randn(16, 8)
        command = torch.randn(16, 2)
        
        # Apply safety filter
        filtered_command, info = safety(state, command, return_info=True)
        
        # Get action with filtered command
        action = policy.get_action(state, filtered_command)
        
        assert action.shape == (16,)
    
    def test_gradient_flow_end_to_end(self):
        """Test gradients flow through entire model."""
        policy = UDRLPolicy(
            state_dim=8,
            action_dim=4,
            hidden_dim=64,
            action_type="discrete",
        )
        
        state = torch.randn(32, 8, requires_grad=True)
        command = torch.randn(32, 2, requires_grad=True)
        action = torch.randint(0, 4, (32,))
        
        loss_dict = policy.get_loss(state, command, action)
        loss_dict["loss"].backward()
        
        # Check gradients exist
        for param in policy.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()
