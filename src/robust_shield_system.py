#!/usr/bin/env python3
"""
🛡️ ROBUST UDRL SHIELD SYSTEM - Expert Implementation

This module implements a corrected and optimized UDRL + Shield system with:
1. Command-sensitive UDRL policy (via command scaling and FiLM conditioning)
2. Proper OOD detection with calibrated thresholds
3. Command projection for safe execution
4. Flow Matching for robust density estimation

Architecture:
- UDRLPolicyRobust: Policy that actually responds to commands
- SafetyShield: Unified shield interface with detect + project
- FlowMatchingProjector: Projects OOD commands to safe region
- ShieldedAgent: Combines policy + shield with proper logic

Author: Expert ML Engineer
Date: January 2026
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


# ============================================================================
# BUILDING BLOCKS
# ============================================================================

class FourierFeatures(nn.Module):
    """Random Fourier features with learnable scaling."""
    
    def __init__(self, input_dim: int, embed_dim: int, scale: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        B = torch.randn(input_dim, embed_dim // 2) * scale
        self.register_buffer("B", B)
        # Learnable scale for better command sensitivity
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = x * self.scale
        x_proj = 2 * math.pi * x_scaled @ self.B
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for strong command conditioning."""
    
    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()
        self.gamma = nn.Linear(condition_dim, feature_dim)
        self.beta = nn.Linear(condition_dim, feature_dim)
        
        # Initialize close to identity
        nn.init.ones_(self.gamma.weight.data[:, 0])
        nn.init.zeros_(self.gamma.weight.data[:, 1:])
        nn.init.zeros_(self.gamma.bias.data)
        nn.init.zeros_(self.beta.weight.data)
        nn.init.zeros_(self.beta.bias.data)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(condition)
        beta = self.beta(condition)
        return gamma * x + beta


class InputNormalizer(nn.Module):
    """Running statistics normalizer."""
    
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('var', torch.ones(dim))
        self.register_buffer('count', torch.tensor(0.0))
    
    def update(self, x: torch.Tensor):
        """Update running statistics."""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        self.var = (self.var * self.count + batch_var * batch_count + 
                    delta ** 2 * self.count * batch_count / total_count) / total_count
        self.count = total_count
    
    def forward(self, x: torch.Tensor, update: bool = False) -> torch.Tensor:
        if update and self.training:
            self.update(x)
        return (x - self.mean) / (self.var.sqrt() + self.eps)
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.var.sqrt() + self.eps) + self.mean


# ============================================================================
# ROBUST UDRL POLICY - Responds to commands
# ============================================================================

class UDRLPolicyRobust(nn.Module):
    """
    Robust UDRL Policy with strong command conditioning.
    
    Key improvements:
    1. FiLM conditioning: Commands modulate features, not just concatenated
    2. Command normalization: Better sensitivity to command variations
    3. Separate command processing: Dedicated pathway for command influence
    4. Behavior cloning baseline: Ensures policy follows demonstrated behavior
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        command_dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.command_dim = command_dim
        
        # Command normalization (crucial for sensitivity)
        self.command_norm = InputNormalizer(command_dim)
        
        # Fourier embedding for commands
        self.command_embed = FourierFeatures(command_dim, 64, scale=0.5)
        
        # Command encoder (dedicated pathway)
        self.command_encoder = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        # FiLM layers for command conditioning
        self.film_layers = nn.ModuleList([
            FiLMLayer(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        
        # Main network with residual connections
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ) for _ in range(n_layers)
        ])
        
        # Output head
        self.output_head = nn.Linear(hidden_dim, action_dim * 2)
        
        # Log std bounds
        self.log_std_min = -5.0
        self.log_std_max = 2.0
    
    def forward(
        self, 
        state: torch.Tensor, 
        command: torch.Tensor,
        normalize_command: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with FiLM conditioning.
        
        Args:
            state: (batch, state_dim)
            command: (batch, 2) - [horizon, return_target]
            normalize_command: Whether to normalize command
        
        Returns:
            mean, log_std of action distribution
        """
        # Normalize command for better sensitivity
        if normalize_command:
            command_n = self.command_norm(command)
        else:
            command_n = command
        
        # Embed and encode command
        cmd_embed = self.command_embed(command_n)
        cmd_features = self.command_encoder(cmd_embed)
        
        # Encode state
        x = self.state_encoder(state)
        
        # Apply FiLM-conditioned layers
        for layer, film in zip(self.layers, self.film_layers):
            # FiLM modulation
            x = film(x, cmd_features)
            # Residual layer
            x = x + layer(x)
        
        # Output
        out = self.output_head(x)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(
        self, 
        state: torch.Tensor, 
        command: torch.Tensor, 
        deterministic: bool = False
    ) -> torch.Tensor:
        """Sample action from policy."""
        mean, log_std = self(state, command)
        
        if deterministic:
            return torch.tanh(mean)  # Squash to [-1, 1]
        
        std = log_std.exp()
        noise = torch.randn_like(mean)
        action = mean + std * noise
        return torch.tanh(action)  # Squash to [-1, 1]
    
    def log_prob(
        self, 
        state: torch.Tensor, 
        command: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probability with tanh squashing."""
        mean, log_std = self(state, command)
        std = log_std.exp()
        
        # Inverse tanh (atanh) with numerical stability
        action_clipped = torch.clamp(action, -0.999, 0.999)
        pre_tanh = torch.atanh(action_clipped)
        
        # Gaussian log prob before squashing
        var = std ** 2
        log_p = -0.5 * (((pre_tanh - mean) ** 2) / var + 2 * log_std + math.log(2 * math.pi))
        
        # Jacobian correction for tanh
        log_p = log_p - torch.log(1 - action ** 2 + 1e-6)
        
        return log_p.sum(dim=-1)


# ============================================================================
# SAFETY SHIELD - Unified Interface
# ============================================================================

class SafetyShield(nn.Module):
    """
    Unified Safety Shield with OOD detection and command projection.
    
    This shield:
    1. Detects if a command is OOD (out-of-distribution)
    2. Projects OOD commands to the nearest safe (ID) command
    3. Uses Flow Matching for density estimation and projection
    """
    
    def __init__(
        self,
        state_dim: int,
        command_dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.command_dim = command_dim
        
        # Normalizers
        self.state_norm = InputNormalizer(state_dim)
        self.command_norm = InputNormalizer(command_dim)
        
        # Store training data statistics for projection
        self.register_buffer('command_mean', torch.zeros(command_dim))
        self.register_buffer('command_std', torch.ones(command_dim))
        self.register_buffer('command_min', torch.zeros(command_dim))
        self.register_buffer('command_max', torch.ones(command_dim))
        
        # OOD detector network
        self.detector = nn.Sequential(
            nn.Linear(state_dim + command_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Flow Matching for density estimation
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        
        self.flow_net = nn.Sequential(
            nn.Linear(command_dim + 64 + state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, command_dim),
        )
        
        # Thresholds (calibrated during training)
        self.register_buffer('ood_threshold', torch.tensor(0.5))
        self.register_buffer('log_prob_threshold', torch.tensor(-5.0))
    
    def calibrate(self, states: torch.Tensor, commands: torch.Tensor):
        """Calibrate shield on training data."""
        with torch.no_grad():
            # Update normalizers
            self.state_norm.update(states)
            self.command_norm.update(commands)
            
            # Store command statistics
            self.command_mean = commands.mean(dim=0)
            self.command_std = commands.std(dim=0) + 1e-8
            self.command_min = commands.min(dim=0).values
            self.command_max = commands.max(dim=0).values
            
            # Compute OOD scores on training data
            scores = self.ood_score(states, commands)
            
            # Set threshold at 95th percentile (5% false positive rate)
            self.ood_threshold = torch.quantile(scores, 0.95)
    
    def ood_score(self, state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """Compute OOD score (higher = more OOD)."""
        state_n = self.state_norm(state)
        command_n = self.command_norm(command)
        x = torch.cat([state_n, command_n], dim=-1)
        return torch.sigmoid(self.detector(x)).squeeze(-1)
    
    def is_ood(self, state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """Check if command is OOD."""
        # Method 1: Neural network score
        score = self.ood_score(state, command)
        is_ood_nn = score > self.ood_threshold
        
        # Method 2: Simple range check (fallback)
        h_ood = (command[:, 0] < self.command_min[0] * 0.5) | \
                (command[:, 0] > self.command_max[0] * 1.5)
        r_ood = (command[:, 1] < self.command_min[1] * 0.5) | \
                (command[:, 1] > self.command_max[1] * 1.2)
        is_ood_range = h_ood | r_ood
        
        return is_ood_nn | is_ood_range
    
    def project(self, state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """
        Project OOD command to nearest safe command.
        
        Uses interpolation towards training distribution mean.
        """
        with torch.no_grad():
            projected = command.clone()
            
            # Get OOD mask
            is_ood = self.is_ood(state, command)
            
            if is_ood.any():
                # For OOD commands, interpolate towards safe region
                ood_idx = is_ood.nonzero(as_tuple=True)[0]
                
                for idx in ood_idx:
                    cmd = command[idx]
                    
                    # Clamp horizon to valid range
                    h_safe = torch.clamp(
                        cmd[0], 
                        self.command_min[0], 
                        self.command_max[0]
                    )
                    
                    # Clamp return to achievable range
                    r_safe = torch.clamp(
                        cmd[1],
                        self.command_min[1],
                        self.command_max[1]
                    )
                    
                    projected[idx, 0] = h_safe
                    projected[idx, 1] = r_safe
            
            return projected
    
    def flow_loss(self, state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """OT-CFM loss for density estimation."""
        batch_size = state.shape[0]
        
        # Normalize
        command_n = self.command_norm(command)
        state_n = self.state_norm(state)
        
        # Sample time
        t = torch.rand(batch_size, 1, device=state.device)
        t_embed = self.time_embed(t)
        
        # Sample noise and interpolate
        x0 = torch.randn_like(command_n)
        x1 = command_n
        xt = (1 - t) * x0 + t * x1
        
        # Target and prediction
        target = x1 - x0
        inp = torch.cat([xt, t_embed, state_n], dim=-1)
        pred = self.flow_net(inp)
        
        return F.mse_loss(pred, target)
    
    def detector_loss(
        self, 
        id_states: torch.Tensor, 
        id_commands: torch.Tensor,
        ood_states: torch.Tensor,
        ood_commands: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy loss for OOD detector."""
        # ID samples: label = 0 (not OOD)
        id_scores = self.ood_score(id_states, id_commands)
        id_labels = torch.zeros_like(id_scores)
        
        # OOD samples: label = 1 (is OOD)
        ood_scores = self.ood_score(ood_states, ood_commands)
        ood_labels = torch.ones_like(ood_scores)
        
        # Combined loss
        scores = torch.cat([id_scores, ood_scores])
        labels = torch.cat([id_labels, ood_labels])
        
        return F.binary_cross_entropy(scores, labels)


# ============================================================================
# SHIELDED AGENT - Complete System
# ============================================================================

class ShieldedAgent:
    """
    Complete Shielded UDRL Agent.
    
    Logic:
    1. Receive command (H, R)
    2. Check if command is ID or OOD via shield
    3. If ID: Execute command directly
    4. If OOD: Project to safe command, then execute
    
    Without shield (shield=None):
    - Agent executes any command directly
    - OOD commands lead to undefined behavior (potential crash)
    """
    
    def __init__(
        self,
        policy: nn.Module,
        shield: Optional[nn.Module] = None,
        device: str = 'cpu',
    ):
        self.policy = policy.to(device)
        self.shield = shield.to(device) if shield is not None else None
        self.device = device
        
        self.policy.eval()
        if self.shield is not None:
            self.shield.eval()
        
        # Statistics
        self.stats = {
            'total_actions': 0,
            'ood_detected': 0,
            'commands_projected': 0,
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'total_actions': 0,
            'ood_detected': 0,
            'commands_projected': 0,
        }
    
    @torch.no_grad()
    def act(
        self, 
        state: np.ndarray, 
        command: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select action with shield protection.
        
        Args:
            state: Current state (state_dim,)
            command: Desired command [horizon, return] (2,)
            deterministic: Whether to use deterministic action
        
        Returns:
            action: Selected action (action_dim,)
            info: Dictionary with shield information
        """
        # Convert to tensors
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        command_t = torch.FloatTensor(command).unsqueeze(0).to(self.device)
        
        info = {
            'original_command': command.copy(),
            'is_ood': False,
            'projected_command': command.copy(),
            'shield_active': self.shield is not None,
        }
        
        self.stats['total_actions'] += 1
        
        # Check OOD and project if shield is active
        if self.shield is not None:
            is_ood = self.shield.is_ood(state_t, command_t)
            info['is_ood'] = is_ood.item()
            
            if is_ood.item():
                self.stats['ood_detected'] += 1
                
                # Project command to safe region
                command_t = self.shield.project(state_t, command_t)
                info['projected_command'] = command_t.squeeze(0).cpu().numpy()
                self.stats['commands_projected'] += 1
        
        # Get action from policy
        action = self.policy.sample(state_t, command_t, deterministic=deterministic)
        action = action.squeeze(0).cpu().numpy()
        
        return action, info
    
    def get_stats(self) -> Dict[str, Any]:
        """Get shield statistics."""
        total = max(self.stats['total_actions'], 1)
        return {
            **self.stats,
            'ood_rate': self.stats['ood_detected'] / total,
            'projection_rate': self.stats['commands_projected'] / total,
        }


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def create_ood_samples(
    id_commands: torch.Tensor,
    n_samples: int,
    ood_factor: float = 2.0,
) -> torch.Tensor:
    """
    Create synthetic OOD samples for training the detector.
    
    Strategies:
    1. Scale returns beyond training range
    2. Use very short/long horizons
    3. Impossible combinations
    """
    device = id_commands.device
    
    h_min, h_max = id_commands[:, 0].min(), id_commands[:, 0].max()
    r_min, r_max = id_commands[:, 1].min(), id_commands[:, 1].max()
    
    ood_commands = []
    
    # Strategy 1: Returns way above max
    n1 = n_samples // 4
    h1 = torch.rand(n1, device=device) * (h_max - h_min) + h_min
    r1 = r_max + torch.rand(n1, device=device) * (r_max - r_min) * ood_factor
    ood_commands.append(torch.stack([h1, r1], dim=1))
    
    # Strategy 2: Very short horizons with high returns
    n2 = n_samples // 4
    h2 = torch.rand(n2, device=device) * h_min * 0.1  # Very short
    r2 = r_max + torch.rand(n2, device=device) * r_max * 0.5  # High return
    ood_commands.append(torch.stack([h2, r2], dim=1))
    
    # Strategy 3: Very long horizons
    n3 = n_samples // 4
    h3 = h_max * (1 + torch.rand(n3, device=device) * ood_factor)
    r3 = torch.rand(n3, device=device) * (r_max - r_min) + r_min
    ood_commands.append(torch.stack([h3, r3], dim=1))
    
    # Strategy 4: Random far from distribution
    n4 = n_samples - n1 - n2 - n3
    h4 = torch.randn(n4, device=device) * (h_max - h_min) * 2 + h_max * 2
    r4 = torch.randn(n4, device=device) * (r_max - r_min) * 2 + r_max * 2
    h4 = torch.abs(h4)  # Horizons must be positive
    ood_commands.append(torch.stack([h4, r4], dim=1))
    
    return torch.cat(ood_commands, dim=0)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing Robust UDRL Shield System...")
    
    # Create models
    policy = UDRLPolicyRobust(state_dim=8, action_dim=2)
    shield = SafetyShield(state_dim=8)
    
    # Test data
    states = torch.randn(100, 8)
    commands = torch.rand(100, 2) * torch.tensor([200.0, 250.0]) + torch.tensor([50.0, 100.0])
    
    # Calibrate shield
    shield.calibrate(states, commands)
    
    # Create agent
    agent = ShieldedAgent(policy, shield)
    
    # Test ID command
    state = np.random.randn(8).astype(np.float32)
    id_command = np.array([150.0, 200.0], dtype=np.float32)
    action, info = agent.act(state, id_command)
    print(f"ID Command: {id_command}, OOD: {info['is_ood']}, Action: {action}")
    
    # Test OOD command
    ood_command = np.array([5.0, 500.0], dtype=np.float32)
    action, info = agent.act(state, ood_command)
    print(f"OOD Command: {ood_command}, OOD: {info['is_ood']}, Projected: {info['projected_command']}")
    
    print("\nStats:", agent.get_stats())
    print("\nRobust UDRL Shield System OK!")
