"""
Quantile Regression Safety Module.

Baseline method that learns quantile bounds on achievable returns.
Used for comparison with Flow Matching approach.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.mlp import MLP
from .base import BaseSafetyModule


class QuantileNetwork(nn.Module):
    """
    Neural network for quantile regression.
    
    Predicts quantiles of the return distribution conditioned on state.
    
    Args:
        state_dim: State dimension
        hidden_dim: Hidden layer dimension
        n_layers: Number of hidden layers
        n_quantiles: Number of quantiles to predict
        
    Example:
        >>> net = QuantileNetwork(8, hidden_dim=256, n_quantiles=5)
        >>> state = torch.randn(32, 8)
        >>> quantiles = net(state)  # (32, 5)
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        n_quantiles: int = 1,
        activation: str = "relu",
    ):
        super().__init__()
        
        self.n_quantiles = n_quantiles
        
        self.net = MLP(
            input_dim=state_dim,
            output_dim=n_quantiles,
            hidden_dims=hidden_dim,
            n_layers=n_layers,
            activation=activation,
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict quantiles.
        
        Args:
            state: State tensor (batch, state_dim)
            
        Returns:
            Quantile predictions (batch, n_quantiles)
        """
        return self.net(state)


class QuantileSafetyModule(BaseSafetyModule):
    """
    Quantile Regression-based safety module.
    
    Learns the τ-quantile of achievable returns from each state.
    Commands with returns above this quantile are considered unsafe
    and clamped to the safe region.
    
    This is a baseline method - simpler than Flow Matching but less
    expressive (only models marginal return distribution, not full
    joint distribution of horizon and return).
    
    Args:
        state_dim: State observation dimension
        command_dim: Command dimension (default 2: horizon, return)
        hidden_dim: Hidden layer dimension
        n_layers: Number of hidden layers
        tau: Quantile level (e.g., 0.9 for 90th percentile)
        n_quantiles: Number of quantiles to learn
        horizon_max: Maximum horizon for clamping
        
    Example:
        >>> safety = QuantileSafetyModule(state_dim=8, tau=0.9)
        >>> state = torch.randn(32, 8)
        >>> command = torch.tensor([[20, 100.0]])  # Possibly too optimistic
        >>> safe_command, info = safety(state, command)
    """
    
    def __init__(
        self,
        state_dim: int,
        command_dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 3,
        tau: float = 0.9,
        n_quantiles: int = 1,
        horizon_max: float = 100.0,
        activation: str = "relu",
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.command_dim = command_dim
        self.tau = tau
        self.n_quantiles = n_quantiles
        self.horizon_max = horizon_max
        
        # Quantile network for return upper bound
        self.return_quantile_net = QuantileNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_quantiles=n_quantiles,
            activation=activation,
        )
        
        # Optionally, learn return lower bound too
        self.return_lower_net = QuantileNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_quantiles=n_quantiles,
            activation=activation,
        )
        
        # For horizon bounds (simpler - often just clip to data range)
        self.horizon_bounds = nn.Parameter(
            torch.tensor([1.0, horizon_max]),
            requires_grad=False,
        )
        
        # Learned statistics for sampling
        self.register_buffer("return_mean", torch.tensor(0.0))
        self.register_buffer("return_std", torch.tensor(1.0))
    
    def get_return_bounds(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get return bounds for given states.
        
        Args:
            state: State tensor (batch, state_dim)
            
        Returns:
            Tuple of (lower_bound, upper_bound) each (batch,) or (batch, n_quantiles)
        """
        upper = self.return_quantile_net(state)
        lower = self.return_lower_net(state)
        
        # If single quantile, squeeze
        if self.n_quantiles == 1:
            upper = upper.squeeze(-1)
            lower = lower.squeeze(-1)
        
        return lower, upper
    
    def is_safe(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check if commands are within safe bounds.
        
        A command is safe if:
        1. Horizon is within [1, horizon_max]
        2. Return is within [lower_quantile, upper_quantile]
        """
        horizon = command[:, 0]
        target_return = command[:, 1]
        
        # Get bounds
        lower, upper = self.get_return_bounds(state)
        if lower.dim() > 1:
            lower = lower[:, 0]  # Use first quantile
            upper = upper[:, -1]  # Use last quantile
        
        # Check safety
        horizon_safe = (horizon >= self.horizon_bounds[0]) & (horizon <= self.horizon_bounds[1])
        return_safe = (target_return >= lower) & (target_return <= upper)
        
        return horizon_safe & return_safe
    
    def project(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project commands to safe region by clamping.
        """
        horizon = command[:, 0]
        target_return = command[:, 1]
        
        # Get bounds
        lower, upper = self.get_return_bounds(state)
        if lower.dim() > 1:
            lower = lower[:, 0]
            upper = upper[:, -1]
        
        # Clamp horizon
        safe_horizon = torch.clamp(horizon, self.horizon_bounds[0], self.horizon_bounds[1])
        
        # Clamp return
        safe_return = torch.clamp(target_return, lower, upper)
        
        return torch.stack([safe_horizon, safe_return], dim=-1)
    
    def get_safety_score(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute safety score based on distance from bounds.
        
        Score is higher when command is well within bounds.
        """
        target_return = command[:, 1]
        
        lower, upper = self.get_return_bounds(state)
        if lower.dim() > 1:
            lower = lower[:, 0]
            upper = upper[:, -1]
        
        # Compute normalized position within bounds
        range_size = upper - lower + 1e-6
        normalized_pos = (target_return - lower) / range_size
        
        # Score: higher near middle of distribution
        # Using negative distance from center
        center = 0.5
        score = -torch.abs(normalized_pos - center)
        
        # Penalize being outside bounds
        outside_lower = (target_return < lower).float()
        outside_upper = (target_return > upper).float()
        distance_outside = (
            outside_lower * (lower - target_return) +
            outside_upper * (target_return - upper)
        )
        
        score = score - distance_outside
        
        return score
    
    def sample_commands(
        self,
        state: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """
        Sample commands by sampling within bounds.
        """
        batch_size = state.shape[0]
        
        lower, upper = self.get_return_bounds(state)
        if lower.dim() > 1:
            lower = lower[:, 0]
            upper = upper[:, -1]
        
        # Sample returns uniformly within bounds
        u = torch.rand(batch_size, n_samples, device=state.device)
        returns = lower.unsqueeze(1) + u * (upper - lower).unsqueeze(1)
        
        # Sample horizons uniformly
        horizons = (
            self.horizon_bounds[0] +
            torch.rand(batch_size, n_samples, device=state.device) *
            (self.horizon_bounds[1] - self.horizon_bounds[0])
        )
        
        return torch.stack([horizons, returns], dim=-1)
    
    def compute_loss(
        self,
        state: torch.Tensor,
        target_return: torch.Tensor,
        tau: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute quantile regression loss.
        
        Uses the pinball loss (asymmetric L1 loss) for quantile regression.
        
        Args:
            state: State tensor (batch, state_dim)
            target_return: Actual returns achieved (batch,)
            tau: Override quantile level
            
        Returns:
            Dictionary with loss values
        """
        tau = tau or self.tau
        
        # Predict quantiles
        pred_upper = self.return_quantile_net(state)
        pred_lower = self.return_lower_net(state)
        
        if pred_upper.dim() > 1:
            pred_upper = pred_upper.squeeze(-1)
            pred_lower = pred_lower.squeeze(-1)
        
        # Pinball loss for upper quantile (τ)
        error_upper = target_return - pred_upper
        loss_upper = torch.where(
            error_upper > 0,
            tau * error_upper,
            (tau - 1) * error_upper,
        ).mean()
        
        # Pinball loss for lower quantile (1 - τ)
        tau_lower = 1 - tau
        error_lower = target_return - pred_lower
        loss_lower = torch.where(
            error_lower > 0,
            tau_lower * error_lower,
            (tau_lower - 1) * error_lower,
        ).mean()
        
        total_loss = loss_upper + loss_lower
        
        return {
            "loss": total_loss,
            "loss_upper": loss_upper,
            "loss_lower": loss_lower,
            "pred_upper_mean": pred_upper.mean(),
            "pred_lower_mean": pred_lower.mean(),
        }
    
    def update_statistics(
        self,
        returns: torch.Tensor,
        horizons: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update running statistics for sampling.
        
        Args:
            returns: Return values from dataset
            horizons: Horizon values from dataset
        """
        self.return_mean = returns.mean()
        self.return_std = returns.std() + 1e-6
        
        if horizons is not None:
            self.horizon_bounds[0] = horizons.min()
            self.horizon_bounds[1] = horizons.max()
