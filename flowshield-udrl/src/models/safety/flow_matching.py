"""
Flow Matching Safety Module.

The main innovation of FlowShield-UDRL: using Flow Matching to model
the distribution p(g|s) of achievable commands from each state.

Flow Matching enables:
1. Exact density estimation via ODE integration
2. Efficient sampling of achievable commands
3. Gradient-based projection to the data manifold
"""

import math
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.embeddings import SinusoidalTimeEmbedding
from ..components.mlp import MLP, ResidualMLP
from .base import BaseSafetyModule


class VectorFieldNetwork(nn.Module):
    """
    Neural network for the flow velocity field v_θ(g_t, s, t).
    
    Takes a noisy command, conditioning state, and time, and predicts
    the velocity that transports from noise to the data distribution.
    
    Args:
        command_dim: Dimension of command (2: horizon, return)
        state_dim: Dimension of state for conditioning
        hidden_dim: Hidden layer dimension
        n_layers: Number of hidden layers
        time_embed_dim: Dimension of time embedding
        
    Example:
        >>> net = VectorFieldNetwork(command_dim=2, state_dim=8)
        >>> g_t = torch.randn(32, 2)
        >>> state = torch.randn(32, 8)
        >>> t = torch.rand(32)
        >>> velocity = net(g_t, state, t)  # (32, 2)
    """
    
    def __init__(
        self,
        command_dim: int = 2,
        state_dim: int = 8,
        hidden_dim: int = 256,
        n_layers: int = 4,
        time_embed_dim: int = 64,
        activation: str = "gelu",
        use_residual: bool = True,
    ):
        super().__init__()
        
        self.command_dim = command_dim
        self.state_dim = state_dim
        
        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        
        # Input dimension: command + state + time_embed
        input_dim = command_dim + state_dim + time_embed_dim
        
        # Main network
        if use_residual:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                ResidualMLP(
                    input_dim=hidden_dim,
                    output_dim=command_dim,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers - 1,
                    activation=activation,
                ),
            )
        else:
            self.net = MLP(
                input_dim=input_dim,
                output_dim=command_dim,
                hidden_dims=hidden_dim,
                n_layers=n_layers,
                activation=activation,
            )
    
    def forward(
        self,
        g_t: torch.Tensor,
        state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity field.
        
        Args:
            g_t: Noisy command at time t, shape (batch, command_dim)
            state: Conditioning state, shape (batch, state_dim)
            t: Time in [0, 1], shape (batch,) or (batch, 1)
            
        Returns:
            Velocity vector, shape (batch, command_dim)
        """
        # Time embedding
        if t.dim() == 2:
            t = t.squeeze(-1)
        t_emb = self.time_embed(t)
        
        # Concatenate inputs
        x = torch.cat([g_t, state, t_emb], dim=-1)
        
        # Predict velocity
        return self.net(x)


class FlowMatchingModel(nn.Module):
    """
    Flow Matching model for conditional command generation.
    
    Learns to transport samples from a prior distribution (Gaussian)
    to the data distribution p(g|s) of achievable commands.
    
    Uses the conditional flow matching objective:
    L = E_{t, (s,g)~data, noise~N(0,I)} [||v_θ(g_t, s, t) - u_t(g|g_0)||²]
    
    where g_t = (1-t)·noise + t·g is the interpolated sample
    and u_t = g - noise is the target velocity.
    
    Args:
        command_dim: Dimension of command
        state_dim: Dimension of conditioning state
        hidden_dim: Hidden layer dimension
        n_layers: Number of layers
        sigma_min: Minimum noise level (for numerical stability)
        
    Example:
        >>> model = FlowMatchingModel(command_dim=2, state_dim=8)
        >>> # Training
        >>> loss = model.compute_loss(states, commands)
        >>> # Sampling
        >>> samples = model.sample(states, n_samples=10)
        >>> # Density estimation
        >>> log_prob = model.log_prob(states, commands)
    """
    
    def __init__(
        self,
        command_dim: int = 2,
        state_dim: int = 8,
        hidden_dim: int = 256,
        n_layers: int = 4,
        time_embed_dim: int = 64,
        sigma_min: float = 0.001,
        solver: str = "euler",
        n_steps: int = 100,
    ):
        super().__init__()
        
        self.command_dim = command_dim
        self.state_dim = state_dim
        self.sigma_min = sigma_min
        self.solver = solver
        self.n_steps = n_steps
        
        # Vector field network
        self.vector_field = VectorFieldNetwork(
            command_dim=command_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            time_embed_dim=time_embed_dim,
        )
    
    def compute_loss(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute flow matching loss.
        
        Uses the conditional flow matching objective with optimal transport.
        
        Args:
            state: State tensor (batch, state_dim)
            command: Command tensor (batch, command_dim)
            noise: Optional noise tensor (batch, command_dim)
            
        Returns:
            Dictionary with loss and diagnostics
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Sample noise
        if noise is None:
            noise = torch.randn_like(command)
        
        # Sample time uniformly
        t = torch.rand(batch_size, device=device)
        
        # Interpolate: g_t = (1-t) * noise + t * command
        # Plus small noise for stability
        t_expanded = t.unsqueeze(-1)
        sigma_t = self.sigma_min + (1 - self.sigma_min) * (1 - t_expanded)
        g_t = (1 - t_expanded) * noise + t_expanded * command
        g_t = g_t + sigma_t * torch.randn_like(g_t) * 0.01  # Small perturbation
        
        # Target velocity: u_t = command - noise
        target_velocity = command - noise
        
        # Predicted velocity
        pred_velocity = self.vector_field(g_t, state, t)
        
        # MSE loss
        loss = F.mse_loss(pred_velocity, target_velocity)
        
        # Diagnostics
        velocity_norm = pred_velocity.norm(dim=-1).mean()
        target_norm = target_velocity.norm(dim=-1).mean()
        
        return {
            "loss": loss,
            "velocity_norm": velocity_norm,
            "target_norm": target_norm,
        }
    
    @torch.no_grad()
    def sample(
        self,
        state: torch.Tensor,
        n_samples: int = 1,
        n_steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample commands by integrating the flow ODE.
        
        Integrates from t=0 (noise) to t=1 (data).
        
        Args:
            state: Conditioning state (batch, state_dim)
            n_samples: Number of samples per state
            n_steps: Number of integration steps
            return_trajectory: Whether to return full trajectory
            
        Returns:
            Samples (batch, n_samples, command_dim)
            If return_trajectory: also returns trajectory (batch, n_samples, n_steps, command_dim)
        """
        n_steps = n_steps or self.n_steps
        batch_size = state.shape[0]
        device = state.device
        
        # Expand state for multiple samples
        state_expanded = state.unsqueeze(1).expand(-1, n_samples, -1)
        state_flat = state_expanded.reshape(-1, self.state_dim)
        
        # Initial noise
        g = torch.randn(batch_size * n_samples, self.command_dim, device=device)
        
        # Integration
        dt = 1.0 / n_steps
        trajectory = [g.reshape(batch_size, n_samples, -1)] if return_trajectory else None
        
        for step in range(n_steps):
            t = torch.full((batch_size * n_samples,), step * dt, device=device)
            
            # Velocity
            v = self.vector_field(g, state_flat, t)
            
            # Euler step
            g = g + v * dt
            
            if return_trajectory:
                trajectory.append(g.reshape(batch_size, n_samples, -1))
        
        # Reshape output
        samples = g.reshape(batch_size, n_samples, self.command_dim)
        
        if return_trajectory:
            trajectory = torch.stack(trajectory, dim=2)  # (batch, n_samples, n_steps+1, dim)
            return samples, trajectory
        
        return samples
    
    def log_prob(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
        n_steps: Optional[int] = None,
        hutchinson_samples: int = 10,
    ) -> torch.Tensor:
        """
        Estimate log probability via continuous normalizing flow.
        
        Integrates the instantaneous change of variables formula:
        log p(g) = log p(g_0) - ∫₀¹ div(v_θ(g_t, s, t)) dt
        
        Uses Hutchinson trace estimator for divergence.
        
        Args:
            state: State tensor (batch, state_dim)
            command: Command tensor (batch, command_dim)
            n_steps: Number of integration steps
            hutchinson_samples: Samples for trace estimation
            
        Returns:
            Log probability (batch,)
        """
        n_steps = n_steps or self.n_steps
        batch_size = state.shape[0]
        device = state.device
        
        # Integrate backwards from data to noise
        g = command.clone()
        log_det = torch.zeros(batch_size, device=device)
        
        dt = -1.0 / n_steps  # Negative for backward integration
        
        for step in range(n_steps):
            t = torch.full((batch_size,), 1 - step / n_steps, device=device)
            
            # Compute velocity and divergence using Hutchinson estimator
            v, div = self._compute_velocity_and_divergence(
                g, state, t, hutchinson_samples
            )
            
            # Update
            g = g + v * dt
            log_det = log_det + div * (-dt)  # Accumulate log det
        
        # Base distribution log prob (standard normal)
        log_prior = -0.5 * (
            self.command_dim * math.log(2 * math.pi) +
            (g ** 2).sum(dim=-1)
        )
        
        # Total log prob
        log_p = log_prior + log_det
        
        return log_p
    
    def _compute_velocity_and_divergence(
        self,
        g: torch.Tensor,
        state: torch.Tensor,
        t: torch.Tensor,
        hutchinson_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute velocity and its divergence using Hutchinson estimator.
        
        div(v) ≈ E_ε[ε^T · ∂v/∂g · ε]
        """
        g = g.requires_grad_(True)
        v = self.vector_field(g, state, t)
        
        # Hutchinson trace estimator
        div = torch.zeros(g.shape[0], device=g.device)
        
        for _ in range(hutchinson_samples):
            # Random vector
            eps = torch.randn_like(g)
            
            # Vector-Jacobian product
            vjp = torch.autograd.grad(
                outputs=v,
                inputs=g,
                grad_outputs=eps,
                create_graph=False,
                retain_graph=True,
            )[0]
            
            # Trace estimate
            div = div + (eps * vjp).sum(dim=-1)
        
        div = div / hutchinson_samples
        
        return v.detach(), div.detach()


class FlowMatchingSafetyModule(BaseSafetyModule):
    """
    Flow Matching-based safety module.
    
    Uses Flow Matching to model the distribution p(g|s) of achievable
    commands from each state. This enables:
    
    1. OOD Detection: Commands with low log p(g|s) are flagged as unsafe
    2. Projection: Gradient-based projection to high-density regions
    3. Sampling: Generate achievable commands for any state
    
    Args:
        state_dim: State observation dimension
        command_dim: Command dimension (default 2)
        hidden_dim: Hidden layer dimension
        n_layers: Number of layers in vector field
        ood_threshold: Log-probability threshold for OOD detection
        n_integration_steps: Steps for ODE integration
        projection_method: "gradient" or "bisection"
        projection_steps: Maximum projection steps
        
    Example:
        >>> safety = FlowMatchingSafetyModule(state_dim=8)
        >>> state = torch.randn(1, 8)
        >>> command = torch.tensor([[50, 100.0]])  # Ambitious command
        >>> safe_command, info = safety(state, command)
        >>> print(info["is_safe"], info["projection_distance"])
    """
    
    def __init__(
        self,
        state_dim: int,
        command_dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 4,
        time_embed_dim: int = 64,
        sigma_min: float = 0.001,
        ood_threshold: float = -5.0,
        n_integration_steps: int = 100,
        projection_method: str = "gradient",
        projection_steps: int = 50,
        projection_lr: float = 0.1,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.command_dim = command_dim
        self.ood_threshold = ood_threshold
        self.n_integration_steps = n_integration_steps
        self.projection_method = projection_method
        self.projection_steps = projection_steps
        self.projection_lr = projection_lr
        
        # Flow Matching model
        self.flow_model = FlowMatchingModel(
            command_dim=command_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            time_embed_dim=time_embed_dim,
            sigma_min=sigma_min,
            n_steps=n_integration_steps,
        )
        
        # Statistics for normalization
        self.register_buffer("command_mean", torch.zeros(command_dim))
        self.register_buffer("command_std", torch.ones(command_dim))
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
    
    def is_safe(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check if commands are safe based on log probability.
        
        Commands with log p(g|s) > threshold are considered safe.
        """
        log_prob = self.get_safety_score(state, command)
        return log_prob > self.ood_threshold
    
    def get_safety_score(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute safety score as log probability.
        """
        return self.flow_model.log_prob(
            state, command,
            n_steps=self.n_integration_steps,
        )
    
    def project(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project commands to safe region.
        
        Uses gradient ascent on log p(g|s) to move towards
        high-density regions.
        """
        is_safe = self.is_safe(state, command)
        
        # Don't project already-safe commands
        if is_safe.all():
            return command
        
        if self.projection_method == "gradient":
            return self._gradient_projection(state, command, is_safe)
        elif self.projection_method == "sample":
            return self._sample_projection(state, command, is_safe)
        else:
            raise ValueError(f"Unknown projection method: {self.projection_method}")
    
    def _gradient_projection(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
        is_safe: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project using gradient ascent on log probability.
        """
        projected = command.clone()
        
        # Only project unsafe commands
        unsafe_mask = ~is_safe
        if not unsafe_mask.any():
            return projected
        
        unsafe_idx = unsafe_mask.nonzero(as_tuple=True)[0]
        g = command[unsafe_idx].clone().requires_grad_(True)
        s = state[unsafe_idx]
        
        for step in range(self.projection_steps):
            # Compute log prob
            log_prob = self.flow_model.log_prob(s, g)
            
            # Check if now safe
            newly_safe = log_prob > self.ood_threshold
            if newly_safe.all():
                break
            
            # Gradient ascent
            grad = torch.autograd.grad(log_prob.sum(), g)[0]
            
            # Update
            with torch.no_grad():
                g = g + self.projection_lr * grad
                g = g.requires_grad_(True)
        
        # Update projected commands
        projected[unsafe_idx] = g.detach()
        
        return projected
    
    def _sample_projection(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
        is_safe: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project by sampling nearby achievable commands.
        """
        projected = command.clone()
        
        unsafe_mask = ~is_safe
        if not unsafe_mask.any():
            return projected
        
        unsafe_idx = unsafe_mask.nonzero(as_tuple=True)[0]
        s = state[unsafe_idx]
        
        # Sample multiple commands and choose closest
        samples = self.sample_commands(s, n_samples=100)  # (n_unsafe, 100, dim)
        
        # Find closest sample to original command
        original = command[unsafe_idx].unsqueeze(1)  # (n_unsafe, 1, dim)
        distances = torch.norm(samples - original, dim=-1)  # (n_unsafe, 100)
        closest_idx = distances.argmin(dim=-1)  # (n_unsafe,)
        
        # Gather closest samples
        batch_idx = torch.arange(len(unsafe_idx), device=command.device)
        projected[unsafe_idx] = samples[batch_idx, closest_idx]
        
        return projected
    
    def sample_commands(
        self,
        state: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """
        Sample achievable commands from the learned distribution.
        """
        return self.flow_model.sample(state, n_samples, n_steps=self.n_integration_steps)
    
    def compute_loss(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss for the flow model.
        """
        return self.flow_model.compute_loss(state, command)
    
    def set_statistics(
        self,
        command_mean: torch.Tensor,
        command_std: torch.Tensor,
        state_mean: Optional[torch.Tensor] = None,
        state_std: Optional[torch.Tensor] = None,
    ) -> None:
        """Set normalization statistics."""
        self.command_mean = command_mean
        self.command_std = command_std
        if state_mean is not None:
            self.state_mean = state_mean
        if state_std is not None:
            self.state_std = state_std
