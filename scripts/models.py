"""
Shared Models and Utilities for FlowShield-UDRL.

This module contains all the neural network architectures used across scripts.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from pathlib import Path


# ============================================================================
# Embedding Layers
# ============================================================================

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion/flow models."""
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * 
            torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.unsqueeze(-1) * freqs
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class FourierFeatures(nn.Module):
    """Random Fourier features for command embedding."""
    
    def __init__(self, input_dim: int, embed_dim: int, scale: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        B = torch.randn(input_dim, embed_dim // 2) * scale
        self.register_buffer("B", B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


# ============================================================================
# MLP Backbone
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with pre-activation."""
    
    def __init__(self, dim: int, activation: nn.Module = None, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.activation = activation if activation else nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.activation(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x + residual


class MLP(nn.Module):
    """Multi-layer perceptron with residual connections."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 4,
        activation: nn.Module = None,
        dropout: float = 0.0,
        use_residual: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual and (n_layers > 2)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.activation = activation if activation else nn.SiLU()
        
        # Residual blocks or simple layers
        if self.use_residual:
            self.layers = nn.ModuleList([
                ResidualBlock(hidden_dim, self.activation, dropout)
                for _ in range(n_layers - 2)
            ])
        else:
            layers = []
            for _ in range(n_layers - 2):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation])
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            self.layers = nn.Sequential(*layers) if layers else nn.Identity()
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.activation(x)
        
        if self.use_residual:
            for layer in self.layers:
                x = layer(x)
        else:
            x = self.layers(x)
        
        x = self.output_norm(x)
        return self.output_proj(x)


# ============================================================================
# Input Normalization
# ============================================================================

class InputNormalizer(nn.Module):
    """Online input normalization with running statistics."""
    
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
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input."""
        return (x - self.mean) / (torch.sqrt(self.var) + self.eps)
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize to original scale."""
        return x * (torch.sqrt(self.var) + self.eps) + self.mean
    
    def forward(self, x: torch.Tensor, update: bool = False) -> torch.Tensor:
        if update and self.training:
            self.update(x)
        return self.normalize(x)


# ============================================================================
# UDRL Policy
# ============================================================================

class UDRLPolicy(nn.Module):
    """
    Upside-Down RL Policy: π(a|s, g)
    
    Takes state and command (horizon, return) and outputs action distribution.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        command_dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 4,
        command_embed_dim: int = 64,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.command_dim = command_dim
        
        # Fourier features for command embedding
        self.command_embed = FourierFeatures(command_dim, command_embed_dim)
        
        # Policy network
        input_dim = state_dim + command_embed_dim
        self.net = MLP(input_dim, hidden_dim, action_dim * 2, n_layers)
    
    def forward(self, state: torch.Tensor, command: torch.Tensor):
        """Return mean and log_std of action distribution."""
        cmd_embed = self.command_embed(command)
        x = torch.cat([state, cmd_embed], dim=-1)
        out = self.net(x)
        
        mean, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -5, 2)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor, command: torch.Tensor, deterministic: bool = False):
        """Sample action from policy."""
        mean, log_std = self(state, command)
        
        if deterministic:
            return mean
        
        std = log_std.exp()
        eps = torch.randn_like(mean)
        return mean + std * eps
    
    def log_prob(self, state: torch.Tensor, command: torch.Tensor, action: torch.Tensor):
        """Compute log probability of action."""
        mean, log_std = self(state, command)
        std = log_std.exp()
        
        # Gaussian log probability
        var = std ** 2
        log_p = -0.5 * (((action - mean) ** 2) / var + 2 * log_std + math.log(2 * math.pi))
        return log_p.sum(dim=-1)


# ============================================================================
# Quantile Shield
# ============================================================================

class QuantileShield(nn.Module):
    """
    Quantile Shield: Quantile regression for pessimistic return estimation.
    Now with adaptive thresholding based on training data statistics.
    """
    
    def __init__(
        self,
        state_dim: int,
        command_dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 4,
        tau: float = 0.9,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.command_dim = command_dim
        self.tau = tau
        
        # Input normalization
        self.state_norm = InputNormalizer(state_dim)
        self.command_norm = InputNormalizer(command_dim)
        
        # Network predicts quantile of returns
        self.net = MLP(state_dim + command_dim, hidden_dim, 1, n_layers)
        
        # Adaptive threshold (calibrated during training)
        self.register_buffer('ood_threshold_ratio', torch.tensor(1.0))
        self.register_buffer('return_mean', torch.tensor(0.0))
        self.register_buffer('return_std', torch.tensor(1.0))
    
    def calibrate(self, states: torch.Tensor, commands: torch.Tensor):
        """Calibrate OOD threshold based on training data."""
        with torch.no_grad():
            # Update normalizers
            self.state_norm.update(states)
            self.command_norm.update(commands)
            
            # Compute prediction errors on training data
            returns = commands[:, 1]
            self.return_mean = returns.mean()
            self.return_std = returns.std() + 1e-8
            
            # Set threshold at 95th percentile of returns
            percentile_95 = torch.quantile(returns, 0.95)
            self.ood_threshold_ratio = percentile_95 / (self.return_mean + 1e-8)
    
    def forward(self, state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """Predict tau-quantile of returns."""
        state_n = self.state_norm(state)
        command_n = self.command_norm(command)
        x = torch.cat([state_n, command_n], dim=-1)
        return self.net(x).squeeze(-1)
    
    def loss(self, state: torch.Tensor, command: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """Pinball loss for quantile regression."""
        pred = self(state, command)
        error = returns - pred
        
        # Pinball loss
        loss = torch.where(
            error > 0,
            self.tau * error,
            (self.tau - 1) * error
        )
        return loss.mean()
    
    def is_ood(self, state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """Check if command is OOD based on quantile prediction with adaptive threshold."""
        with torch.no_grad():
            predicted_return = self(state, command)
            requested_return = command[:, 1]
            
            # Adaptive threshold: use calibrated ratio or fall back to 1.5 std above mean
            threshold = predicted_return + 1.5 * self.return_std
            return requested_return > threshold
    
    def project(self, state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """Project command to in-distribution by clamping return."""
        with torch.no_grad():
            predicted_return = self(state, command)
            new_command = command.clone()
            # Clamp to predicted quantile with small margin
            new_command[:, 1] = torch.minimum(command[:, 1], predicted_return * 0.95)
            return new_command


# ============================================================================
# Flow Matching Shield
# ============================================================================

class FlowMatchingShield(nn.Module):
    """
    Flow Matching Shield: Conditional Flow Matching for density estimation.
    
    CORRECTED IMPLEMENTATION with proper:
    1. OT-CFM training (Optimal Transport Conditional Flow Matching)
    2. Multiple sample averaging for robust OOD detection
    3. Smart projection using interpolation towards safe samples
    
    Flow Matching models p(command | state) by learning to transport
    from noise N(0,I) to the data distribution.
    """
    
    def __init__(
        self,
        state_dim: int,
        command_dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 4,
        ood_threshold: float = -5.0,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.command_dim = command_dim
        
        # Input normalization (running statistics)
        self.state_norm = InputNormalizer(state_dim)
        self.command_norm = InputNormalizer(command_dim)
        
        # Time embedding
        self.time_embed = SinusoidalEmbedding(64)
        
        # Velocity network: v(x, t, s) predicts dx/dt at time t
        input_dim = command_dim + 64 + state_dim
        self.net = MLP(input_dim, hidden_dim, command_dim, n_layers)
        
        # OOD detection parameters
        self.register_buffer('ood_threshold', torch.tensor(ood_threshold))
        self.register_buffer('log_prob_mean', torch.tensor(0.0))
        self.register_buffer('log_prob_std', torch.tensor(1.0))
    
    def calibrate(self, states: torch.Tensor, commands: torch.Tensor, 
                  n_samples: int = 1000, percentile: float = 5.0):
        """
        Calibrate OOD threshold on training data.
        
        Sets threshold at given percentile of training log probs.
        Commands with log_prob below this are considered OOD.
        """
        with torch.no_grad():
            # Update normalization statistics
            self.state_norm.update(states)
            self.command_norm.update(commands)
            
            # Compute log probs on random subset
            idx = torch.randperm(len(states))[:n_samples]
            log_probs = self.log_prob_multi(states[idx], commands[idx], n_samples=10)
            
            self.log_prob_mean = log_probs.mean()
            self.log_prob_std = log_probs.std() + 1e-8
            
            # Set threshold at percentile
            self.ood_threshold = torch.tensor(
                np.percentile(log_probs.cpu().numpy(), percentile),
                device=states.device
            )
            
            return {
                'mean': self.log_prob_mean.item(),
                'std': self.log_prob_std.item(),
                'threshold': self.ood_threshold.item(),
            }
    
    def velocity(self, x: torch.Tensor, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity field v(x, t, s).
        
        Args:
            x: Current position in command space (batch, command_dim)
            t: Time in [0, 1] (batch,)
            state: Conditioning state (batch, state_dim)
        
        Returns:
            Predicted velocity (batch, command_dim)
        """
        t_embed = self.time_embed(t)
        state_n = self.state_norm(state)
        inp = torch.cat([x, t_embed, state_n], dim=-1)
        return self.net(inp)
    
    def loss(self, state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """
        OT-CFM loss (Optimal Transport Conditional Flow Matching).
        
        Trains velocity network to predict the conditional OT path:
        x_t = (1-t) * x_0 + t * x_1, where x_0 ~ N(0,I), x_1 = command
        Target velocity: v*(x_t, t) = x_1 - x_0
        """
        batch_size = state.shape[0]
        
        # Normalize command
        command_n = self.command_norm(command, update=True)
        
        # Sample time uniformly
        t = torch.rand(batch_size, device=state.device)
        
        # Sample noise
        x0 = torch.randn_like(command_n)
        x1 = command_n
        
        # Interpolate along OT path
        t_expanded = t.unsqueeze(-1)
        xt = (1 - t_expanded) * x0 + t_expanded * x1
        
        # Target velocity is constant along OT path
        target_velocity = x1 - x0
        
        # Predict velocity
        pred_velocity = self.velocity(xt, t, state)
        
        # MSE loss
        loss = F.mse_loss(pred_velocity, target_velocity)
        
        return loss
    
    @torch.no_grad()
    def sample(self, state: torch.Tensor, n_steps: int = 50) -> torch.Tensor:
        """
        Sample from p(command | state) using Euler integration.
        
        Integrates the ODE: dx/dt = v(x, t, s) from t=0 to t=1.
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Start from noise
        x = torch.randn(batch_size, self.command_dim, device=device)
        dt = 1.0 / n_steps
        
        # Forward integration
        for i in range(n_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            v = self.velocity(x, t, state)
            x = x + dt * v
        
        # Denormalize to original scale
        x = self.command_norm.denormalize(x)
        return x
    
    @torch.no_grad()
    def sample_multiple(self, state: torch.Tensor, n_samples: int = 10, 
                        n_steps: int = 50) -> torch.Tensor:
        """
        Sample multiple commands per state.
        
        Returns: (batch, n_samples, command_dim)
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Expand state for multiple samples
        state_exp = state.unsqueeze(1).expand(-1, n_samples, -1)
        state_flat = state_exp.reshape(-1, self.state_dim)
        
        # Sample
        samples_flat = self.sample(state_flat, n_steps)
        samples = samples_flat.reshape(batch_size, n_samples, self.command_dim)
        
        return samples
    
    @torch.no_grad()
    def log_prob_multi(self, state: torch.Tensor, command: torch.Tensor, 
                       n_samples: int = 10, n_steps: int = 30) -> torch.Tensor:
        """
        Estimate log p(command | state) using multiple samples.
        
        Uses a Gaussian kernel density estimate based on samples from the model.
        More robust than single-sample reconstruction error.
        """
        batch_size = state.shape[0]
        
        # Sample multiple commands per state
        samples = self.sample_multiple(state, n_samples, n_steps)  # (batch, n_samples, dim)
        
        # Normalize command and samples
        command_n = self.command_norm(command)  # (batch, dim)
        samples_n = (samples - self.command_norm.mean) / (torch.sqrt(self.command_norm.var) + 1e-8)
        
        # Compute distances to all samples
        command_exp = command_n.unsqueeze(1)  # (batch, 1, dim)
        sq_dists = ((samples_n - command_exp) ** 2).sum(dim=-1)  # (batch, n_samples)
        
        # Kernel density estimate with adaptive bandwidth
        bandwidth = sq_dists.mean(dim=-1, keepdim=True).sqrt() + 0.1
        
        # Log-sum-exp for numerical stability
        log_kernels = -0.5 * sq_dists / (bandwidth ** 2)
        log_prob = torch.logsumexp(log_kernels, dim=-1) - np.log(n_samples)
        
        return log_prob
    
    def log_prob(self, state: torch.Tensor, command: torch.Tensor, 
                 n_steps: int = 50) -> torch.Tensor:
        """
        Compute log p(command | state) via CNF change of variables.
        
        Uses Hutchinson trace estimator for divergence computation.
        This version supports gradients through command for projection.
        """
        batch_size = state.shape[0]
        
        # Normalize command (keep gradients if command requires_grad)
        mean = self.command_norm.mean
        std = torch.sqrt(self.command_norm.var) + 1e-8
        command_n = (command - mean) / std
        
        # Start from data, integrate backwards to noise
        x = command_n
        log_det = torch.zeros(batch_size, device=state.device)
        dt = 1.0 / n_steps
        
        # Reverse ODE: from t=1 to t=0
        for i in range(n_steps):
            t_val = 1.0 - i * dt
            t = torch.full((batch_size,), t_val, device=state.device)
            
            # Need gradients for divergence
            x_for_div = x.detach().requires_grad_(True)
            v = self.velocity(x_for_div, t, state)
            
            # Hutchinson trace estimator
            eps = torch.randn_like(x_for_div)
            vjp = torch.autograd.grad(
                outputs=(v * eps).sum(),
                inputs=x_for_div,
                create_graph=False
            )[0]
            div_v = (vjp * eps).sum(dim=-1)
            
            # Update log determinant
            log_det = log_det - dt * div_v
            
            # Euler step (use original x, not x_for_div)
            with torch.no_grad():
                v_step = self.velocity(x, t, state)
            x = x - dt * v_step
        
        # Log probability of base distribution
        log_p_base = -0.5 * (x ** 2).sum(dim=-1) - 0.5 * self.command_dim * np.log(2 * np.pi)
        
        return log_p_base + log_det
    
    def is_ood(self, state: torch.Tensor, command: torch.Tensor, 
               fast: bool = True) -> torch.Tensor:
        """Check if command is OOD (out of distribution)."""
        with torch.no_grad():
            if fast:
                log_p = self.log_prob_multi(state, command, n_samples=5)
            else:
                log_p = self.log_prob(state, command)
            return log_p < self.ood_threshold
    
    def project(self, state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """
        Project OOD commands to the safe (in-distribution) region.
        
        Strategy: For OOD commands, interpolate towards the closest
        sampled safe command while preserving the intended direction.
        """
        with torch.no_grad():
            # Detect OOD commands
            is_ood = self.is_ood(state, command, fast=True)
            
            if not is_ood.any():
                return command
            
            projected = command.clone()
            unsafe_idx = is_ood.nonzero(as_tuple=True)[0]
            
            # Sample multiple safe commands for OOD states
            n_candidates = 20
            safe_samples = self.sample_multiple(
                state[unsafe_idx], n_samples=n_candidates, n_steps=50
            )  # (n_unsafe, n_candidates, dim)
            
            # Find closest safe sample to original command
            orig_cmd = command[unsafe_idx].unsqueeze(1)  # (n_unsafe, 1, dim)
            distances = ((safe_samples - orig_cmd) ** 2).sum(dim=-1)  # (n_unsafe, n_candidates)
            closest_idx = distances.argmin(dim=-1)  # (n_unsafe,)
            
            # Get closest samples
            batch_idx = torch.arange(len(unsafe_idx), device=command.device)
            closest_samples = safe_samples[batch_idx, closest_idx]  # (n_unsafe, dim)
            
            # Interpolate: move 80% towards safe sample
            # This preserves some of the original intent while ensuring safety
            alpha = 0.8
            projected[unsafe_idx] = (1 - alpha) * command[unsafe_idx] + alpha * closest_samples
            
            return projected


# ============================================================================
# Diffusion Shield
# ============================================================================

class DiffusionShield(nn.Module):
    """
    Diffusion Shield: DDPM-based density estimation.
    Improved with input normalization, calibrated OOD detection, and better projection.
    """
    
    def __init__(
        self,
        state_dim: int,
        command_dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.command_dim = command_dim
        self.n_timesteps = n_timesteps
        
        # Input normalization
        self.state_norm = InputNormalizer(state_dim)
        self.command_norm = InputNormalizer(command_dim)
        
        # Noise schedule
        betas = torch.linspace(beta_start, beta_end, n_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        
        # OOD threshold (calibrated)
        self.register_buffer("ood_threshold", torch.tensor(1.0))
        self.register_buffer("error_mean", torch.tensor(0.0))
        self.register_buffer("error_std", torch.tensor(1.0))
        
        # Time embedding
        self.time_embed = SinusoidalEmbedding(64)
        
        # Noise prediction network
        input_dim = command_dim + 64 + state_dim
        self.net = MLP(input_dim, hidden_dim, command_dim, n_layers)
    
    def calibrate(self, states: torch.Tensor, commands: torch.Tensor, n_samples: int = 200):
        """Calibrate OOD threshold based on training data."""
        with torch.no_grad():
            self.state_norm.update(states)
            self.command_norm.update(commands)
            
            # Compute reconstruction errors on training data
            idx = torch.randperm(len(states))[:n_samples]
            errors = self.denoising_score(states[idx], commands[idx])
            
            self.error_mean = errors.mean()
            self.error_std = errors.std() + 1e-8
            
            # Set threshold at 2 std above mean
            self.ood_threshold = self.error_mean + 2.0 * self.error_std
    
    def predict_noise(self, x: torch.Tensor, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Predict noise at timestep t."""
        t_embed = self.time_embed(t.float())
        state_n = self.state_norm(state)
        inp = torch.cat([x, t_embed, state_n], dim=-1)
        return self.net(inp)
    
    def loss(self, state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """DDPM training loss with normalized inputs."""
        batch_size = state.shape[0]
        
        # Normalize commands
        command_n = self.command_norm(command, update=True)
        
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=state.device)
        noise = torch.randn_like(command_n)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        
        x_t = sqrt_alpha * command_n + sqrt_one_minus_alpha * noise
        pred_noise = self.predict_noise(x_t, t, state)
        
        loss = F.mse_loss(pred_noise, noise)
        return loss
    
    def sample(self, state: torch.Tensor) -> torch.Tensor:
        """Sample using DDPM."""
        x = torch.randn(state.shape[0], self.command_dim, device=state.device)
        
        with torch.no_grad():
            for t in reversed(range(self.n_timesteps)):
                t_batch = torch.full((state.shape[0],), t, device=state.device)
                pred_noise = self.predict_noise(x, t_batch, state)
                
                alpha = self.alphas[t]
                alpha_cumprod = self.alphas_cumprod[t]
                beta = self.betas[t]
                
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = 0
                
                x = (1 / torch.sqrt(alpha)) * (
                    x - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_noise
                ) + torch.sqrt(beta) * noise
        
        # Denormalize output
        x = x * (torch.sqrt(self.command_norm.var) + 1e-8) + self.command_norm.mean
        return x
    
    def denoising_score(self, state: torch.Tensor, command: torch.Tensor, t_test: int = None) -> torch.Tensor:
        """
        Compute denoising score matching error as OOD metric.
        Better than reconstruction error for detecting OOD samples.
        """
        if t_test is None:
            t_test = self.n_timesteps // 2  # Use middle timestep by default
        t_test = min(t_test, self.n_timesteps - 1)  # Ensure valid index
        
        with torch.no_grad():
            command_n = self.command_norm(command)
            
            # Add noise at timestep t_test
            t = torch.full((state.shape[0],), t_test, device=state.device, dtype=torch.long)
            noise = torch.randn_like(command_n)
            
            sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
            sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
            
            x_t = sqrt_alpha * command_n + sqrt_one_minus_alpha * noise
            
            # Predict noise
            pred_noise = self.predict_noise(x_t, t, state)
            
            # Denoising error
            error = ((pred_noise - noise) ** 2).sum(dim=-1)
            
            return error
    
    def reconstruction_error(self, state: torch.Tensor, command: torch.Tensor, n_samples: int = 3) -> torch.Tensor:
        """Compute reconstruction error for OOD detection."""
        errors = []
        with torch.no_grad():
            for _ in range(n_samples):
                sampled = self.sample(state)
                error = ((sampled - command) ** 2).sum(dim=-1)
                errors.append(error)
        return torch.stack(errors).mean(dim=0)
    
    def is_ood(self, state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """Check if command is OOD using denoising score."""
        score = self.denoising_score(state, command)
        return score > self.ood_threshold
    
    def project(self, state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """Project to in-distribution with adaptive interpolation."""
        with torch.no_grad():
            sampled = self.sample(state)
            
            # Compute how OOD the command is
            score = self.denoising_score(state, command)
            
            # Compute interpolation weight based on OOD score
            # More OOD = more weight to sampled
            normalized_score = (score - self.error_mean) / self.error_std
            weight = torch.sigmoid(normalized_score - 2.0)  # Centered at 2 std
            weight = weight.unsqueeze(-1)
            
            # Interpolate
            projected = (1 - weight) * command + weight * sampled
            
        return projected


# ============================================================================
# Data Utilities
# ============================================================================

def load_data(data_path: str):
    """Load data from npz file."""
    data = np.load(data_path, allow_pickle=True)
    
    # Handle env_name which may be stored as array
    env_name = data.get('env_name', 'unknown')
    if hasattr(env_name, 'item'):
        env_name = str(env_name.item())
    elif isinstance(env_name, np.ndarray):
        env_name = str(env_name)
    
    return {
        'states': data['states'],
        'actions': data['actions'],
        'commands': data['commands'],
        'rewards': data['rewards'],
        'episode_returns': data['episode_returns'],
        'state_dim': int(data['state_dim']),
        'action_dim': int(data['action_dim']),
        'env_name': env_name,
    }


def normalize_commands(commands: np.ndarray):
    """Normalize commands for training."""
    mean = commands.mean(axis=0)
    std = commands.std(axis=0) + 1e-8
    return (commands - mean) / std, mean, std


def create_dataloaders(states, actions, commands, batch_size=256, val_split=0.1):
    """Create training and validation dataloaders."""
    n = len(states)
    n_val = int(n * val_split)
    
    indices = np.random.permutation(n)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(states[train_idx], dtype=torch.float32),
        torch.tensor(actions[train_idx], dtype=torch.float32),
        torch.tensor(commands[train_idx], dtype=torch.float32),
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(states[val_idx], dtype=torch.float32),
        torch.tensor(actions[val_idx], dtype=torch.float32),
        torch.tensor(commands[val_idx], dtype=torch.float32),
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader


# ============================================================================
# Results Directory Management
# ============================================================================

def get_results_dir(env_name: str, model_name: str = None) -> Path:
    """
    Get the results directory path.
    
    Structure:
        results/
            <env_name>/
                models/
                    policy.pt
                    quantile_shield.pt
                    flow_shield.pt
                    diffusion_shield.pt
                figures/
                    training_curves.png
                    comparison.png
                    ...
                logs/
                    training.log
                metrics/
                    results.json
    """
    base = Path("results") / env_name
    
    if model_name:
        return base / "models" / f"{model_name}.pt"
    
    return base


def ensure_results_dirs(env_name: str):
    """Create all result directories."""
    base = Path("results") / env_name
    (base / "models").mkdir(parents=True, exist_ok=True)
    (base / "figures").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)
    (base / "metrics").mkdir(parents=True, exist_ok=True)
    return base


def save_model(model: nn.Module, env_name: str, model_name: str):
    """Save a model to the results directory."""
    ensure_results_dirs(env_name)
    path = get_results_dir(env_name, model_name)
    torch.save(model.state_dict(), path)
    print(f"Saved {model_name} to {path}")


def load_model(model: nn.Module, env_name: str, model_name: str):
    """Load a model from the results directory."""
    path = get_results_dir(env_name, model_name)
    model.load_state_dict(torch.load(path, weights_only=True))
    print(f"Loaded {model_name} from {path}")
    return model


def load_diffusion_shield(env_name: str, state_dim: int) -> 'DiffusionShield':
    """
    Load a DiffusionShield with the correct n_timesteps from checkpoint.
    """
    path = get_results_dir(env_name, "diffusion_shield")
    state_dict = torch.load(path, weights_only=True)
    
    # Infer n_timesteps from saved betas shape
    n_timesteps = state_dict['betas'].shape[0]
    
    # Create model with correct n_timesteps
    model = DiffusionShield(state_dim, n_timesteps=n_timesteps)
    model.load_state_dict(state_dict)
    print(f"Loaded diffusion_shield from {path} (n_timesteps={n_timesteps})")
    return model


# ============================================================================
# Training Utilities
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, score: float, model: nn.Module = None) -> bool:
        if self.mode == 'min':
            score = -score
        
        if self.best_score is None:
            self.best_score = score
            if model is not None:
                self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            if model is not None:
                self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        return self.early_stop
    
    def restore_best(self, model: nn.Module):
        """Restore best model weights."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


class CosineScheduler:
    """
    Cosine annealing learning rate scheduler with warmup.
    """
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            scale = self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * scale)
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


class GradientClipper:
    """Gradient clipping utility."""
    
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
    
    def __call__(self, model: nn.Module):
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)


# ============================================================================
# Training Functions with Improvements
# ============================================================================

def train_model_with_early_stopping(
    model: nn.Module,
    train_fn,
    val_fn,
    epochs: int = 100,
    patience: int = 15,
    scheduler=None,
    verbose: bool = True,
    logger=None,
):
    """
    Generic training loop with early stopping and scheduling.
    
    Args:
        model: PyTorch model to train
        train_fn: Function that runs one training epoch, returns loss
        val_fn: Function that runs validation, returns loss
        epochs: Maximum number of epochs
        patience: Early stopping patience
        scheduler: Optional learning rate scheduler
        verbose: Print progress
        logger: Optional TensorBoard logger
    
    Returns:
        train_losses, val_losses, best_epoch
    """
    early_stopping = EarlyStopping(patience=patience, mode='min')
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        # Training
        train_loss = train_fn()
        train_losses.append(train_loss)
        
        # Validation
        val_loss = val_fn()
        val_losses.append(val_loss)
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_lr()
        else:
            current_lr = None
        
        # Logging
        if logger is not None:
            logger.log_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)
            if current_lr is not None:
                logger.log_scalar('lr', current_lr, epoch)
        
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            lr_str = f" | LR: {current_lr:.2e}" if current_lr else ""
            print(f"Epoch {epoch}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}{lr_str}")
        
        # Early stopping check
        if early_stopping(val_loss, model):
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break
    
    # Restore best model
    early_stopping.restore_best(model)
    best_epoch = epochs - early_stopping.counter
    
    return train_losses, val_losses, best_epoch
