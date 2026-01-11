"""
FlowShield-UDRL: Safety Shields for Command Projection.

This module implements three approaches for detecting and projecting 
Out-of-Distribution (OOD) commands in UDRL:

1. QuantileShield: Baseline using pinball loss for conservative estimation
2. DiffusionShield: DDPM-based conditional generative model
3. FlowMatchShield: Optimal Transport Conditional Flow Matching

Mathematical Foundations:
- We model p(g|s) where g = (horizon, return-to-go), s = state
- OOD detection: g is OOD if log p(g|s) < threshold
- Projection: map g_ood → g_safe on the learned manifold

References:
- Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
- Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- Koenker & Bassett, "Regression Quantiles" (Econometrica 1978)
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Utility Modules
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for time/diffusion step.
    
    Maps scalar t ∈ [0, 1] to high-dimensional embedding using:
        PE(t, 2i) = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))
    """
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        # Precompute frequencies
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("freqs", freqs)
    
    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: Time tensor of shape (batch_size,) or (batch_size, 1)
        Returns:
            Embedding of shape (batch_size, dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # (batch_size, half_dim)
        args = t * self.freqs.unsqueeze(0) * 2 * math.pi
        
        # Interleave sin and cos
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        
        return embedding


class MLP(nn.Module):
    """Multi-layer perceptron with configurable activation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 3,
        activation: str = "silu",
        dropout: float = 0.0,
        output_activation: Optional[str] = None,
    ):
        super().__init__()
        
        act_fn = {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }[activation]
        
        layers = []
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            if i < len(dims) - 2:  # Not last layer
                layers.append(act_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        if output_activation:
            layers.append(act_fn() if output_activation == activation else nn.Identity())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConditionedMLP(nn.Module):
    """
    MLP conditioned on state via concatenation + optional time embedding.
    
    Architecture:
        [state, command, time_embed] → MLP → output
    """
    
    def __init__(
        self,
        state_dim: int,
        command_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 4,
        time_embed_dim: int = 128,
        use_time: bool = True,
        activation: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.use_time = use_time
        
        # Time embedding
        if use_time:
            self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
            self.time_mlp = nn.Sequential(
                nn.Linear(time_embed_dim, time_embed_dim * 2),
                nn.SiLU(),
                nn.Linear(time_embed_dim * 2, time_embed_dim),
            )
            input_dim = state_dim + command_dim + time_embed_dim
        else:
            input_dim = state_dim + command_dim
        
        # Main network
        self.net = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            activation=activation,
            dropout=dropout,
        )
    
    def forward(
        self,
        state: Tensor,
        command: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            state: (batch_size, state_dim)
            command: (batch_size, command_dim)
            t: (batch_size,) time in [0, 1], required if use_time=True
        """
        if self.use_time:
            assert t is not None, "Time t required for time-conditioned network"
            t_embed = self.time_mlp(self.time_embed(t))
            x = torch.cat([state, command, t_embed], dim=-1)
        else:
            x = torch.cat([state, command], dim=-1)
        
        return self.net(x)


# =============================================================================
# Base Shield Class
# =============================================================================

class BaseShield(ABC, nn.Module):
    """
    Abstract base class for command safety shields.
    
    A shield has two main functions:
    1. is_safe(state, command) → bool: Detect if command is OOD
    2. project(state, command) → safe_command: Map OOD to safe command
    """
    
    def __init__(
        self,
        state_dim: int,
        command_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.command_dim = command_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
    
    @abstractmethod
    def compute_loss(
        self,
        state: Tensor,
        command: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute training loss."""
        pass
    
    @abstractmethod
    def get_safety_score(
        self,
        state: Tensor,
        command: Tensor,
    ) -> Tensor:
        """
        Compute safety score for (state, command) pairs.
        Higher score = more likely in-distribution.
        """
        pass
    
    @abstractmethod
    def is_safe(
        self,
        state: Tensor,
        command: Tensor,
        threshold: float,
    ) -> Tensor:
        """Return boolean mask indicating safe commands."""
        pass
    
    @abstractmethod
    def project(
        self,
        state: Tensor,
        command: Tensor,
        **kwargs,
    ) -> Tensor:
        """Project OOD command to the safe manifold."""
        pass
    
    def log_prob(
        self,
        state: Tensor,
        command: Tensor,
    ) -> Tensor:
        """
        Compute log probability of command given state.
        Default implementation uses safety score.
        Override for true density estimation.
        """
        return self.get_safety_score(state, command)
    
    def forward(
        self,
        state: Tensor,
        command: Tensor,
        threshold: float = -5.0,
        **kwargs,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Apply shield: detect OOD and project if necessary.
        
        Returns:
            safe_command: Projected command (or original if safe)
            info: Dict with is_safe, score, projection_distance
        """
        score = self.get_safety_score(state, command)
        safe_mask = self.is_safe(state, command, threshold)
        
        # Project unsafe commands
        projected = self.project(state, command, **kwargs)
        
        # Use original if safe, projected if OOD
        safe_command = torch.where(
            safe_mask.unsqueeze(-1),
            command,
            projected,
        )
        
        info = {
            "is_safe": safe_mask,
            "score": score,
            "projection_distance": (command - safe_command).norm(dim=-1),
        }
        
        return safe_command, info


# =============================================================================
# 1. Quantile Shield (Baseline)
# =============================================================================

class QuantileShield(BaseShield):
    """
    Quantile regression baseline for conservative command estimation.
    
    Learns q_τ(s) = τ-quantile of return achievable from state s.
    Uses pinball loss (asymmetric L1) for training.
    
    Pinball Loss:
        L_τ(y, ŷ) = τ·max(y - ŷ, 0) + (1-τ)·max(ŷ - y, 0)
    
    This penalizes over-prediction more when τ > 0.5 (conservative).
    """
    
    def __init__(
        self,
        state_dim: int,
        command_dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 4,
        tau: float = 0.9,
        activation: str = "silu",
    ):
        super().__init__(state_dim, command_dim, hidden_dim, n_layers)
        
        self.tau = tau
        
        # Predicts (max_horizon, max_return) achievable from state
        self.net = MLP(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=command_dim,
            n_layers=n_layers,
            activation=activation,
        )
    
    def compute_loss(
        self,
        state: Tensor,
        command: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Pinball loss for quantile regression.
        
        Args:
            state: (batch_size, state_dim)
            command: (batch_size, command_dim) - observed (horizon, return)
        """
        pred = self.net(state)  # (batch_size, command_dim)
        
        # Pinball loss: asymmetric L1
        residual = command - pred
        loss = torch.where(
            residual >= 0,
            self.tau * residual,
            (self.tau - 1) * residual,
        )
        loss = loss.mean()
        
        return {
            "loss": loss,
            "mae": residual.abs().mean(),
            "prediction": pred.detach(),
        }
    
    def get_safety_score(
        self,
        state: Tensor,
        command: Tensor,
    ) -> Tensor:
        """
        Safety score: negative distance to quantile prediction.
        Score = -||command - q_τ(s)||_+ (only penalize exceeding quantile)
        """
        with torch.no_grad():
            pred = self.net(state)
        
        # Only penalize if command exceeds predicted quantile
        excess = F.relu(command - pred)
        score = -excess.sum(dim=-1)
        
        return score
    
    def is_safe(
        self,
        state: Tensor,
        command: Tensor,
        threshold: float = -1.0,
    ) -> Tensor:
        """Command is safe if it doesn't exceed quantile by more than threshold."""
        score = self.get_safety_score(state, command)
        return score >= threshold
    
    def project(
        self,
        state: Tensor,
        command: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Project by clipping to quantile prediction.
        Simple and fast: just cap the command at predicted limits.
        """
        with torch.no_grad():
            pred = self.net(state)
        
        # Clip command to not exceed predicted maximum
        projected = torch.minimum(command, pred)
        
        return projected


# =============================================================================
# 2. Diffusion Shield (DDPM)
# =============================================================================

class DiffusionShield(BaseShield):
    """
    Denoising Diffusion Probabilistic Model for conditional p(g|s).
    
    Forward process (fixed):
        q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1 - ᾱ_t) I)
    
    Reverse process (learned):
        p_θ(x_{t-1} | x_t, s) = N(x_{t-1}; μ_θ(x_t, t, s), σ_t² I)
    
    Training: predict noise ε from (x_t, t, s)
        L = ||ε - ε_θ(x_t, t, s)||²
    
    Inference: iterative denoising from x_T ~ N(0, I) to x_0
    """
    
    def __init__(
        self,
        state_dim: int,
        command_dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        time_embed_dim: int = 128,
        n_inference_steps: int = 20,
        activation: str = "silu",
    ):
        super().__init__(state_dim, command_dim, hidden_dim, n_layers)
        
        self.n_timesteps = n_timesteps
        self.n_inference_steps = n_inference_steps
        
        # Noise schedule
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, n_timesteps)
        elif beta_schedule == "cosine":
            # Cosine schedule from Nichol & Dhariwal
            steps = torch.arange(n_timesteps + 1) / n_timesteps
            alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            betas = betas.clamp(max=0.999)
        elif beta_schedule == "quadratic":
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, n_timesteps) ** 2
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        
        # For posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod)
        )
        
        # Noise prediction network: ε_θ(x_t, t, s)
        self.noise_net = ConditionedMLP(
            state_dim=state_dim,
            command_dim=command_dim,
            hidden_dim=hidden_dim,
            output_dim=command_dim,
            n_layers=n_layers,
            time_embed_dim=time_embed_dim,
            use_time=True,
            activation=activation,
        )
    
    def _extract(self, coef: Tensor, t: Tensor, shape: Tuple) -> Tensor:
        """Extract coefficients at timestep t and reshape for broadcasting."""
        batch_size = t.shape[0]
        out = coef.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))
    
    def q_sample(
        self,
        x_0: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward diffusion: q(x_t | x_0) = N(√ᾱ_t x_0, (1 - ᾱ_t) I)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    
    def compute_loss(
        self,
        state: Tensor,
        command: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Denoising score matching loss.
        
        L = E_{t, ε} ||ε - ε_θ(x_t, t, s)||²
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Sample random timesteps
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=device)
        t_normalized = t.float() / self.n_timesteps
        
        # Sample noise and corrupt
        noise = torch.randn_like(command)
        x_t = self.q_sample(command, t, noise)
        
        # Predict noise
        noise_pred = self.noise_net(state, x_t, t_normalized)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        return {
            "loss": loss,
            "mse": loss.detach(),
        }
    
    @torch.no_grad()
    def p_sample(
        self,
        x_t: Tensor,
        t: Tensor,
        state: Tensor,
        clip_denoised: bool = True,
    ) -> Tensor:
        """
        Single reverse diffusion step: p_θ(x_{t-1} | x_t, s)
        """
        batch_size = x_t.shape[0]
        t_normalized = t.float() / self.n_timesteps
        
        # Predict noise
        noise_pred = self.noise_net(state, x_t, t_normalized)
        
        # Compute x_0 prediction
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        x_0_pred = (x_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
        
        if clip_denoised:
            x_0_pred = x_0_pred.clamp(-10.0, 10.0)
        
        # Compute posterior mean
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = coef1 * x_0_pred + coef2 * x_t
        
        # Add noise (except at t=0)
        posterior_var = self._extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.dim() - 1)))
        
        return posterior_mean + nonzero_mask * posterior_var.sqrt() * noise
    
    @torch.no_grad()
    def sample(
        self,
        state: Tensor,
        n_steps: Optional[int] = None,
    ) -> Tensor:
        """
        Generate command samples via iterative denoising.
        
        Args:
            state: (batch_size, state_dim) conditioning
            n_steps: Number of denoising steps (default: n_inference_steps)
        """
        n_steps = n_steps or self.n_inference_steps
        batch_size = state.shape[0]
        device = state.device
        
        # DDIM-style step selection
        step_size = self.n_timesteps // n_steps
        timesteps = torch.arange(0, self.n_timesteps, step_size, device=device)
        timesteps = timesteps.flip(0)  # Reverse order
        
        # Start from noise
        x = torch.randn(batch_size, self.command_dim, device=device)
        
        for t in timesteps:
            t_batch = t.expand(batch_size)
            x = self.p_sample(x, t_batch, state)
        
        return x
    
    def get_safety_score(
        self,
        state: Tensor,
        command: Tensor,
    ) -> Tensor:
        """
        Approximate log p(g|s) via denoising score matching.
        
        Uses ELBO approximation: higher score if less noise is predicted
        at low diffusion levels.
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Evaluate at small noise level (near data)
        t = torch.full((batch_size,), 10, device=device, dtype=torch.long)
        t_normalized = t.float() / self.n_timesteps
        
        noise = torch.randn_like(command)
        x_t = self.q_sample(command, t, noise)
        
        with torch.no_grad():
            noise_pred = self.noise_net(state, x_t, t_normalized)
        
        # Lower prediction error = higher likelihood
        mse = ((noise_pred - noise) ** 2).sum(dim=-1)
        score = -mse
        
        return score
    
    def is_safe(
        self,
        state: Tensor,
        command: Tensor,
        threshold: float = -5.0,
    ) -> Tensor:
        score = self.get_safety_score(state, command)
        return score >= threshold
    
    def project(
        self,
        state: Tensor,
        command: Tensor,
        n_steps: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
        """
        Project by denoising the command.
        
        Add noise to OOD command, then denoise conditioned on state.
        This maps it back to the learned manifold.
        """
        n_steps = n_steps or (self.n_inference_steps // 2)
        device = state.device
        batch_size = state.shape[0]
        
        # Add moderate noise to command
        t_start = min(self.n_timesteps // 4, 250)
        t = torch.full((batch_size,), t_start, device=device, dtype=torch.long)
        
        noise = torch.randn_like(command)
        x = self.q_sample(command, t, noise)
        
        # Denoise
        step_size = t_start // n_steps
        timesteps = torch.arange(0, t_start, step_size, device=device).flip(0)
        
        for t_val in timesteps:
            t_batch = t_val.expand(batch_size)
            x = self.p_sample(x, t_batch, state)
        
        return x


# =============================================================================
# 3. Flow Matching Shield (Optimal Transport CFM)
# =============================================================================

class FlowMatchShield(BaseShield):
    """
    Conditional Flow Matching using Optimal Transport paths.
    
    Learns the vector field v_θ(x, t, s) that transports noise to data:
        dx/dt = v_θ(x_t, t, s)
    
    Using OT-CFM (Lipman et al., 2023):
        - Probability path: p_t = [(1-t) + tσ_min] N(0,I) + t·δ_{x_1}
        - Interpolant: x_t = (1 - t) x_0 + t x_1  (straight line)
        - Target velocity: u_t(x|x_1) = x_1 - x_0
        - Loss: ||v_θ(x_t, t, s) - (x_1 - x_0)||²
    
    Advantages over diffusion:
        - Simulation-free training
        - Faster inference (ODE vs SDE)
        - Straighter paths = fewer integration steps
    """
    
    def __init__(
        self,
        state_dim: int,
        command_dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_integration_steps: int = 50,
        sigma_min: float = 1e-4,
        solver: str = "euler",
        time_embed_dim: int = 128,
        activation: str = "silu",
    ):
        super().__init__(state_dim, command_dim, hidden_dim, n_layers)
        
        self.n_integration_steps = n_integration_steps
        self.sigma_min = sigma_min
        self.solver = solver
        
        # Vector field network: v_θ(x, t, s)
        self.velocity_net = ConditionedMLP(
            state_dim=state_dim,
            command_dim=command_dim,
            hidden_dim=hidden_dim,
            output_dim=command_dim,
            n_layers=n_layers,
            time_embed_dim=time_embed_dim,
            use_time=True,
            activation=activation,
        )
    
    def _ot_interpolant(
        self,
        x_0: Tensor,
        x_1: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Optimal Transport interpolant: x_t = (1-t)x_0 + t·x_1
        With small noise for numerical stability.
        """
        t = t.view(-1, 1)
        sigma_t = 1 - (1 - self.sigma_min) * t
        x_t = t * x_1 + (1 - t) * x_0
        return x_t + sigma_t * torch.randn_like(x_0) * 0  # Optional: add small noise
    
    def compute_loss(
        self,
        state: Tensor,
        command: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Flow Matching loss with OT path.
        
        L = E_{t, x_0} ||v_θ(x_t, t, s) - (x_1 - x_0)||²
        
        where:
            x_0 ~ N(0, I) (source noise)
            x_1 = command (target data)
            x_t = (1-t)x_0 + t·x_1 (OT interpolant)
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Sample source noise
        x_0 = torch.randn_like(command)
        x_1 = command
        
        # Sample time uniformly
        t = torch.rand(batch_size, device=device)
        
        # OT interpolant
        x_t = self._ot_interpolant(x_0, x_1, t)
        
        # Target velocity (straight line from noise to data)
        target_velocity = x_1 - x_0
        
        # Predict velocity
        pred_velocity = self.velocity_net(state, x_t, t)
        
        # MSE loss
        loss = F.mse_loss(pred_velocity, target_velocity)
        
        return {
            "loss": loss,
            "mse": loss.detach(),
        }
    
    def _euler_step(
        self,
        x: Tensor,
        t: float,
        dt: float,
        state: Tensor,
    ) -> Tensor:
        """Euler integration step."""
        t_tensor = torch.full((x.shape[0],), t, device=x.device)
        v = self.velocity_net(state, x, t_tensor)
        return x + dt * v
    
    def _midpoint_step(
        self,
        x: Tensor,
        t: float,
        dt: float,
        state: Tensor,
    ) -> Tensor:
        """Midpoint (RK2) integration step."""
        t_tensor = torch.full((x.shape[0],), t, device=x.device)
        t_mid_tensor = torch.full((x.shape[0],), t + dt/2, device=x.device)
        
        v1 = self.velocity_net(state, x, t_tensor)
        x_mid = x + (dt / 2) * v1
        v2 = self.velocity_net(state, x_mid, t_mid_tensor)
        
        return x + dt * v2
    
    def _rk4_step(
        self,
        x: Tensor,
        t: float,
        dt: float,
        state: Tensor,
    ) -> Tensor:
        """RK4 integration step."""
        t1 = torch.full((x.shape[0],), t, device=x.device)
        t2 = torch.full((x.shape[0],), t + dt/2, device=x.device)
        t3 = torch.full((x.shape[0],), t + dt, device=x.device)
        
        k1 = self.velocity_net(state, x, t1)
        k2 = self.velocity_net(state, x + (dt/2) * k1, t2)
        k3 = self.velocity_net(state, x + (dt/2) * k2, t2)
        k4 = self.velocity_net(state, x + dt * k3, t3)
        
        return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    @torch.no_grad()
    def sample(
        self,
        state: Tensor,
        n_steps: Optional[int] = None,
    ) -> Tensor:
        """
        Generate samples by integrating ODE from t=0 to t=1.
        
        dx/dt = v_θ(x, t, s),  x(0) ~ N(0, I)
        """
        n_steps = n_steps or self.n_integration_steps
        batch_size = state.shape[0]
        device = state.device
        
        # Start from noise
        x = torch.randn(batch_size, self.command_dim, device=device)
        
        dt = 1.0 / n_steps
        step_fn = {
            "euler": self._euler_step,
            "midpoint": self._midpoint_step,
            "rk4": self._rk4_step,
        }[self.solver]
        
        for i in range(n_steps):
            t = i / n_steps
            x = step_fn(x, t, dt, state)
        
        return x
    
    @torch.no_grad()
    def log_prob(
        self,
        state: Tensor,
        command: Tensor,
        n_steps: Optional[int] = None,
        n_hutchinson: int = 10,
    ) -> Tensor:
        """
        Compute log p(command | state) via change of variables.
        
        log p(x_1) = log p(x_0) - ∫₀¹ tr(∂v/∂x) dt
        
        Uses Hutchinson trace estimator:
            tr(J) ≈ E_ε[ε^T J ε] where ε ~ N(0, I)
        """
        n_steps = n_steps or self.n_integration_steps
        batch_size = state.shape[0]
        device = command.device
        
        # Integrate backward from x_1 to x_0
        x = command.clone()
        log_det = torch.zeros(batch_size, device=device)
        
        dt = -1.0 / n_steps  # Negative for reverse integration
        
        for i in range(n_steps, 0, -1):
            t = i / n_steps
            t_tensor = torch.full((batch_size,), t, device=device)
            
            # Enable gradients for trace estimation
            x_input = x.detach().requires_grad_(True)
            v = self.velocity_net(state, x_input, t_tensor)
            
            # Hutchinson trace estimator
            trace_est = torch.zeros(batch_size, device=device)
            for _ in range(n_hutchinson):
                eps = torch.randn_like(x_input)
                vjp = torch.autograd.grad(
                    v, x_input, eps,
                    create_graph=False, retain_graph=True
                )[0]
                trace_est += (eps * vjp).sum(dim=-1)
            trace_est = trace_est / n_hutchinson
            
            # Update log determinant (note: dt is negative)
            log_det = log_det - dt * trace_est
            
            # Euler step backward
            x = x + dt * v.detach()
        
        # Log prob under base (standard Gaussian)
        log_p0 = -0.5 * (x ** 2).sum(dim=-1) - self.command_dim * 0.5 * math.log(2 * math.pi)
        
        return log_p0 + log_det
    
    def get_safety_score(
        self,
        state: Tensor,
        command: Tensor,
    ) -> Tensor:
        """Log probability as safety score."""
        return self.log_prob(state, command)
    
    def is_safe(
        self,
        state: Tensor,
        command: Tensor,
        threshold: float = -5.0,
    ) -> Tensor:
        score = self.get_safety_score(state, command)
        return score >= threshold
    
    def project(
        self,
        state: Tensor,
        command: Tensor,
        n_steps: int = 50,
        lr: float = 0.1,
        momentum: float = 0.9,
        **kwargs,
    ) -> Tensor:
        """
        Gradient-based projection to maximize log p(g|s).
        
        Iteratively move command toward higher probability region:
            g ← g + lr · ∇_g log p(g|s)
        """
        x = command.clone().detach().requires_grad_(True)
        velocity = torch.zeros_like(x)
        
        for _ in range(n_steps):
            # Compute log prob and gradient
            log_p = self.log_prob(state, x, n_steps=20, n_hutchinson=5)
            
            grad = torch.autograd.grad(log_p.sum(), x, retain_graph=False)[0]
            
            # Momentum update
            velocity = momentum * velocity + (1 - momentum) * grad
            x = x + lr * velocity
            x = x.detach().requires_grad_(True)
        
        return x.detach()


# =============================================================================
# Shield Factory
# =============================================================================

def create_shield(
    method: str,
    state_dim: int,
    command_dim: int = 2,
    config: dict = None,
    **kwargs,
) -> BaseShield:
    """
    Factory function to create a shield by name.
    
    Args:
        method: "quantile", "diffusion", or "flow_matching"
        state_dim: Dimension of state space
        command_dim: Dimension of command (default: 2 for horizon, return)
        config: Method-specific configuration dict
        **kwargs: Additional method-specific arguments
    
    Returns:
        Instantiated shield module
    """
    shields = {
        "quantile": QuantileShield,
        "diffusion": DiffusionShield,
        "flow_matching": FlowMatchShield,
    }
    
    if method not in shields:
        raise ValueError(f"Unknown shield method: {method}. Choose from {list(shields.keys())}")
    
    # Merge config dict with kwargs
    if config is not None:
        # Remove keys that shouldn't go to constructor
        config = {k: v for k, v in config.items() if k not in ["method", "command_dim"]}
        kwargs = {**config, **kwargs}
    
    return shields[method](state_dim=state_dim, command_dim=command_dim, **kwargs)

