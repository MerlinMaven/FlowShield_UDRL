"""
Complete Pipeline Runner for FlowShield-UDRL.

Executes the full experiment pipeline:
1. Data collection
2. UDRL policy training
3. Shield training (Quantile, Diffusion, Flow Matching)
4. Evaluation and comparison
5. Visualization and analysis

Usage:
    python run_experiments.py --env lunarlander --n-episodes 2000
    python run_experiments.py --env highway --quick
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import gymnasium, fallback to gym
try:
    import gymnasium as gym
except ImportError:
    import gym

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =============================================================================
# Environment Wrappers
# =============================================================================

class LunarLanderWrapper:
    """Wrapper for LunarLander with continuous actions."""
    
    def __init__(self, render_mode: Optional[str] = None):
        # Try v3 first (newer gymnasium), then fallback to v2
        try:
            self.env = gym.make("LunarLander-v3", continuous=True, render_mode=render_mode)
        except:
            try:
                self.env = gym.make("LunarLanderContinuous-v2", render_mode=render_mode)
            except:
                self.env = gym.make("LunarLander-v2", continuous=True)
        
        self.state_dim = 8
        self.action_dim = 2
        self.max_steps = 1000
        
        # Normalization stats (computed from data)
        self.state_mean = np.zeros(8)
        self.state_std = np.ones(8)
    
    def reset(self, seed=None):
        if seed is not None:
            result = self.env.reset(seed=seed)
        else:
            result = self.env.reset()
        
        if isinstance(result, tuple):
            return result[0]
        return result
    
    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            state, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            state, reward, done, info = result
        return state, reward, done, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()


def make_env(env_name: str = "lunarlander", render_mode: Optional[str] = None):
    """Create environment by name."""
    if env_name.lower() in ["lunarlander", "lunar", "ll"]:
        return LunarLanderWrapper(render_mode=render_mode)
    else:
        raise ValueError(f"Unknown environment: {env_name}")


# =============================================================================
# Data Collection
# =============================================================================

def collect_trajectories(
    env,
    n_episodes: int,
    policy=None,
    seed: int = 42,
    show_progress: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Collect trajectories from environment.
    
    Returns:
        Dictionary with states, actions, rewards, dones, returns, horizons
    """
    np.random.seed(seed)
    
    all_states = []
    all_actions = []
    all_rewards = []
    all_dones = []
    all_returns = []
    all_horizons = []
    
    episode_returns = []
    episode_lengths = []
    
    iterator = range(n_episodes)
    if show_progress:
        iterator = tqdm(iterator, desc="Collecting trajectories")
    
    for ep in iterator:
        state = env.reset(seed=seed + ep)
        
        states = []
        actions = []
        rewards = []
        
        done = False
        step = 0
        
        while not done and step < env.max_steps:
            if policy is None:
                # Random policy with smoothing
                action = np.random.uniform(-1, 1, size=env.action_dim)
            else:
                action = policy(state)
            
            next_state, reward, done, info = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
            step += 1
        
        # Compute hindsight commands for each timestep
        episode_return = sum(rewards)
        episode_length = len(rewards)
        
        for t in range(episode_length):
            remaining_horizon = episode_length - t
            remaining_return = sum(rewards[t:])
            
            all_states.append(states[t])
            all_actions.append(actions[t])
            all_rewards.append(rewards[t])
            all_dones.append(t == episode_length - 1)
            all_returns.append(remaining_return)
            all_horizons.append(remaining_horizon)
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
    
    data = {
        "states": np.array(all_states, dtype=np.float32),
        "actions": np.array(all_actions, dtype=np.float32),
        "rewards": np.array(all_rewards, dtype=np.float32),
        "dones": np.array(all_dones, dtype=np.float32),
        "returns": np.array(all_returns, dtype=np.float32),
        "horizons": np.array(all_horizons, dtype=np.float32),
    }
    
    print(f"\n=== Data Collection Summary ===")
    print(f"Episodes: {n_episodes}")
    print(f"Total transitions: {len(all_states)}")
    print(f"Mean episode return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Return range: [{min(episode_returns):.1f}, {max(episode_returns):.1f}]")
    
    return data


# =============================================================================
# Neural Network Components
# =============================================================================

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal time/position embedding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class FourierFeatures(nn.Module):
    """Random Fourier features for continuous inputs."""
    
    def __init__(self, input_dim: int, embed_dim: int, scale: float = 1.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(input_dim, embed_dim // 2) * scale, requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


class MLP(nn.Module):
    """Multi-layer perceptron with SiLU activation."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# UDRL Policy
# =============================================================================

class UDRLPolicy(nn.Module):
    """
    Upside-Down RL Policy: π(a|s, g) where g = (H, R).
    
    Maps (state, command) -> action distribution.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        command_dim: int = 2,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.command_dim = command_dim
        
        # Command embedding with Fourier features
        self.command_embed = FourierFeatures(command_dim, 64, scale=1.0)
        
        # Main network
        self.net = MLP(
            input_dim=state_dim + 64,  # state + command embedding
            output_dim=action_dim * 2,  # mean + log_std
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )
        
        self.log_std_min = -5.0
        self.log_std_max = 2.0
    
    def forward(self, state: torch.Tensor, command: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute action distribution parameters."""
        command_emb = self.command_embed(command)
        x = torch.cat([state, command_emb], dim=-1)
        out = self.net(x)
        
        mean, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor, command: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample action from policy."""
        mean, log_std = self.forward(state, command)
        
        if deterministic:
            return torch.tanh(mean)
        
        std = torch.exp(log_std)
        noise = torch.randn_like(mean)
        action = torch.tanh(mean + std * noise)
        
        return action
    
    def log_prob(self, state: torch.Tensor, command: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of action."""
        mean, log_std = self.forward(state, command)
        std = torch.exp(log_std)
        
        # Inverse tanh
        action_clipped = torch.clamp(action, -0.999, 0.999)
        pre_tanh = 0.5 * (torch.log1p(action_clipped) - torch.log1p(-action_clipped))
        
        # Gaussian log prob
        log_prob = -0.5 * (((pre_tanh - mean) / std) ** 2 + 2 * log_std + np.log(2 * np.pi))
        
        # Jacobian correction
        log_prob = log_prob - torch.log(1 - action ** 2 + 1e-6)
        
        return log_prob.sum(dim=-1)


# =============================================================================
# Safety Shields
# =============================================================================

class QuantileShield(nn.Module):
    """
    Quantile Shield: Estimates command bounds using pinball loss.
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
        
        self.tau = tau
        self.command_dim = command_dim
        
        self.net = MLP(
            input_dim=state_dim,
            output_dim=command_dim * 2,  # upper and lower bounds
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict quantile bounds."""
        out = self.net(state)
        lower, upper = out.chunk(2, dim=-1)
        return lower, upper
    
    def compute_loss(self, state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """Compute pinball loss for both quantiles."""
        lower, upper = self.forward(state)
        
        # Upper quantile (tau)
        error_upper = command - upper
        loss_upper = torch.where(
            error_upper >= 0,
            self.tau * error_upper,
            (self.tau - 1) * error_upper
        )
        
        # Lower quantile (1 - tau)
        error_lower = command - lower
        loss_lower = torch.where(
            error_lower >= 0,
            (1 - self.tau) * error_lower,
            -self.tau * error_lower
        )
        
        return (loss_upper.mean() + loss_lower.mean()) / 2
    
    def is_ood(self, state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """Check if command is out of distribution."""
        lower, upper = self.forward(state)
        ood_upper = (command > upper).any(dim=-1)
        ood_lower = (command < lower).any(dim=-1)
        return ood_upper | ood_lower
    
    def project(self, state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """Project command to within bounds."""
        lower, upper = self.forward(state)
        return torch.clamp(command, lower, upper)


class FlowMatchingShield(nn.Module):
    """
    Flow Matching Shield: OT-CFM based density estimation.
    
    Learns velocity field v(x, t, s) for flow from N(0,I) to p(g|s).
    """
    
    def __init__(
        self,
        state_dim: int,
        command_dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 4,
        sigma_min: float = 1e-4,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.command_dim = command_dim
        self.sigma_min = sigma_min
        
        # Time embedding
        self.time_embed = SinusoidalEmbedding(64)
        
        # Velocity network
        self.net = MLP(
            input_dim=state_dim + command_dim + 64,  # state + command + time
            output_dim=command_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )
    
    def velocity(self, x: torch.Tensor, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Compute velocity at (x, t) conditioned on state."""
        t_emb = self.time_embed(t)
        inp = torch.cat([state, x, t_emb], dim=-1)
        return self.net(inp)
    
    def compute_loss(self, state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """
        Compute CFM loss: ||v_θ(x_t, t, s) - (x_1 - x_0)||²
        
        Uses OT interpolation: x_t = (1-t)x_0 + t*x_1
        Target velocity: x_1 - x_0
        """
        batch_size = state.shape[0]
        
        # Sample time uniformly
        t = torch.rand(batch_size, device=state.device)
        
        # Sample from prior
        x0 = torch.randn_like(command)
        
        # Target is the data
        x1 = command
        
        # OT interpolation
        t_expanded = t.unsqueeze(-1)
        xt = (1 - t_expanded) * x0 + t_expanded * x1
        
        # Target velocity
        target_velocity = x1 - x0
        
        # Predicted velocity
        pred_velocity = self.velocity(xt, t, state)
        
        # MSE loss
        loss = F.mse_loss(pred_velocity, target_velocity)
        
        return loss
    
    def sample(self, state: torch.Tensor, n_steps: int = 50) -> torch.Tensor:
        """Sample from flow using Euler integration."""
        x = torch.randn(state.shape[0], self.command_dim, device=state.device)
        dt = 1.0 / n_steps
        
        for i in range(n_steps):
            t = torch.full((state.shape[0],), i * dt, device=state.device)
            v = self.velocity(x, t, state)
            x = x + dt * v
        
        return x
    
    def log_prob(self, state: torch.Tensor, command: torch.Tensor, n_steps: int = 50) -> torch.Tensor:
        """
        Compute log probability via reverse ODE with Hutchinson trace estimator.
        """
        # Ensure command requires grad for Jacobian computation
        x = command.clone().detach().requires_grad_(True)
        state = state.detach()  # State doesn't need gradients
        
        log_p = torch.zeros(state.shape[0], device=state.device)
        
        dt = 1.0 / n_steps
        
        for i in range(n_steps, 0, -1):
            t = torch.full((state.shape[0],), i * dt, device=state.device)
            
            # Compute velocity with gradient tracking
            v = self.velocity(x, t, state)
            
            # Hutchinson trace estimator for divergence
            eps = torch.randn_like(x)
            
            # Compute Jacobian-vector product
            v_dot_eps = (v * eps).sum()
            grad_v = torch.autograd.grad(v_dot_eps, x, create_graph=False, retain_graph=True)[0]
            div_v = (grad_v * eps).sum(dim=-1)
            
            # Update position (detach to avoid building huge graph)
            with torch.no_grad():
                x_new = x - dt * v
            x = x_new.requires_grad_(True)
            
            log_p = log_p + dt * div_v.detach()
        
        # Add prior log prob
        log_p_prior = -0.5 * (x.detach() ** 2).sum(dim=-1) - 0.5 * self.command_dim * np.log(2 * np.pi)
        
        return log_p_prior + log_p
    
    def is_ood(self, state: torch.Tensor, command: torch.Tensor, threshold: float = -6.0) -> torch.Tensor:
        """Check if command is OOD based on log probability."""
        # Don't use no_grad - log_prob needs gradients for Hutchinson estimator
        log_p = self.log_prob(state, command)
        return log_p.detach() < threshold
    
    def project(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
        n_steps: int = 20,
        lr: float = 0.1,
    ) -> torch.Tensor:
        """
        Project OOD command to in-distribution using sampling.
        Instead of gradient ascent (which has gradient issues), we use 
        a weighted combination of the original command and a sampled in-distribution command.
        """
        with torch.no_grad():
            # Sample a fresh in-distribution command
            sampled = self.sample(state, n_steps=50)
            
            # Compute weights based on how OOD the original command is
            # Use lower n_steps for faster log_prob (approximate)
            # Skip log_prob since it has gradient issues, just blend toward sample
            
            # Simple projection: move 50% toward the sampled in-distribution command
            # This is a conservative but robust approach
            projected = 0.5 * command + 0.5 * sampled
            
        return projected


class DiffusionShield(nn.Module):
    """
    Diffusion Shield: DDPM-based density estimation.
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
        
        # Noise schedule
        betas = torch.linspace(beta_start, beta_end, n_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        
        # Time embedding
        self.time_embed = SinusoidalEmbedding(64)
        
        # Noise prediction network
        self.net = MLP(
            input_dim=state_dim + command_dim + 64,
            output_dim=command_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Predict noise."""
        t_emb = self.time_embed(t.float())
        inp = torch.cat([state, x, t_emb], dim=-1)
        return self.net(inp)
    
    def compute_loss(self, state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """Compute denoising loss."""
        batch_size = state.shape[0]
        
        # Sample timesteps
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=state.device)
        
        # Sample noise
        noise = torch.randn_like(command)
        
        # Add noise to command
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        noisy_command = sqrt_alpha * command + sqrt_one_minus_alpha * noise
        
        # Predict noise
        pred_noise = self.forward(noisy_command, t, state)
        
        # MSE loss
        loss = F.mse_loss(pred_noise, noise)
        
        return loss
    
    def sample(self, state: torch.Tensor, n_steps: int = None) -> torch.Tensor:
        """Sample via reverse diffusion."""
        if n_steps is None:
            n_steps = self.n_timesteps
        
        x = torch.randn(state.shape[0], self.command_dim, device=state.device)
        
        for t in reversed(range(n_steps)):
            t_batch = torch.full((state.shape[0],), t, device=state.device)
            
            # Predict noise
            pred_noise = self.forward(x, t_batch, state)
            
            # Compute mean
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            mean = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_noise
            )
            
            # Add noise (except last step)
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta) * noise
            else:
                x = mean
        
        return x
    
    def is_ood(self, state: torch.Tensor, command: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Check OOD via reconstruction error."""
        # Add noise and denoise
        t = self.n_timesteps // 2
        t_batch = torch.full((state.shape[0],), t, device=state.device)
        
        noise = torch.randn_like(command)
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        noisy = sqrt_alpha * command + sqrt_one_minus_alpha * noise
        
        # Denoise
        reconstructed = self.sample(state)
        
        # Reconstruction error
        error = ((reconstructed - command) ** 2).sum(dim=-1)
        
        return error > threshold
    
    def project(self, state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """Project by sampling from conditional distribution."""
        return self.sample(state)


# =============================================================================
# Training Functions
# =============================================================================

def train_udrl(
    policy: UDRLPolicy,
    data: Dict[str, np.ndarray],
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = "cuda",
) -> List[float]:
    """Train UDRL policy with behavior cloning."""
    
    # Prepare data
    states = torch.tensor(data["states"], dtype=torch.float32)
    actions = torch.tensor(data["actions"], dtype=torch.float32)
    returns = torch.tensor(data["returns"], dtype=torch.float32)
    horizons = torch.tensor(data["horizons"], dtype=torch.float32)
    
    # Normalize commands
    commands = torch.stack([horizons, returns], dim=-1)
    cmd_mean = commands.mean(dim=0)
    cmd_std = commands.std(dim=0) + 1e-6
    commands = (commands - cmd_mean) / cmd_std
    
    # Create dataset and dataloader
    dataset = TensorDataset(states, actions, commands)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    policy = policy.to(device)
    optimizer = AdamW(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    losses = []
    best_val_loss = float("inf")
    
    print("\n=== Training UDRL Policy ===")
    
    for epoch in range(n_epochs):
        # Training
        policy.train()
        train_loss = 0
        
        for states_b, actions_b, commands_b in train_loader:
            states_b = states_b.to(device)
            actions_b = actions_b.to(device)
            commands_b = commands_b.to(device)
            
            optimizer.zero_grad()
            
            mean, log_std = policy(states_b, commands_b)
            std = torch.exp(log_std)
            
            # MSE loss on mean
            loss = F.mse_loss(torch.tanh(mean), actions_b)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        losses.append(train_loss)
        
        # Validation
        policy.eval()
        val_loss = 0
        with torch.no_grad():
            for states_b, actions_b, commands_b in val_loader:
                states_b = states_b.to(device)
                actions_b = actions_b.to(device)
                commands_b = commands_b.to(device)
                
                mean, log_std = policy(states_b, commands_b)
                loss = F.mse_loss(torch.tanh(mean), actions_b)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    return losses


def train_shield(
    shield: nn.Module,
    data: Dict[str, np.ndarray],
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-4,
    device: str = "cuda",
) -> List[float]:
    """Train a safety shield."""
    
    # Prepare data
    states = torch.tensor(data["states"], dtype=torch.float32)
    returns = torch.tensor(data["returns"], dtype=torch.float32)
    horizons = torch.tensor(data["horizons"], dtype=torch.float32)
    
    # Normalize commands
    commands = torch.stack([horizons, returns], dim=-1)
    cmd_mean = commands.mean(dim=0)
    cmd_std = commands.std(dim=0) + 1e-6
    commands = (commands - cmd_mean) / cmd_std
    
    # Store normalization params
    shield.register_buffer("cmd_mean", cmd_mean)
    shield.register_buffer("cmd_std", cmd_std)
    
    # Create dataset
    dataset = TensorDataset(states, commands)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    shield = shield.to(device)
    optimizer = AdamW(shield.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    losses = []
    
    shield_name = shield.__class__.__name__
    print(f"\n=== Training {shield_name} ===")
    
    for epoch in range(n_epochs):
        # Training
        shield.train()
        train_loss = 0
        
        for states_b, commands_b in train_loader:
            states_b = states_b.to(device)
            commands_b = commands_b.to(device)
            
            optimizer.zero_grad()
            loss = shield.compute_loss(states_b, commands_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(shield.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        losses.append(train_loss)
        
        scheduler.step()
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Loss: {train_loss:.4f}")
    
    return losses


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_policy(
    policy: UDRLPolicy,
    env,
    commands: List[Tuple[float, float]],
    shield: Optional[nn.Module] = None,
    n_episodes: int = 10,
    device: str = "cuda",
    threshold: float = -6.0,
) -> Dict[str, float]:
    """Evaluate policy with optional shield."""
    
    policy.eval()
    if shield is not None:
        shield.eval()
    
    all_returns = []
    all_lengths = []
    all_crashes = []
    ood_detections = 0
    projections = 0
    
    for h, r in commands:
        for ep in range(n_episodes):
            state = env.reset()
            
            # Normalize command (using stored stats if available)
            cmd = torch.tensor([[h, r]], dtype=torch.float32, device=device)
            if hasattr(shield, "cmd_mean"):
                cmd = (cmd - shield.cmd_mean.to(device)) / shield.cmd_std.to(device)
            
            original_cmd = cmd.clone()
            
            # Check OOD and project if needed
            if shield is not None:
                state_t = torch.tensor([state], dtype=torch.float32, device=device)
                
                if isinstance(shield, FlowMatchingShield):
                    is_ood = shield.is_ood(state_t, cmd, threshold=threshold)
                else:
                    is_ood = shield.is_ood(state_t, cmd)
                
                if is_ood.any():
                    ood_detections += 1
                    cmd = shield.project(state_t, cmd)
                    projections += 1
            
            episode_return = 0
            episode_length = 0
            crashed = False
            
            for step in range(env.max_steps):
                state_t = torch.tensor([state], dtype=torch.float32, device=device)
                
                with torch.no_grad():
                    action = policy.sample(state_t, cmd, deterministic=True)
                
                action_np = action.cpu().numpy()[0]
                next_state, reward, done, info = env.step(action_np)
                
                episode_return += reward
                episode_length += 1
                state = next_state
                
                if done:
                    if reward < -50:  # Crash
                        crashed = True
                    break
            
            all_returns.append(episode_return)
            all_lengths.append(episode_length)
            all_crashes.append(crashed)
    
    results = {
        "mean_return": np.mean(all_returns),
        "std_return": np.std(all_returns),
        "mean_length": np.mean(all_lengths),
        "success_rate": np.mean([r > 100 for r in all_returns]),
        "crash_rate": np.mean(all_crashes),
        "ood_detections": ood_detections,
        "projections": projections,
    }
    
    return results


def run_comparison(
    policy: UDRLPolicy,
    shields: Dict[str, nn.Module],
    env,
    test_commands: List[Tuple[float, float]],
    n_episodes: int = 10,
    device: str = "cuda",
) -> Dict[str, Dict[str, float]]:
    """Compare different shield methods."""
    
    results = {}
    
    # No shield baseline
    print("\nEvaluating without shield...")
    results["No Shield"] = evaluate_policy(
        policy, env, test_commands, shield=None,
        n_episodes=n_episodes, device=device,
    )
    
    # Each shield
    for name, shield in shields.items():
        print(f"Evaluating with {name}...")
        results[name] = evaluate_policy(
            policy, env, test_commands, shield=shield,
            n_episodes=n_episodes, device=device,
        )
    
    return results


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_training_curves(
    losses_dict: Dict[str, List[float]],
    save_path: Optional[str] = None,
):
    """Plot training loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, losses in losses_dict.items():
        ax.plot(losses, label=name, linewidth=2)
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss Curves", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_comparison_results(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
):
    """Plot comparison bar charts."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = list(results.keys())
    x = np.arange(len(methods))
    width = 0.6
    
    # Return comparison
    returns = [results[m]["mean_return"] for m in methods]
    stds = [results[m]["std_return"] for m in methods]
    colors = ["#FF6B6B" if m == "No Shield" else "#4ECDC4" if "Flow" in m else "#45B7D1" 
              for m in methods]
    
    axes[0].bar(x, returns, width, yerr=stds, color=colors, capsize=5, alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=15, ha="right")
    axes[0].set_ylabel("Mean Episode Return")
    axes[0].set_title("Return Comparison")
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    
    # Success rate
    success = [results[m]["success_rate"] * 100 for m in methods]
    axes[1].bar(x, success, width, color=colors, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=15, ha="right")
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].set_title("Success Rate Comparison")
    axes[1].set_ylim(0, 100)
    
    # Crash rate
    crash = [results[m]["crash_rate"] * 100 for m in methods]
    axes[2].bar(x, crash, width, color=colors, alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, rotation=15, ha="right")
    axes[2].set_ylabel("Crash Rate (%)")
    axes[2].set_title("Crash Rate Comparison (Lower is Better)")
    axes[2].set_ylim(0, 100)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_command_distribution(
    data: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
):
    """Plot command distribution from training data."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    horizons = data["horizons"]
    returns = data["returns"]
    
    # Horizon distribution
    axes[0].hist(horizons, bins=50, color="#4ECDC4", alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Horizon (steps)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Horizon Distribution")
    
    # Return distribution
    axes[1].hist(returns, bins=50, color="#FF6B6B", alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Return")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Return Distribution")
    
    # Joint distribution
    h = axes[2].hist2d(horizons, returns, bins=30, cmap="viridis")
    axes[2].set_xlabel("Horizon")
    axes[2].set_ylabel("Return")
    axes[2].set_title("Joint Command Distribution")
    plt.colorbar(h[3], ax=axes[2], label="Count")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_ood_detection(
    shield: FlowMatchingShield,
    data: Dict[str, np.ndarray],
    device: str = "cuda",
    save_path: Optional[str] = None,
):
    """Visualize OOD detection boundaries."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Sample some states
    n_samples = 100
    indices = np.random.choice(len(data["states"]), n_samples, replace=False)
    states = torch.tensor(data["states"][indices], dtype=torch.float32, device=device)
    
    # Create grid of commands
    h_range = np.linspace(0, 500, 50)
    r_range = np.linspace(-200, 300, 50)
    H, R = np.meshgrid(h_range, r_range)
    
    # Use first state for visualization
    state = states[0:1]
    
    log_probs = np.zeros_like(H)
    
    shield.eval()
    with torch.no_grad():
        for i in range(len(h_range)):
            for j in range(len(r_range)):
                cmd = torch.tensor([[H[j, i], R[j, i]]], dtype=torch.float32, device=device)
                if hasattr(shield, "cmd_mean"):
                    cmd = (cmd - shield.cmd_mean.to(device)) / shield.cmd_std.to(device)
                
                try:
                    log_p = shield.log_prob(state, cmd, n_steps=20)
                    log_probs[j, i] = log_p.cpu().numpy()[0]
                except:
                    log_probs[j, i] = -10
    
    # Log probability heatmap
    im = axes[0].contourf(H, R, log_probs, levels=20, cmap="RdYlGn")
    axes[0].set_xlabel("Horizon")
    axes[0].set_ylabel("Return")
    axes[0].set_title("Log Probability p(g|s)")
    plt.colorbar(im, ax=axes[0], label="Log p(g|s)")
    
    # Mark training data
    axes[0].scatter(data["horizons"][indices], data["returns"][indices],
                   c="blue", s=5, alpha=0.5, label="Training data")
    axes[0].legend()
    
    # OOD boundary
    threshold = -6.0
    ood_mask = log_probs < threshold
    axes[1].contourf(H, R, ood_mask.astype(float), levels=[0, 0.5, 1],
                    colors=["#90EE90", "#FF6B6B"], alpha=0.5)
    axes[1].contour(H, R, log_probs, levels=[threshold], colors=["black"], linewidths=2)
    axes[1].set_xlabel("Horizon")
    axes[1].set_ylabel("Return")
    axes[1].set_title(f"OOD Detection (threshold={threshold})")
    axes[1].scatter(data["horizons"][indices], data["returns"][indices],
                   c="blue", s=5, alpha=0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#90EE90", alpha=0.5, label="In-distribution"),
        Patch(facecolor="#FF6B6B", alpha=0.5, label="Out-of-distribution"),
    ]
    axes[1].legend(handles=legend_elements)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_results_table(results: Dict[str, Dict[str, float]]):
    """Print results as formatted table."""
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    headers = ["Method", "Return", "Success %", "Crash %", "OOD Det.", "Projections"]
    row_format = "{:<20} {:>12} {:>12} {:>12} {:>12} {:>12}"
    
    print(row_format.format(*headers))
    print("-" * 80)
    
    for method, metrics in results.items():
        row = [
            method,
            f"{metrics['mean_return']:.1f} ± {metrics['std_return']:.1f}",
            f"{metrics['success_rate']*100:.1f}%",
            f"{metrics['crash_rate']*100:.1f}%",
            str(metrics['ood_detections']),
            str(metrics['projections']),
        ]
        print(row_format.format(*row))
    
    print("=" * 80)


# =============================================================================
# Main Pipeline
# =============================================================================

def run_full_pipeline(
    env_name: str = "lunarlander",
    n_episodes: int = 2000,
    n_epochs_policy: int = 100,
    n_epochs_shield: int = 80,
    batch_size: int = 256,
    seed: int = 42,
    quick: bool = False,
    output_dir: str = "results",
):
    """Run the complete FlowShield-UDRL pipeline."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / f"{env_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Quick mode for testing
    if quick:
        n_episodes = 500
        n_epochs_policy = 30
        n_epochs_shield = 30
        print("QUICK MODE: Using reduced settings for faster testing")
    
    print("\n" + "=" * 60)
    print("FlowShield-UDRL Complete Pipeline")
    print("=" * 60)
    print(f"Environment: {env_name}")
    print(f"Episodes: {n_episodes}")
    print(f"Device: {DEVICE}")
    print(f"Output: {run_dir}")
    print("=" * 60)
    
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # ==========================================================================
    # Step 1: Create Environment
    # ==========================================================================
    print("\n[1/6] Creating environment...")
    env = make_env(env_name)
    state_dim = env.state_dim
    action_dim = env.action_dim
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # ==========================================================================
    # Step 2: Collect Data
    # ==========================================================================
    print("\n[2/6] Collecting training data...")
    data = collect_trajectories(env, n_episodes=n_episodes, seed=seed)
    
    # Save data
    np.savez(run_dir / "data.npz", **data)
    
    # Plot command distribution
    print("\nPlotting command distribution...")
    plot_command_distribution(data, save_path=run_dir / "command_distribution.png")
    
    # ==========================================================================
    # Step 3: Train UDRL Policy
    # ==========================================================================
    print("\n[3/6] Training UDRL policy...")
    policy = UDRLPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        n_layers=4,
    )
    
    policy_losses = train_udrl(
        policy, data,
        n_epochs=n_epochs_policy,
        batch_size=batch_size,
        device=str(DEVICE),
    )
    
    torch.save(policy.state_dict(), run_dir / "policy.pt")
    
    # ==========================================================================
    # Step 4: Train Safety Shields
    # ==========================================================================
    print("\n[4/6] Training safety shields...")
    
    shields = {}
    shield_losses = {}
    
    # Quantile Shield
    print("\nTraining Quantile Shield...")
    quantile_shield = QuantileShield(
        state_dim=state_dim,
        hidden_dim=256,
        tau=0.9,
    )
    shield_losses["Quantile"] = train_shield(
        quantile_shield, data,
        n_epochs=n_epochs_shield,
        device=str(DEVICE),
    )
    shields["Quantile"] = quantile_shield
    torch.save(quantile_shield.state_dict(), run_dir / "quantile_shield.pt")
    
    # Flow Matching Shield
    print("\nTraining Flow Matching Shield...")
    flow_shield = FlowMatchingShield(
        state_dim=state_dim,
        hidden_dim=256,
        n_layers=4,
    )
    shield_losses["Flow Matching"] = train_shield(
        flow_shield, data,
        n_epochs=n_epochs_shield,
        device=str(DEVICE),
    )
    shields["Flow Matching"] = flow_shield
    torch.save(flow_shield.state_dict(), run_dir / "flow_shield.pt")
    
    # Diffusion Shield
    print("\nTraining Diffusion Shield...")
    diffusion_shield = DiffusionShield(
        state_dim=state_dim,
        hidden_dim=256,
        n_timesteps=50 if quick else 100,
    )
    shield_losses["Diffusion"] = train_shield(
        diffusion_shield, data,
        n_epochs=n_epochs_shield,
        device=str(DEVICE),
    )
    shields["Diffusion"] = diffusion_shield
    torch.save(diffusion_shield.state_dict(), run_dir / "diffusion_shield.pt")
    
    # Plot training curves
    print("\nPlotting training curves...")
    all_losses = {"UDRL Policy": policy_losses, **shield_losses}
    plot_training_curves(all_losses, save_path=run_dir / "training_curves.png")
    
    # ==========================================================================
    # Step 5: Evaluate and Compare
    # ==========================================================================
    print("\n[5/6] Evaluating and comparing methods...")
    
    # Define test commands (mix of ID and OOD)
    # Based on training data statistics
    h_mean = data["horizons"].mean()
    h_std = data["horizons"].std()
    r_mean = data["returns"].mean()
    r_std = data["returns"].std()
    
    test_commands = [
        # In-distribution
        (h_mean, r_mean),
        (h_mean - h_std, r_mean),
        (h_mean + h_std, r_mean + r_std),
        # Mild OOD
        (h_mean - 2*h_std, r_mean + 2*r_std),
        (50, r_mean + 1.5*r_std),
        # Strong OOD
        (20, 200),  # Very short horizon, high return
        (30, 250),  # Extremely ambitious
        (h_mean, r_mean + 3*r_std),  # Very high return
    ]
    
    print(f"\nTest commands (mix of ID and OOD):")
    for i, (h, r) in enumerate(test_commands):
        print(f"  {i+1}. H={h:.0f}, R={r:.1f}")
    
    # Run comparison
    results = run_comparison(
        policy, shields, env,
        test_commands=test_commands,
        n_episodes=5 if quick else 10,
        device=str(DEVICE),
    )
    
    # Print results table
    print_results_table(results)
    
    # Save results
    with open(run_dir / "results.json", "w") as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
    
    # Plot comparison
    print("\nPlotting comparison results...")
    plot_comparison_results(results, save_path=run_dir / "comparison.png")
    
    # ==========================================================================
    # Step 6: Visualize OOD Detection
    # ==========================================================================
    print("\n[6/6] Visualizing OOD detection...")
    try:
        plot_ood_detection(flow_shield, data, device=str(DEVICE), 
                          save_path=run_dir / "ood_detection.png")
    except Exception as e:
        print(f"Warning: Could not plot OOD detection: {e}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {run_dir}")
    print(f"\nFiles generated:")
    for f in sorted(run_dir.iterdir()):
        print(f"  - {f.name}")
    
    print("\n=== KEY FINDINGS ===")
    
    # Find best method
    best_method = max(results.keys(), key=lambda k: results[k]["mean_return"])
    worst_crash = max(results.keys(), key=lambda k: results[k]["crash_rate"])
    
    print(f"Best method by return: {best_method} ({results[best_method]['mean_return']:.1f})")
    print(f"Worst crash rate: {worst_crash} ({results[worst_crash]['crash_rate']*100:.1f}%)")
    
    if "Flow Matching" in results and "No Shield" in results:
        improvement = results["Flow Matching"]["mean_return"] - results["No Shield"]["mean_return"]
        crash_reduction = results["No Shield"]["crash_rate"] - results["Flow Matching"]["crash_rate"]
        print(f"\nFlow Matching improvement over No Shield:")
        print(f"  Return: +{improvement:.1f}")
        print(f"  Crash rate reduction: {crash_reduction*100:.1f}%")
    
    return results, run_dir


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlowShield-UDRL Pipeline")
    parser.add_argument("--env", type=str, default="lunarlander", help="Environment name")
    parser.add_argument("--n-episodes", type=int, default=2000, help="Number of episodes to collect")
    parser.add_argument("--n-epochs-policy", type=int, default=100, help="Policy training epochs")
    parser.add_argument("--n-epochs-shield", type=int, default=80, help="Shield training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick mode with reduced settings")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    results, run_dir = run_full_pipeline(
        env_name=args.env,
        n_episodes=args.n_episodes,
        n_epochs_policy=args.n_epochs_policy,
        n_epochs_shield=args.n_epochs_shield,
        batch_size=args.batch_size,
        seed=args.seed,
        quick=args.quick,
        output_dir=args.output_dir,
    )
