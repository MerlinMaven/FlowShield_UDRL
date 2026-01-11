"""
Embedding modules for neural networks.

Provides:
- SinusoidalTimeEmbedding: Time embeddings for diffusion/flow models
- PositionalEncoding: Standard positional encodings
- StateEmbedding: State/observation embeddings
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding as used in diffusion models.
    
    Maps scalar time t ∈ [0, 1] to high-dimensional embedding.
    
    Args:
        dim: Output embedding dimension
        max_period: Maximum period for sinusoidal functions
        
    Example:
        >>> emb = SinusoidalTimeEmbedding(64)
        >>> t = torch.rand(32)  # (batch,)
        >>> e = emb(t)  # (batch, 64)
    """
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        
        self.dim = dim
        self.max_period = max_period
        
        # Optional learned projection
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute time embedding.
        
        Args:
            t: Time tensor of shape (batch,) or (batch, 1)
            
        Returns:
            Embedding of shape (batch, dim)
        """
        if t.dim() == 2:
            t = t.squeeze(-1)
        
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * 
            torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim
        )
        
        # (batch,) x (half_dim,) -> (batch, half_dim)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        
        # Concatenate sin and cos
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        # Handle odd dimensions
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        # Project through MLP
        return self.proj(embedding)


class PositionalEncoding(nn.Module):
    """
    Standard positional encoding for sequence models.
    
    Adds positional information to input embeddings.
    
    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
        
    Example:
        >>> pe = PositionalEncoding(64, max_len=100)
        >>> x = torch.randn(32, 50, 64)  # (batch, seq, dim)
        >>> y = pe(x)  # (batch, seq, dim)
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class StateEmbedding(nn.Module):
    """
    Learnable state/observation embedding.
    
    Projects state observations to an embedding space with optional
    normalization and dropout.
    
    Args:
        state_dim: Input state dimension
        embed_dim: Output embedding dimension
        hidden_dim: Hidden layer dimension (if None, no hidden layer)
        layer_norm: Whether to apply layer normalization
        dropout: Dropout probability
        
    Example:
        >>> emb = StateEmbedding(8, 64, hidden_dim=128)
        >>> s = torch.randn(32, 8)
        >>> e = emb(s)  # (32, 64)
    """
    
    def __init__(
        self,
        state_dim: int,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        
        if hidden_dim is not None:
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, embed_dim),
            )
        else:
            self.net = nn.Linear(state_dim, embed_dim)
        
        self.norm = nn.LayerNorm(embed_dim) if layer_norm else nn.Identity()
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute state embedding."""
        return self.norm(self.net(state))


class FourierFeatures(nn.Module):
    """
    Random Fourier features for positional encoding.
    
    Maps low-dimensional inputs to high-dimensional space using
    random sinusoidal projections.
    
    Args:
        input_dim: Input dimension
        mapping_size: Output dimension (number of features)
        scale: Scale of random frequencies
        
    Example:
        >>> ff = FourierFeatures(2, 128)
        >>> x = torch.randn(32, 2)
        >>> f = ff(x)  # (32, 256) - sin and cos concatenated
    """
    
    def __init__(
        self,
        input_dim: int,
        mapping_size: int = 256,
        scale: float = 10.0,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        
        # Random frequency matrix (not learned)
        B = torch.randn(input_dim, mapping_size) * scale
        self.register_buffer('B', B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Fourier features.
        
        Args:
            x: Input tensor (..., input_dim)
            
        Returns:
            Fourier features (..., 2 * mapping_size)
        """
        # Project to frequency space
        x_proj = 2 * math.pi * x @ self.B
        
        # Return sin and cos
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class CommandEmbedding(nn.Module):
    """
    Embedding for UDRL commands (horizon, return).
    
    Processes the command tuple and projects to embedding space.
    Can use different strategies for horizon vs return.
    
    Args:
        embed_dim: Output embedding dimension
        horizon_embedding: Type of horizon embedding ("linear", "sinusoidal")
        return_embedding: Type of return embedding ("linear", "fourier")
        max_horizon: Maximum horizon for normalization
        return_scale: Scale for return normalization
        
    Example:
        >>> emb = CommandEmbedding(64)
        >>> cmd = torch.tensor([[10, 5.0], [20, -3.0]])  # (batch, 2)
        >>> e = emb(cmd)  # (batch, 64)
    """
    
    def __init__(
        self,
        embed_dim: int,
        horizon_embedding: str = "sinusoidal",
        return_embedding: str = "linear",
        max_horizon: int = 100,
        return_scale: float = 10.0,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_horizon = max_horizon
        self.return_scale = return_scale
        
        half_dim = embed_dim // 2
        
        # Horizon embedding
        if horizon_embedding == "sinusoidal":
            self.horizon_emb = SinusoidalTimeEmbedding(half_dim)
        else:
            self.horizon_emb = nn.Sequential(
                nn.Linear(1, half_dim * 2),
                nn.GELU(),
                nn.Linear(half_dim * 2, half_dim),
            )
        
        # Return embedding
        if return_embedding == "fourier":
            self.return_emb = nn.Sequential(
                FourierFeatures(1, half_dim // 2),
                nn.Linear(half_dim, half_dim),
            )
        else:
            self.return_emb = nn.Sequential(
                nn.Linear(1, half_dim * 2),
                nn.GELU(),
                nn.Linear(half_dim * 2, half_dim),
            )
        
        # Final projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, command: torch.Tensor) -> torch.Tensor:
        """
        Compute command embedding.
        
        Args:
            command: Command tensor (batch, 2) with [horizon, return]
            
        Returns:
            Embedding (batch, embed_dim)
        """
        horizon = command[:, 0:1] / self.max_horizon  # Normalize
        ret = command[:, 1:2] / self.return_scale  # Normalize
        
        # Get embeddings
        if hasattr(self.horizon_emb, 'proj'):  # SinusoidalTimeEmbedding
            h_emb = self.horizon_emb(horizon.squeeze(-1))
        else:
            h_emb = self.horizon_emb(horizon)
        
        r_emb = self.return_emb(ret)
        
        # Concatenate and project
        combined = torch.cat([h_emb, r_emb], dim=-1)
        return self.norm(self.proj(combined))
