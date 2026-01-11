"""
Attention mechanisms for neural networks.

Optional components for more expressive architectures.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        
    Example:
        >>> attn = MultiHeadAttention(64, n_heads=8)
        >>> x = torch.randn(32, 10, 64)  # (batch, seq, dim)
        >>> y, weights = attn(x)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        if return_attention:
            return output, attention_weights
        return output, None


class CrossAttention(nn.Module):
    """
    Cross-attention for conditioning on external context.
    
    Args:
        d_model: Query model dimension
        d_context: Context dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        
    Example:
        >>> attn = CrossAttention(64, d_context=32)
        >>> x = torch.randn(32, 10, 64)  # Query
        >>> ctx = torch.randn(32, 5, 32)  # Context
        >>> y, _ = attn(x, ctx)
    """
    
    def __init__(
        self,
        d_model: int,
        d_context: Optional[int] = None,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        d_context = d_context or d_model
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_context, d_model)
        self.W_v = nn.Linear(d_context, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Query tensor (batch, seq_len, d_model)
            context: Context tensor (batch, ctx_len, d_context)
            mask: Optional attention mask
            
        Returns:
            Output tensor and attention weights
        """
        batch_size, seq_len, _ = x.shape
        ctx_len = context.shape[1]
        
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(context).view(batch_size, ctx_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(context).view(batch_size, ctx_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        return output, attention_weights


class TransformerBlock(nn.Module):
    """
    Standard Transformer encoder block.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        
    Example:
        >>> block = TransformerBlock(64, n_heads=4, d_ff=256)
        >>> x = torch.randn(32, 10, 64)
        >>> y = block(x)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        d_ff = d_ff or d_model * 4
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-norm architecture."""
        # Self-attention with residual
        attended, _ = self.attention(self.norm1(x), mask)
        x = x + attended
        
        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))
        
        return x
