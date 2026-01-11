"""
Multi-Layer Perceptron (MLP) building blocks.

Provides flexible MLP architectures with various activation functions,
normalization options, and residual connections.
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """
    Get activation function by name.
    
    Args:
        name: Activation name ("relu", "gelu", "tanh", "silu", "leaky_relu")
        
    Returns:
        Activation module
    """
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "silu": nn.SiLU(),
        "swish": nn.SiLU(),
        "leaky_relu": nn.LeakyReLU(0.1),
        "elu": nn.ELU(),
        "softplus": nn.Softplus(),
        "mish": nn.Mish(),
        "none": nn.Identity(),
    }
    
    name = name.lower()
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    
    return activations[name]


def create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    activation: str = "relu",
    output_activation: str = "none",
    dropout: float = 0.0,
    layer_norm: bool = False,
    batch_norm: bool = False,
    bias: bool = True,
) -> nn.Sequential:
    """
    Create a simple MLP.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: List of hidden layer dimensions
        activation: Activation function name
        output_activation: Activation for output layer
        dropout: Dropout probability
        layer_norm: Whether to use layer normalization
        batch_norm: Whether to use batch normalization
        bias: Whether to use bias in linear layers
        
    Returns:
        nn.Sequential MLP
        
    Example:
        >>> mlp = create_mlp(10, 5, [64, 64], activation="relu")
        >>> x = torch.randn(32, 10)
        >>> y = mlp(x)  # shape: (32, 5)
    """
    layers = []
    dims = [input_dim] + hidden_dims
    
    for i in range(len(dims) - 1):
        # Linear layer
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
        
        # Normalization
        if batch_norm:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
        elif layer_norm:
            layers.append(nn.LayerNorm(dims[i + 1]))
        
        # Activation
        layers.append(get_activation(activation))
        
        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    
    # Output layer
    layers.append(nn.Linear(dims[-1], output_dim, bias=bias))
    
    # Output activation
    if output_activation != "none":
        layers.append(get_activation(output_activation))
    
    return nn.Sequential(*layers)


class MLP(nn.Module):
    """
    Configurable Multi-Layer Perceptron.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: List of hidden layer dimensions
        activation: Activation function name
        output_activation: Activation for output layer
        dropout: Dropout probability
        layer_norm: Whether to use layer normalization
        
    Example:
        >>> mlp = MLP(10, 5, [64, 64])
        >>> x = torch.randn(32, 10)
        >>> y = mlp(x)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Union[List[int], int] = 256,
        n_layers: int = 2,
        activation: str = "relu",
        output_activation: str = "none",
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        
        # Handle hidden_dims specification
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * n_layers
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        self.net = create_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            output_activation=output_activation,
            dropout=dropout,
            layer_norm=layer_norm,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


class ResidualBlock(nn.Module):
    """
    Residual MLP block.
    
    Implements: output = activation(LayerNorm(x + MLP(x)))
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        
        hidden_dim = hidden_dim or dim * 2
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        
        self.norm = nn.LayerNorm(dim)
        self.activation = get_activation(activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        return self.activation(self.norm(x + self.net(x)))


class ResidualMLP(nn.Module):
    """
    MLP with residual connections.
    
    Better gradient flow for deeper networks.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dim: Hidden dimension (constant across layers)
        n_layers: Number of residual blocks
        activation: Activation function
        dropout: Dropout probability
        
    Example:
        >>> mlp = ResidualMLP(10, 5, hidden_dim=64, n_layers=4)
        >>> x = torch.randn(32, 10)
        >>> y = mlp(x)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, activation=activation, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
        
        return self.output_proj(x)


class ConditionalMLP(nn.Module):
    """
    MLP conditioned on an additional input.
    
    Useful for conditioning on state, time, or commands.
    
    Args:
        input_dim: Main input dimension
        condition_dim: Condition input dimension
        output_dim: Output dimension
        hidden_dim: Hidden dimension
        n_layers: Number of layers
        condition_method: How to incorporate condition ("concat", "film", "add")
        
    Example:
        >>> mlp = ConditionalMLP(10, 5, 3, hidden_dim=64)
        >>> x = torch.randn(32, 10)
        >>> c = torch.randn(32, 5)
        >>> y = mlp(x, c)
    """
    
    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        condition_method: str = "concat",
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.condition_method = condition_method
        
        if condition_method == "concat":
            # Concatenate input and condition
            self.net = MLP(
                input_dim=input_dim + condition_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dim,
                n_layers=n_layers,
                activation=activation,
                dropout=dropout,
            )
        elif condition_method == "add":
            # Project both to same dim and add
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.cond_proj = nn.Linear(condition_dim, hidden_dim)
            self.net = MLP(
                input_dim=hidden_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dim,
                n_layers=n_layers - 1,
                activation=activation,
                dropout=dropout,
            )
        elif condition_method == "film":
            # FiLM conditioning (scale and shift)
            self.net = FiLMConditionedMLP(
                input_dim=input_dim,
                condition_dim=condition_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                activation=activation,
            )
        else:
            raise ValueError(f"Unknown condition method: {condition_method}")
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass with condition."""
        if self.condition_method == "concat":
            combined = torch.cat([x, condition], dim=-1)
            return self.net(combined)
        elif self.condition_method == "add":
            x_proj = self.input_proj(x)
            c_proj = self.cond_proj(condition)
            return self.net(x_proj + c_proj)
        else:  # film
            return self.net(x, condition)


class FiLMConditionedMLP(nn.Module):
    """
    MLP with FiLM (Feature-wise Linear Modulation) conditioning.
    
    Each layer is modulated by condition: y = gamma * x + beta
    where gamma and beta are predicted from the condition.
    """
    
    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        activation: str = "relu",
    ):
        super().__init__()
        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layers - 1)
        ])
        
        # FiLM generators (one for each layer)
        self.film_generators = nn.ModuleList([
            nn.Linear(condition_dim, 2 * hidden_dim)  # gamma and beta
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self.activation = get_activation(activation)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward with FiLM conditioning."""
        # Input projection with first FiLM
        h = self.input_proj(x)
        film_params = self.film_generators[0](condition)
        gamma, beta = film_params.chunk(2, dim=-1)
        h = gamma * h + beta
        h = self.activation(h)
        
        # Hidden layers with FiLM
        for i, layer in enumerate(self.layers):
            h = layer(h)
            film_params = self.film_generators[i + 1](condition)
            gamma, beta = film_params.chunk(2, dim=-1)
            h = gamma * h + beta
            h = self.activation(h)
        
        return self.output_proj(h)
