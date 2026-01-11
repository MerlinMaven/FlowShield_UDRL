"""
UDRL Policy: π(a | s, g)

Upside-Down Reinforcement Learning policy that takes a state and command
(horizon, target_return) and outputs an action distribution.

Architecture:
- StateEncoder: MLP to encode the observation
- CommandEncoder: MLP to encode the command (horizon, return)
- Fusion: Combine encodings (add, concat, or FiLM)
- ActionHead: Output action distribution (discrete or continuous)
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from ..components.mlp import MLP, create_mlp, get_activation


class UDRLPolicy(nn.Module):
    """
    Command-conditioned policy for Upside-Down RL.
    
    Learns to map (state, command) -> action, where command = (horizon, return).
    Trained via behavioral cloning on offline data.
    
    Args:
        state_dim: Dimension of state observations
        action_dim: Number of discrete actions OR dimension of continuous actions
        command_dim: Dimension of command (default: 2 for horizon + return)
        hidden_dim: Size of hidden layers
        n_layers: Number of hidden layers
        activation: Activation function name
        continuous: Whether actions are continuous
        fusion_method: How to combine state and command ("add", "concat", "film")
        dropout: Dropout probability
        log_std_min: Min log std for continuous actions
        log_std_max: Max log std for continuous actions
        
    Example:
        >>> policy = UDRLPolicy(state_dim=8, action_dim=4, continuous=False)
        >>> state = torch.randn(32, 8)
        >>> command = torch.randn(32, 2)
        >>> action = policy.get_action(state, command)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        command_dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 3,
        activation: str = "relu",
        continuous: bool = False,
        fusion_method: str = "concat",
        dropout: float = 0.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.command_dim = command_dim
        self.hidden_dim = hidden_dim
        self.continuous = continuous
        self.fusion_method = fusion_method
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # State encoder
        self.state_encoder = MLP(
            input_dim=state_dim,
            output_dim=hidden_dim,
            hidden_dims=hidden_dim,
            n_layers=max(1, n_layers // 2),
            activation=activation,
            dropout=dropout,
        )
        
        # Command encoder
        self.command_encoder = MLP(
            input_dim=command_dim,
            output_dim=hidden_dim,
            hidden_dims=hidden_dim,
            n_layers=max(1, n_layers // 2),
            activation=activation,
            dropout=dropout,
        )
        
        # Fusion and action head
        if fusion_method == "concat":
            fusion_dim = hidden_dim * 2
        elif fusion_method == "add":
            fusion_dim = hidden_dim
        elif fusion_method == "film":
            fusion_dim = hidden_dim
            # FiLM: command generates scale and shift for state features
            self.film_generator = nn.Linear(hidden_dim, hidden_dim * 2)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Action network
        if continuous:
            # Output mean and log_std for Gaussian policy
            self.action_net = MLP(
                input_dim=fusion_dim,
                output_dim=action_dim * 2,  # mean and log_std
                hidden_dims=hidden_dim,
                n_layers=max(1, n_layers // 2),
                activation=activation,
                dropout=dropout,
            )
        else:
            # Output logits for categorical
            self.action_net = MLP(
                input_dim=fusion_dim,
                output_dim=action_dim,
                hidden_dims=hidden_dim,
                n_layers=max(1, n_layers // 2),
                activation=activation,
                dropout=dropout,
            )
    
    def _fuse(
        self,
        state_emb: torch.Tensor,
        command_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse state and command embeddings."""
        if self.fusion_method == "concat":
            return torch.cat([state_emb, command_emb], dim=-1)
        elif self.fusion_method == "add":
            return state_emb + command_emb
        elif self.fusion_method == "film":
            film_params = self.film_generator(command_emb)
            gamma, beta = film_params.chunk(2, dim=-1)
            return gamma * state_emb + beta
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def forward(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            state: State tensor (batch, state_dim)
            command: Command tensor (batch, command_dim)
            
        Returns:
            If discrete: logits (batch, action_dim)
            If continuous: tuple of (mean, log_std) each (batch, action_dim)
        """
        # Encode
        state_emb = self.state_encoder(state)
        command_emb = self.command_encoder(command)
        
        # Fuse
        fused = self._fuse(state_emb, command_emb)
        
        # Action output
        action_output = self.action_net(fused)
        
        if self.continuous:
            mean, log_std = action_output.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            return mean, log_std
        else:
            return action_output  # logits
    
    def get_action(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Sample an action from the policy.
        
        Args:
            state: State tensor (batch, state_dim) or (state_dim,)
            command: Command tensor (batch, command_dim) or (command_dim,)
            deterministic: If True, return mode of distribution
            
        Returns:
            Action tensor
        """
        # Handle single state
        squeeze = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            command = command.unsqueeze(0)
            squeeze = True
        
        if self.continuous:
            mean, log_std = self.forward(state, command)
            
            if deterministic:
                action = mean
            else:
                std = log_std.exp()
                dist = Normal(mean, std)
                action = dist.sample()
            
            # Squash to [-1, 1] using tanh
            action = torch.tanh(action)
        else:
            logits = self.forward(state, command)
            
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                dist = Categorical(logits=logits)
                action = dist.sample()
        
        if squeeze:
            action = action.squeeze(0)
        
        return action
    
    def get_distribution(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
    ) -> Union[Categorical, Normal]:
        """
        Get the action distribution.
        
        Args:
            state: State tensor
            command: Command tensor
            
        Returns:
            Categorical or Normal distribution
        """
        if self.continuous:
            mean, log_std = self.forward(state, command)
            return Normal(mean, log_std.exp())
        else:
            logits = self.forward(state, command)
            return Categorical(logits=logits)
    
    def log_prob(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of actions.
        
        Args:
            state: State tensor (batch, state_dim)
            command: Command tensor (batch, command_dim)
            action: Action tensor (batch,) or (batch, action_dim)
            
        Returns:
            Log probability tensor (batch,)
        """
        if self.continuous:
            mean, log_std = self.forward(state, command)
            std = log_std.exp()
            
            # For tanh-squashed actions, need to account for change of variables
            # Assuming action is already in [-1, 1]
            
            # Inverse tanh to get unsquashed action
            # Clamp to avoid numerical issues at boundaries
            action_clamped = torch.clamp(action, -0.999, 0.999)
            unsquashed = torch.atanh(action_clamped)
            
            # Gaussian log prob
            dist = Normal(mean, std)
            log_prob = dist.log_prob(unsquashed).sum(dim=-1)
            
            # Jacobian correction for tanh
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
            
            return log_prob
        else:
            logits = self.forward(state, command)
            dist = Categorical(logits=logits)
            return dist.log_prob(action)
    
    def entropy(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute entropy of action distribution.
        
        Args:
            state: State tensor
            command: Command tensor
            
        Returns:
            Entropy tensor (batch,)
        """
        dist = self.get_distribution(state, command)
        return dist.entropy()
    
    def get_loss(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
        action: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute behavioral cloning loss (negative log-likelihood).
        
        Args:
            state: State tensor (batch, state_dim)
            command: Command tensor (batch, command_dim)
            action: Target action tensor
            reduction: "mean", "sum", or "none"
            
        Returns:
            Loss tensor
        """
        log_prob = self.log_prob(state, command, action)
        nll = -log_prob
        
        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        else:
            return nll


class UDRLPolicyWithValue(UDRLPolicy):
    """
    UDRL Policy with optional value function head.
    
    Can be used for policy evaluation or actor-critic style training.
    
    Args:
        All args from UDRLPolicy, plus:
        value_hidden_dim: Hidden dim for value network
        value_n_layers: Number of layers for value network
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        command_dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 3,
        activation: str = "relu",
        continuous: bool = False,
        fusion_method: str = "concat",
        dropout: float = 0.0,
        value_hidden_dim: int = 256,
        value_n_layers: int = 2,
        **kwargs,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            command_dim=command_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
            continuous=continuous,
            fusion_method=fusion_method,
            dropout=dropout,
            **kwargs,
        )
        
        # Value network (takes state and command)
        if fusion_method == "concat":
            value_input_dim = hidden_dim * 2
        else:
            value_input_dim = hidden_dim
        
        self.value_net = MLP(
            input_dim=value_input_dim,
            output_dim=1,
            hidden_dims=value_hidden_dim,
            n_layers=value_n_layers,
            activation=activation,
        )
    
    def get_value(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate value for state-command pair.
        
        Args:
            state: State tensor
            command: Command tensor
            
        Returns:
            Value estimate (batch, 1)
        """
        state_emb = self.state_encoder(state)
        command_emb = self.command_encoder(command)
        fused = self._fuse(state_emb, command_emb)
        return self.value_net(fused)
    
    def forward_with_value(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
    ) -> Tuple[Union[torch.Tensor, Tuple], torch.Tensor]:
        """
        Forward pass returning both action output and value.
        
        Returns:
            Tuple of (action_output, value)
        """
        state_emb = self.state_encoder(state)
        command_emb = self.command_encoder(command)
        fused = self._fuse(state_emb, command_emb)
        
        action_output = self.action_net(fused)
        value = self.value_net(fused)
        
        if self.continuous:
            mean, log_std = action_output.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            return (mean, log_std), value
        else:
            return action_output, value
