"""
Abstract base class for safety modules.

Defines the interface that all safety modules must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class BaseSafetyModule(ABC, nn.Module):
    """
    Abstract interface for safety modules.
    
    Each safety module must be able to:
    1. Evaluate if a command is "safe" (in-distribution)
    2. Project an unsafe command to the safe region
    3. Provide detailed information about safety status
    
    The safety module learns the distribution p(g|s) of achievable
    commands from a given state, and uses this to detect and correct
    out-of-distribution (OOD) commands.
    """
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def is_safe(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check if commands are within the safe (in-distribution) region.
        
        Args:
            state: State tensor (batch, state_dim)
            command: Command tensor (batch, command_dim)
            
        Returns:
            Boolean tensor (batch,) indicating if each command is safe
        """
        pass
    
    @abstractmethod
    def project(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project commands to the safe region.
        
        For safe commands, returns the original command.
        For unsafe commands, returns the nearest safe command.
        
        Args:
            state: State tensor (batch, state_dim)
            command: Command tensor (batch, command_dim)
            
        Returns:
            Projected command tensor (batch, command_dim)
        """
        pass
    
    @abstractmethod
    def get_safety_score(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get a continuous safety score for commands.
        
        Higher scores indicate safer (more in-distribution) commands.
        Can be log-probability, quantile position, etc.
        
        Args:
            state: State tensor (batch, state_dim)
            command: Command tensor (batch, command_dim)
            
        Returns:
            Safety score tensor (batch,)
        """
        pass
    
    def get_info(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Get detailed information about command safety.
        
        Args:
            state: State tensor
            command: Command tensor
            
        Returns:
            Dictionary with:
            - is_safe: Boolean tensor
            - safety_score: Continuous score
            - projected_command: Projected command if unsafe
            - projection_distance: Distance moved during projection
        """
        is_safe = self.is_safe(state, command)
        safety_score = self.get_safety_score(state, command)
        projected = self.project(state, command)
        
        # Compute projection distance
        distance = torch.norm(command - projected, dim=-1)
        
        return {
            "is_safe": is_safe,
            "safety_score": safety_score,
            "projected_command": projected,
            "projection_distance": distance,
            "original_command": command,
        }
    
    def __call__(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
        return_info: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Safety shield: check and project commands if needed.
        
        This is the main interface for using the safety module.
        
        Args:
            state: State tensor (batch, state_dim) or (state_dim,)
            command: Command tensor (batch, command_dim) or (command_dim,)
            return_info: Whether to return detailed info
            
        Returns:
            Tuple of (safe_command, info_dict)
        """
        # Handle single state
        squeeze = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            command = command.unsqueeze(0)
            squeeze = True
        
        # Get info (includes projection)
        info = self.get_info(state, command)
        
        # Use projected command
        safe_command = info["projected_command"]
        
        if squeeze:
            safe_command = safe_command.squeeze(0)
            for key in info:
                if isinstance(info[key], torch.Tensor):
                    info[key] = info[key].squeeze(0)
        
        if return_info:
            return safe_command, info
        else:
            return safe_command, {"is_safe": info["is_safe"]}
    
    @abstractmethod
    def sample_commands(
        self,
        state: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """
        Sample achievable commands from p(g|s).
        
        Args:
            state: State tensor (batch, state_dim)
            n_samples: Number of samples per state
            
        Returns:
            Command samples (batch, n_samples, command_dim)
        """
        pass
    
    def get_command_bounds(
        self,
        state: torch.Tensor,
        n_samples: int = 100,
        quantile_low: float = 0.05,
        quantile_high: float = 0.95,
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate bounds on achievable commands for given states.
        
        Uses sampling to estimate quantiles of the command distribution.
        
        Args:
            state: State tensor (batch, state_dim)
            n_samples: Number of samples for estimation
            quantile_low: Lower quantile for bounds
            quantile_high: Upper quantile for bounds
            
        Returns:
            Dictionary with 'lower' and 'upper' bounds
        """
        # Sample commands
        samples = self.sample_commands(state, n_samples)  # (batch, n_samples, dim)
        
        # Compute quantiles
        lower = torch.quantile(samples, quantile_low, dim=1)
        upper = torch.quantile(samples, quantile_high, dim=1)
        
        return {
            "lower": lower,
            "upper": upper,
            "mean": samples.mean(dim=1),
            "std": samples.std(dim=1),
        }
