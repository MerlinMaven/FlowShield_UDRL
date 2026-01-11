"""
Command Projector utilities.

Provides various projection methods for moving out-of-distribution
commands to the safe (in-distribution) region.
"""

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn


class CommandProjector:
    """
    Utility class for projecting commands to safe regions.
    
    Provides multiple projection strategies:
    1. Gradient ascent: Move towards higher density
    2. Bisection: Binary search between original and sampled safe command
    3. Boundary: Project to distribution boundary
    4. Nearest: Find nearest command in dataset
    
    Args:
        safety_fn: Function returning safety score (higher = safer)
        threshold: Safety threshold
        method: Projection method
        max_steps: Maximum projection steps
        step_size: Step size for gradient methods
        tolerance: Convergence tolerance
        
    Example:
        >>> projector = CommandProjector(
        ...     safety_fn=safety_module.get_safety_score,
        ...     threshold=-5.0,
        ... )
        >>> safe_cmd = projector.project(state, command)
    """
    
    def __init__(
        self,
        safety_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        threshold: float = 0.0,
        method: str = "gradient",
        max_steps: int = 100,
        step_size: float = 0.1,
        tolerance: float = 1e-4,
    ):
        self.safety_fn = safety_fn
        self.threshold = threshold
        self.method = method
        self.max_steps = max_steps
        self.step_size = step_size
        self.tolerance = tolerance
    
    def project(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
        safe_reference: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Project commands to safe region.
        
        Args:
            state: Conditioning state
            command: Original command to project
            safe_reference: Optional known-safe command for bisection
            
        Returns:
            Projected command
        """
        if self.method == "gradient":
            return self._gradient_projection(state, command)
        elif self.method == "bisection":
            if safe_reference is None:
                raise ValueError("Bisection requires safe_reference")
            return self._bisection_projection(state, command, safe_reference)
        elif self.method == "boundary":
            return self._boundary_projection(state, command)
        else:
            raise ValueError(f"Unknown projection method: {self.method}")
    
    def _gradient_projection(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gradient ascent on safety score.
        """
        g = command.clone().requires_grad_(True)
        
        for step in range(self.max_steps):
            score = self.safety_fn(state, g)
            
            # Check convergence
            if (score > self.threshold).all():
                break
            
            # Gradient step
            grad = torch.autograd.grad(score.sum(), g)[0]
            
            with torch.no_grad():
                g_new = g + self.step_size * grad
                
                # Check improvement
                score_new = self.safety_fn(state, g_new)
                
                # Line search (simple backtracking)
                lr = self.step_size
                for _ in range(5):
                    if score_new.mean() >= score.mean():
                        break
                    lr *= 0.5
                    g_new = g + lr * grad
                    score_new = self.safety_fn(state, g_new)
                
                g = g_new.requires_grad_(True)
        
        return g.detach()
    
    def _bisection_projection(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
        safe_reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Binary search between unsafe command and safe reference.
        """
        # Start with midpoint
        low = torch.zeros(command.shape[0], device=command.device)
        high = torch.ones(command.shape[0], device=command.device)
        
        for step in range(self.max_steps):
            mid = (low + high) / 2
            mid_expanded = mid.unsqueeze(-1)
            
            # Interpolate
            g = command * (1 - mid_expanded) + safe_reference * mid_expanded
            
            # Check safety
            score = self.safety_fn(state, g)
            is_safe = score > self.threshold
            
            # Update bounds
            high = torch.where(is_safe, mid, high)
            low = torch.where(is_safe, low, mid)
            
            # Check convergence
            if (high - low).max() < self.tolerance:
                break
        
        # Use the safe side
        final_alpha = high.unsqueeze(-1)
        return command * (1 - final_alpha) + safe_reference * final_alpha
    
    def _boundary_projection(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project to boundary of safe region.
        
        First finds the direction towards safe region, then
        moves command exactly to the boundary.
        """
        # Get gradient direction
        g = command.clone().requires_grad_(True)
        score = self.safety_fn(state, g)
        grad = torch.autograd.grad(score.sum(), g)[0]
        
        # Normalize direction
        direction = grad / (grad.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Binary search for boundary
        low = torch.zeros(command.shape[0], device=command.device)
        high = torch.ones(command.shape[0], device=command.device) * 10.0  # Max distance
        
        for step in range(self.max_steps):
            mid = (low + high) / 2
            mid_expanded = mid.unsqueeze(-1)
            
            # Move in gradient direction
            g = command + direction * mid_expanded
            
            # Check safety
            score = self.safety_fn(state, g)
            is_safe = score > self.threshold
            
            # Update bounds
            high = torch.where(is_safe, mid, high)
            low = torch.where(is_safe, low, mid)
            
            if (high - low).max() < self.tolerance:
                break
        
        # Use boundary point
        final_dist = high.unsqueeze(-1)
        return command + direction * final_dist


class ClampingProjector:
    """
    Simple clamping-based projector.
    
    Clips commands to given bounds. Used for baseline comparison.
    
    Args:
        lower_bound: Lower bound for each command dimension
        upper_bound: Upper bound for each command dimension
    """
    
    def __init__(
        self,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
    ):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def project(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        """Clamp command to bounds."""
        return torch.clamp(
            command,
            min=self.lower_bound,
            max=self.upper_bound,
        )
    
    def is_safe(self, command: torch.Tensor) -> torch.Tensor:
        """Check if command is within bounds."""
        in_lower = (command >= self.lower_bound).all(dim=-1)
        in_upper = (command <= self.upper_bound).all(dim=-1)
        return in_lower & in_upper


class AdaptiveProjector:
    """
    Projector with state-dependent bounds.
    
    Combines a base projector with learned state-dependent adjustments.
    """
    
    def __init__(
        self,
        base_projector: CommandProjector,
        bound_network: Optional[nn.Module] = None,
    ):
        self.base_projector = base_projector
        self.bound_network = bound_network
    
    def project(
        self,
        state: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        """Project with state-dependent adjustments."""
        if self.bound_network is not None:
            # Get state-dependent bounds
            bounds = self.bound_network(state)
            lower, upper = bounds.chunk(2, dim=-1)
            
            # First clamp to state-dependent bounds
            command = torch.clamp(command, lower, upper)
        
        # Then apply base projection
        return self.base_projector.project(state, command)
