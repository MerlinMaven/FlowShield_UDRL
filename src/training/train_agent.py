"""
UDRL Agent trainer.

Trains the command-conditioned policy π(a|s,g) using supervised learning
on expert demonstrations or collected trajectories.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from ..models.agent import UDRLPolicy
# UDRLPolicyWithValue removed - use separate value network if needed
from ..models.safety import BaseSafetyModule
from ..utils.io import CheckpointManager
from ..utils.logging import MetricsLogger
from .trainer import BaseTrainer


class UDRLTrainer(BaseTrainer):
    """
    Trainer for UDRL (Upside-Down RL) policy.
    
    Trains the policy to predict actions given (state, command) pairs.
    Optionally integrates safety module for command filtering.
    
    Args:
        policy: UDRL policy network
        optimizer: Optimizer
        scheduler: Optional LR scheduler
        safety_module: Optional safety module for command filtering
        device: Device to train on
        logger: Optional W&B logger
        use_value_head: Whether to train with value predictions
        value_loss_coef: Coefficient for value loss
        entropy_coef: Coefficient for entropy bonus
        label_smoothing: Label smoothing for classification
        
    Example:
        >>> policy = UDRLPolicy(state_dim=8, action_dim=4)
        >>> optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        >>> trainer = UDRLTrainer(policy, optimizer)
        >>> trainer.train(train_loader, n_epochs=100)
    """
    
    def __init__(
        self,
        policy: UDRLPolicy,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        safety_module: Optional[BaseSafetyModule] = None,
        device: str = "cuda",
        logger: Optional[MetricsLogger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        use_value_head: bool = False,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        label_smoothing: float = 0.0,
        command_noise_std: float = 0.0,
        log_interval: int = 100,
        eval_interval: int = 1000,
        max_grad_norm: float = 1.0,
    ):
        super().__init__(
            model=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            checkpoint_manager=checkpoint_manager,
            log_interval=log_interval,
            eval_interval=eval_interval,
            max_grad_norm=max_grad_norm,
        )
        
        self.policy = policy
        self.safety_module = safety_module
        self.use_value_head = False  # Value head removed from current implementation
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.label_smoothing = label_smoothing
        self.command_noise_std = command_noise_std
        
        if self.safety_module:
            self.safety_module.to(self.device)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
        
        Batch should contain:
        - state: (batch, state_dim)
        - action: (batch,) for discrete or (batch, action_dim) for continuous
        - command: (batch, 2) with [horizon, return]
        - return_to_go: (batch,) optional, for value prediction
        """
        self.optimizer.zero_grad()
        
        state = batch["state"]
        action = batch["action"]
        command = batch["command"]
        
        # Optional: Apply safety filter to commands
        if self.safety_module is not None:
            with torch.no_grad():
                command, _ = self.safety_module(state, command)
        
        # Optional: Add noise to commands for regularization
        if self.training and self.command_noise_std > 0:
            command = command + torch.randn_like(command) * self.command_noise_std
        
        # Compute policy loss
        loss_dict = self.policy.get_loss(state, command, action)
        policy_loss = loss_dict["loss"]
        
        total_loss = policy_loss
        
        # Value loss (if applicable)
        value_loss = torch.tensor(0.0, device=self.device)
        if self.use_value_head and "return_to_go" in batch:
            return_to_go = batch["return_to_go"]
            value_pred = self.policy.get_value(state, command)
            value_loss = F.mse_loss(value_pred.squeeze(-1), return_to_go)
            total_loss = total_loss + self.value_loss_coef * value_loss
        
        # Entropy bonus
        if self.entropy_coef > 0:
            entropy = loss_dict.get("entropy", torch.tensor(0.0))
            total_loss = total_loss - self.entropy_coef * entropy
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        grad_norm = self.clip_gradients()
        
        # Optimizer step
        self.optimizer.step()
        
        # Prepare metrics
        metrics = {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
        }
        
        if self.use_value_head:
            metrics["value_loss"] = value_loss.item()
        
        if "entropy" in loss_dict:
            metrics["entropy"] = loss_dict["entropy"].item()
        
        if "accuracy" in loss_dict:
            metrics["accuracy"] = loss_dict["accuracy"].item()
        
        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm
        
        return metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_value_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            batch = self._move_batch_to_device(batch)
            
            state = batch["state"]
            action = batch["action"]
            command = batch["command"]
            
            # Compute policy loss
            loss_dict = self.policy.get_loss(state, command, action)
            total_loss += loss_dict["loss"].item()
            
            if "accuracy" in loss_dict:
                total_accuracy += loss_dict["accuracy"].item()
            
            # Value loss
            if self.use_value_head and "return_to_go" in batch:
                value_pred = self.policy.get_value(state, command)
                value_loss = F.mse_loss(value_pred.squeeze(-1), batch["return_to_go"])
                total_value_loss += value_loss.item()
            
            n_batches += 1
        
        self.model.train()
        
        metrics = {
            "loss": total_loss / n_batches,
            "accuracy": total_accuracy / n_batches if total_accuracy > 0 else 0.0,
        }
        
        if self.use_value_head:
            metrics["value_loss"] = total_value_loss / n_batches
        
        return metrics
    
    def evaluate_environment(
        self,
        env,
        command: torch.Tensor,
        n_episodes: int = 10,
        max_steps: int = 1000,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate policy in environment.
        
        Args:
            env: Gymnasium environment
            command: Command to use (horizon, return)
            n_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode
            deterministic: Use deterministic actions
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        total_return = 0.0
        total_length = 0
        successes = 0
        
        command = command.to(self.device)
        
        for episode in range(n_episodes):
            state, info = env.reset()
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            episode_return = 0.0
            current_command = command.clone()
            
            for step in range(max_steps):
                # Apply safety filter
                if self.safety_module is not None:
                    current_command, _ = self.safety_module(state, current_command)
                
                # Get action
                action = self.policy.get_action(
                    state, current_command,
                    deterministic=deterministic,
                )
                
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                
                # Environment step
                next_state, reward, terminated, truncated, info = env.step(
                    action.squeeze() if action.ndim > 0 else action
                )
                
                episode_return += reward
                done = terminated or truncated
                
                if done:
                    if info.get("success", False) or (terminated and reward > 0):
                        successes += 1
                    break
                
                state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                
                # Update command (decrement horizon, adjust return)
                current_command[0, 0] = max(1, current_command[0, 0] - 1)
                current_command[0, 1] = current_command[0, 1] - reward
            
            total_return += episode_return
            total_length += step + 1
        
        self.model.train()
        
        return {
            "mean_return": total_return / n_episodes,
            "mean_length": total_length / n_episodes,
            "success_rate": successes / n_episodes,
        }


class UDRLHindsightTrainer(UDRLTrainer):
    """
    UDRL trainer with hindsight command relabeling.
    
    Uses the actual achieved outcomes to create additional training
    data with correctly-matched commands.
    
    Args:
        relabeling_strategy: "future", "episode", or "final"
        relabeling_fraction: Fraction of batch to relabel
    """
    
    def __init__(
        self,
        policy: UDRLPolicy,
        optimizer: Optimizer,
        relabeling_strategy: str = "future",
        relabeling_fraction: float = 0.5,
        **kwargs,
    ):
        super().__init__(policy, optimizer, **kwargs)
        self.relabeling_strategy = relabeling_strategy
        self.relabeling_fraction = relabeling_fraction
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step with hindsight relabeling."""
        # Relabel a fraction of commands with actual outcomes
        if "return_to_go" in batch and "timesteps_to_go" in batch:
            batch = self._relabel_commands(batch)
        
        return super().train_step(batch)
    
    def _relabel_commands(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Relabel commands with actual achieved outcomes."""
        batch_size = batch["state"].shape[0]
        n_relabel = int(batch_size * self.relabeling_fraction)
        
        if n_relabel == 0:
            return batch
        
        # Select samples to relabel
        indices = torch.randperm(batch_size)[:n_relabel]
        
        # Create new commands from actual outcomes
        new_command = batch["command"].clone()
        
        # Use actual return-to-go and timesteps-to-go as the command
        new_command[indices, 0] = batch["timesteps_to_go"][indices]
        new_command[indices, 1] = batch["return_to_go"][indices]
        
        batch["command"] = new_command
        
        return batch
