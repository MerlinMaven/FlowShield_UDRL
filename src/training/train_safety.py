"""
Safety module trainers.

Trainers for:
- Flow Matching safety module
- Quantile regression baseline
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from ..models.safety import FlowMatchingSafetyModule, QuantileSafetyModule
from ..utils.io import CheckpointManager
from ..utils.logging import MetricsLogger
from .trainer import BaseTrainer


class SafetyTrainer(BaseTrainer):
    """
    Base trainer for safety modules.
    
    Handles common training logic for modules that learn p(g|s).
    """
    
    def __init__(
        self,
        safety_module: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: str = "cuda",
        logger: Optional[MetricsLogger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        log_interval: int = 100,
        max_grad_norm: float = 1.0,
    ):
        super().__init__(
            model=safety_module,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            checkpoint_manager=checkpoint_manager,
            log_interval=log_interval,
            max_grad_norm=max_grad_norm,
        )


class FlowMatchingTrainer(SafetyTrainer):
    """
    Trainer for Flow Matching safety module.
    
    Trains the vector field network using the conditional flow matching
    objective. The resulting model can estimate log p(g|s) and sample
    from the distribution.
    
    Args:
        safety_module: FlowMatchingSafetyModule to train
        optimizer: Optimizer
        scheduler: Optional LR scheduler
        ema_decay: Exponential moving average decay for model weights
        n_samples_eval: Samples for density evaluation
        
    Example:
        >>> safety = FlowMatchingSafetyModule(state_dim=8)
        >>> optimizer = torch.optim.AdamW(safety.parameters(), lr=1e-4)
        >>> trainer = FlowMatchingTrainer(safety, optimizer)
        >>> trainer.train(command_loader, n_epochs=100)
    """
    
    def __init__(
        self,
        safety_module: FlowMatchingSafetyModule,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: str = "cuda",
        logger: Optional[MetricsLogger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        ema_decay: float = 0.999,
        n_samples_eval: int = 100,
        log_interval: int = 100,
        max_grad_norm: float = 1.0,
    ):
        super().__init__(
            safety_module=safety_module,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            checkpoint_manager=checkpoint_manager,
            log_interval=log_interval,
            max_grad_norm=max_grad_norm,
        )
        
        self.safety_module = safety_module
        self.ema_decay = ema_decay
        self.n_samples_eval = n_samples_eval
        
        # EMA model
        if ema_decay > 0:
            self.ema_model = self._create_ema_model()
        else:
            self.ema_model = None
    
    def _create_ema_model(self) -> FlowMatchingSafetyModule:
        """Create EMA copy of model."""
        ema = FlowMatchingSafetyModule(
            state_dim=self.safety_module.state_dim,
            command_dim=self.safety_module.command_dim,
        ).to(self.device)
        ema.load_state_dict(self.safety_module.state_dict())
        for param in ema.parameters():
            param.requires_grad = False
        return ema
    
    def _update_ema(self) -> None:
        """Update EMA model weights."""
        if self.ema_model is None:
            return
        
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(),
                self.safety_module.parameters(),
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    param.data, alpha=1 - self.ema_decay
                )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single flow matching training step.
        
        Batch should contain:
        - state: (batch, state_dim)
        - command: (batch, 2)
        """
        self.optimizer.zero_grad()
        
        state = batch["state"]
        command = batch["command"]
        
        # Compute flow matching loss
        loss_dict = self.safety_module.compute_loss(state, command)
        loss = loss_dict["loss"]
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        grad_norm = self.clip_gradients()
        
        # Optimizer step
        self.optimizer.step()
        
        # Update EMA
        self._update_ema()
        
        metrics = {
            "loss": loss.item(),
            "velocity_norm": loss_dict.get("velocity_norm", torch.tensor(0.0)).item(),
        }
        
        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm
        
        return metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate flow matching quality."""
        model = self.ema_model if self.ema_model else self.safety_module
        model.eval()
        
        total_loss = 0.0
        total_log_prob = 0.0
        n_batches = 0
        
        for batch in dataloader:
            batch = self._move_batch_to_device(batch)
            state = batch["state"]
            command = batch["command"]
            
            # Flow matching loss
            loss_dict = model.compute_loss(state, command)
            total_loss += loss_dict["loss"].item()
            
            # Log probability (expensive - subsample)
            if n_batches < 10:  # Only compute for first few batches
                log_prob = model.get_safety_score(state[:32], command[:32])
                total_log_prob += log_prob.mean().item()
            
            n_batches += 1
        
        model.train()
        
        return {
            "loss": total_loss / n_batches,
            "mean_log_prob": total_log_prob / min(n_batches, 10),
        }
    
    def evaluate_ood_detection(
        self,
        in_distribution_loader: DataLoader,
        out_of_distribution_commands: torch.Tensor,
        states: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate OOD detection performance.
        
        Args:
            in_distribution_loader: Loader with in-distribution data
            out_of_distribution_commands: Known OOD commands
            states: States to condition on
            
        Returns:
            OOD detection metrics (AUROC, AUPRC, etc.)
        """
        model = self.ema_model if self.ema_model else self.safety_module
        model.eval()
        
        # Collect in-distribution scores
        id_scores = []
        for batch in in_distribution_loader:
            batch = self._move_batch_to_device(batch)
            state = batch["state"][:32]
            command = batch["command"][:32]
            
            score = model.get_safety_score(state, command)
            id_scores.append(score)
        
        id_scores = torch.cat(id_scores)
        
        # OOD scores
        states = states.to(self.device)
        ood_commands = out_of_distribution_commands.to(self.device)
        ood_scores = model.get_safety_score(states, ood_commands)
        
        # Compute metrics
        threshold = id_scores.quantile(0.05).item()  # 5th percentile of ID
        
        # Detection rate
        ood_detected = (ood_scores < threshold).float().mean().item()
        id_false_positive = (id_scores < threshold).float().mean().item()
        
        # AUROC (simple approximation)
        all_scores = torch.cat([id_scores, ood_scores])
        all_labels = torch.cat([
            torch.ones_like(id_scores),
            torch.zeros_like(ood_scores),
        ])
        
        # Sort by score
        sorted_idx = all_scores.argsort(descending=True)
        sorted_labels = all_labels[sorted_idx]
        
        # ROC-AUC approximation
        tpr_at_fpr_5 = (sorted_labels[:int(0.95 * len(sorted_labels))]).mean().item()
        
        model.train()
        
        return {
            "ood_detection_rate": ood_detected,
            "false_positive_rate": id_false_positive,
            "tpr_at_fpr_5pct": tpr_at_fpr_5,
            "id_score_mean": id_scores.mean().item(),
            "ood_score_mean": ood_scores.mean().item(),
            "score_gap": id_scores.mean().item() - ood_scores.mean().item(),
        }


class QuantileTrainer(SafetyTrainer):
    """
    Trainer for Quantile Regression safety module.
    
    Trains quantile networks using the pinball loss to estimate
    quantiles of the return distribution.
    
    Args:
        safety_module: QuantileSafetyModule to train
        optimizer: Optimizer
        tau: Quantile level to train (default from module)
        
    Example:
        >>> safety = QuantileSafetyModule(state_dim=8, tau=0.9)
        >>> optimizer = torch.optim.Adam(safety.parameters(), lr=1e-4)
        >>> trainer = QuantileTrainer(safety, optimizer)
        >>> trainer.train(loader, n_epochs=50)
    """
    
    def __init__(
        self,
        safety_module: QuantileSafetyModule,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: str = "cuda",
        logger: Optional[MetricsLogger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        tau: Optional[float] = None,
        log_interval: int = 100,
        max_grad_norm: float = 1.0,
    ):
        super().__init__(
            safety_module=safety_module,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            checkpoint_manager=checkpoint_manager,
            log_interval=log_interval,
            max_grad_norm=max_grad_norm,
        )
        
        self.safety_module = safety_module
        self.tau = tau or safety_module.tau
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single quantile regression training step.
        
        Batch should contain:
        - state: (batch, state_dim)
        - command: (batch, 2) or just return values
        """
        self.optimizer.zero_grad()
        
        state = batch["state"]
        
        # Extract return values
        if "return" in batch:
            target_return = batch["return"]
        elif "command" in batch:
            target_return = batch["command"][:, 1]  # Second element is return
        else:
            raise ValueError("Batch must contain 'return' or 'command'")
        
        # Compute quantile loss
        loss_dict = self.safety_module.compute_loss(state, target_return, tau=self.tau)
        loss = loss_dict["loss"]
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        grad_norm = self.clip_gradients()
        
        # Optimizer step
        self.optimizer.step()
        
        metrics = {
            "loss": loss.item(),
            "loss_upper": loss_dict["loss_upper"].item(),
            "loss_lower": loss_dict["loss_lower"].item(),
            "pred_upper_mean": loss_dict["pred_upper_mean"].item(),
            "pred_lower_mean": loss_dict["pred_lower_mean"].item(),
        }
        
        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm
        
        return metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate quantile predictions."""
        self.safety_module.eval()
        
        total_loss = 0.0
        all_preds_upper = []
        all_preds_lower = []
        all_targets = []
        n_batches = 0
        
        for batch in dataloader:
            batch = self._move_batch_to_device(batch)
            state = batch["state"]
            
            if "return" in batch:
                target_return = batch["return"]
            else:
                target_return = batch["command"][:, 1]
            
            loss_dict = self.safety_module.compute_loss(state, target_return)
            total_loss += loss_dict["loss"].item()
            
            # Get predictions for coverage analysis
            lower, upper = self.safety_module.get_return_bounds(state)
            all_preds_upper.append(upper)
            all_preds_lower.append(lower)
            all_targets.append(target_return)
            
            n_batches += 1
        
        # Compute coverage
        all_preds_upper = torch.cat(all_preds_upper)
        all_preds_lower = torch.cat(all_preds_lower)
        all_targets = torch.cat(all_targets)
        
        coverage = ((all_targets >= all_preds_lower) & (all_targets <= all_preds_upper)).float().mean()
        interval_width = (all_preds_upper - all_preds_lower).mean()
        
        self.safety_module.train()
        
        return {
            "loss": total_loss / n_batches,
            "coverage": coverage.item(),
            "interval_width": interval_width.item(),
        }
    
    def update_dataset_statistics(
        self,
        dataloader: DataLoader,
    ) -> None:
        """Update safety module with dataset statistics."""
        all_returns = []
        all_horizons = []
        
        for batch in dataloader:
            if "return" in batch:
                all_returns.append(batch["return"])
            elif "command" in batch:
                all_returns.append(batch["command"][:, 1])
                all_horizons.append(batch["command"][:, 0])
        
        returns = torch.cat(all_returns)
        horizons = torch.cat(all_horizons) if all_horizons else None
        
        self.safety_module.update_statistics(returns, horizons)
