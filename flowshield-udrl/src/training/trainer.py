"""
Base trainer class with common training utilities.
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from ..utils.io import CheckpointManager
from ..utils.logging import MetricsLogger


class BaseTrainer(ABC):
    """
    Abstract base class for trainers.
    
    Provides common functionality:
    - Training loop structure
    - Logging and metrics
    - Checkpointing
    - Early stopping
    - Learning rate scheduling
    
    Args:
        model: Neural network model
        optimizer: Optimizer
        scheduler: Optional LR scheduler
        device: Device to train on
        logger: Optional metrics logger (W&B)
        checkpoint_manager: Optional checkpoint manager
        log_interval: Steps between logging
        eval_interval: Steps between evaluation
        checkpoint_interval: Steps between checkpoints
        
    Example:
        >>> class MyTrainer(BaseTrainer):
        ...     def train_step(self, batch): ...
        ...     def evaluate(self, dataloader): ...
        >>> trainer = MyTrainer(model, optimizer)
        >>> trainer.train(train_loader, n_epochs=100)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: str = "cuda",
        logger: Optional[MetricsLogger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        log_interval: int = 100,
        eval_interval: int = 1000,
        checkpoint_interval: int = 5000,
        max_grad_norm: Optional[float] = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval
        self.max_grad_norm = max_grad_norm
        
        # Move model to device
        self.model.to(self.device)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float("-inf")
        self.early_stop = False
        
        # Metrics tracking
        self.train_metrics: Dict[str, List[float]] = {}
        self.eval_metrics: Dict[str, List[float]] = {}
    
    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Dictionary of metrics
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            dataloader: Evaluation dataloader
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    def train(
        self,
        train_loader: DataLoader,
        n_epochs: int,
        eval_loader: Optional[DataLoader] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_metric: str = "loss",
        minimize_metric: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training dataloader
            n_epochs: Number of training epochs
            eval_loader: Optional evaluation dataloader
            early_stopping_patience: Epochs without improvement before stopping
            early_stopping_metric: Metric to monitor for early stopping
            minimize_metric: Whether to minimize (True) or maximize (False)
            
        Returns:
            Training history dictionary
        """
        patience_counter = 0
        best_metric_value = float("inf") if minimize_metric else float("-inf")
        
        print(f"Starting training for {n_epochs} epochs on {self.device}")
        print(f"Train batches: {len(train_loader)}")
        if eval_loader:
            print(f"Eval batches: {len(eval_loader)}")
        
        for epoch in range(n_epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            # Training epoch
            epoch_metrics = self._train_epoch(train_loader)
            
            epoch_time = time.time() - epoch_start
            
            # Logging
            print(f"Epoch {epoch + 1}/{n_epochs} ({epoch_time:.1f}s) - ", end="")
            for key, value in epoch_metrics.items():
                print(f"{key}: {value:.4f} ", end="")
            print()
            
            if self.logger:
                self.logger.log({f"train/{k}": v for k, v in epoch_metrics.items()})
                self.logger.log({"epoch": epoch, "epoch_time": epoch_time})
            
            # Evaluation
            if eval_loader and (epoch + 1) % (self.eval_interval // len(train_loader) + 1) == 0:
                eval_metrics = self.evaluate(eval_loader)
                
                print(f"  Eval - ", end="")
                for key, value in eval_metrics.items():
                    print(f"{key}: {value:.4f} ", end="")
                print()
                
                if self.logger:
                    self.logger.log({f"eval/{k}": v for k, v in eval_metrics.items()})
                
                # Early stopping check
                if early_stopping_patience:
                    metric_value = eval_metrics.get(early_stopping_metric, epoch_metrics.get(early_stopping_metric, 0))
                    
                    improved = (
                        metric_value < best_metric_value if minimize_metric
                        else metric_value > best_metric_value
                    )
                    
                    if improved:
                        best_metric_value = metric_value
                        patience_counter = 0
                        
                        # Save best model
                        if self.checkpoint_manager:
                            self.checkpoint_manager.save_checkpoint(
                                {
                                    "model": self.model.state_dict(),
                                    "optimizer": self.optimizer.state_dict(),
                                    "epoch": epoch,
                                    "global_step": self.global_step,
                                    "best_metric": best_metric_value,
                                },
                                filename="best_model.pt",
                            )
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(f"Early stopping after {epoch + 1} epochs")
                            break
            
            # Checkpointing
            if self.checkpoint_manager and (epoch + 1) % (self.checkpoint_interval // len(train_loader) + 1) == 0:
                self.checkpoint_manager.save_checkpoint(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                        "epoch": epoch,
                        "global_step": self.global_step,
                    },
                    filename=f"checkpoint_epoch{epoch + 1}.pt",
                )
        
        return self.train_metrics
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_metrics: Dict[str, float] = {}
        n_batches = 0
        
        for batch in train_loader:
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Training step
            metrics = self.train_step(batch)
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value
            
            epoch_loss += metrics.get("loss", 0.0)
            n_batches += 1
            self.global_step += 1
            
            # Step-level logging
            if self.global_step % self.log_interval == 0:
                if self.logger:
                    self.logger.log({f"step/{k}": v for k, v in metrics.items()})
                    self.logger.log({"step": self.global_step})
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
        
        # Store history
        for key, value in epoch_metrics.items():
            if key not in self.train_metrics:
                self.train_metrics[key] = []
            self.train_metrics[key].append(value)
        
        # Learning rate step
        if self.scheduler:
            self.scheduler.step()
            if self.logger:
                self.logger.log({"lr": self.scheduler.get_last_lr()[0]})
        
        return epoch_metrics
    
    def _move_batch_to_device(self, batch: Any) -> Any:
        """Move batch to device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self._move_batch_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._move_batch_to_device(x) for x in batch)
        else:
            return batch
    
    def clip_gradients(self) -> Optional[float]:
        """Clip gradients and return norm."""
        if self.max_grad_norm:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
            )
            return grad_norm.item()
        return None
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        if "scheduler" in checkpoint and checkpoint["scheduler"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        
        self.epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        
        print(f"Loaded checkpoint from epoch {self.epoch}, step {self.global_step}")
    
    def save_model(self, path: str) -> None:
        """Save only model weights."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
