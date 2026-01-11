"""
Model save/load utilities.

This module provides functions for saving and loading model checkpoints
with metadata, optimizer states, and training progress.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a model checkpoint with optional training state.
    
    Args:
        model: PyTorch model to save
        path: Path for the checkpoint file
        optimizer: Optional optimizer state
        scheduler: Optional learning rate scheduler
        epoch: Current epoch number
        step: Current training step
        metrics: Dictionary of metrics to save
        config: Configuration dictionary
        extra: Any extra data to save
    
    Returns:
        Path to saved checkpoint
    
    Example:
        >>> save_checkpoint(
        ...     model=policy,
        ...     path="checkpoints/epoch_10.pt",
        ...     optimizer=optimizer,
        ...     epoch=10,
        ...     metrics={"loss": 0.1, "accuracy": 0.95}
        ... )
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "metrics": metrics or {},
        "config": config or {},
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if extra is not None:
        checkpoint["extra"] = extra
    
    torch.save(checkpoint, path)
    
    return path


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a model checkpoint.
    
    Args:
        path: Path to the checkpoint file
        model: Optional model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load tensors to
        strict: Whether to strictly enforce state dict keys match
    
    Returns:
        Dictionary with checkpoint contents
    
    Example:
        >>> checkpoint = load_checkpoint(
        ...     "checkpoints/best_model.pt",
        ...     model=policy,
        ...     device="cuda"
        ... )
        >>> print(f"Loaded from epoch {checkpoint['epoch']}")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint


def save_model(
    model: nn.Module,
    path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save just the model weights (for inference).
    
    Args:
        model: PyTorch model to save
        path: Output path
        config: Optional config to save alongside
    
    Returns:
        Path to saved model
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), path)
    
    # Save config as JSON alongside
    if config:
        config_path = path.with_suffix(".json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    return path


def load_model(
    model: nn.Module,
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> nn.Module:
    """
    Load model weights for inference.
    
    Args:
        model: Model instance to load weights into
        path: Path to saved weights
        device: Device to load to
        strict: Whether to strictly enforce key matching
    
    Returns:
        Model with loaded weights
    """
    path = Path(path)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict, strict=strict)
    return model


class CheckpointManager:
    """
    Manager for handling multiple checkpoints with best model tracking.
    
    Example:
        >>> manager = CheckpointManager("checkpoints/", keep_best=3)
        >>> manager.save(model, optimizer, metrics={"val_loss": 0.5}, epoch=1)
        >>> manager.save(model, optimizer, metrics={"val_loss": 0.3}, epoch=2)
        >>> best = manager.load_best(model)
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        keep_best: int = 3,
        keep_last: int = 1,
        metric_name: str = "val_loss",
        mode: str = "min",
    ):
        """
        Initialize CheckpointManager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            keep_best: Number of best checkpoints to keep
            keep_last: Number of most recent checkpoints to keep
            metric_name: Metric to use for determining best
            mode: "min" or "max" (whether lower or higher is better)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_best = keep_best
        self.keep_last = keep_last
        self.metric_name = metric_name
        self.mode = mode
        
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.checkpoints: list = []
        self.best_checkpoints: list = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save a checkpoint and manage best/last tracking.
        
        Returns:
            Path to saved checkpoint
        """
        metrics = metrics or {}
        
        # Create checkpoint filename
        metric_str = ""
        if self.metric_name in metrics:
            metric_str = f"_{self.metric_name}_{metrics[self.metric_name]:.4f}"
        filename = f"checkpoint_epoch_{epoch:04d}{metric_str}.pt"
        path = self.checkpoint_dir / filename
        
        # Save checkpoint
        save_checkpoint(
            model=model,
            path=path,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=step,
            metrics=metrics,
            config=config,
        )
        
        # Track checkpoint
        self.checkpoints.append((path, epoch, metrics))
        
        # Check if best
        if self.metric_name in metrics:
            value = metrics[self.metric_name]
            is_best = (self.mode == "min" and value < self.best_value) or \
                      (self.mode == "max" and value > self.best_value)
            
            if is_best:
                self.best_value = value
                best_path = self.checkpoint_dir / "best_model.pt"
                save_checkpoint(model=model, path=best_path, metrics=metrics, config=config)
            
            self.best_checkpoints.append((path, value))
            self.best_checkpoints.sort(key=lambda x: x[1], reverse=(self.mode == "max"))
        
        # Cleanup old checkpoints
        self._cleanup()
        
        return path
    
    def _cleanup(self) -> None:
        """Remove old checkpoints beyond keep limits."""
        # Keep best
        best_to_keep = set(p for p, _ in self.best_checkpoints[:self.keep_best])
        
        # Keep last
        last_to_keep = set(p for p, _, _ in self.checkpoints[-self.keep_last:])
        
        # Always keep best_model.pt
        best_model_path = self.checkpoint_dir / "best_model.pt"
        
        # Remove others
        for path, _, _ in self.checkpoints[:-self.keep_last]:
            if path not in best_to_keep and path != best_model_path:
                if path.exists():
                    path.unlink()
    
    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> Dict[str, Any]:
        """Load the best checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pt"
        return load_checkpoint(best_path, model, optimizer, device=device)
    
    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> Dict[str, Any]:
        """Load the most recent checkpoint."""
        if not self.checkpoints:
            raise FileNotFoundError("No checkpoints available")
        latest_path = self.checkpoints[-1][0]
        return load_checkpoint(latest_path, model, optimizer, device=device)
