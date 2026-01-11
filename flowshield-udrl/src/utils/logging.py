"""
Logging utilities with Weights & Biases integration.

This module provides:
- Standard Python logging setup
- W&B experiment tracking integration
- Metric logging helpers
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Try to import wandb, but don't fail if not available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_str: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging output
        format_str: Custom format string
    
    Returns:
        Root logger instance
    
    Example:
        >>> logger = setup_logging(level="DEBUG", log_file="logs/experiment.log")
        >>> logger.info("Starting experiment")
    """
    if format_str is None:
        format_str = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Create formatter
    formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")
    """
    return logging.getLogger(name)


def setup_wandb(
    project: str,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
    notes: Optional[str] = None,
    group: Optional[str] = None,
    mode: str = "online",
    dir: Optional[Union[str, Path]] = None,
    reinit: bool = False,
    resume: Optional[str] = None,
) -> Optional[Any]:
    """
    Initialize Weights & Biases run.
    
    Args:
        project: W&B project name
        name: Run name (auto-generated if None)
        config: Configuration dictionary to log
        tags: List of tags for the run
        notes: Notes/description for the run
        group: Group name for organizing runs
        mode: "online", "offline", or "disabled"
        dir: Directory for W&B files
        reinit: Allow reinitializing in the same process
        resume: Resume mode ("allow", "must", "never", run_id)
    
    Returns:
        W&B run object, or None if W&B not available
    
    Example:
        >>> run = setup_wandb(
        ...     project="flowshield-udrl",
        ...     name="lunarlander-flow-matching",
        ...     config={"lr": 0.001, "hidden_dim": 256}
        ... )
    """
    if not WANDB_AVAILABLE:
        logging.warning("wandb not installed. Experiment tracking disabled.")
        return None
    
    # Generate run name if not provided
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"run_{timestamp}"
    
    # Set directory
    if dir:
        dir = str(Path(dir))
    
    try:
        run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            group=group,
            mode=mode,
            dir=dir,
            reinit=reinit,
            resume=resume,
        )
        return run
    except Exception as e:
        logging.warning(f"Failed to initialize wandb: {e}")
        return None


def log_metrics(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    prefix: Optional[str] = None,
) -> None:
    """
    Log metrics to W&B (if available).
    
    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number
        prefix: Optional prefix for all metric names
    
    Example:
        >>> log_metrics({"loss": 0.5, "accuracy": 0.95}, step=100)
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return
    
    # Add prefix if specified
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    
    wandb.log(metrics, step=step)


def log_artifact(
    name: str,
    type: str,
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    aliases: Optional[list] = None,
) -> None:
    """
    Log an artifact to W&B.
    
    Args:
        name: Artifact name
        type: Artifact type (e.g., "model", "dataset")
        path: Path to file or directory
        metadata: Optional metadata dictionary
        aliases: Optional list of aliases
    
    Example:
        >>> log_artifact(
        ...     name="udrl-policy",
        ...     type="model",
        ...     path="checkpoints/best_model.pt"
        ... )
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return
    
    artifact = wandb.Artifact(name=name, type=type, metadata=metadata)
    
    path = Path(path)
    if path.is_dir():
        artifact.add_dir(str(path))
    else:
        artifact.add_file(str(path))
    
    wandb.log_artifact(artifact, aliases=aliases)


def finish_wandb() -> None:
    """
    Finish the current W&B run.
    
    Should be called at the end of the experiment.
    """
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


class MetricsLogger:
    """
    Helper class for accumulating and logging metrics.
    
    Example:
        >>> logger = MetricsLogger()
        >>> for batch in dataloader:
        ...     loss = train_step(batch)
        ...     logger.update({"loss": loss.item()})
        >>> epoch_metrics = logger.compute()
        >>> logger.reset()
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self._metrics: Dict[str, list] = {}
        self._counts: Dict[str, int] = {}
    
    def update(
        self,
        metrics: Dict[str, float],
        count: int = 1,
    ) -> None:
        """
        Update metrics with new values.
        
        Args:
            metrics: Dictionary of metric values
            count: Number of samples (for weighted averaging)
        """
        for key, value in metrics.items():
            if key not in self._metrics:
                self._metrics[key] = []
                self._counts[key] = 0
            self._metrics[key].append(value * count)
            self._counts[key] += count
    
    def compute(self) -> Dict[str, float]:
        """
        Compute average metrics.
        
        Returns:
            Dictionary of averaged metrics
        """
        result = {}
        for key in self._metrics:
            total = sum(self._metrics[key])
            count = self._counts[key]
            result[key] = total / count if count > 0 else 0.0
        return result
    
    def log(self, step: Optional[int] = None, prefix: Optional[str] = None) -> Dict[str, float]:
        """
        Compute and log metrics to W&B.
        
        Args:
            step: Optional step number
            prefix: Optional prefix for metric names
        
        Returns:
            Computed metrics
        """
        metrics = self.compute()
        log_metrics(metrics, step=step, prefix=prefix)
        return metrics
