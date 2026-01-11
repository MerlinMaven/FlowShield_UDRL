"""
Reproducibility utilities.

This module provides functions to set random seeds across all libraries
to ensure reproducible experiments.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    This function sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - CUDA backend settings (if deterministic=True)
    
    Args:
        seed: Integer seed value
        deterministic: If True, enables deterministic algorithms in PyTorch.
                      This may reduce performance but ensures reproducibility.
    
    Example:
        >>> from src.utils import set_seed
        >>> set_seed(42)
        >>> # All random operations are now reproducible
    
    Note:
        Some operations may still be non-deterministic even with these settings.
        See PyTorch documentation for more details on reproducibility.
    """
    # Python random
    random.seed(seed)
    
    # Environment variable (for some libraries)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
        if deterministic:
            # Enable deterministic algorithms
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # PyTorch 1.8+ deterministic algorithms
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True)
                except RuntimeError:
                    # Some operations don't have deterministic implementations
                    pass


def get_random_state() -> dict:
    """
    Capture current random state for all libraries.
    
    Useful for saving random state during checkpointing.
    
    Returns:
        Dictionary containing random states for all libraries.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    
    return state


def set_random_state(state: dict) -> None:
    """
    Restore random state for all libraries.
    
    Args:
        state: Dictionary from get_random_state()
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def worker_init_fn(worker_id: int, base_seed: Optional[int] = None) -> None:
    """
    Initialize random seeds for DataLoader workers.
    
    Use this as the worker_init_fn argument in DataLoader to ensure
    reproducibility with multiple workers.
    
    Args:
        worker_id: Worker ID (provided by DataLoader)
        base_seed: Base seed to combine with worker_id
    
    Example:
        >>> from functools import partial
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=partial(worker_init_fn, base_seed=42)
        ... )
    """
    if base_seed is None:
        base_seed = torch.initial_seed() % (2**32)
    
    seed = base_seed + worker_id
    set_seed(seed, deterministic=False)
