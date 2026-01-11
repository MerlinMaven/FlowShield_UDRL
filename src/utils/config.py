"""
Configuration loading and management utilities.

This module provides functions for loading YAML configurations
using Hydra/OmegaConf and merging them with command-line overrides.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> DictConfig:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        overrides: Optional dictionary of values to override in the config
    
    Returns:
        OmegaConf DictConfig object
    
    Example:
        >>> config = load_config("configs/base.yaml", overrides={"seed": 123})
        >>> print(config.seed)
        123
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load base config
    config = OmegaConf.load(config_path)
    
    # Apply overrides
    if overrides:
        override_config = OmegaConf.create(overrides)
        config = OmegaConf.merge(config, override_config)
    
    return config


def merge_configs(*configs: Union[DictConfig, Dict]) -> DictConfig:
    """
    Merge multiple configurations.
    
    Later configs override earlier ones.
    
    Args:
        *configs: Variable number of config dicts or DictConfigs
    
    Returns:
        Merged DictConfig
    
    Example:
        >>> base = {"a": 1, "b": 2}
        >>> override = {"b": 3, "c": 4}
        >>> merged = merge_configs(base, override)
        >>> print(merged)
        {"a": 1, "b": 3, "c": 4}
    """
    result = OmegaConf.create()
    for config in configs:
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        result = OmegaConf.merge(result, config)
    return result


def load_config_with_defaults(
    config_path: Union[str, Path],
    defaults_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> DictConfig:
    """
    Load a configuration with default values from a base config.
    
    Args:
        config_path: Path to the main configuration file
        defaults_path: Path to default/base configuration (optional)
        overrides: Command-line or programmatic overrides
    
    Returns:
        Merged DictConfig
    """
    configs_to_merge = []
    
    # Load defaults if provided
    if defaults_path:
        defaults_path = Path(defaults_path)
        if defaults_path.exists():
            configs_to_merge.append(OmegaConf.load(defaults_path))
    
    # Load main config
    configs_to_merge.append(OmegaConf.load(config_path))
    
    # Add overrides
    if overrides:
        configs_to_merge.append(OmegaConf.create(overrides))
    
    return merge_configs(*configs_to_merge)


def config_to_dict(config: DictConfig) -> Dict[str, Any]:
    """
    Convert OmegaConf config to plain Python dictionary.
    
    Useful for logging or serialization.
    
    Args:
        config: OmegaConf DictConfig
    
    Returns:
        Plain Python dictionary
    """
    return OmegaConf.to_container(config, resolve=True)


def save_config(config: DictConfig, path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, path)


def validate_config(config: DictConfig, required_keys: List[str]) -> None:
    """
    Validate that required keys exist in configuration.
    
    Args:
        config: Configuration to validate
        required_keys: List of required key paths (e.g., ["model.hidden_dim"])
    
    Raises:
        ValueError: If a required key is missing
    """
    for key in required_keys:
        try:
            OmegaConf.select(config, key, throw_on_missing=True)
        except Exception:
            raise ValueError(f"Required configuration key missing: {key}")


def print_config(config: DictConfig, resolve: bool = True) -> None:
    """
    Pretty-print configuration to console.
    
    Args:
        config: Configuration to print
        resolve: If True, resolve interpolations
    """
    print(OmegaConf.to_yaml(config, resolve=resolve))


class ConfigManager:
    """
    Configuration manager for experiments.
    
    Handles loading, merging, and accessing configurations.
    
    Example:
        >>> manager = ConfigManager("configs/base.yaml")
        >>> manager.load_experiment("lunarlander")
        >>> print(manager.config.env.name)
    """
    
    def __init__(
        self,
        base_config_path: Union[str, Path],
        configs_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize ConfigManager.
        
        Args:
            base_config_path: Path to base configuration
            configs_dir: Directory containing experiment configs
        """
        self.base_config_path = Path(base_config_path)
        self.configs_dir = Path(configs_dir) if configs_dir else self.base_config_path.parent
        
        self.base_config = load_config(self.base_config_path)
        self.config = self.base_config.copy()
    
    def load_experiment(
        self,
        experiment_name: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> DictConfig:
        """
        Load experiment-specific configuration.
        
        Args:
            experiment_name: Name of experiment (corresponds to YAML file)
            overrides: Additional overrides
        
        Returns:
            Merged configuration
        """
        experiment_path = self.configs_dir / f"{experiment_name}.yaml"
        
        if experiment_path.exists():
            experiment_config = OmegaConf.load(experiment_path)
            self.config = merge_configs(self.base_config, experiment_config)
        else:
            self.config = self.base_config.copy()
        
        if overrides:
            self.config = merge_configs(self.config, overrides)
        
        return self.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key path.
        
        Args:
            key: Dot-separated key path
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        try:
            return OmegaConf.select(self.config, key)
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key path.
        
        Args:
            key: Dot-separated key path
            value: Value to set
        """
        OmegaConf.update(self.config, key, value)
