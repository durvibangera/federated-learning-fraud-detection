"""
Configuration System with YAML Support

This module implements a robust configuration management system with:
- YAML parsing and validation
- Schema validation
- Environment-specific overrides
- Default value handling
- Hot-reloading for non-critical parameters
- Privacy budget enforcement per bank
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import copy
from loguru import logger


class ConfigError(Exception):
    """Configuration-related errors."""
    pass


class ParameterType(Enum):
    """Parameter criticality types."""
    CRITICAL = "critical"  # Requires restart
    NON_CRITICAL = "non_critical"  # Can be hot-reloaded


@dataclass
class FederatedLearningConfig:
    """Federated learning configuration."""
    num_rounds: int = 30
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 3
    strategy: str = "FedProx"
    proximal_mu: float = 0.01
    local_epochs: int = 3
    
    def validate(self) -> None:
        """Validate federated learning configuration."""
        if self.num_rounds <= 0:
            raise ConfigError("num_rounds must be positive")
        if self.min_fit_clients < 1:
            raise ConfigError("min_fit_clients must be at least 1")
        if self.min_evaluate_clients < 1:
            raise ConfigError("min_evaluate_clients must be at least 1")
        if self.min_available_clients < self.min_fit_clients:
            raise ConfigError("min_available_clients must be >= min_fit_clients")
        if self.strategy not in ["FedProx", "FedAvg", "FedAdam"]:
            raise ConfigError(f"Invalid strategy: {self.strategy}")
        if self.proximal_mu < 0:
            raise ConfigError("proximal_mu must be non-negative")
        if self.local_epochs < 1:
            raise ConfigError("local_epochs must be at least 1")


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    embedding_dim: int = 50
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 1024
    weight_decay: float = 1e-5
    
    def validate(self) -> None:
        """Validate model configuration."""
        if self.embedding_dim <= 0:
            raise ConfigError("embedding_dim must be positive")
        if not self.hidden_dims or any(d <= 0 for d in self.hidden_dims):
            raise ConfigError("hidden_dims must be non-empty with positive values")
        if not 0 <= self.dropout_rate < 1:
            raise ConfigError("dropout_rate must be in [0, 1)")
        if self.learning_rate <= 0:
            raise ConfigError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ConfigError("batch_size must be positive")
        if self.weight_decay < 0:
            raise ConfigError("weight_decay must be non-negative")


@dataclass
class PrivacyConfig:
    """Privacy configuration."""
    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.1
    target_epsilons: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0, 8.0])
    
    # Per-bank privacy budgets (optional)
    bank_budgets: Optional[Dict[str, float]] = None
    
    def validate(self) -> None:
        """Validate privacy configuration."""
        if self.epsilon <= 0:
            raise ConfigError("epsilon must be positive")
        if self.delta <= 0 or self.delta >= 1:
            raise ConfigError("delta must be in (0, 1)")
        if self.max_grad_norm <= 0:
            raise ConfigError("max_grad_norm must be positive")
        if self.noise_multiplier < 0:
            raise ConfigError("noise_multiplier must be non-negative")
        if not self.target_epsilons or any(e <= 0 for e in self.target_epsilons):
            raise ConfigError("target_epsilons must be non-empty with positive values")
        
        # Validate per-bank budgets if specified
        if self.bank_budgets:
            for bank_id, budget in self.bank_budgets.items():
                if budget <= 0:
                    raise ConfigError(f"Privacy budget for {bank_id} must be positive")


@dataclass
class DataConfig:
    """Data processing configuration."""
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    missing_threshold: float = 0.5
    random_seed: int = 42
    categorical_features: List[str] = field(default_factory=list)
    
    def validate(self) -> None:
        """Validate data configuration."""
        total_split = self.train_split + self.val_split + self.test_split
        if not (0.99 <= total_split <= 1.01):  # Allow small floating point error
            raise ConfigError(f"Data splits must sum to 1.0, got {total_split}")
        if not 0 < self.train_split < 1:
            raise ConfigError("train_split must be in (0, 1)")
        if not 0 < self.val_split < 1:
            raise ConfigError("val_split must be in (0, 1)")
        if not 0 < self.test_split < 1:
            raise ConfigError("test_split must be in (0, 1)")
        if not 0 < self.missing_threshold <= 1:
            raise ConfigError("missing_threshold must be in (0, 1]")


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    mlflow_tracking_uri: str = "http://localhost:5000"
    prometheus_port: int = 8000
    grafana_port: int = 3000
    log_level: str = "INFO"
    experiment_name: str = "federated_fraud_detection"
    
    def validate(self) -> None:
        """Validate monitoring configuration."""
        if self.prometheus_port < 1024 or self.prometheus_port > 65535:
            raise ConfigError("prometheus_port must be in [1024, 65535]")
        if self.grafana_port < 1024 or self.grafana_port > 65535:
            raise ConfigError("grafana_port must be in [1024, 65535]")
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ConfigError(f"Invalid log_level: {self.log_level}")


@dataclass
class PathsConfig:
    """Paths configuration."""
    data_raw: str = "data/raw"
    data_splits: str = "data/splits"
    models: str = "models"
    logs: str = "logs"
    results: str = "results"
    
    def validate(self) -> None:
        """Validate paths configuration."""
        # Paths are validated at runtime when accessed
        pass


@dataclass
class SystemConfig:
    """System configuration."""
    num_workers: int = 4
    device: str = "auto"
    memory_limit_gb: int = 8
    checkpoint_frequency: int = 5
    
    def validate(self) -> None:
        """Validate system configuration."""
        if self.num_workers < 0:
            raise ConfigError("num_workers must be non-negative")
        if self.device not in ["auto", "cpu", "cuda"]:
            raise ConfigError(f"Invalid device: {self.device}")
        if self.memory_limit_gb <= 0:
            raise ConfigError("memory_limit_gb must be positive")
        if self.checkpoint_frequency <= 0:
            raise ConfigError("checkpoint_frequency must be positive")


@dataclass
class SystemConfiguration:
    """Complete system configuration."""
    federated_learning: FederatedLearningConfig = field(default_factory=FederatedLearningConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    data: DataConfig = field(default_factory=DataConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def validate(self) -> None:
        """Validate entire configuration."""
        self.federated_learning.validate()
        self.model.validate()
        self.privacy.validate()
        self.data.validate()
        self.monitoring.validate()
        self.paths.validate()
        self.system.validate()


class Configuration_System:
    """
    Configuration management system with YAML support.
    
    Features:
    - YAML parsing with schema validation
    - Environment-specific overrides
    - Default value handling
    - Hot-reloading for non-critical parameters
    - Privacy budget enforcement per bank
    
    Attributes:
        config: Current system configuration
        config_path: Path to configuration file
        env: Environment name (dev, staging, prod)
    """
    
    # Define which parameters can be hot-reloaded
    HOT_RELOADABLE_PARAMS = {
        'monitoring.log_level',
        'monitoring.experiment_name',
        'system.checkpoint_frequency',
        'federated_learning.local_epochs',
        'model.learning_rate',
        'model.dropout_rate',
    }
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        env: str = "dev"
    ):
        """
        Initialize Configuration_System.
        
        Args:
            config_path: Path to YAML configuration file
            env: Environment name (dev, staging, prod)
        """
        self.config_path = Path(config_path) if config_path else Path("config/config.yaml")
        self.env = env
        self.config: Optional[SystemConfiguration] = None
        self._config_cache: Dict[str, Any] = {}
        
        # Load configuration
        self.load_config()
        
        logger.info(f"Configuration_System initialized for environment: {env}")
    
    def load_config(self) -> SystemConfiguration:
        """
        Load configuration from YAML file.
        
        Returns:
            Loaded and validated configuration
            
        Raises:
            ConfigError: If configuration is invalid
        """
        try:
            # Load base configuration
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Apply environment-specific overrides
            config_dict = self._apply_env_overrides(config_dict)
            
            # Apply environment variable overrides
            config_dict = self._apply_env_var_overrides(config_dict)
            
            # Parse into dataclasses
            self.config = self._parse_config(config_dict)
            
            # Validate configuration
            self.config.validate()
            
            # Cache configuration
            self._config_cache = config_dict
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return self.config
            
        except FileNotFoundError:
            raise ConfigError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to load configuration: {e}")
    
    def _parse_config(self, config_dict: Dict[str, Any]) -> SystemConfiguration:
        """
        Parse configuration dictionary into dataclasses.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Parsed configuration
        """
        return SystemConfiguration(
            federated_learning=FederatedLearningConfig(
                **config_dict.get('federated_learning', {})
            ),
            model=ModelConfig(**config_dict.get('model', {})),
            privacy=PrivacyConfig(**config_dict.get('privacy', {})),
            data=DataConfig(**config_dict.get('data', {})),
            monitoring=MonitoringConfig(**config_dict.get('monitoring', {})),
            paths=PathsConfig(**config_dict.get('paths', {})),
            system=SystemConfig(**config_dict.get('system', {}))
        )
    
    def _apply_env_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment-specific configuration overrides.
        
        Args:
            config_dict: Base configuration dictionary
            
        Returns:
            Configuration with environment overrides applied
        """
        env_config_path = self.config_path.parent / f"config.{self.env}.yaml"
        
        if env_config_path.exists():
            try:
                with open(env_config_path, 'r') as f:
                    env_overrides = yaml.safe_load(f)
                
                # Deep merge overrides
                config_dict = self._deep_merge(config_dict, env_overrides)
                logger.info(f"Applied environment overrides from {env_config_path}")
                
            except Exception as e:
                logger.warning(f"Failed to load environment overrides: {e}")
        
        return config_dict
    
    def _apply_env_var_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides.
        
        Environment variables should be prefixed with FL_ and use double underscores
        for nesting. Example: FL_PRIVACY__EPSILON=2.0
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Configuration with environment variable overrides applied
        """
        prefix = "FL_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to nested path
                config_key = key[len(prefix):].lower().replace('__', '.')
                
                # Set value in config dict
                self._set_nested_value(config_dict, config_key, value)
                logger.debug(f"Applied environment variable override: {config_key}")
        
        return config_dict
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _set_nested_value(self, d: Dict, path: str, value: Any) -> None:
        """
        Set a nested dictionary value using dot notation.
        
        Args:
            d: Dictionary to modify
            path: Dot-separated path (e.g., 'privacy.epsilon')
            value: Value to set
        """
        keys = path.split('.')
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        
        # Convert value to appropriate type
        final_key = keys[-1]
        if final_key in d:
            # Try to match existing type
            existing_type = type(d[final_key])
            try:
                d[final_key] = existing_type(value)
            except:
                d[final_key] = value
        else:
            d[final_key] = value
    
    def reload_config(self, hot_reload: bool = True) -> SystemConfiguration:
        """
        Reload configuration from file.
        
        Args:
            hot_reload: If True, only reload hot-reloadable parameters
            
        Returns:
            Reloaded configuration
            
        Raises:
            ConfigError: If hot_reload=True and critical parameters changed
        """
        old_config = copy.deepcopy(self.config)
        
        # Load new configuration
        self.load_config()
        
        if hot_reload:
            # Check if any critical parameters changed
            critical_changes = self._detect_critical_changes(old_config, self.config)
            
            if critical_changes:
                # Revert to old configuration
                self.config = old_config
                raise ConfigError(
                    f"Cannot hot-reload: critical parameters changed: {critical_changes}"
                )
            
            logger.info("Configuration hot-reloaded successfully")
        else:
            logger.info("Configuration reloaded (full restart required)")
        
        return self.config
    
    def _detect_critical_changes(
        self,
        old_config: SystemConfiguration,
        new_config: SystemConfiguration
    ) -> List[str]:
        """
        Detect changes to critical parameters.
        
        Args:
            old_config: Old configuration
            new_config: New configuration
            
        Returns:
            List of changed critical parameters
        """
        changes = []
        
        old_dict = asdict(old_config)
        new_dict = asdict(new_config)
        
        def check_dict(old_d, new_d, prefix=''):
            for key, old_value in old_d.items():
                new_value = new_d.get(key)
                full_key = f"{prefix}.{key}" if prefix else key
                
                if isinstance(old_value, dict) and isinstance(new_value, dict):
                    check_dict(old_value, new_value, full_key)
                elif old_value != new_value:
                    # Check if this is a hot-reloadable parameter
                    if full_key not in self.HOT_RELOADABLE_PARAMS:
                        changes.append(full_key)
        
        check_dict(old_dict, new_dict)
        return changes
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path.
        
        Args:
            path: Dot-separated path (e.g., 'privacy.epsilon')
            default: Default value if path not found
            
        Returns:
            Configuration value
        """
        try:
            value = self.config
            for key in path.split('.'):
                value = getattr(value, key)
            return value
        except (AttributeError, KeyError):
            return default
    
    def set(self, path: str, value: Any, hot_reload: bool = True) -> None:
        """
        Set configuration value by dot-separated path.
        
        Args:
            path: Dot-separated path (e.g., 'monitoring.log_level')
            value: Value to set
            hot_reload: If True, only allow hot-reloadable parameters
            
        Raises:
            ConfigError: If trying to hot-reload a critical parameter
        """
        if hot_reload and path not in self.HOT_RELOADABLE_PARAMS:
            raise ConfigError(f"Parameter '{path}' cannot be hot-reloaded")
        
        # Set value
        obj = self.config
        keys = path.split('.')
        for key in keys[:-1]:
            obj = getattr(obj, key)
        setattr(obj, keys[-1], value)
        
        # Validate configuration
        self.config.validate()
        
        logger.info(f"Configuration parameter updated: {path} = {value}")
    
    def enforce_privacy_budget(self, bank_id: str, epsilon_spent: float) -> bool:
        """
        Enforce privacy budget for a specific bank.
        
        Args:
            bank_id: Bank identifier
            epsilon_spent: Epsilon spent so far
            
        Returns:
            True if within budget, False if budget exhausted
        """
        # Get bank-specific budget if configured
        if self.config.privacy.bank_budgets and bank_id in self.config.privacy.bank_budgets:
            budget = self.config.privacy.bank_budgets[bank_id]
        else:
            # Use global budget
            budget = self.config.privacy.epsilon
        
        if epsilon_spent >= budget:
            logger.warning(f"Privacy budget exhausted for {bank_id}: {epsilon_spent:.4f}/{budget:.4f}")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return asdict(self.config)
    
    def to_yaml(self, path: Optional[Union[str, Path]] = None) -> str:
        """
        Export configuration to YAML.
        
        Args:
            path: Optional path to save YAML file
            
        Returns:
            YAML string
        """
        config_dict = self.to_dict()
        yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
        
        if path:
            with open(path, 'w') as f:
                f.write(yaml_str)
            logger.info(f"Configuration exported to {path}")
        
        return yaml_str
    
    def validate_config(self) -> bool:
        """
        Validate current configuration.
        
        Returns:
            True if valid
            
        Raises:
            ConfigError: If configuration is invalid
        """
        self.config.validate()
        return True
