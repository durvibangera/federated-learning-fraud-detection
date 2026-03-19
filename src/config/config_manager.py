"""
Configuration Management System for Federated Fraud Detection

Handles YAML configuration loading, validation, and environment overrides.
Implements Requirements 12.1, 12.2, 12.3, 12.4.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class FederatedLearningConfig:
    """Federated learning configuration parameters."""

    num_rounds: int = 30
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 3
    strategy: str = "FedProx"
    proximal_mu: float = 0.01
    local_epochs: int = 3


@dataclass
class ModelConfig:
    """Model architecture configuration parameters."""

    embedding_dim: int = 50
    hidden_dims: list = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 1024
    weight_decay: float = 1e-5


@dataclass
class PrivacyConfig:
    """Privacy and differential privacy configuration."""

    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.1
    target_epsilons: list = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0, 8.0])


@dataclass
class DataConfig:
    """Data processing configuration parameters."""

    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    missing_threshold: float = 0.5
    random_seed: int = 42
    categorical_features: list = field(
        default_factory=lambda: [
            "card1",
            "card2",
            "card3",
            "card4",
            "card5",
            "card6",
            "P_emaildomain",
            "R_emaildomain",
            "DeviceInfo",
            "DeviceType",
            "ProductCD",
        ]
    )


@dataclass
class MonitoringConfig:
    """Monitoring and MLOps configuration."""

    mlflow_tracking_uri: str = "http://localhost:5000"
    prometheus_port: int = 8000
    grafana_port: int = 3000
    log_level: str = "INFO"
    experiment_name: str = "federated_fraud_detection"


@dataclass
class PathsConfig:
    """File paths configuration."""

    data_raw: str = "data/raw"
    data_splits: str = "data/splits"
    models: str = "models"
    logs: str = "logs"
    results: str = "results"


@dataclass
class SystemConfig:
    """System resource configuration."""

    num_workers: int = 4
    device: str = "auto"
    memory_limit_gb: int = 8
    checkpoint_frequency: int = 5


@dataclass
class Config:
    """Main configuration class containing all subsections."""

    federated_learning: FederatedLearningConfig = field(default_factory=FederatedLearningConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    data: DataConfig = field(default_factory=DataConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    system: SystemConfig = field(default_factory=SystemConfig)


class ConfigManager:
    """
    Configuration manager with YAML loading, validation, and environment overrides.

    Implements:
    - YAML configuration parsing with schema validation (Req 12.1, 12.2)
    - Environment-specific configuration overrides (Req 12.3)
    - Default value handling for optional parameters (Req 12.4)
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_path = config_path or "config/config.yaml"
        self.config: Optional[Config] = None

    def load_config(self) -> Config:
        """
        Load configuration from YAML file with environment overrides.

        Returns:
            Config: Validated configuration object

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If configuration validation fails
        """
        try:
            # Load base configuration
            config_dict = self._load_yaml_config()

            # Apply environment overrides
            config_dict = self._apply_environment_overrides(config_dict)

            # Validate and create config object
            self.config = self._create_config_object(config_dict)

            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return self.config

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        config_path = Path(self.config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
            return config_dict or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in {config_path}: {e}")

    def _apply_environment_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.

        Environment variables should be prefixed with 'FFD_' (Federated Fraud Detection)
        and use double underscores for nested keys.

        Example: FFD_MODEL__LEARNING_RATE=0.01
        """
        env_prefix = "FFD_"

        for key, value in os.environ.items():
            if not key.startswith(env_prefix):
                continue

            # Remove prefix and convert to lowercase
            config_key = key[len(env_prefix) :].lower()

            # Handle nested keys (double underscore separator)
            keys = config_key.split("__")

            # Navigate to the correct nested dictionary
            current_dict = config_dict
            for k in keys[:-1]:
                if k not in current_dict:
                    current_dict[k] = {}
                current_dict = current_dict[k]

            # Set the value (attempt type conversion)
            final_key = keys[-1]
            current_dict[final_key] = self._convert_env_value(value)

            logger.info(f"Applied environment override: {config_key} = {value}")

        return config_dict

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Try boolean
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Try integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Try scientific notation (for delta values like 1e-5)
        try:
            if "e" in value.lower():
                return float(value)
        except ValueError:
            pass

        # Try list (comma-separated)
        if "," in value:
            return [item.strip() for item in value.split(",")]

        # Return as string
        return value

    def _create_config_object(self, config_dict: Dict[str, Any]) -> Config:
        """Create and validate configuration object from dictionary."""
        try:
            # Create configuration sections
            fl_config = FederatedLearningConfig(**config_dict.get("federated_learning", {}))
            model_config = ModelConfig(**config_dict.get("model", {}))
            privacy_config = PrivacyConfig(**config_dict.get("privacy", {}))
            data_config = DataConfig(**config_dict.get("data", {}))
            monitoring_config = MonitoringConfig(**config_dict.get("monitoring", {}))
            paths_config = PathsConfig(**config_dict.get("paths", {}))
            system_config = SystemConfig(**config_dict.get("system", {}))

            # Validate configuration
            self._validate_config(fl_config, model_config, privacy_config, data_config)

            return Config(
                federated_learning=fl_config,
                model=model_config,
                privacy=privacy_config,
                data=data_config,
                monitoring=monitoring_config,
                paths=paths_config,
                system=system_config,
            )

        except TypeError as e:
            raise ValueError(f"Configuration validation failed: {e}")

    def _validate_config(
        self,
        fl_config: FederatedLearningConfig,
        model_config: ModelConfig,
        privacy_config: PrivacyConfig,
        data_config: DataConfig,
    ) -> None:
        """Validate configuration parameters."""
        errors = []

        # Validate federated learning config
        if fl_config.num_rounds <= 0:
            errors.append("num_rounds must be positive")
        if fl_config.min_fit_clients <= 0:
            errors.append("min_fit_clients must be positive")
        if fl_config.proximal_mu < 0:
            errors.append("proximal_mu must be non-negative")

        # Validate model config
        if model_config.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if model_config.batch_size <= 0:
            errors.append("batch_size must be positive")
        if not (0 <= model_config.dropout_rate <= 1):
            errors.append("dropout_rate must be between 0 and 1")

        # Validate privacy config
        if privacy_config.epsilon <= 0:
            errors.append("epsilon must be positive")
        if privacy_config.delta <= 0:
            errors.append("delta must be positive")

        # Validate data config
        total_split = data_config.train_split + data_config.val_split + data_config.test_split
        if abs(total_split - 1.0) > 1e-6:
            errors.append("train_split + val_split + test_split must equal 1.0")

        if errors:
            raise ValueError("Configuration validation errors: " + "; ".join(errors))

    def save_config(self, config: Config, output_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            "federated_learning": config.federated_learning.__dict__,
            "model": config.model.__dict__,
            "privacy": config.privacy.__dict__,
            "data": config.data.__dict__,
            "monitoring": config.monitoring.__dict__,
            "paths": config.paths.__dict__,
            "system": config.system.__dict__,
        }

        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to {output_path}")


# Global configuration instance
config_manager = ConfigManager()


def get_config() -> Config:
    """Get the global configuration instance."""
    if config_manager.config is None:
        config_manager.load_config()
    return config_manager.config


def reload_config() -> Config:
    """Reload configuration from file."""
    return config_manager.load_config()
