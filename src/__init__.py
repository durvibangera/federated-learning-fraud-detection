"""
Federated Fraud Intelligence Network

A privacy-preserving federated learning system for fraud detection
using the IEEE-CIS dataset with differential privacy guarantees.
"""

__version__ = "1.0.0"
__author__ = "Federated Fraud Detection Team"

# Initialize logging on import
from src.utils.logging_setup import setup_logging
from src.config.config_manager import get_config

# Set up logging with configuration
try:
    config = get_config()
    setup_logging(
        log_level=config.monitoring.log_level,
        log_dir=config.paths.logs
    )
except Exception:
    # Fallback to default logging if config fails
    setup_logging()