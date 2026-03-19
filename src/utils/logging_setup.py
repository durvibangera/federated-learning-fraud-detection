"""
Logging Setup for Federated Fraud Detection System

Configures structured JSON logging with appropriate levels and formatting.
"""

import sys
import json
from pathlib import Path
from loguru import logger
from typing import Dict, Any


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """
    Set up structured logging for the federated fraud detection system.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Remove default logger
    logger.remove()

    # Console logger with colored output
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        colorize=True,
    )

    # File logger with JSON format for structured logging
    logger.add(
        log_path / "federated_fraud_detection.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        rotation="100 MB",
        retention="30 days",
        compression="zip",
    )

    # Separate JSON logger for structured data
    logger.add(
        log_path / "federated_fraud_detection.json",
        level=log_level,
        serialize=True,  # Use built-in JSON serialization
        rotation="100 MB",
        retention="30 days",
    )

    logger.info(f"Logging initialized with level {log_level}")


def log_experiment_metrics(metrics: Dict[str, Any], round_number: int = None) -> None:
    """
    Log experiment metrics in structured format.

    Args:
        metrics: Dictionary of metrics to log
        round_number: Optional federated learning round number
    """
    extra_data = {"metrics": metrics}
    if round_number is not None:
        extra_data["round"] = round_number

    logger.bind(**extra_data).info("Experiment metrics logged")


def log_privacy_budget(epsilon_spent: float, delta: float, round_number: int) -> None:
    """
    Log privacy budget consumption.

    Args:
        epsilon_spent: Cumulative epsilon spent
        delta: Delta parameter
        round_number: Federated learning round number
    """
    logger.bind(privacy_budget={"epsilon_spent": epsilon_spent, "delta": delta, "round": round_number}).info(
        "Privacy budget updated"
    )


def log_federated_round(round_num: int, participating_clients: list, global_metrics: Dict[str, float]) -> None:
    """
    Log federated learning round completion.

    Args:
        round_num: Round number
        participating_clients: List of client IDs that participated
        global_metrics: Global model performance metrics
    """
    logger.bind(federated_round={"round": round_num, "clients": participating_clients, "metrics": global_metrics}).info(
        f"Federated learning round {round_num} completed"
    )


def log_error_with_context(error: Exception, context: Dict[str, Any]) -> None:
    """
    Log error with additional context information.

    Args:
        error: Exception that occurred
        context: Additional context information
    """
    logger.bind(error_context=context, error_type=type(error).__name__).error(f"Error occurred: {str(error)}")
