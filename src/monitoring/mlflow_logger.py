"""
MLflow Logger Implementation for Experiment Tracking

This module implements comprehensive experiment tracking for federated learning
using MLflow, including metrics logging, model artifacts, and hyperparameters.
"""

import mlflow
import mlflow.pytorch
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import torch.nn as nn
from loguru import logger
import json


class MLflow_Logger:
    """
    MLflow-based experiment tracking for federated learning.

    This class provides comprehensive logging of FL experiments including:
    - Round-by-round metrics (AUPRC, AUROC, loss)
    - Privacy budget tracking
    - Model artifacts with versioning
    - Hyperparameters and configuration
    - Reproducible experiment tracking

    Attributes:
        experiment_name: Name of the MLflow experiment
        tracking_uri: MLflow tracking server URI
        run_id: Current MLflow run ID
        run_name: Name of the current run
    """

    def __init__(
        self,
        experiment_name: str = "federated_fraud_detection",
        tracking_uri: str = "http://localhost:5000",
        run_name: Optional[str] = None,
    ):
        """
        Initialize MLflow_Logger.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI of MLflow tracking server
            run_name: Optional name for the run
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_id = None
        self.run_name = run_name

        # Set MLflow tracking URI
        try:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI set to: {tracking_uri}")
        except Exception as e:
            logger.warning(f"Failed to set MLflow tracking URI: {e}")
            logger.info("Using local file-based tracking")

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created MLflow experiment: {experiment_name}")
        except:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                self.experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {experiment_name}")
            else:
                self.experiment_id = None
                logger.warning("Could not create or find MLflow experiment")

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new MLflow run.

        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run

        Returns:
            Run ID
        """
        if run_name:
            self.run_name = run_name

        try:
            if self.experiment_id:
                mlflow.start_run(experiment_id=self.experiment_id, run_name=self.run_name, tags=tags)
            else:
                mlflow.start_run(run_name=self.run_name, tags=tags)

            self.run_id = mlflow.active_run().info.run_id
            logger.info(f"Started MLflow run: {self.run_id}")

            return self.run_id

        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            raise

    def end_run(self) -> None:
        """End the current MLflow run."""
        try:
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {self.run_id}")
            self.run_id = None
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters for the experiment.

        Args:
            params: Dictionary of hyperparameters
        """
        try:
            # Flatten nested dictionaries
            flat_params = self._flatten_dict(params)
            mlflow.log_params(flat_params)
            logger.info(f"Logged {len(flat_params)} hyperparameters")
        except Exception as e:
            logger.error(f"Failed to log hyperparameters: {e}")

    def log_fl_round_metrics(self, round_num: int, metrics: Dict[str, float], client_id: Optional[str] = None) -> None:
        """
        Log metrics for a federated learning round.

        Args:
            round_num: FL round number
            metrics: Dictionary of metrics (loss, auprc, auroc, etc.)
            client_id: Optional client identifier for client-specific metrics
        """
        try:
            # Add prefix for client-specific metrics
            prefix = f"{client_id}_" if client_id else "global_"

            # Log each metric with round number as step
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(key=f"{prefix}{metric_name}", value=metric_value, step=round_num)

            logger.debug(f"Logged {len(metrics)} metrics for round {round_num}")

        except Exception as e:
            logger.error(f"Failed to log FL round metrics: {e}")

    def log_privacy_budget(self, round_num: int, epsilon: float, delta: float, client_id: Optional[str] = None) -> None:
        """
        Log privacy budget consumption.

        Args:
            round_num: FL round number
            epsilon: Epsilon value (privacy budget spent)
            delta: Delta value
            client_id: Optional client identifier
        """
        try:
            prefix = f"{client_id}_" if client_id else "global_"

            mlflow.log_metric(f"{prefix}epsilon", epsilon, step=round_num)
            mlflow.log_metric(f"{prefix}delta", delta, step=round_num)

            logger.debug(f"Logged privacy budget: ε={epsilon:.4f}, δ={delta:.2e}")

        except Exception as e:
            logger.error(f"Failed to log privacy budget: {e}")

    def log_model_artifact(
        self, model: nn.Module, artifact_path: str = "model", round_num: Optional[int] = None
    ) -> None:
        """
        Log PyTorch model as artifact.

        Args:
            model: PyTorch model to log
            artifact_path: Path within MLflow artifacts
            round_num: Optional round number for versioning
        """
        try:
            if round_num is not None:
                artifact_path = f"{artifact_path}_round_{round_num}"

            mlflow.pytorch.log_model(model, artifact_path)
            logger.info(f"Logged model artifact: {artifact_path}")

        except Exception as e:
            logger.error(f"Failed to log model artifact: {e}")

    def log_model_state_dict(
        self, state_dict: Dict[str, Any], filename: str = "model_state.pth", round_num: Optional[int] = None
    ) -> None:
        """
        Log model state dict as artifact.

        Args:
            state_dict: Model state dictionary
            filename: Filename for the state dict
            round_num: Optional round number for versioning
        """
        try:
            if round_num is not None:
                filename = f"model_state_round_{round_num}.pth"

            # Save to temporary file
            temp_path = Path(f"/tmp/{filename}")
            torch.save(state_dict, temp_path)

            # Log as artifact
            mlflow.log_artifact(str(temp_path))
            logger.info(f"Logged model state dict: {filename}")

            # Clean up
            temp_path.unlink()

        except Exception as e:
            logger.error(f"Failed to log model state dict: {e}")

    def log_config(self, config: Dict[str, Any], filename: str = "config.json") -> None:
        """
        Log configuration as JSON artifact.

        Args:
            config: Configuration dictionary
            filename: Filename for the config
        """
        try:
            # Save to temporary file
            temp_path = Path(f"/tmp/{filename}")
            with open(temp_path, "w") as f:
                json.dump(config, f, indent=2)

            # Log as artifact
            mlflow.log_artifact(str(temp_path))
            logger.info(f"Logged configuration: {filename}")

            # Clean up
            temp_path.unlink()

        except Exception as e:
            logger.error(f"Failed to log configuration: {e}")

    def log_convergence_metrics(self, round_num: int, convergence_score: float, is_converged: bool) -> None:
        """
        Log convergence tracking metrics.

        Args:
            round_num: FL round number
            convergence_score: Convergence metric value
            is_converged: Whether model has converged
        """
        try:
            mlflow.log_metric("convergence_score", convergence_score, step=round_num)
            mlflow.log_metric("is_converged", float(is_converged), step=round_num)

            logger.debug(f"Logged convergence: score={convergence_score:.4f}, " f"converged={is_converged}")

        except Exception as e:
            logger.error(f"Failed to log convergence metrics: {e}")

    def log_system_metrics(self, round_num: int, metrics: Dict[str, float]) -> None:
        """
        Log system performance metrics (memory, CPU, etc.).

        Args:
            round_num: FL round number
            metrics: Dictionary of system metrics
        """
        try:
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"system_{metric_name}", metric_value, step=round_num)

            logger.debug(f"Logged {len(metrics)} system metrics")

        except Exception as e:
            logger.error(f"Failed to log system metrics: {e}")

    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Set tags for the current run.

        Args:
            tags: Dictionary of tags
        """
        try:
            mlflow.set_tags(tags)
            logger.debug(f"Set {len(tags)} tags")
        except Exception as e:
            logger.error(f"Failed to set tags: {e}")

    def log_artifact_file(self, filepath: str) -> None:
        """
        Log an arbitrary file as artifact.

        Args:
            filepath: Path to file to log
        """
        try:
            mlflow.log_artifact(filepath)
            logger.info(f"Logged artifact: {filepath}")
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")

    def get_run_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current run.

        Returns:
            Dictionary with run information or None if no active run
        """
        try:
            active_run = mlflow.active_run()
            if active_run:
                return {
                    "run_id": active_run.info.run_id,
                    "run_name": active_run.info.run_name,
                    "experiment_id": active_run.info.experiment_id,
                    "status": active_run.info.status,
                    "start_time": active_run.info.start_time,
                    "artifact_uri": active_run.info.artifact_uri,
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get run info: {e}")
            return None

    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """
        Flatten nested dictionary for MLflow logging.

        Args:
            d: Dictionary to flatten
            parent_key: Parent key for recursion
            sep: Separator for nested keys

        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLflow_Logger._flatten_dict(v, new_key, sep=sep).items())
            else:
                # Convert to string if not a basic type
                if not isinstance(v, (int, float, str, bool)):
                    v = str(v)
                items.append((new_key, v))
        return dict(items)

    def __enter__(self):
        """Context manager entry."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()
