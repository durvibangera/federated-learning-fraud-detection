"""
Evaluation System for Comprehensive Performance Metrics

This module implements comprehensive evaluation for federated learning models
including AUPRC, AUROC with confidence intervals, convergence tracking, and
baseline comparisons.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix
from scipy import stats
from loguru import logger
import json
from pathlib import Path


class Evaluation_System:
    """
    Comprehensive evaluation system for federated learning models.

    This class provides:
    - AUPRC and AUROC computation with confidence intervals
    - Evaluation on individual bank test sets and combined data
    - Convergence tracking across FL rounds
    - Centralized baseline comparison
    - Performance degradation analysis

    Attributes:
        model: PyTorch model to evaluate
        device: Device to run evaluation on
        history: Dictionary storing evaluation history across rounds
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        """
        Initialize Evaluation_System.

        Args:
            model: PyTorch model to evaluate
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.device = device
        self.history: Dict[int, Dict[str, Any]] = {}
        self.baseline_metrics: Optional[Dict[str, float]] = None

        logger.info("Initialized Evaluation_System")

    def evaluate_model(
        self, dataloader: DataLoader, dataset_name: str = "test", compute_ci: bool = True, ci_confidence: float = 0.95
    ) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.

        Args:
            dataloader: DataLoader for evaluation
            dataset_name: Name of the dataset (for logging)
            compute_ci: Whether to compute confidence intervals
            ci_confidence: Confidence level for intervals (default 0.95)

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating model on {dataset_name} dataset...")

        self.model.eval()
        all_predictions = []
        all_targets = []
        all_losses = []

        criterion = nn.BCEWithLogitsLoss(reduction="none")

        with torch.no_grad():
            for features, targets in dataloader:
                # Move data to device
                if isinstance(features, dict):
                    features = {k: v.to(self.device) for k, v in features.items()}
                else:
                    features = features.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(features)
                loss = criterion(outputs, targets)

                # Collect predictions and targets
                probs = torch.sigmoid(outputs)
                all_predictions.extend(probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_losses.extend(loss.cpu().numpy())

        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        losses = np.array(all_losses)

        # Compute metrics
        metrics = self._compute_metrics(
            predictions, targets, losses, compute_ci=compute_ci, ci_confidence=ci_confidence
        )

        metrics["dataset_name"] = dataset_name
        metrics["num_samples"] = len(targets)

        logger.info(
            f"{dataset_name} - AUPRC: {metrics['auprc']:.4f}, "
            f"AUROC: {metrics['auroc']:.4f}, Loss: {metrics['loss']:.4f}"
        )

        return metrics

    def evaluate_per_bank(
        self, bank_dataloaders: Dict[str, DataLoader], compute_ci: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate model on individual bank test sets.

        Args:
            bank_dataloaders: Dictionary mapping bank IDs to their dataloaders
            compute_ci: Whether to compute confidence intervals

        Returns:
            Dictionary mapping bank IDs to their evaluation metrics
        """
        logger.info(f"Evaluating model on {len(bank_dataloaders)} bank test sets...")

        bank_metrics = {}
        for bank_id, dataloader in bank_dataloaders.items():
            metrics = self.evaluate_model(dataloader, dataset_name=f"{bank_id}_test", compute_ci=compute_ci)
            bank_metrics[bank_id] = metrics

        # Compute aggregate statistics
        avg_auprc = np.mean([m["auprc"] for m in bank_metrics.values()])
        avg_auroc = np.mean([m["auroc"] for m in bank_metrics.values()])

        logger.info(f"Average across banks - AUPRC: {avg_auprc:.4f}, AUROC: {avg_auroc:.4f}")

        return bank_metrics

    def track_round_metrics(self, round_num: int, metrics: Dict[str, Any]) -> None:
        """
        Track metrics for a specific FL round.

        Args:
            round_num: FL round number
            metrics: Dictionary of metrics for this round
        """
        self.history[round_num] = metrics
        logger.debug(f"Tracked metrics for round {round_num}")

    def compute_convergence(
        self, metric_name: str = "auprc", window_size: int = 3, threshold: float = 0.01
    ) -> Tuple[bool, float]:
        """
        Check if model has converged based on metric stability.

        Args:
            metric_name: Metric to check for convergence
            window_size: Number of recent rounds to consider
            threshold: Maximum allowed change for convergence

        Returns:
            Tuple of (is_converged, convergence_score)
        """
        if len(self.history) < window_size:
            return False, 1.0

        # Get recent metric values
        recent_rounds = sorted(self.history.keys())[-window_size:]
        recent_values = [self.history[r][metric_name] for r in recent_rounds]

        # Compute standard deviation as convergence score
        convergence_score = np.std(recent_values)
        is_converged = convergence_score < threshold

        logger.debug(f"Convergence check: score={convergence_score:.4f}, " f"converged={is_converged}")

        return is_converged, convergence_score

    def set_baseline_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Set centralized baseline metrics for comparison.

        Args:
            metrics: Dictionary of baseline metrics
        """
        self.baseline_metrics = metrics
        logger.info(
            f"Set baseline metrics: AUPRC={metrics.get('auprc', 0):.4f}, " f"AUROC={metrics.get('auroc', 0):.4f}"
        )

    def compare_to_baseline(self, current_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Compare current metrics to centralized baseline.

        Args:
            current_metrics: Current model metrics

        Returns:
            Dictionary with comparison metrics (degradation percentages)
        """
        if self.baseline_metrics is None:
            logger.warning("No baseline metrics set for comparison")
            return {}

        comparison = {}
        for metric_name in ["auprc", "auroc", "loss"]:
            if metric_name in current_metrics and metric_name in self.baseline_metrics:
                baseline_val = self.baseline_metrics[metric_name]
                current_val = current_metrics[metric_name]

                if metric_name == "loss":
                    # For loss, lower is better
                    degradation = ((current_val - baseline_val) / baseline_val) * 100
                else:
                    # For AUPRC/AUROC, higher is better
                    degradation = ((baseline_val - current_val) / baseline_val) * 100

                comparison[f"{metric_name}_degradation_pct"] = degradation
                comparison[f"{metric_name}_baseline"] = baseline_val
                comparison[f"{metric_name}_current"] = current_val

        logger.info(f"Baseline comparison - AUPRC degradation: " f"{comparison.get('auprc_degradation_pct', 0):.2f}%")

        return comparison

    def get_convergence_history(self, metric_name: str = "auprc") -> Dict[str, List]:
        """
        Get convergence history for plotting.

        Args:
            metric_name: Metric to get history for

        Returns:
            Dictionary with rounds and metric values
        """
        if not self.history:
            return {"rounds": [], "values": []}

        rounds = sorted(self.history.keys())
        values = [self.history[r][metric_name] for r in rounds]

        return {"rounds": rounds, "values": values}

    def export_evaluation_report(self, filepath: str, include_history: bool = True) -> None:
        """
        Export comprehensive evaluation report to JSON.

        Args:
            filepath: Path to save report
            include_history: Whether to include full history
        """
        logger.info(f"Exporting evaluation report to {filepath}")

        report = {"baseline_metrics": self.baseline_metrics, "num_rounds_evaluated": len(self.history)}

        if include_history:
            report["history"] = self.history

        # Add final round metrics if available
        if self.history:
            final_round = max(self.history.keys())
            report["final_metrics"] = self.history[final_round]

            # Add convergence info
            is_converged, conv_score = self.compute_convergence()
            report["convergence"] = {"is_converged": is_converged, "convergence_score": float(conv_score)}

            # Add baseline comparison if available
            if self.baseline_metrics:
                report["baseline_comparison"] = self.compare_to_baseline(self.history[final_round])

        # Save to file
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Evaluation report exported to {filepath}")

    def _compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        losses: np.ndarray,
        compute_ci: bool = True,
        ci_confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics.

        Args:
            predictions: Model predictions (probabilities)
            targets: Ground truth labels
            losses: Per-sample losses
            compute_ci: Whether to compute confidence intervals
            ci_confidence: Confidence level for intervals

        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        auprc = average_precision_score(targets, predictions)
        auroc = roc_auc_score(targets, predictions)
        loss = np.mean(losses)

        # Binary predictions for confusion matrix
        binary_preds = (predictions > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(targets, binary_preds).ravel()

        # Compute precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics = {
            "auprc": float(auprc),
            "auroc": float(auroc),
            "loss": float(loss),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }

        # Compute confidence intervals using bootstrap
        if compute_ci:
            ci_metrics = self._bootstrap_confidence_intervals(predictions, targets, confidence=ci_confidence)
            metrics.update(ci_metrics)

        return metrics

    def _bootstrap_confidence_intervals(
        self, predictions: np.ndarray, targets: np.ndarray, confidence: float = 0.95, n_bootstrap: int = 1000
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute confidence intervals using bootstrap resampling.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            confidence: Confidence level
            n_bootstrap: Number of bootstrap samples

        Returns:
            Dictionary with confidence intervals for metrics
        """
        n_samples = len(predictions)
        auprc_scores = []
        auroc_scores = []

        # Bootstrap resampling
        rng = np.random.RandomState(42)
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            pred_sample = predictions[indices]
            target_sample = targets[indices]

            # Skip if only one class present
            if len(np.unique(target_sample)) < 2:
                continue

            try:
                auprc_scores.append(average_precision_score(target_sample, pred_sample))
                auroc_scores.append(roc_auc_score(target_sample, pred_sample))
            except:
                continue

        # Compute confidence intervals
        alpha = 1 - confidence
        auprc_ci = np.percentile(auprc_scores, [alpha / 2 * 100, (1 - alpha / 2) * 100])
        auroc_ci = np.percentile(auroc_scores, [alpha / 2 * 100, (1 - alpha / 2) * 100])

        return {
            "auprc_ci_lower": float(auprc_ci[0]),
            "auprc_ci_upper": float(auprc_ci[1]),
            "auroc_ci_lower": float(auroc_ci[0]),
            "auroc_ci_upper": float(auroc_ci[1]),
            "ci_confidence": confidence,
        }

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics across all evaluated rounds.

        Returns:
            Dictionary with summary statistics
        """
        if not self.history:
            return {}

        # Extract metrics across rounds
        auprc_values = [m["auprc"] for m in self.history.values()]
        auroc_values = [m["auroc"] for m in self.history.values()]
        loss_values = [m["loss"] for m in self.history.values()]

        summary = {
            "num_rounds": len(self.history),
            "auprc": {
                "mean": float(np.mean(auprc_values)),
                "std": float(np.std(auprc_values)),
                "min": float(np.min(auprc_values)),
                "max": float(np.max(auprc_values)),
                "final": float(auprc_values[-1]),
            },
            "auroc": {
                "mean": float(np.mean(auroc_values)),
                "std": float(np.std(auroc_values)),
                "min": float(np.min(auroc_values)),
                "max": float(np.max(auroc_values)),
                "final": float(auroc_values[-1]),
            },
            "loss": {
                "mean": float(np.mean(loss_values)),
                "std": float(np.std(loss_values)),
                "min": float(np.min(loss_values)),
                "max": float(np.max(loss_values)),
                "final": float(loss_values[-1]),
            },
        }

        # Add convergence info
        is_converged, conv_score = self.compute_convergence()
        summary["convergence"] = {"is_converged": is_converged, "score": float(conv_score)}

        return summary
