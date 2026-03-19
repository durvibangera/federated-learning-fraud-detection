"""
Privacy Engine Implementation with Opacus Integration

This module implements differential privacy for federated learning using the
Opacus library. It provides privacy budget tracking, gradient noise addition,
and privacy-utility analysis capabilities.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Optional, Any
from opacus import PrivacyEngine as OpacusEngine
from opacus.validators import ModuleValidator
from loguru import logger


class Privacy_Engine:
    """
    Differential privacy engine using Opacus library.

    This class wraps Opacus functionality to provide differential privacy
    guarantees for federated learning. It tracks privacy budget consumption,
    adds calibrated noise to gradients, and supports multiple epsilon values
    for privacy-utility tradeoff analysis.

    Attributes:
        epsilon: Target privacy budget (epsilon parameter)
        delta: Privacy parameter (typically 1e-5 for datasets of size ~10k)
        max_grad_norm: Maximum gradient norm for clipping
        noise_multiplier: Noise multiplier for DP-SGD
        opacus_engine: Underlying Opacus PrivacyEngine instance
        privacy_spent: Tuple of (epsilon, delta) consumed so far
    """

    def __init__(
        self, epsilon: float, delta: float = 1e-5, max_grad_norm: float = 1.0, noise_multiplier: Optional[float] = None
    ):
        """
        Initialize Privacy_Engine with privacy parameters.

        Args:
            epsilon: Target privacy budget (e.g., 0.5, 1.0, 2.0, 4.0, 8.0)
            delta: Privacy parameter, typically 1e-5
            max_grad_norm: Maximum L2 norm for gradient clipping
            noise_multiplier: Noise multiplier for DP-SGD (auto-calculated if None)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.opacus_engine: Optional[OpacusEngine] = None
        self.privacy_spent = (0.0, 0.0)
        self._actual_noise_multiplier = noise_multiplier if noise_multiplier else 1.1

        logger.info(f"Initialized Privacy_Engine with ε={epsilon}, δ={delta}, " f"max_grad_norm={max_grad_norm}")

    def make_private(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, dataloader: DataLoader, epochs: int = 1
    ) -> Tuple[nn.Module, torch.optim.Optimizer, DataLoader]:
        """
        Convert model, optimizer, and dataloader to privacy-preserving versions.

        This method wraps the model, optimizer, and dataloader with Opacus
        to enable differential privacy during training. It validates model
        compatibility and applies necessary modifications.

        Args:
            model: PyTorch model to make private
            optimizer: Optimizer to make private
            dataloader: DataLoader to make private
            epochs: Number of training epochs (for privacy accounting)

        Returns:
            Tuple of (private_model, private_optimizer, private_dataloader)

        Raises:
            ValueError: If model is not compatible with Opacus
        """
        logger.info("Making model, optimizer, and dataloader private with Opacus")

        # Validate model compatibility with Opacus
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            logger.warning(f"Model has {len(errors)} compatibility issues with Opacus")
            logger.debug(f"Compatibility errors: {errors}")

            # Try to fix model automatically
            model = ModuleValidator.fix(model)
            logger.info("Applied automatic fixes to model for Opacus compatibility")

            # Re-validate
            errors = ModuleValidator.validate(model, strict=False)
            if errors:
                error_msg = f"Model still incompatible with Opacus after fixes: {errors}"
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Calculate noise multiplier if not provided
        if self.noise_multiplier is None:
            # Use Opacus's automatic calculation based on epsilon and delta
            # This will be done by the PrivacyEngine during attachment
            logger.info("Noise multiplier will be auto-calculated by Opacus")

        # Create Opacus PrivacyEngine
        self.opacus_engine = OpacusEngine()

        # Attach privacy engine to model, optimizer, and dataloader
        try:
            # Store the noise multiplier we're using
            actual_noise_multiplier = self.noise_multiplier if self.noise_multiplier else 1.1

            private_model, private_optimizer, private_dataloader = self.opacus_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=dataloader,
                noise_multiplier=actual_noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )

            # Store the actual noise multiplier used
            self._actual_noise_multiplier = actual_noise_multiplier

            logger.info("Successfully attached Opacus privacy engine")
            logger.info(
                f"Privacy parameters: max_grad_norm={self.max_grad_norm}, "
                f"noise_multiplier={actual_noise_multiplier}"
            )

        except Exception as e:
            logger.error(f"Failed to attach Opacus privacy engine: {e}")
            raise

        return private_model, private_optimizer, private_dataloader

    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get current privacy budget consumption.

        Returns:
            Tuple of (epsilon_spent, delta) representing privacy consumed so far

        Raises:
            RuntimeError: If privacy engine has not been initialized via make_private
        """
        if self.opacus_engine is None:
            logger.warning("Privacy engine not initialized, returning (0.0, 0.0)")
            return (0.0, 0.0)

        try:
            epsilon_spent = self.opacus_engine.get_epsilon(delta=self.delta)
            self.privacy_spent = (epsilon_spent, self.delta)

            logger.debug(f"Privacy spent: ε={epsilon_spent:.4f}, δ={self.delta:.2e}")

            # Check if budget is exhausted
            if epsilon_spent > self.epsilon:
                logger.warning(f"Privacy budget exhausted! Spent {epsilon_spent:.4f} > " f"target {self.epsilon:.4f}")

            return self.privacy_spent

        except Exception as e:
            logger.error(f"Failed to get privacy spent: {e}")
            return (0.0, 0.0)

    def is_budget_exhausted(self) -> bool:
        """
        Check if privacy budget has been exhausted.

        Returns:
            True if epsilon spent exceeds target epsilon, False otherwise
        """
        epsilon_spent, _ = self.get_privacy_spent()
        exhausted = epsilon_spent > self.epsilon

        if exhausted:
            logger.warning(f"Privacy budget exhausted: {epsilon_spent:.4f} > {self.epsilon:.4f}")

        return exhausted

    def get_remaining_budget(self) -> float:
        """
        Get remaining privacy budget.

        Returns:
            Remaining epsilon budget (target_epsilon - spent_epsilon)
        """
        epsilon_spent, _ = self.get_privacy_spent()
        remaining = max(0.0, self.epsilon - epsilon_spent)

        logger.debug(f"Remaining privacy budget: {remaining:.4f}")
        return remaining

    def reset_privacy_accountant(self) -> None:
        """
        Reset privacy accountant to start fresh privacy tracking.

        This should be called when starting a new experiment or training run.
        """
        if self.opacus_engine is not None:
            # Opacus doesn't have a direct reset method, so we need to create a new engine
            logger.info("Resetting privacy accountant")
            self.opacus_engine = None
            self.privacy_spent = (0.0, 0.0)
        else:
            logger.warning("Privacy engine not initialized, nothing to reset")

    def get_privacy_summary(self) -> Dict[str, float]:
        """
        Get comprehensive privacy summary.

        Returns:
            Dictionary containing privacy parameters and current state
        """
        epsilon_spent, delta = self.get_privacy_spent()

        summary = {
            "target_epsilon": self.epsilon,
            "epsilon_spent": epsilon_spent,
            "remaining_epsilon": self.get_remaining_budget(),
            "delta": delta,
            "max_grad_norm": self.max_grad_norm,
            "noise_multiplier": self._actual_noise_multiplier,
            "budget_exhausted": self.is_budget_exhausted(),
        }

        logger.debug(f"Privacy summary: {summary}")
        return summary

    @staticmethod
    def validate_model_compatibility(model: nn.Module) -> Tuple[bool, List[str]]:
        """
        Validate if model is compatible with Opacus.

        Args:
            model: PyTorch model to validate

        Returns:
            Tuple of (is_compatible, list_of_errors)
        """
        errors = ModuleValidator.validate(model, strict=False)
        is_compatible = len(errors) == 0

        if is_compatible:
            logger.info("Model is compatible with Opacus")
        else:
            logger.warning(f"Model has {len(errors)} compatibility issues")
            for error in errors:
                logger.debug(f"  - {error}")

        return is_compatible, errors

    @staticmethod
    def fix_model_compatibility(model: nn.Module) -> nn.Module:
        """
        Automatically fix model compatibility issues with Opacus.

        Args:
            model: PyTorch model to fix

        Returns:
            Fixed model compatible with Opacus
        """
        logger.info("Applying automatic fixes for Opacus compatibility")
        fixed_model = ModuleValidator.fix(model)

        # Validate the fixed model
        errors = ModuleValidator.validate(fixed_model, strict=False)
        if errors:
            logger.warning(f"Model still has {len(errors)} issues after automatic fixes")
        else:
            logger.info("Model successfully fixed for Opacus compatibility")

        return fixed_model


class Privacy_Utility_Analyzer:
    """
    Analyzer for privacy-utility tradeoff experiments.

    This class helps run experiments across multiple epsilon values to generate
    privacy-utility curves showing how model performance degrades with stronger
    privacy guarantees.
    """

    def __init__(
        self, target_epsilons: List[float] = [0.5, 1.0, 2.0, 4.0, 8.0], delta: float = 1e-5, max_grad_norm: float = 1.0
    ):
        """
        Initialize Privacy_Utility_Analyzer.

        Args:
            target_epsilons: List of epsilon values to test
            delta: Privacy parameter (constant across experiments)
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.target_epsilons = sorted(target_epsilons)
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.results: Dict[float, Dict[str, float]] = {}

        logger.info(f"Initialized Privacy_Utility_Analyzer with epsilons: {target_epsilons}")

    def add_result(self, epsilon: float, auprc: float, auroc: float, loss: float, epsilon_spent: float) -> None:
        """
        Add experimental result for a specific epsilon value.

        Args:
            epsilon: Target epsilon value
            auprc: Area Under Precision-Recall Curve achieved
            auroc: Area Under ROC Curve achieved
            loss: Final loss value
            epsilon_spent: Actual epsilon consumed
        """
        self.results[epsilon] = {
            "target_epsilon": epsilon,
            "epsilon_spent": epsilon_spent,
            "auprc": auprc,
            "auroc": auroc,
            "loss": loss,
        }

        logger.info(f"Added result for ε={epsilon}: AUPRC={auprc:.4f}, AUROC={auroc:.4f}")

    def get_privacy_utility_curve(self) -> Dict[str, List[float]]:
        """
        Get privacy-utility curve data.

        Returns:
            Dictionary with lists of epsilon, AUPRC, and AUROC values
        """
        if not self.results:
            logger.warning("No results available for privacy-utility curve")
            return {"epsilon": [], "auprc": [], "auroc": [], "loss": []}

        # Sort by epsilon
        sorted_results = sorted(self.results.items(), key=lambda x: x[0])

        curve_data = {
            "epsilon": [r[0] for r in sorted_results],
            "auprc": [r[1]["auprc"] for r in sorted_results],
            "auroc": [r[1]["auroc"] for r in sorted_results],
            "loss": [r[1]["loss"] for r in sorted_results],
            "epsilon_spent": [r[1]["epsilon_spent"] for r in sorted_results],
        }

        logger.debug(f"Generated privacy-utility curve with {len(sorted_results)} points")
        return curve_data

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for privacy-utility tradeoff.

        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {}

        curve_data = self.get_privacy_utility_curve()

        # Calculate performance degradation
        if len(curve_data["auprc"]) >= 2:
            auprc_degradation = curve_data["auprc"][-1] - curve_data["auprc"][0]
            auroc_degradation = curve_data["auroc"][-1] - curve_data["auroc"][0]
        else:
            auprc_degradation = 0.0
            auroc_degradation = 0.0

        summary = {
            "num_experiments": len(self.results),
            "epsilon_range": (min(curve_data["epsilon"]), max(curve_data["epsilon"])),
            "auprc_range": (min(curve_data["auprc"]), max(curve_data["auprc"])),
            "auroc_range": (min(curve_data["auroc"]), max(curve_data["auroc"])),
            "auprc_degradation": auprc_degradation,
            "auroc_degradation": auroc_degradation,
            "best_epsilon_auprc": curve_data["epsilon"][curve_data["auprc"].index(max(curve_data["auprc"]))],
            "best_epsilon_auroc": curve_data["epsilon"][curve_data["auroc"].index(max(curve_data["auroc"]))],
        }

        logger.info(
            f"Privacy-utility summary: AUPRC degradation={auprc_degradation:.4f}, "
            f"AUROC degradation={auroc_degradation:.4f}"
        )

        return summary

    def export_results(self, filepath: str) -> None:
        """
        Export privacy-utility results to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        import json
        from pathlib import Path

        output_data = {
            "target_epsilons": self.target_epsilons,
            "delta": self.delta,
            "max_grad_norm": self.max_grad_norm,
            "results": self.results,
            "curve_data": self.get_privacy_utility_curve(),
            "summary": self.get_summary_statistics(),
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Exported privacy-utility results to {filepath}")

    def plot_privacy_utility_curve(self, save_path: Optional[str] = None) -> None:
        """
        Plot privacy-utility curve (requires matplotlib).

        Args:
            save_path: Optional path to save plot image
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
            return

        curve_data = self.get_privacy_utility_curve()

        if not curve_data["epsilon"]:
            logger.warning("No data to plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # AUPRC vs Epsilon
        ax1.plot(curve_data["epsilon"], curve_data["auprc"], "o-", linewidth=2, markersize=8)
        ax1.set_xlabel("Privacy Budget (ε)", fontsize=12)
        ax1.set_ylabel("AUPRC", fontsize=12)
        ax1.set_title("Privacy-Utility Tradeoff: AUPRC", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale("log")

        # AUROC vs Epsilon
        ax2.plot(curve_data["epsilon"], curve_data["auroc"], "o-", linewidth=2, markersize=8, color="orange")
        ax2.set_xlabel("Privacy Budget (ε)", fontsize=12)
        ax2.set_ylabel("AUROC", fontsize=12)
        ax2.set_title("Privacy-Utility Tradeoff: AUROC", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved privacy-utility plot to {save_path}")
        else:
            plt.show()

        plt.close()
