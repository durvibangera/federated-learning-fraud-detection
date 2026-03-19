"""
Explainability Engine Implementation with SHAP Integration

This module implements model explainability using SHAP (SHapley Additive exPlanations)
for regulatory compliance and audit trails in fraud detection systems.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
from loguru import logger

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP library not available. Install with: pip install shap")


class Explainability_Engine:
    """
    SHAP-based explainability engine for fraud detection models.

    This class provides local and global explanations for model predictions
    using SHAP values. It supports regulatory compliance by generating
    interpretable explanations and maintaining audit trails.

    Attributes:
        model: PyTorch model to explain
        feature_names: List of feature names for interpretability
        background_data: Background dataset for SHAP explainer
        explainer: SHAP explainer instance
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str],
        background_data: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ):
        """
        Initialize Explainability_Engine with model and background data.

        Args:
            model: PyTorch model to explain
            feature_names: List of feature names matching model inputs
            background_data: Background dataset for SHAP (optional, will use summary if None)
            device: Device to run explanations on

        Raises:
            ImportError: If SHAP library is not installed
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library required. Install with: pip install shap")

        self.model = model.to(device)
        self.model.eval()
        self.feature_names = feature_names
        self.device = device
        self.background_data = background_data
        self.explainer = None

        # Initialize SHAP explainer
        if background_data is not None:
            self._initialize_explainer(background_data)

        logger.info(f"Initialized Explainability_Engine with {len(feature_names)} features")

    def _initialize_explainer(self, background_data: torch.Tensor) -> None:
        """
        Initialize SHAP explainer with background data.

        Args:
            background_data: Background dataset for SHAP baseline
        """
        logger.info("Initializing SHAP explainer...")

        # Convert model to a function that SHAP can use
        def model_predict(x):
            """Wrapper function for SHAP."""
            with torch.no_grad():
                if isinstance(x, np.ndarray):
                    x = torch.tensor(x, dtype=torch.float32).to(self.device)

                # Handle different input formats
                if hasattr(self.model, "forward"):
                    # If model expects dict format, convert
                    try:
                        outputs = self.model(x)
                    except:
                        # Try dict format
                        features = {"numerical": x}
                        outputs = self.model(features)
                else:
                    outputs = self.model(x)

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs)
                return probs.cpu().numpy()

        # Use a subset of background data for efficiency
        if len(background_data) > 100:
            background_sample = background_data[:100]
            logger.info(f"Using {len(background_sample)} background samples for SHAP")
        else:
            background_sample = background_data

        # Create SHAP explainer (using KernelExplainer for model-agnostic explanations)
        try:
            self.explainer = shap.KernelExplainer(
                model_predict,
                background_sample.cpu().numpy() if isinstance(background_sample, torch.Tensor) else background_sample,
            )
            logger.info("SHAP explainer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            raise

    def explain_prediction(self, input_data: torch.Tensor, top_k: int = 10) -> Dict[str, Any]:
        """
        Generate local explanation for a single prediction.

        Args:
            input_data: Input features for prediction (single sample or batch)
            top_k: Number of top contributing features to return

        Returns:
            Dictionary containing SHAP values, feature contributions, and prediction
        """
        if self.explainer is None:
            raise RuntimeError("SHAP explainer not initialized. Provide background_data.")

        logger.debug(f"Generating explanation for input shape: {input_data.shape}")

        # Ensure input is 2D (batch_size, features)
        if input_data.dim() == 1:
            input_data = input_data.unsqueeze(0)

        # Convert to numpy for SHAP
        input_np = input_data.cpu().numpy() if isinstance(input_data, torch.Tensor) else input_data

        # Compute SHAP values
        try:
            shap_values = self.explainer.shap_values(input_np)

            # Handle single sample
            if len(input_np) == 1:
                shap_values_sample = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
            else:
                shap_values_sample = shap_values

            # Get prediction
            with torch.no_grad():
                if isinstance(input_data, np.ndarray):
                    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(self.device)
                else:
                    input_tensor = input_data.to(self.device)

                try:
                    outputs = self.model(input_tensor)
                except:
                    features = {"numerical": input_tensor}
                    outputs = self.model(features)

                prediction = torch.sigmoid(outputs).cpu().numpy()[0]

            # Get top contributing features
            feature_contributions = []
            for i, (feature_name, shap_value) in enumerate(zip(self.feature_names, shap_values_sample)):
                feature_contributions.append(
                    {
                        "feature": feature_name,
                        "shap_value": float(shap_value),
                        "feature_value": float(input_np[0, i]),
                        "abs_contribution": abs(float(shap_value)),
                    }
                )

            # Sort by absolute contribution
            feature_contributions.sort(key=lambda x: x["abs_contribution"], reverse=True)
            top_features = feature_contributions[:top_k]

            explanation = {
                "prediction": float(prediction),
                "prediction_class": "fraud" if prediction > 0.5 else "legitimate",
                "confidence": float(max(prediction, 1 - prediction)),
                "top_features": top_features,
                "all_shap_values": shap_values_sample.tolist(),
                "base_value": (
                    float(self.explainer.expected_value) if hasattr(self.explainer, "expected_value") else 0.0
                ),
            }

            logger.info(
                f"Generated explanation: prediction={prediction:.4f}, " f"top feature={top_features[0]['feature']}"
            )

            return explanation

        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            raise

    def get_global_feature_importance(self, test_data: torch.Tensor, num_samples: int = 100) -> Dict[str, float]:
        """
        Compute global feature importance across multiple samples.

        Args:
            test_data: Test dataset to compute importance on
            num_samples: Number of samples to use for importance calculation

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.explainer is None:
            raise RuntimeError("SHAP explainer not initialized. Provide background_data.")

        logger.info(f"Computing global feature importance on {num_samples} samples...")

        # Sample data if needed
        if len(test_data) > num_samples:
            indices = torch.randperm(len(test_data))[:num_samples]
            sample_data = test_data[indices]
        else:
            sample_data = test_data

        # Convert to numpy
        sample_np = sample_data.cpu().numpy() if isinstance(sample_data, torch.Tensor) else sample_data

        # Compute SHAP values
        try:
            shap_values = self.explainer.shap_values(sample_np)

            # Compute mean absolute SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            # Create feature importance dictionary
            feature_importance = {}
            for feature_name, importance in zip(self.feature_names, mean_abs_shap):
                feature_importance[feature_name] = float(importance)

            # Sort by importance
            sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

            logger.info(f"Global feature importance computed. " f"Top feature: {list(sorted_importance.keys())[0]}")

            return sorted_importance

        except Exception as e:
            logger.error(f"Failed to compute global feature importance: {e}")
            raise

    def export_explanations(
        self, explanations: List[Dict[str, Any]], filepath: str, include_metadata: bool = True
    ) -> None:
        """
        Export explanations to JSON file for audit trails.

        Args:
            explanations: List of explanation dictionaries
            filepath: Path to save JSON file
            include_metadata: Whether to include model and system metadata
        """
        logger.info(f"Exporting {len(explanations)} explanations to {filepath}")

        output_data = {"explanations": explanations, "num_explanations": len(explanations)}

        if include_metadata:
            output_data["metadata"] = {
                "model_type": type(self.model).__name__,
                "num_features": len(self.feature_names),
                "feature_names": self.feature_names,
                "device": self.device,
            }

        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Write JSON
        with open(filepath, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Explanations exported to {filepath}")

    def generate_explanation_summary(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics from multiple explanations.

        Args:
            explanations: List of explanation dictionaries

        Returns:
            Summary statistics dictionary
        """
        if not explanations:
            return {}

        # Aggregate statistics
        predictions = [e["prediction"] for e in explanations]
        fraud_count = sum(1 for e in explanations if e["prediction_class"] == "fraud")

        # Aggregate feature importance
        feature_importance_sum = {}
        for explanation in explanations:
            for feature_contrib in explanation["top_features"]:
                feature = feature_contrib["feature"]
                abs_contrib = feature_contrib["abs_contribution"]
                feature_importance_sum[feature] = feature_importance_sum.get(feature, 0) + abs_contrib

        # Average importance
        feature_importance_avg = {k: v / len(explanations) for k, v in feature_importance_sum.items()}

        # Sort by importance
        top_features_overall = sorted(feature_importance_avg.items(), key=lambda x: x[1], reverse=True)[:10]

        summary = {
            "num_explanations": len(explanations),
            "fraud_predictions": fraud_count,
            "legitimate_predictions": len(explanations) - fraud_count,
            "fraud_rate": fraud_count / len(explanations),
            "avg_prediction_score": float(np.mean(predictions)),
            "top_features_overall": [{"feature": f, "avg_importance": float(imp)} for f, imp in top_features_overall],
        }

        logger.info(f"Generated summary for {len(explanations)} explanations")
        return summary

    def create_audit_trail(
        self,
        transaction_id: str,
        explanation: Dict[str, Any],
        user_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create audit trail entry for regulatory compliance.

        Args:
            transaction_id: Unique transaction identifier
            explanation: Explanation dictionary from explain_prediction
            user_id: Optional user ID who requested explanation
            timestamp: Optional timestamp (auto-generated if None)

        Returns:
            Audit trail entry dictionary
        """
        from datetime import datetime

        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        audit_entry = {
            "transaction_id": transaction_id,
            "timestamp": timestamp,
            "user_id": user_id,
            "prediction": explanation["prediction"],
            "prediction_class": explanation["prediction_class"],
            "confidence": explanation["confidence"],
            "top_contributing_features": explanation["top_features"][:5],  # Top 5 for audit
            "model_type": type(self.model).__name__,
        }

        logger.debug(f"Created audit trail for transaction {transaction_id}")
        return audit_entry

    @staticmethod
    def validate_explanation_completeness(explanation: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that explanation contains all required fields.

        Args:
            explanation: Explanation dictionary to validate

        Returns:
            Tuple of (is_valid, list_of_missing_fields)
        """
        required_fields = ["prediction", "prediction_class", "confidence", "top_features", "all_shap_values"]

        missing_fields = [field for field in required_fields if field not in explanation]
        is_valid = len(missing_fields) == 0

        if not is_valid:
            logger.warning(f"Explanation missing fields: {missing_fields}")

        return is_valid, missing_fields
