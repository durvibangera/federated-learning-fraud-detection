"""
PyTorch Dataset for IEEE-CIS Fraud Detection

This module provides a custom PyTorch Dataset class optimized for fraud detection
with class imbalance handling, Opacus compatibility, and fraud-specific features.
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class PyTorch_Dataset(Dataset):
    """
    Custom PyTorch Dataset for IEEE-CIS fraud detection.

    Optimized for federated learning with differential privacy support,
    class imbalance handling, and fraud-specific feature engineering.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str = "isFraud",
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        device: str = "cpu",
    ):
        """
        Initialize PyTorch dataset for fraud detection.

        Args:
            df: Preprocessed DataFrame with encoded features
            target_column: Name of target column (default: 'isFraud')
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            device: Device to store tensors on ('cpu' or 'cuda')
        """
        self.df = df.copy()
        self.target_column = target_column
        self.device = device

        # Identify feature columns
        self.categorical_columns = categorical_columns or self._identify_categorical_columns()
        self.numerical_columns = numerical_columns or self._identify_numerical_columns()

        # Prepare features and targets
        self.features, self.targets = self._prepare_tensors()

        # Calculate class weights for imbalance handling
        self.class_weights = self._calculate_class_weights()

        logger.info(
            f"Dataset initialized: {len(self)} samples, "
            f"{len(self.categorical_columns)} categorical features, "
            f"{len(self.numerical_columns)} numerical features"
        )

    def _identify_categorical_columns(self) -> List[str]:
        """Identify categorical columns from the dataset."""
        categorical_cols = []

        # Known categorical columns from IEEE-CIS dataset
        known_categorical = [
            "ProductCD",
            "card4",
            "card6",
            "P_emaildomain",
            "R_emaildomain",
            "DeviceType",
            "DeviceInfo",
        ] + [f"id_{i:02d}" for i in [11, 12, 15, 16, 20]]

        for col in self.df.columns:
            if col == self.target_column or col == "TransactionID":
                continue

            # Check if it's a known categorical column or has been label encoded
            if col in known_categorical or (
                self.df[col].dtype in ["int64", "int8"]
                and col.startswith(("ProductCD", "card", "email", "Device", "id_"))
            ):
                categorical_cols.append(col)

        return categorical_cols

    def _identify_numerical_columns(self) -> List[str]:
        """Identify numerical columns from the dataset."""
        numerical_cols = []

        for col in self.df.columns:
            if (
                col != self.target_column
                and col != "TransactionID"
                and col not in self.categorical_columns
                and self.df[col].dtype in ["int64", "int8", "float64", "float32"]
            ):
                numerical_cols.append(col)

        return numerical_cols

    def _prepare_tensors(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Prepare feature and target tensors."""
        features = {}

        # Prepare categorical features
        if self.categorical_columns:
            cat_data = self.df[self.categorical_columns].values.astype(np.int64)
            features["categorical"] = torch.tensor(cat_data, dtype=torch.long, device=self.device)

        # Prepare numerical features
        if self.numerical_columns:
            num_data = self.df[self.numerical_columns].values.astype(np.float32)
            # Handle any remaining NaN values
            num_data = np.nan_to_num(num_data, nan=0.0, posinf=1e6, neginf=-1e6)
            features["numerical"] = torch.tensor(num_data, dtype=torch.float32, device=self.device)

        # Prepare targets
        if self.target_column in self.df.columns:
            targets = torch.tensor(
                self.df[self.target_column].values.astype(np.float32), dtype=torch.float32, device=self.device
            )
        else:
            # For test sets without labels
            targets = torch.zeros(len(self.df), dtype=torch.float32, device=self.device)

        return features, targets

    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced dataset."""
        if self.target_column not in self.df.columns:
            return torch.tensor([1.0, 1.0], device=self.device)

        # Count classes
        class_counts = self.df[self.target_column].value_counts().sort_index()

        # Calculate inverse frequency weights
        total_samples = len(self.df)
        weights = []

        for class_idx in [0, 1]:  # Non-fraud, Fraud
            if class_idx in class_counts.index:
                weight = total_samples / (2.0 * class_counts[class_idx])
            else:
                weight = 1.0
            weights.append(weight)

        class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        logger.info(f"Class weights calculated: Non-fraud={weights[0]:.3f}, Fraud={weights[1]:.3f}")
        return class_weights

    def get_sample_weights(self) -> torch.Tensor:
        """Get sample weights for WeightedRandomSampler."""
        if self.target_column not in self.df.columns:
            return torch.ones(len(self), device=self.device)

        sample_weights = torch.zeros(len(self), device=self.device)

        for idx, target in enumerate(self.targets):
            class_idx = int(target.item())
            sample_weights[idx] = self.class_weights[class_idx]

        return sample_weights

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features_dict, target)
        """
        features = {}

        if "categorical" in self.features:
            features["categorical"] = self.features["categorical"][idx]

        if "numerical" in self.features:
            features["numerical"] = self.features["numerical"][idx]

        target = self.targets[idx]

        return features, target

    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about dataset features."""
        info = {
            "total_features": len(self.categorical_columns) + len(self.numerical_columns),
            "categorical_features": len(self.categorical_columns),
            "numerical_features": len(self.numerical_columns),
            "categorical_columns": self.categorical_columns,
            "numerical_columns": self.numerical_columns,
            "samples": len(self),
            "has_targets": self.target_column in self.df.columns,
        }

        if info["has_targets"]:
            fraud_rate = self.df[self.target_column].mean()
            info["fraud_rate"] = fraud_rate
            info["class_distribution"] = self.df[self.target_column].value_counts().to_dict()

        return info


def create_fraud_dataloader(
    dataset: PyTorch_Dataset,
    batch_size: int = 256,
    shuffle: bool = True,
    use_weighted_sampling: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create DataLoader optimized for fraud detection and Opacus compatibility.

    Args:
        dataset: PyTorch_Dataset instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle data (ignored if using weighted sampling)
        use_weighted_sampling: Use WeightedRandomSampler for class imbalance
        drop_last: Drop last incomplete batch (required for Opacus)
        num_workers: Number of worker processes

    Returns:
        Configured DataLoader
    """
    sampler = None

    if use_weighted_sampling and dataset.target_column in dataset.df.columns:
        # Use weighted sampling for class imbalance
        sample_weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)
        shuffle = False  # Sampler handles shuffling

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last,  # Required for Opacus compatibility
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        f"DataLoader created: batch_size={batch_size}, "
        f"weighted_sampling={use_weighted_sampling}, "
        f"drop_last={drop_last}"
    )

    return dataloader


def get_categorical_embedding_dims(dataset: PyTorch_Dataset) -> Dict[str, Tuple[int, int]]:
    """
    Calculate embedding dimensions for categorical features.

    Args:
        dataset: PyTorch_Dataset instance

    Returns:
        Dictionary mapping column names to (vocab_size, embedding_dim) tuples
    """
    embedding_dims = {}

    for col in dataset.categorical_columns:
        if col in dataset.df.columns:
            # Get vocabulary size - must be max_value + 1 to accommodate all indices
            # (e.g., if max value is 10, we need vocab_size of 11 for indices 0-10)
            max_value = int(dataset.df[col].max())
            vocab_size = max_value + 1

            # Calculate embedding dimension using rule of thumb: min(50, vocab_size//2)
            embedding_dim = min(50, max(1, vocab_size // 2))

            embedding_dims[col] = (vocab_size, embedding_dim)

    logger.info(f"Calculated embedding dimensions for {len(embedding_dims)} categorical features")
    return embedding_dims


def validate_dataset_compatibility(dataset: PyTorch_Dataset) -> Dict[str, Any]:
    """
    Validate dataset compatibility with Opacus and federated learning requirements.

    Args:
        dataset: PyTorch_Dataset instance

    Returns:
        Validation results dictionary
    """
    validation_results = {"opacus_compatible": True, "federated_ready": True, "issues": [], "recommendations": []}

    # Check for NaN values in tensors
    for feature_type, tensor in dataset.features.items():
        if torch.isnan(tensor).any():
            validation_results["opacus_compatible"] = False
            validation_results["issues"].append(f"NaN values found in {feature_type} features")

    # Check target tensor
    if torch.isnan(dataset.targets).any():
        validation_results["opacus_compatible"] = False
        validation_results["issues"].append("NaN values found in targets")

    # Check for infinite values
    for feature_type, tensor in dataset.features.items():
        if feature_type == "numerical" and torch.isinf(tensor).any():
            validation_results["opacus_compatible"] = False
            validation_results["issues"].append(f"Infinite values found in {feature_type} features")

    # Check dataset size for federated learning
    if len(dataset) < 100:
        validation_results["federated_ready"] = False
        validation_results["issues"].append("Dataset too small for effective federated learning")

    # Check class balance
    if dataset.target_column in dataset.df.columns:
        fraud_rate = dataset.df[dataset.target_column].mean()
        if fraud_rate < 0.001 or fraud_rate > 0.999:
            validation_results["recommendations"].append(
                f"Extreme class imbalance detected (fraud rate: {fraud_rate:.3f})"
            )

    # Check feature counts
    total_features = len(dataset.categorical_columns) + len(dataset.numerical_columns)
    if total_features == 0:
        validation_results["federated_ready"] = False
        validation_results["issues"].append("No features found in dataset")

    return validation_results
