"""
Model package for fraud detection.

This package contains PyTorch implementations for fraud detection including:
- PyTorch_Dataset: Custom dataset class with fraud-specific features
- FraudMLP: Neural network architecture optimized for fraud detection
- Supporting utilities for model creation and validation
"""

from .pytorch_dataset import (
    PyTorch_Dataset,
    create_fraud_dataloader,
    get_categorical_embedding_dims,
    validate_dataset_compatibility,
)

from .fraud_mlp import FraudMLP, FraudLoss, create_fraud_model, validate_model_architecture, calculate_model_flops

__all__ = [
    "PyTorch_Dataset",
    "create_fraud_dataloader",
    "get_categorical_embedding_dims",
    "validate_dataset_compatibility",
    "FraudMLP",
    "FraudLoss",
    "create_fraud_model",
    "validate_model_architecture",
    "calculate_model_flops",
]
