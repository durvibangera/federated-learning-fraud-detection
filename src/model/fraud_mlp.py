"""
FraudMLP Neural Network Architecture

This module implements a Multi-Layer Perceptron optimized for fraud detection
with embedding layers for categorical features, GroupNorm for Opacus compatibility,
and proper handling of class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GroupNorm, Dropout, Linear, Embedding
from typing import Dict, List, Tuple, Optional, Any
import logging
import math

logger = logging.getLogger(__name__)


class FraudMLP(nn.Module):
    """
    Multi-Layer Perceptron for fraud detection with embedding layers.

    Designed for federated learning with differential privacy support using
    GroupNorm instead of BatchNorm for Opacus compatibility.
    """

    def __init__(
        self,
        categorical_embedding_dims: Dict[str, Tuple[int, int]],
        numerical_input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.3,
        num_groups: int = 8,
        activation: str = "relu",
    ):
        """
        Initialize FraudMLP model.

        Args:
            categorical_embedding_dims: Dict mapping feature names to (vocab_size, embed_dim)
            numerical_input_dim: Number of numerical input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            num_groups: Number of groups for GroupNorm (must divide hidden dimensions)
            activation: Activation function ('relu', 'gelu', 'leaky_relu')
        """
        super(FraudMLP, self).__init__()

        self.categorical_embedding_dims = categorical_embedding_dims
        self.numerical_input_dim = numerical_input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.num_groups = num_groups
        self.activation = activation

        # Create embedding layers for categorical features
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0

        for feature_name, (vocab_size, embed_dim) in categorical_embedding_dims.items():
            self.embeddings[feature_name] = Embedding(
                num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=None
            )
            total_embedding_dim += embed_dim

        # Calculate total input dimension
        self.total_input_dim = total_embedding_dim + numerical_input_dim

        # Build MLP layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Input layer
        prev_dim = self.total_input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            self.layers.append(Linear(prev_dim, hidden_dim))

            # GroupNorm layer (Opacus compatible)
            # Ensure num_groups divides hidden_dim
            groups = min(self.num_groups, hidden_dim)
            while hidden_dim % groups != 0 and groups > 1:
                groups -= 1

            self.norms.append(GroupNorm(num_groups=groups, num_channels=hidden_dim))

            # Dropout layer
            self.dropouts.append(Dropout(dropout_rate))

            prev_dim = hidden_dim

        # Output layer (binary classification)
        self.output_layer = Linear(prev_dim, 1)

        # Initialize weights
        self._initialize_weights()

        logger.info(
            f"FraudMLP initialized: {self.total_input_dim} input features, "
            f"{len(hidden_dims)} hidden layers, "
            f"{total_embedding_dim} embedding dimensions"
        )

    def _initialize_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, Embedding):
                nn.init.xavier_uniform_(module.weight)

    def _get_activation_fn(self):
        """Get activation function based on configuration."""
        if self.activation == "relu":
            return F.relu
        elif self.activation == "gelu":
            return F.gelu
        elif self.activation == "leaky_relu":
            return F.leaky_relu
        else:
            return F.relu

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            features: Dictionary with 'categorical' and/or 'numerical' tensors

        Returns:
            Output logits (before sigmoid)
        """
        embeddings_list = []

        # Process categorical features through embeddings
        if "categorical" in features and len(self.embeddings) > 0:
            categorical_features = features["categorical"]  # Shape: (batch_size, num_categorical)

            # Apply embeddings for each categorical feature
            for i, (feature_name, embedding_layer) in enumerate(self.embeddings.items()):
                if i < categorical_features.size(1):
                    # Get embedding for this feature
                    feature_embedding = embedding_layer(categorical_features[:, i])
                    embeddings_list.append(feature_embedding)

        # Combine all embeddings
        if embeddings_list:
            categorical_embeddings = torch.cat(embeddings_list, dim=1)
        else:
            categorical_embeddings = torch.empty(
                features.get("numerical", features.get("categorical")).size(0), 0, device=next(self.parameters()).device
            )

        # Process numerical features
        if "numerical" in features:
            numerical_features = features["numerical"]
        else:
            numerical_features = torch.empty(categorical_embeddings.size(0), 0, device=categorical_embeddings.device)

        # Concatenate all features
        if categorical_embeddings.size(1) > 0 and numerical_features.size(1) > 0:
            x = torch.cat([categorical_embeddings, numerical_features], dim=1)
        elif categorical_embeddings.size(1) > 0:
            x = categorical_embeddings
        elif numerical_features.size(1) > 0:
            x = numerical_features
        else:
            raise ValueError("No features provided to the model")

        # Pass through MLP layers
        activation_fn = self._get_activation_fn()

        for layer, norm, dropout in zip(self.layers, self.norms, self.dropouts):
            x = layer(x)
            x = norm(x)
            x = activation_fn(x)
            x = dropout(x)

        # Output layer
        logits = self.output_layer(x)

        return logits.squeeze(-1)  # Remove last dimension for binary classification

    def predict_proba(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get prediction probabilities.

        Args:
            features: Dictionary with 'categorical' and/or 'numerical' tensors

        Returns:
            Probabilities in [0,1] range
        """
        with torch.no_grad():
            logits = self.forward(features)
            probabilities = torch.sigmoid(logits)
            return probabilities

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        embedding_info = {}
        for name, (vocab_size, embed_dim) in self.categorical_embedding_dims.items():
            embedding_info[name] = {
                "vocab_size": vocab_size,
                "embedding_dim": embed_dim,
                "parameters": vocab_size * embed_dim,
            }

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "total_input_dim": self.total_input_dim,
            "numerical_input_dim": self.numerical_input_dim,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "num_groups": self.num_groups,
            "embedding_info": embedding_info,
            "opacus_compatible": True,  # Uses GroupNorm instead of BatchNorm
        }


class FraudLoss(nn.Module):
    """
    Custom loss function for fraud detection with class imbalance handling.

    Uses Binary Cross Entropy with configurable positive class weighting.
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        """
        Initialize fraud detection loss.

        Args:
            pos_weight: Weight for positive class (fraud) to handle imbalance
            reduction: Loss reduction method ('mean', 'sum', 'none')
        """
        super(FraudLoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for fraud detection.

        Args:
            logits: Model output logits (before sigmoid)
            targets: Ground truth labels (0 or 1)

        Returns:
            Computed loss
        """
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction=self.reduction)


def create_fraud_model(
    categorical_embedding_dims: Dict[str, Tuple[int, int]],
    numerical_input_dim: int,
    class_weights: Optional[torch.Tensor] = None,
    hidden_dims: List[int] = [256, 128, 64],
    dropout_rate: float = 0.3,
    device: str = "cpu",
) -> Tuple[FraudMLP, FraudLoss, torch.optim.Optimizer]:
    """
    Create complete fraud detection model with loss and optimizer.

    Args:
        categorical_embedding_dims: Embedding dimensions for categorical features
        numerical_input_dim: Number of numerical features
        class_weights: Weights for handling class imbalance
        hidden_dims: Hidden layer dimensions
        dropout_rate: Dropout probability
        device: Device to place model on

    Returns:
        Tuple of (model, loss_function, optimizer)
    """
    # Create model
    model = FraudMLP(
        categorical_embedding_dims=categorical_embedding_dims,
        numerical_input_dim=numerical_input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
    ).to(device)

    # Create loss function with class weighting
    pos_weight = None
    if class_weights is not None and len(class_weights) >= 2:
        # pos_weight should be weight_of_positive_class / weight_of_negative_class
        pos_weight = (class_weights[1] / class_weights[0]).to(device)

    loss_fn = FraudLoss(pos_weight=pos_weight)

    # Create optimizer (Adam with weight decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5, betas=(0.9, 0.999))

    logger.info(f"Fraud model created: {model.get_model_info()['total_parameters']} parameters")

    return model, loss_fn, optimizer


def validate_model_architecture(model: FraudMLP, sample_features: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Validate model architecture and compatibility.

    Args:
        model: FraudMLP model instance
        sample_features: Sample input features for testing

    Returns:
        Validation results
    """
    validation_results = {
        "architecture_valid": True,
        "opacus_compatible": True,
        "forward_pass_successful": False,
        "output_shape_correct": False,
        "issues": [],
        "model_info": model.get_model_info(),
    }

    try:
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(sample_features)
            validation_results["forward_pass_successful"] = True

            # Check output shape
            batch_size = next(iter(sample_features.values())).size(0)
            if output.shape == (batch_size,):
                validation_results["output_shape_correct"] = True
            else:
                validation_results["issues"].append(
                    f"Incorrect output shape: expected ({batch_size},), got {output.shape}"
                )

            # Check output range after sigmoid
            probabilities = torch.sigmoid(output)
            if torch.all(probabilities >= 0) and torch.all(probabilities <= 1):
                validation_results["output_range_valid"] = True
            else:
                validation_results["issues"].append("Output probabilities not in [0,1] range")

    except Exception as e:
        validation_results["issues"].append(f"Forward pass failed: {str(e)}")

    # Check for BatchNorm layers (not Opacus compatible)
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            validation_results["opacus_compatible"] = False
            validation_results["issues"].append(f"Found BatchNorm layer: {name}")

    # Check GroupNorm configuration
    for name, module in model.named_modules():
        if isinstance(module, GroupNorm):
            if module.num_channels % module.num_groups != 0:
                validation_results["issues"].append(
                    f"Invalid GroupNorm configuration in {name}: "
                    f"{module.num_channels} channels, {module.num_groups} groups"
                )

    return validation_results


def calculate_model_flops(model: FraudMLP, sample_features: Dict[str, torch.Tensor]) -> int:
    """
    Estimate FLOPs (Floating Point Operations) for the model.

    Args:
        model: FraudMLP model instance
        sample_features: Sample input features

    Returns:
        Estimated FLOPs count
    """
    flops = 0

    # Embedding lookups (minimal FLOPs, mainly memory access)
    if "categorical" in sample_features:
        for feature_name, (vocab_size, embed_dim) in model.categorical_embedding_dims.items():
            flops += embed_dim  # Lookup operation per sample

    # Linear layer FLOPs: input_dim * output_dim * 2 (multiply-add)
    prev_dim = model.total_input_dim

    for hidden_dim in model.hidden_dims:
        flops += prev_dim * hidden_dim * 2  # Linear layer
        flops += hidden_dim  # GroupNorm (approximation)
        prev_dim = hidden_dim

    # Output layer
    flops += prev_dim * 1 * 2  # Binary classification

    return flops
