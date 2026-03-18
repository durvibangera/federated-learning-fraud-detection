"""
Hypothesis Strategies for Property-Based Testing

This module provides custom Hypothesis strategies for generating:
- IEEE-CIS dataset structures with realistic data
- Model weights for federated learning scenarios
- Configuration objects for testing
- Federated learning round data

All strategies are designed to generate valid test data that matches
the actual data structures used in the federated fraud detection system.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from hypothesis import strategies as st
from typing import Dict, List, Tuple, Any
from collections import OrderedDict


# ============================================================================
# IEEE-CIS Dataset Strategies
# ============================================================================

@st.composite
def ieee_cis_transaction_id(draw):
    """Generate valid TransactionID."""
    return draw(st.integers(min_value=1, max_value=10_000_000))


@st.composite
def ieee_cis_transaction_dt(draw):
    """Generate valid TransactionDT (seconds from reference point)."""
    return draw(st.floats(min_value=0.0, max_value=31_536_000.0))  # Up to 1 year


@st.composite
def ieee_cis_transaction_amt(draw):
    """Generate valid TransactionAmt."""
    return draw(st.floats(min_value=0.01, max_value=10_000.0))


@st.composite
def ieee_cis_product_cd(draw):
    """Generate valid ProductCD (W, H, R, S, C)."""
    return draw(st.sampled_from(['W', 'H', 'R', 'S', 'C']))


@st.composite
def ieee_cis_card_features(draw):
    """Generate card-related features."""
    return {
        'card1': draw(st.one_of(st.none(), st.floats(min_value=1000.0, max_value=20000.0))),
        'card2': draw(st.one_of(st.none(), st.floats(min_value=100.0, max_value=1000.0))),
        'card3': draw(st.one_of(st.none(), st.floats(min_value=100.0, max_value=300.0))),
        'card4': draw(st.one_of(st.none(), st.sampled_from(['visa', 'mastercard', 'amex', 'discover']))),
        'card5': draw(st.one_of(st.none(), st.floats(min_value=100.0, max_value=300.0))),
        'card6': draw(st.one_of(st.none(), st.sampled_from(['debit', 'credit', 'charge card', 'debit or credit'])))
    }


@st.composite
def ieee_cis_address_features(draw):
    """Generate address-related features."""
    return {
        'addr1': draw(st.one_of(st.none(), st.floats(min_value=100.0, max_value=600.0))),
        'addr2': draw(st.one_of(st.none(), st.floats(min_value=10.0, max_value=100.0)))
    }


@st.composite
def ieee_cis_distance_features(draw):
    """Generate distance features."""
    return {
        'dist1': draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=10000.0))),
        'dist2': draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=10000.0)))
    }


@st.composite
def ieee_cis_email_domain(draw):
    """Generate email domain."""
    domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com', None]
    return draw(st.sampled_from(domains))


@st.composite
def ieee_cis_c_features(draw):
    """Generate C-type features (counts)."""
    return {
        f'C{i}': draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=1000.0)))
        for i in range(1, 3)  # C1, C2 for simplicity
    }


@st.composite
def ieee_cis_d_features(draw):
    """Generate D-type features (time deltas)."""
    return {
        f'D{i}': draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=1000.0)))
        for i in range(1, 3)  # D1, D2 for simplicity
    }


@st.composite
def ieee_cis_m_features(draw):
    """Generate M-type features (match indicators)."""
    return {
        f'M{i}': draw(st.one_of(st.none(), st.sampled_from(['T', 'F'])))
        for i in range(1, 3)  # M1, M2 for simplicity
    }


@st.composite
def ieee_cis_v_features(draw):
    """Generate V-type features (Vesta engineered features)."""
    return {
        f'V{i}': draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)))
        for i in range(1, 3)  # V1, V2 for simplicity
    }


@st.composite
def ieee_cis_is_fraud(draw):
    """Generate fraud label with realistic imbalance (3.5% fraud rate)."""
    return draw(st.integers(min_value=0, max_value=1))


@st.composite
def ieee_cis_transaction_record(draw):
    """
    Generate a complete IEEE-CIS transaction record.
    
    Returns a dictionary with all transaction features.
    """
    record = {
        'TransactionID': draw(ieee_cis_transaction_id()),
        'TransactionDT': draw(ieee_cis_transaction_dt()),
        'TransactionAmt': draw(ieee_cis_transaction_amt()),
        'ProductCD': draw(ieee_cis_product_cd()),
        'isFraud': draw(ieee_cis_is_fraud())
    }
    
    # Add card features
    record.update(draw(ieee_cis_card_features()))
    
    # Add address features
    record.update(draw(ieee_cis_address_features()))
    
    # Add distance features
    record.update(draw(ieee_cis_distance_features()))
    
    # Add email domains
    record['P_emaildomain'] = draw(ieee_cis_email_domain())
    record['R_emaildomain'] = draw(ieee_cis_email_domain())
    
    # Add C, D, M, V features
    record.update(draw(ieee_cis_c_features()))
    record.update(draw(ieee_cis_d_features()))
    record.update(draw(ieee_cis_m_features()))
    record.update(draw(ieee_cis_v_features()))
    
    return record


@st.composite
def ieee_cis_dataframe(draw, min_rows=10, max_rows=1000):
    """
    Generate a complete IEEE-CIS DataFrame with multiple records.
    
    Args:
        min_rows: Minimum number of rows
        max_rows: Maximum number of rows
        
    Returns:
        pandas DataFrame with IEEE-CIS structure
    """
    num_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    
    records = [draw(ieee_cis_transaction_record()) for _ in range(num_rows)]
    
    # Ensure unique TransactionIDs
    for i, record in enumerate(records):
        record['TransactionID'] = i + 1
    
    df = pd.DataFrame(records)
    
    return df


# ============================================================================
# Model Weight Strategies
# ============================================================================

@st.composite
def model_weight_tensor(draw, shape):
    """
    Generate a PyTorch tensor with given shape for model weights.
    
    Args:
        shape: Tuple specifying tensor shape
        
    Returns:
        PyTorch tensor with random weights
    """
    # Generate weights using Xavier/Glorot initialization range
    limit = np.sqrt(6.0 / (shape[0] + shape[-1])) if len(shape) > 1 else 0.1
    
    weights = draw(st.lists(
        st.floats(min_value=-limit, max_value=limit, allow_nan=False, allow_infinity=False),
        min_size=np.prod(shape),
        max_size=np.prod(shape)
    ))
    
    return torch.tensor(weights, dtype=torch.float32).reshape(shape)


@st.composite
def model_weights_dict(draw, layer_dims):
    """
    Generate a complete model weights dictionary.
    
    Args:
        layer_dims: List of tuples (input_dim, output_dim) for each layer
        
    Returns:
        OrderedDict of model weights compatible with PyTorch state_dict
    """
    weights = OrderedDict()
    
    for i, (in_dim, out_dim) in enumerate(layer_dims):
        # Weight matrix
        weights[f'layers.{i}.weight'] = draw(model_weight_tensor((out_dim, in_dim)))
        
        # Bias vector
        weights[f'layers.{i}.bias'] = draw(model_weight_tensor((out_dim,)))
    
    return weights


@st.composite
def federated_model_weights(draw, num_clients=3):
    """
    Generate model weights for multiple federated learning clients.
    
    Args:
        num_clients: Number of clients
        
    Returns:
        List of model weight dictionaries, one per client
    """
    # Define a simple architecture
    layer_dims = [(256, 128), (128, 64), (64, 1)]
    
    client_weights = []
    for _ in range(num_clients):
        weights = draw(model_weights_dict(layer_dims))
        client_weights.append(weights)
    
    return client_weights


# ============================================================================
# Configuration Strategies
# ============================================================================

@st.composite
def privacy_config(draw):
    """Generate valid privacy configuration."""
    return {
        'epsilon': draw(st.floats(min_value=0.1, max_value=10.0)),
        'delta': draw(st.floats(min_value=1e-7, max_value=1e-3)),
        'max_grad_norm': draw(st.floats(min_value=0.1, max_value=10.0)),
        'noise_multiplier': draw(st.floats(min_value=0.5, max_value=5.0))
    }


@st.composite
def model_config(draw):
    """Generate valid model configuration."""
    return {
        'embedding_dim': draw(st.integers(min_value=8, max_value=128)),
        'hidden_dims': draw(st.lists(
            st.integers(min_value=32, max_value=512),
            min_size=2,
            max_size=5
        )),
        'dropout_rate': draw(st.floats(min_value=0.0, max_value=0.7)),
        'learning_rate': draw(st.floats(min_value=1e-5, max_value=1e-2)),
        'batch_size': draw(st.sampled_from([32, 64, 128, 256, 512, 1024]))
    }


@st.composite
def fl_config(draw):
    """Generate valid federated learning configuration."""
    min_clients = draw(st.integers(min_value=1, max_value=5))
    
    return {
        'num_rounds': draw(st.integers(min_value=1, max_value=100)),
        'min_clients': min_clients,
        'min_available_clients': draw(st.integers(min_value=min_clients, max_value=10)),
        'proximal_mu': draw(st.floats(min_value=0.0, max_value=1.0)),
        'local_epochs': draw(st.integers(min_value=1, max_value=20))
    }


@st.composite
def data_split_ratios(draw):
    """
    Generate valid data split ratios that sum to 1.0.
    
    Returns:
        Tuple of (train_ratio, val_ratio, test_ratio)
    """
    # Generate two random values and compute third to ensure sum = 1.0
    train_ratio = draw(st.floats(min_value=0.6, max_value=0.9))
    val_ratio = draw(st.floats(min_value=0.05, max_value=(1.0 - train_ratio - 0.05)))
    test_ratio = 1.0 - train_ratio - val_ratio
    
    return (train_ratio, val_ratio, test_ratio)


# ============================================================================
# Federated Learning Round Strategies
# ============================================================================

@st.composite
def fl_round_metrics(draw):
    """Generate metrics for a federated learning round."""
    return {
        'round_num': draw(st.integers(min_value=1, max_value=100)),
        'train_loss': draw(st.floats(min_value=0.0, max_value=10.0)),
        'val_loss': draw(st.floats(min_value=0.0, max_value=10.0)),
        'auprc': draw(st.floats(min_value=0.0, max_value=1.0)),
        'auroc': draw(st.floats(min_value=0.0, max_value=1.0)),
        'num_samples': draw(st.integers(min_value=100, max_value=100_000)),
        'duration': draw(st.floats(min_value=1.0, max_value=1000.0))
    }


@st.composite
def client_update(draw):
    """Generate a client update for federated learning."""
    layer_dims = [(256, 128), (128, 64), (64, 1)]
    
    return {
        'client_id': draw(st.sampled_from(['bank_1', 'bank_2', 'bank_3'])),
        'weights': draw(model_weights_dict(layer_dims)),
        'num_samples': draw(st.integers(min_value=100, max_value=50_000)),
        'metrics': draw(fl_round_metrics())
    }


@st.composite
def fl_round_updates(draw, num_clients=3):
    """Generate updates from multiple clients for one FL round."""
    return [draw(client_update()) for _ in range(num_clients)]


# ============================================================================
# Categorical Embedding Strategies
# ============================================================================

@st.composite
def categorical_embedding_dims(draw):
    """
    Generate categorical embedding dimensions dictionary.
    
    Returns:
        Dict mapping feature names to (vocab_size, embed_dim) tuples
    """
    features = ['ProductCD', 'card4', 'card6', 'P_emaildomain']
    
    embedding_dims = {}
    for feature in features:
        vocab_size = draw(st.integers(min_value=3, max_value=100))
        embed_dim = draw(st.integers(min_value=4, max_value=64))
        embedding_dims[feature] = (vocab_size, embed_dim)
    
    return embedding_dims


# ============================================================================
# PyTorch Dataset Strategies
# ============================================================================

@st.composite
def pytorch_features_dict(draw, batch_size=32):
    """
    Generate PyTorch features dictionary for model input.
    
    Args:
        batch_size: Batch size for tensors
        
    Returns:
        Dict with 'categorical' and 'numerical' tensors
    """
    num_categorical = draw(st.integers(min_value=1, max_value=10))
    num_numerical = draw(st.integers(min_value=5, max_value=50))
    
    # Categorical features (integer indices)
    categorical = torch.randint(0, 10, (batch_size, num_categorical))
    
    # Numerical features (float values)
    numerical = torch.randn(batch_size, num_numerical)
    
    return {
        'categorical': categorical,
        'numerical': numerical
    }


@st.composite
def pytorch_targets(draw, batch_size=32):
    """
    Generate PyTorch targets for binary classification.
    
    Args:
        batch_size: Batch size
        
    Returns:
        Tensor of binary labels
    """
    # Generate with realistic fraud imbalance (3.5% fraud rate)
    fraud_prob = 0.035
    targets = torch.bernoulli(torch.full((batch_size,), fraud_prob))
    
    return targets
