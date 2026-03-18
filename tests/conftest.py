"""
Pytest configuration and shared fixtures for property-based testing.

This module provides:
- Hypothesis configuration for all property tests
- Shared data generators for IEEE-CIS dataset structure
- Model weight generators for federated learning scenarios
- Common fixtures for testing infrastructure
"""

import pytest
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from hypothesis import settings, Verbosity

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# ============================================================================
# Hypothesis Configuration
# ============================================================================

# Configure Hypothesis for all property tests
# Minimum 100 iterations per property test as per design document
settings.register_profile("default", max_examples=100, deadline=None)
settings.register_profile("ci", max_examples=200, deadline=None)
settings.register_profile("dev", max_examples=50, deadline=None, verbosity=Verbosity.verbose)
settings.register_profile("debug", max_examples=10, deadline=None, verbosity=Verbosity.verbose)

# Load profile from environment or use default
settings.load_profile("default")


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def device():
    """Get PyTorch device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def random_seed():
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def set_random_seeds(random_seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


@pytest.fixture(scope="session")
def sample_ieee_cis_schema():
    """
    IEEE-CIS dataset schema for testing.
    
    Returns dictionary with column names and types.
    """
    return {
        'TransactionID': 'int64',
        'TransactionDT': 'float64',
        'TransactionAmt': 'float64',
        'ProductCD': 'object',
        'card1': 'float64',
        'card2': 'float64',
        'card3': 'float64',
        'card4': 'object',
        'card5': 'float64',
        'card6': 'object',
        'addr1': 'float64',
        'addr2': 'float64',
        'dist1': 'float64',
        'dist2': 'float64',
        'P_emaildomain': 'object',
        'R_emaildomain': 'object',
        'C1': 'float64',
        'C2': 'float64',
        'D1': 'float64',
        'D2': 'float64',
        'M1': 'object',
        'M2': 'object',
        'V1': 'float64',
        'V2': 'float64',
        'isFraud': 'int64'
    }


@pytest.fixture(scope="session")
def categorical_features():
    """List of categorical features in IEEE-CIS dataset."""
    return [
        'ProductCD', 'card4', 'card6', 
        'P_emaildomain', 'R_emaildomain',
        'M1', 'M2'
    ]


@pytest.fixture(scope="session")
def numerical_features():
    """List of numerical features in IEEE-CIS dataset."""
    return [
        'TransactionDT', 'TransactionAmt',
        'card1', 'card2', 'card3', 'card5',
        'addr1', 'addr2', 'dist1', 'dist2',
        'C1', 'C2', 'D1', 'D2', 'V1', 'V2'
    ]


@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing."""
    return {
        'categorical_embedding_dims': {
            'ProductCD': (5, 10),
            'card4': (4, 8),
            'card6': (4, 8)
        },
        'numerical_input_dim': 16,
        'hidden_dims': [256, 128, 64],
        'dropout_rate': 0.3,
        'num_groups': 8
    }


@pytest.fixture
def sample_privacy_config():
    """Sample privacy configuration for testing."""
    return {
        'epsilon': 1.0,
        'delta': 1e-5,
        'max_grad_norm': 1.0,
        'noise_multiplier': 1.1
    }


@pytest.fixture
def sample_fl_config():
    """Sample federated learning configuration for testing."""
    return {
        'num_rounds': 30,
        'min_clients': 2,
        'min_available_clients': 3,
        'proximal_mu': 0.01,
        'local_epochs': 5
    }


# ============================================================================
# Test Markers
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "property: mark test as a property-based test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_data: mark test as requiring IEEE-CIS dataset"
    )
