"""
Unit tests for Bank_Client federated learning implementation
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from federated.bank_client import Bank_Client
from model.fraud_mlp import FraudMLP


def test_bank_client_initialization():
    """Test Bank_Client can be initialized correctly"""
    # Create dummy model
    embedding_dims = {"cat1": (10, 5), "cat2": (20, 8)}
    model = FraudMLP(
        categorical_embedding_dims=embedding_dims, numerical_input_dim=10, hidden_dims=[32, 16], dropout_rate=0.3
    )

    # Create dummy data
    cat_data = {"cat1": torch.randint(0, 10, (100,)), "cat2": torch.randint(0, 20, (100,))}
    num_data = torch.randn(100, 10)
    targets = torch.randint(0, 2, (100,)).float()

    # Create simple dataset (not using PyTorch_Dataset for simplicity)
    dataset = TensorDataset(num_data, targets)
    train_loader = DataLoader(dataset, batch_size=16)
    val_loader = DataLoader(dataset, batch_size=16)

    # Initialize client
    client = Bank_Client(
        bank_id="test_bank", model=model, train_loader=train_loader, val_loader=val_loader, device="cpu", local_epochs=2
    )

    assert client.bank_id == "test_bank"
    assert client.local_epochs == 2
    print("✓ Bank_Client initialization test passed")


def test_get_parameters():
    """Test parameter extraction from model"""
    embedding_dims = {"cat1": (10, 5)}
    model = FraudMLP(
        categorical_embedding_dims=embedding_dims, numerical_input_dim=5, hidden_dims=[16], dropout_rate=0.3
    )

    dataset = TensorDataset(torch.randn(50, 5), torch.randint(0, 2, (50,)).float())
    loader = DataLoader(dataset, batch_size=16)

    client = Bank_Client(bank_id="test_bank", model=model, train_loader=loader, val_loader=loader)

    # Get parameters
    params = client.get_parameters(config={})

    assert isinstance(params, list)
    assert len(params) > 0
    assert all(isinstance(p, np.ndarray) for p in params)
    print("✓ get_parameters test passed")


def test_set_parameters():
    """Test setting model parameters"""
    embedding_dims = {"cat1": (10, 5)}
    model = FraudMLP(
        categorical_embedding_dims=embedding_dims, numerical_input_dim=5, hidden_dims=[16], dropout_rate=0.3
    )

    dataset = TensorDataset(torch.randn(50, 5), torch.randint(0, 2, (50,)).float())
    loader = DataLoader(dataset, batch_size=16)

    client = Bank_Client(bank_id="test_bank", model=model, train_loader=loader, val_loader=loader)

    # Get original parameters
    original_params = client.get_parameters(config={})

    # Create modified parameters
    modified_params = [p + 0.1 for p in original_params]

    # Set modified parameters
    client.set_parameters(modified_params)

    # Get parameters again
    new_params = client.get_parameters(config={})

    # Check parameters were updated
    assert not np.allclose(original_params[0], new_params[0])
    print("✓ set_parameters test passed")


if __name__ == "__main__":
    print("Running Bank_Client tests...")
    test_bank_client_initialization()
    test_get_parameters()
    test_set_parameters()
    print("\n✅ All Bank_Client tests passed!")
