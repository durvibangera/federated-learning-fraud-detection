"""
Example: Federated Learning with 3 Banks

This script demonstrates how to set up and run federated learning
with the Bank_Client and Aggregation_Server.

Usage:
    # Terminal 1 - Start server:
    python examples/federated_learning_example.py --mode server
    
    # Terminal 2 - Start bank 1:
    python examples/federated_learning_example.py --mode client --bank-id bank_1
    
    # Terminal 3 - Start bank 2:
    python examples/federated_learning_example.py --mode client --bank-id bank_2
    
    # Terminal 4 - Start bank 3:
    python examples/federated_learning_example.py --mode client --bank-id bank_3
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import argparse
import flwr as fl
from pathlib import Path

from federated import Bank_Client, Aggregation_Server
from model import FraudMLP, PyTorch_Dataset, create_fraud_dataloader, get_categorical_embedding_dims
from data.csv_parser import CSV_Parser
from data.preprocessor import Data_Preprocessor


def prepare_data(bank_id: str):
    """
    Prepare data for a specific bank.
    
    Args:
        bank_id: Bank identifier (bank_1, bank_2, or bank_3)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, feature_info)
    """
    print(f"\n{'='*60}")
    print(f"Preparing data for {bank_id}")
    print(f"{'='*60}")
    
    # Load and preprocess data
    parser = CSV_Parser()
    data_dir = Path('data/raw')
    
    train_transaction = parser.parse_csv(data_dir / 'train_transaction.csv', 'transaction')
    train_identity = parser.parse_csv(data_dir / 'train_identity.csv', 'identity')
    
    preprocessor = Data_Preprocessor(missing_threshold=0.5, random_state=42)
    merged_df = preprocessor.merge_datasets(train_transaction, train_identity)
    cleaned_df = preprocessor.handle_missing_values(merged_df)
    
    # Partition data by bank
    bank_partitions = preprocessor.partition_by_product_cd(cleaned_df)
    
    # Map bank_id to partition
    bank_map = {
        'bank_1': 'W',
        'bank_2': 'H',  # or 'R'
        'bank_3': 'S'   # or 'C'
    }
    
    partition_key = bank_map.get(bank_id, 'W')
    bank_data = bank_partitions.get(partition_key, cleaned_df[:10000])  # Fallback to subset
    
    print(f"{bank_id} data: {len(bank_data):,} transactions")
    
    # Split data
    train_df, val_df, test_df = preprocessor.temporal_split(bank_data)
    train_enc, val_enc, test_enc = preprocessor.encode_categorical_features(train_df, val_df, test_df)
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    categorical_prefixes = ("ProductCD", "card4", "card6", "P_email", "R_email", "Device", "id_")
    numerical_cols = [col for col in train_enc.columns 
                     if col not in ["TransactionID", "isFraud"] 
                     and train_enc[col].dtype in ["float64", "int64", "float32", "int32"]
                     and not col.startswith(categorical_prefixes)]
    
    scaler = StandardScaler()
    train_enc[numerical_cols] = scaler.fit_transform(train_enc[numerical_cols])
    val_enc[numerical_cols] = scaler.transform(val_enc[numerical_cols])
    test_enc[numerical_cols] = scaler.transform(test_enc[numerical_cols])
    
    # Create datasets
    train_dataset = PyTorch_Dataset(train_enc, device='cpu')
    val_dataset = PyTorch_Dataset(val_enc, device='cpu')
    test_dataset = PyTorch_Dataset(test_enc, device='cpu')
    
    # Create dataloaders
    train_loader = create_fraud_dataloader(train_dataset, batch_size=256)
    val_loader = create_fraud_dataloader(val_dataset, batch_size=256, shuffle=False, use_weighted_sampling=False)
    test_loader = create_fraud_dataloader(test_dataset, batch_size=256, shuffle=False, use_weighted_sampling=False)
    
    feature_info = train_dataset.get_feature_info()
    
    print(f"✓ Data prepared for {bank_id}")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val: {len(val_dataset):,} samples")
    print(f"  Test: {len(test_dataset):,} samples")
    
    return train_loader, val_loader, test_loader, feature_info


def start_server(num_rounds: int = 30):
    """Start the aggregation server."""
    print("\n" + "="*70)
    print("STARTING AGGREGATION SERVER")
    print("="*70)
    
    server = Aggregation_Server(
        num_rounds=num_rounds,
        min_clients=2,  # Require at least 2 clients
        proximal_mu=0.01,
        server_address="[::]:8080"
    )
    
    history = server.start_federated_learning()
    
    print("\n" + "="*70)
    print("FEDERATED LEARNING COMPLETE")
    print("="*70)
    print(f"Rounds completed: {len(history.losses_distributed)}")
    
    return history


def start_client(bank_id: str):
    """Start a bank client."""
    print(f"\n{'='*70}")
    print(f"STARTING {bank_id.upper()}")
    print(f"{'='*70}")
    
    # Prepare data
    train_loader, val_loader, test_loader, feature_info = prepare_data(bank_id)
    
    # Create model
    embedding_dims = get_categorical_embedding_dims(train_loader.dataset)
    model = FraudMLP(
        categorical_embedding_dims=embedding_dims,
        numerical_input_dim=feature_info['numerical_features'],
        hidden_dims=[512, 256],  # Wider architecture from experiments
        dropout_rate=0.3
    )
    
    print(f"\n✓ Model created: {model.get_model_info()['total_parameters']:,} parameters")
    
    # Create client
    client = Bank_Client(
        bank_id=bank_id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cpu',
        local_epochs=5,
        learning_rate=0.001
    )
    
    print(f"✓ {bank_id} ready to connect to server")
    print(f"  Connecting to: localhost:8080")
    
    # Start client (connects to server)
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client
    )
    
    print(f"\n✓ {bank_id} completed federated learning")


def main():
    parser = argparse.ArgumentParser(description='Federated Learning Example')
    parser.add_argument('--mode', type=str, required=True, choices=['server', 'client'],
                       help='Run as server or client')
    parser.add_argument('--bank-id', type=str, default='bank_1',
                       choices=['bank_1', 'bank_2', 'bank_3'],
                       help='Bank identifier (for client mode)')
    parser.add_argument('--rounds', type=int, default=5,
                       help='Number of FL rounds (for server mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'server':
        start_server(num_rounds=args.rounds)
    else:
        start_client(bank_id=args.bank_id)


if __name__ == '__main__':
    main()
