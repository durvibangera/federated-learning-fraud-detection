"""
Simple Federated Learning Test using Flower Simulation

This runs all clients and server in a single process for easier testing.
"""

import sys
sys.path.append('src')

import torch
from pathlib import Path
import flwr as fl
from flwr.server.strategy import FedProx

from federated import Bank_Client
from model import FraudMLP, PyTorch_Dataset, create_fraud_dataloader, get_categorical_embedding_dims
from data.csv_parser import CSV_Parser
from data.preprocessor import Data_Preprocessor
from sklearn.preprocessing import StandardScaler


def prepare_bank_data(bank_id: str):
    """Prepare data for a specific bank"""
    print(f"\nPreparing data for {bank_id}...")
    
    # Load data
    parser = CSV_Parser()
    data_dir = Path('data/raw')
    
    train_transaction = parser.parse_csv(data_dir / 'train_transaction.csv', 'transaction')
    train_identity = parser.parse_csv(data_dir / 'train_identity.csv', 'identity')
    
    # Preprocess
    preprocessor = Data_Preprocessor(missing_threshold=0.5, random_state=42)
    merged_df = preprocessor.merge_datasets(train_transaction, train_identity)
    cleaned_df = preprocessor.handle_missing_values(merged_df)
    
    # Use subset for faster testing
    bank_data = cleaned_df[:10000]
    
    # Split
    train_df, val_df, test_df = preprocessor.temporal_split(bank_data)
    train_enc, val_enc, test_enc = preprocessor.encode_categorical_features(train_df, val_df, test_df)
    
    # Normalize
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
    
    # Create dataloaders
    train_loader = create_fraud_dataloader(train_dataset, batch_size=256)
    val_loader = create_fraud_dataloader(val_dataset, batch_size=256, shuffle=False, use_weighted_sampling=False)
    
    feature_info = train_dataset.get_feature_info()
    
    print(f"✓ {bank_id}: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    return train_loader, val_loader, feature_info


def create_client_fn(bank_id: str, train_loader, val_loader, feature_info):
    """Create a client function for Flower simulation"""
    def client_fn(cid: str):
        # Create model
        embedding_dims = get_categorical_embedding_dims(train_loader.dataset)
        model = FraudMLP(
            categorical_embedding_dims=embedding_dims,
            numerical_input_dim=feature_info['numerical_features'],
            hidden_dims=[256, 128],  # Smaller for faster testing
            dropout_rate=0.3
        )
        
        # Create client
        client = Bank_Client(
            bank_id=f"{bank_id}_{cid}",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu',
            local_epochs=2,  # Fewer epochs for testing
            learning_rate=0.001
        )
        
        return client
    
    return client_fn


def main():
    print("\n" + "="*70)
    print("FEDERATED LEARNING SIMULATION TEST")
    print("="*70)
    
    # Prepare data for 3 banks
    print("\n[1/3] Preparing data...")
    bank1_train, bank1_val, feature_info = prepare_bank_data("bank_1")
    bank2_train, bank2_val, _ = prepare_bank_data("bank_2")
    bank3_train, bank3_val, _ = prepare_bank_data("bank_3")
    
    # Create client functions
    print("\n[2/3] Creating clients...")
    clients = {
        "0": create_client_fn("bank_1", bank1_train, bank1_val, feature_info)("0"),
        "1": create_client_fn("bank_2", bank2_train, bank2_val, feature_info)("1"),
        "2": create_client_fn("bank_3", bank3_train, bank3_val, feature_info)("2"),
    }
    
    print("✓ Created 3 bank clients")
    
    # Create strategy
    strategy = FedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        proximal_mu=0.01
    )
    
    print("\n[3/3] Starting federated learning...")
    print("  Rounds: 2")
    print("  Strategy: FedProx (μ=0.01)")
    print("  Clients: 3 banks")
    print()
    
    # Run simulation
    try:
        history = fl.simulation.start_simulation(
            client_fn=lambda cid: clients[cid],
            num_clients=3,
            config=fl.server.ServerConfig(num_rounds=2),
            strategy=strategy,
        )
        
        print("\n" + "="*70)
        print("✅ FEDERATED LEARNING COMPLETE!")
        print("="*70)
        print(f"Rounds completed: {len(history.losses_distributed)}")
        print(f"Final loss: {history.losses_distributed[-1][1]:.4f}")
        
        if history.metrics_distributed:
            print("\nFinal metrics:")
            for key, values in history.metrics_distributed.items():
                if values:
                    print(f"  {key}: {values[-1][1]:.4f}")
        
    except Exception as e:
        print(f"\n❌ Error during federated learning: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
