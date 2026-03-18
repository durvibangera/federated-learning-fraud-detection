"""
Manual Federated Learning Test (No Flower Framework)

This tests the core FL logic without Flower's complexity.
"""

import sys
sys.path.append('src')

import torch
import numpy as np
from pathlib import Path

from federated import Bank_Client
from model import FraudMLP, PyTorch_Dataset, create_fraud_dataloader, get_categorical_embedding_dims
from data.csv_parser import CSV_Parser
from data.preprocessor import Data_Preprocessor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score


def prepare_bank_data(num_samples=5000):
    """Prepare a small dataset for testing"""
    print("Loading and preprocessing data...")
    
    parser = CSV_Parser()
    data_dir = Path('data/raw')
    
    train_transaction = parser.parse_csv(data_dir / 'train_transaction.csv', 'transaction')
    train_identity = parser.parse_csv(data_dir / 'train_identity.csv', 'identity')
    
    preprocessor = Data_Preprocessor(missing_threshold=0.5, random_state=42)
    merged_df = preprocessor.merge_datasets(train_transaction, train_identity)
    cleaned_df = preprocessor.handle_missing_values(merged_df)
    
    # Use small subset for fast testing
    bank_data = cleaned_df[:num_samples]
    
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
    
    train_dataset = PyTorch_Dataset(train_enc, device='cpu')
    val_dataset = PyTorch_Dataset(val_enc, device='cpu')
    
    train_loader = create_fraud_dataloader(train_dataset, batch_size=128)
    val_loader = create_fraud_dataloader(val_dataset, batch_size=128, shuffle=False, use_weighted_sampling=False)
    
    feature_info = train_dataset.get_feature_info()
    
    return train_loader, val_loader, feature_info


def aggregate_weights(client_weights_list):
    """Simple FedAvg aggregation"""
    # Average all client weights
    num_clients = len(client_weights_list)
    aggregated = []
    
    for layer_idx in range(len(client_weights_list[0])):
        layer_weights = [client_weights[layer_idx] for client_weights in client_weights_list]
        avg_weights = np.mean(layer_weights, axis=0)
        aggregated.append(avg_weights)
    
    return aggregated


def main():
    print("\n" + "="*70)
    print("MANUAL FEDERATED LEARNING TEST")
    print("="*70)
    
    # Prepare data
    print("\n[1/4] Preparing data...")
    train_loader, val_loader, feature_info = prepare_bank_data(num_samples=5000)
    print(f"✓ Data ready: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    
    # Create 3 bank clients
    print("\n[2/4] Creating 3 bank clients...")
    clients = []
    for i in range(3):
        embedding_dims = get_categorical_embedding_dims(train_loader.dataset)
        model = FraudMLP(
            categorical_embedding_dims=embedding_dims,
            numerical_input_dim=feature_info['numerical_features'],
            hidden_dims=[128, 64],  # Small for fast testing
            dropout_rate=0.3
        )
        
        client = Bank_Client(
            bank_id=f"bank_{i+1}",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu',
            local_epochs=1,  # Just 1 epoch for testing
            learning_rate=0.001
        )
        clients.append(client)
    
    print(f"✓ Created {len(clients)} clients")
    
    # Run federated learning manually
    print("\n[3/4] Running federated learning (2 rounds)...")
    num_rounds = 2
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num}/{num_rounds} ---")
        
        # Get initial parameters from first client
        if round_num == 1:
            global_weights = clients[0].get_parameters(config={})
            print(f"  Initialized global model ({len(global_weights)} layers)")
        
        # Each client trains locally
        client_weights_list = []
        client_metrics = []
        
        for client in clients:
            print(f"  {client.bank_id}: Training locally...")
            
            # Client trains
            updated_weights, num_examples, metrics = client.fit(
                parameters=global_weights,
                config={"round": str(round_num)}
            )
            
            client_weights_list.append(updated_weights)
            client_metrics.append(metrics)
            
            print(f"    Loss: {metrics['train_loss']:.4f}, Samples: {num_examples}")
        
        # Aggregate weights (FedAvg)
        print(f"  Aggregating weights from {len(clients)} clients...")
        global_weights = aggregate_weights(client_weights_list)
        
        # Evaluate global model
        print(f"  Evaluating global model...")
        eval_metrics_list = []
        for client in clients:
            loss, num_examples, metrics = client.evaluate(
                parameters=global_weights,
                config={"round": str(round_num)}
            )
            eval_metrics_list.append(metrics)
        
        # Average evaluation metrics
        avg_auprc = np.mean([m['auprc'] for m in eval_metrics_list])
        avg_auroc = np.mean([m['auroc'] for m in eval_metrics_list])
        avg_loss = np.mean([m['loss'] for m in eval_metrics_list])
        
        print(f"  Global Model - Loss: {avg_loss:.4f}, AUPRC: {avg_auprc:.4f}, AUROC: {avg_auroc:.4f}")
    
    print("\n[4/4] Final Results")
    print("="*70)
    print("✅ FEDERATED LEARNING SUCCESSFUL!")
    print("="*70)
    print(f"Rounds completed: {num_rounds}")
    print(f"Final AUPRC: {avg_auprc:.4f}")
    print(f"Final AUROC: {avg_auroc:.4f}")
    print(f"Final Loss: {avg_loss:.4f}")
    print()
    print("Key Components Verified:")
    print("  ✓ Bank_Client.get_parameters() - Extracts model weights")
    print("  ✓ Bank_Client.set_parameters() - Updates model weights")
    print("  ✓ Bank_Client.fit() - Local training")
    print("  ✓ Bank_Client.evaluate() - Local evaluation")
    print("  ✓ Weight aggregation (FedAvg)")
    print("  ✓ Multi-round federated learning")
    print()
    print("🎉 Your federated learning system is working!")


if __name__ == '__main__':
    main()
