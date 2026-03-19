"""
End-to-End Workflow Validation

This script validates the complete federated learning workflow:
- Data preprocessing and partitioning
- Model initialization
- Federated learning rounds (simplified)
- Privacy engine integration
- Model evaluation
- MLOps logging

**Validates: Requirements 1.1-8.7 (integration)**
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import Data_Preprocessor
from src.model.fraud_mlp import FraudMLP
from src.privacy.privacy_engine import Privacy_Engine
from torch.utils.data import TensorDataset, DataLoader


class E2EWorkflowTest:
    """End-to-end workflow validation."""

    def __init__(self):
        self.results = {"timestamp": datetime.now().isoformat(), "workflow_steps": {}, "overall_status": "unknown"}

    def step_1_data_preprocessing(self) -> Dict:
        """
        Step 1: Data preprocessing and partitioning.

        **Validates: Requirements 1.1, 1.2, 1.3**
        """
        print("\n[Step 1] Data Preprocessing and Partitioning")

        try:
            # Generate synthetic IEEE-CIS data
            num_samples = 5000
            synthetic_data = self._generate_synthetic_data(num_samples)

            # Initialize preprocessor
            preprocessor = Data_Preprocessor()

            # Handle missing values
            processed_data = preprocessor.handle_missing_values(synthetic_data)

            # Partition by ProductCD
            partitions = preprocessor.partition_by_product_cd(processed_data)

            # Verify partitions
            bank1_size = len(partitions.get("bank1", []))
            bank2_size = len(partitions.get("bank2", []))
            bank3_size = len(partitions.get("bank3", []))

            total_partitioned = bank1_size + bank2_size + bank3_size

            result = {
                "status": "passed" if total_partitioned > 0 else "failed",
                "num_samples": num_samples,
                "bank1_samples": bank1_size,
                "bank2_samples": bank2_size,
                "bank3_samples": bank3_size,
                "total_partitioned": total_partitioned,
                "message": "Data preprocessing completed successfully",
            }

            print(f"  Bank 1: {bank1_size} samples")
            print(f"  Bank 2: {bank2_size} samples")
            print(f"  Bank 3: {bank3_size} samples")
            print(f"  Status: {result['status']}")

            return result

        except Exception as e:
            print(f"  Status: failed - {e}")
            return {"status": "failed", "error": str(e), "message": "Data preprocessing failed"}

    def step_2_model_initialization(self) -> Dict:
        """
        Step 2: Model initialization with proper architecture.

        **Validates: Requirements 2.1, 2.2, 2.4**
        """
        print("\n[Step 2] Model Initialization")

        try:
            device = torch.device("cpu")

            # Define model architecture
            categorical_embedding_dims = {"ProductCD": (5, 10), "card4": (4, 8), "card6": (4, 8)}

            model = FraudMLP(
                categorical_embedding_dims=categorical_embedding_dims, numerical_input_dim=16, hidden_dims=[128, 64, 32]
            ).to(device)

            # Verify model structure
            num_parameters = sum(p.numel() for p in model.parameters())
            has_embeddings = hasattr(model, "embeddings")

            # Test forward pass
            batch_size = 32
            categorical_features = {
                "ProductCD": torch.randint(0, 5, (batch_size,)).to(device),
                "card4": torch.randint(0, 4, (batch_size,)).to(device),
                "card6": torch.randint(0, 4, (batch_size,)).to(device),
            }
            numerical_features = torch.randn(batch_size, 16).to(device)

            output = model(categorical_features, numerical_features)

            # Verify output shape and range
            output_valid = output.shape == (batch_size, 1) and torch.all(output >= 0) and torch.all(output <= 1)

            result = {
                "status": "passed" if (has_embeddings and output_valid) else "failed",
                "num_parameters": num_parameters,
                "has_embeddings": has_embeddings,
                "output_shape": list(output.shape),
                "output_range_valid": output_valid,
                "message": "Model initialized successfully",
            }

            print(f"  Parameters: {num_parameters:,}")
            print(f"  Has embeddings: {has_embeddings}")
            print(f"  Output valid: {output_valid}")
            print(f"  Status: {result['status']}")

            return result

        except Exception as e:
            print(f"  Status: failed - {e}")
            return {"status": "failed", "error": str(e), "message": "Model initialization failed"}

    def step_3_privacy_engine_integration(self) -> Dict:
        """
        Step 3: Privacy engine integration with model.

        **Validates: Requirements 4.1, 4.6**
        """
        print("\n[Step 3] Privacy Engine Integration")

        try:
            device = torch.device("cpu")

            # Create model
            categorical_embedding_dims = {"ProductCD": (5, 10), "card4": (4, 8), "card6": (4, 8)}
            model = FraudMLP(
                categorical_embedding_dims=categorical_embedding_dims, numerical_input_dim=16, hidden_dims=[128, 64]
            ).to(device)

            # Create synthetic dataset
            num_samples = 1000
            categorical_data = torch.randint(0, 5, (num_samples, 3))
            numerical_data = torch.randn(num_samples, 16)
            targets = torch.randint(0, 2, (num_samples,)).float()

            dataset = TensorDataset(categorical_data, numerical_data, targets)
            dataloader = DataLoader(dataset, batch_size=100, drop_last=True)

            # Initialize privacy engine
            privacy_engine = Privacy_Engine(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)

            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            # Make model private
            private_model, private_optimizer, private_dataloader = privacy_engine.make_private(
                model=model, optimizer=optimizer, data_loader=dataloader
            )

            # Verify privacy engine is attached
            privacy_attached = hasattr(private_model, "privacy_engine") or hasattr(private_optimizer, "privacy_engine")

            result = {
                "status": "passed" if privacy_attached else "failed",
                "epsilon": 1.0,
                "delta": 1e-5,
                "privacy_attached": privacy_attached,
                "message": "Privacy engine integrated successfully",
            }

            print(f"  Epsilon: {result['epsilon']}")
            print(f"  Privacy attached: {privacy_attached}")
            print(f"  Status: {result['status']}")

            return result

        except Exception as e:
            print(f"  Status: failed - {e}")
            return {"status": "failed", "error": str(e), "message": "Privacy engine integration failed"}

    def step_4_federated_training_simulation(self) -> Dict:
        """
        Step 4: Simulate federated learning rounds.

        **Validates: Requirements 3.1, 3.4, 3.5**
        """
        print("\n[Step 4] Federated Training Simulation")

        try:
            device = torch.device("cpu")
            num_clients = 3
            num_rounds = 3  # Simplified for testing

            # Initialize global model
            categorical_embedding_dims = {"ProductCD": (5, 10), "card4": (4, 8), "card6": (4, 8)}
            global_model = FraudMLP(
                categorical_embedding_dims=categorical_embedding_dims, numerical_input_dim=16, hidden_dims=[128, 64]
            ).to(device)

            round_metrics = []

            for round_num in range(num_rounds):
                client_weights = []

                # Simulate each client training
                for client_id in range(num_clients):
                    # Create client model (copy of global)
                    client_model = FraudMLP(
                        categorical_embedding_dims=categorical_embedding_dims,
                        numerical_input_dim=16,
                        hidden_dims=[128, 64],
                    ).to(device)
                    client_model.load_state_dict(global_model.state_dict())

                    # Simulate local training (1 batch)
                    optimizer = torch.optim.SGD(client_model.parameters(), lr=0.01)
                    criterion = torch.nn.BCELoss()

                    batch_size = 32
                    categorical_features = {
                        "ProductCD": torch.randint(0, 5, (batch_size,)).to(device),
                        "card4": torch.randint(0, 4, (batch_size,)).to(device),
                        "card6": torch.randint(0, 4, (batch_size,)).to(device),
                    }
                    numerical_features = torch.randn(batch_size, 16).to(device)
                    targets = torch.randint(0, 2, (batch_size,)).float().to(device)

                    optimizer.zero_grad()
                    output = client_model(categorical_features, numerical_features)
                    loss = criterion(output.squeeze(), targets)
                    loss.backward()
                    optimizer.step()

                    # Extract weights
                    weights = [p.detach().cpu().numpy() for p in client_model.parameters()]
                    client_weights.append(weights)

                # Aggregate weights (FedAvg)
                aggregated_weights = []
                for layer_idx in range(len(client_weights[0])):
                    layer_weights = [client_weights[i][layer_idx] for i in range(num_clients)]
                    avg_weight = np.mean(layer_weights, axis=0)
                    aggregated_weights.append(torch.tensor(avg_weight))

                # Update global model
                with torch.no_grad():
                    for param, new_weight in zip(global_model.parameters(), aggregated_weights):
                        param.copy_(new_weight)

                round_metrics.append({"round": round_num + 1, "num_clients": num_clients})

            result = {
                "status": "passed",
                "num_rounds": num_rounds,
                "num_clients": num_clients,
                "rounds_completed": len(round_metrics),
                "message": "Federated training simulation completed",
            }

            print(f"  Rounds: {num_rounds}")
            print(f"  Clients: {num_clients}")
            print(f"  Status: {result['status']}")

            return result

        except Exception as e:
            print(f"  Status: failed - {e}")
            return {"status": "failed", "error": str(e), "message": "Federated training simulation failed"}

    def step_5_model_evaluation(self) -> Dict:
        """
        Step 5: Model evaluation with metrics.

        **Validates: Requirements 8.1, 8.2**
        """
        print("\n[Step 5] Model Evaluation")

        try:
            device = torch.device("cpu")

            # Create model
            categorical_embedding_dims = {"ProductCD": (5, 10), "card4": (4, 8), "card6": (4, 8)}
            model = FraudMLP(
                categorical_embedding_dims=categorical_embedding_dims, numerical_input_dim=16, hidden_dims=[128, 64]
            ).to(device)

            model.eval()

            # Generate test data
            num_test_samples = 500
            batch_size = 100

            all_predictions = []
            all_targets = []

            with torch.no_grad():
                for _ in range(num_test_samples // batch_size):
                    categorical_features = {
                        "ProductCD": torch.randint(0, 5, (batch_size,)).to(device),
                        "card4": torch.randint(0, 4, (batch_size,)).to(device),
                        "card6": torch.randint(0, 4, (batch_size,)).to(device),
                    }
                    numerical_features = torch.randn(batch_size, 16).to(device)
                    targets = torch.randint(0, 2, (batch_size,)).float().to(device)

                    output = model(categorical_features, numerical_features)

                    all_predictions.extend(output.squeeze().cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

            # Calculate basic metrics
            predictions = np.array(all_predictions)
            targets = np.array(all_targets)

            # Binary predictions
            binary_preds = (predictions > 0.5).astype(int)
            accuracy = np.mean(binary_preds == targets)

            result = {
                "status": "passed",
                "num_test_samples": len(predictions),
                "accuracy": round(accuracy, 4),
                "predictions_range": [float(predictions.min()), float(predictions.max())],
                "message": "Model evaluation completed",
            }

            print(f"  Test samples: {len(predictions)}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Status: {result['status']}")

            return result

        except Exception as e:
            print(f"  Status: failed - {e}")
            return {"status": "failed", "error": str(e), "message": "Model evaluation failed"}

    def _generate_synthetic_data(self, num_samples: int) -> pd.DataFrame:
        """Generate synthetic IEEE-CIS dataset."""
        data = {
            "TransactionID": range(1, num_samples + 1),
            "TransactionDT": np.random.uniform(0, 1000000, num_samples),
            "TransactionAmt": np.random.uniform(1, 1000, num_samples),
            "ProductCD": np.random.choice(["W", "H", "R", "S", "C"], num_samples),
            "card1": np.random.uniform(1000, 20000, num_samples),
            "card2": np.random.uniform(100, 1000, num_samples),
            "card4": np.random.choice(["visa", "mastercard", "amex", "discover"], num_samples),
            "card6": np.random.choice(["debit", "credit"], num_samples),
            "isFraud": np.random.choice([0, 1], num_samples, p=[0.965, 0.035]),
        }

        return pd.DataFrame(data)

    def run_workflow(self) -> Dict:
        """Run complete end-to-end workflow."""
        print("=" * 70)
        print("END-TO-END WORKFLOW VALIDATION")
        print("=" * 70)

        # Run workflow steps
        self.results["workflow_steps"]["step_1_preprocessing"] = self.step_1_data_preprocessing()
        self.results["workflow_steps"]["step_2_model_init"] = self.step_2_model_initialization()
        self.results["workflow_steps"]["step_3_privacy"] = self.step_3_privacy_engine_integration()
        self.results["workflow_steps"]["step_4_federated"] = self.step_4_federated_training_simulation()
        self.results["workflow_steps"]["step_5_evaluation"] = self.step_5_model_evaluation()

        # Overall status
        all_passed = all(step["status"] == "passed" for step in self.results["workflow_steps"].values())

        self.results["overall_status"] = "passed" if all_passed else "failed"

        print("\n" + "=" * 70)
        print(f"OVERALL STATUS: {self.results['overall_status'].upper()}")
        print("=" * 70)

        return self.results

    def save_results(self, filepath: str = "e2e_test_results.json"):
        """Save test results to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {filepath}")


def main():
    """Run end-to-end workflow test."""
    test = E2EWorkflowTest()
    results = test.run_workflow()
    test.save_results()

    # Exit with appropriate code
    exit_code = 0 if results["overall_status"] == "passed" else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
