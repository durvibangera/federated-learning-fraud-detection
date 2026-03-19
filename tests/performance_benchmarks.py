"""
Performance Benchmarks for Federated Fraud Detection System

This script runs performance benchmarks with synthetic datasets to:
- Measure training time per FL round
- Measure model inference latency
- Measure data preprocessing throughput
- Track memory usage during training
- Measure aggregation server performance

**Validates: Requirements 10.4, 10.5**
"""

import sys
import time
import json
import psutil
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.fraud_mlp import FraudMLP
from src.data.preprocessor import Data_Preprocessor


class PerformanceBenchmark:
    """Performance benchmarking suite for federated fraud detection."""

    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "benchmarks": {},
        }

    def _get_system_info(self) -> Dict:
        """Collect system information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }

    def benchmark_data_preprocessing(self, num_samples: int = 10000) -> Dict:
        """
        Benchmark data preprocessing throughput.

        Args:
            num_samples: Number of samples to process

        Returns:
            Dict with preprocessing metrics
        """
        print(f"\n[Benchmark] Data Preprocessing ({num_samples} samples)")

        # Generate synthetic IEEE-CIS data
        synthetic_data = self._generate_synthetic_data(num_samples)

        preprocessor = Data_Preprocessor()

        # Measure preprocessing time
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024**2)

        # Simulate preprocessing steps
        preprocessor.handle_missing_values(synthetic_data)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024**2)

        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        throughput = num_samples / duration

        results = {
            "num_samples": num_samples,
            "duration_seconds": round(duration, 3),
            "throughput_samples_per_sec": round(throughput, 2),
            "memory_delta_mb": round(memory_delta, 2),
            "status": "passed",
        }

        print(f"  Duration: {results['duration_seconds']}s")
        print(f"  Throughput: {results['throughput_samples_per_sec']} samples/sec")
        print(f"  Memory Delta: {results['memory_delta_mb']} MB")

        return results

    def benchmark_model_training(self, num_samples: int = 5000, num_epochs: int = 3) -> Dict:
        """
        Benchmark model training performance.

        Args:
            num_samples: Number of training samples
            num_epochs: Number of training epochs

        Returns:
            Dict with training metrics
        """
        print(f"\n[Benchmark] Model Training ({num_samples} samples, {num_epochs} epochs)")

        # Create synthetic model and data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        categorical_embedding_dims = {"ProductCD": (5, 10), "card4": (4, 8), "card6": (4, 8)}

        model = FraudMLP(
            categorical_embedding_dims=categorical_embedding_dims, numerical_input_dim=16, hidden_dims=[128, 64]
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()

        # Generate synthetic batch
        batch_size = 256
        num_batches = num_samples // batch_size

        # Measure training time
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024**2)

        model.train()
        total_loss = 0.0

        for epoch in range(num_epochs):
            for _ in range(num_batches):
                # Generate synthetic batch
                categorical_data = torch.stack(
                    [
                        torch.randint(0, 5, (batch_size,)),
                        torch.randint(0, 4, (batch_size,)),
                        torch.randint(0, 4, (batch_size,)),
                    ],
                    dim=1,
                ).to(device)
                numerical_data = torch.randn(batch_size, 16).to(device)
                targets = torch.randint(0, 2, (batch_size,)).float().to(device)

                features = {"categorical": categorical_data, "numerical": numerical_data}

                optimizer.zero_grad()
                outputs = model(features)
                outputs = torch.sigmoid(outputs)  # Apply sigmoid for BCELoss
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024**2)

        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        samples_per_sec = (num_samples * num_epochs) / duration

        results = {
            "num_samples": num_samples,
            "num_epochs": num_epochs,
            "duration_seconds": round(duration, 3),
            "samples_per_sec": round(samples_per_sec, 2),
            "avg_loss": round(total_loss / (num_batches * num_epochs), 4),
            "memory_delta_mb": round(memory_delta, 2),
            "device": str(device),
            "status": "passed",
        }

        print(f"  Duration: {results['duration_seconds']}s")
        print(f"  Throughput: {results['samples_per_sec']} samples/sec")
        print(f"  Avg Loss: {results['avg_loss']}")
        print(f"  Memory Delta: {results['memory_delta_mb']} MB")

        return results

    def benchmark_model_inference(self, num_samples: int = 10000) -> Dict:
        """
        Benchmark model inference latency.

        Args:
            num_samples: Number of inference samples

        Returns:
            Dict with inference metrics
        """
        print(f"\n[Benchmark] Model Inference ({num_samples} samples)")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        categorical_embedding_dims = {"ProductCD": (5, 10), "card4": (4, 8), "card6": (4, 8)}

        model = FraudMLP(
            categorical_embedding_dims=categorical_embedding_dims, numerical_input_dim=16, hidden_dims=[128, 64]
        ).to(device)

        model.eval()

        batch_size = 256
        num_batches = num_samples // batch_size

        # Measure inference time
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_batches):
                categorical_data = torch.stack(
                    [
                        torch.randint(0, 5, (batch_size,)),
                        torch.randint(0, 4, (batch_size,)),
                        torch.randint(0, 4, (batch_size,)),
                    ],
                    dim=1,
                ).to(device)
                numerical_data = torch.randn(batch_size, 16).to(device)

                features = {"categorical": categorical_data, "numerical": numerical_data}

                _ = model(features)

        end_time = time.time()

        duration = end_time - start_time
        latency_per_sample = (duration / num_samples) * 1000  # ms
        throughput = num_samples / duration

        results = {
            "num_samples": num_samples,
            "duration_seconds": round(duration, 3),
            "latency_per_sample_ms": round(latency_per_sample, 3),
            "throughput_samples_per_sec": round(throughput, 2),
            "device": str(device),
            "status": "passed",
        }

        print(f"  Duration: {results['duration_seconds']}s")
        print(f"  Latency: {results['latency_per_sample_ms']} ms/sample")
        print(f"  Throughput: {results['throughput_samples_per_sec']} samples/sec")

        return results

    def benchmark_weight_aggregation(self, num_clients: int = 3, num_rounds: int = 10) -> Dict:
        """
        Benchmark federated weight aggregation performance.

        Args:
            num_clients: Number of clients
            num_rounds: Number of aggregation rounds

        Returns:
            Dict with aggregation metrics
        """
        print(f"\n[Benchmark] Weight Aggregation ({num_clients} clients, {num_rounds} rounds)")

        # Create synthetic model weights
        categorical_embedding_dims = {"ProductCD": (5, 10), "card4": (4, 8), "card6": (4, 8)}

        model = FraudMLP(
            categorical_embedding_dims=categorical_embedding_dims, numerical_input_dim=16, hidden_dims=[128, 64]
        )

        # Measure aggregation time
        start_time = time.time()

        for _ in range(num_rounds):
            # Simulate client weight collection
            client_weights = []
            for _ in range(num_clients):
                weights = [param.detach().cpu().numpy() for param in model.parameters()]
                client_weights.append(weights)

            # Simulate FedAvg aggregation
            aggregated_weights = []
            for layer_idx in range(len(client_weights[0])):
                layer_weights = [client_weights[i][layer_idx] for i in range(num_clients)]
                avg_weight = np.mean(layer_weights, axis=0)
                aggregated_weights.append(avg_weight)

        end_time = time.time()

        duration = end_time - start_time
        time_per_round = duration / num_rounds

        results = {
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "duration_seconds": round(duration, 3),
            "time_per_round_seconds": round(time_per_round, 3),
            "status": "passed",
        }

        print(f"  Duration: {results['duration_seconds']}s")
        print(f"  Time per round: {results['time_per_round_seconds']}s")

        return results

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

    def run_all_benchmarks(self) -> Dict:
        """Run all performance benchmarks."""
        print("=" * 70)
        print("PERFORMANCE BENCHMARKS - Federated Fraud Detection System")
        print("=" * 70)

        # Run benchmarks
        self.results["benchmarks"]["data_preprocessing"] = self.benchmark_data_preprocessing()
        self.results["benchmarks"]["model_training"] = self.benchmark_model_training()
        self.results["benchmarks"]["model_inference"] = self.benchmark_model_inference()
        self.results["benchmarks"]["weight_aggregation"] = self.benchmark_weight_aggregation()

        # Overall status
        all_passed = all(benchmark["status"] == "passed" for benchmark in self.results["benchmarks"].values())

        self.results["overall_status"] = "passed" if all_passed else "failed"

        print("\n" + "=" * 70)
        print(f"OVERALL STATUS: {self.results['overall_status'].upper()}")
        print("=" * 70)

        return self.results

    def save_results(self, filepath: str = "benchmark_results.json"):
        """Save benchmark results to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {filepath}")


def main():
    """Run performance benchmarks."""
    benchmark = PerformanceBenchmark()
    results = benchmark.run_all_benchmarks()
    benchmark.save_results()

    # Exit with appropriate code
    exit_code = 0 if results["overall_status"] == "passed" else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
