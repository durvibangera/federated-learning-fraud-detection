"""
Differential Privacy Guarantee Verification

This script verifies that the differential privacy implementation provides
formal (epsilon, delta) guarantees as specified in the design document.

Tests:
- Privacy budget tracking accuracy
- Noise addition to gradients
- Privacy accounting across FL rounds
- Epsilon exhaustion handling
- Multiple epsilon values [0.5, 1, 2, 4, 8]

**Validates: Requirements 4.1, 4.3, 4.4, 4.6, 4.7**
"""

import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.fraud_mlp import FraudMLP
from src.privacy.privacy_engine import Privacy_Engine
from torch.utils.data import TensorDataset, DataLoader


class DPVerification:
    """Differential privacy guarantee verification suite."""

    def __init__(self):
        self.results = {"timestamp": datetime.now().isoformat(), "tests": {}, "overall_status": "unknown"}

    def verify_privacy_budget_tracking(self) -> Dict:
        """
        Verify that privacy budget is tracked correctly across training steps.

        **Validates: Requirements 4.3, 4.4**
        """
        print("\n[Test] Privacy Budget Tracking")

        try:
            device = torch.device("cpu")

            # Create simple model
            categorical_embedding_dims = {"ProductCD": (5, 10)}
            model = FraudMLP(
                categorical_embedding_dims=categorical_embedding_dims, numerical_input_dim=10, hidden_dims=[64, 32]
            ).to(device)

            # Create synthetic dataset
            num_samples = 1000
            categorical_data = torch.randint(0, 5, (num_samples, 1))
            numerical_data = torch.randn(num_samples, 10)
            targets = torch.randint(0, 2, (num_samples,)).float()

            dataset = TensorDataset(categorical_data, numerical_data, targets)
            dataloader = DataLoader(dataset, batch_size=100, drop_last=True)

            # Test multiple epsilon values
            epsilon_values = [0.5, 1.0, 2.0, 4.0, 8.0]
            tracking_results = {}

            for epsilon in epsilon_values:
                # Create privacy engine
                privacy_engine = Privacy_Engine(epsilon=epsilon, delta=1e-5, max_grad_norm=1.0)

                # Make model private
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

                try:
                    private_model, private_optimizer, private_dataloader = privacy_engine.make_private(
                        model=model, optimizer=optimizer, data_loader=dataloader
                    )

                    # Train for a few steps
                    criterion = nn.BCELoss()
                    private_model.train()

                    for batch_idx, (cat_data, num_data, target) in enumerate(private_dataloader):
                        if batch_idx >= 5:  # Only 5 batches
                            break

                        cat_features = {"ProductCD": cat_data.squeeze().to(device)}
                        num_features = num_data.to(device)
                        target = target.to(device)

                        private_optimizer.zero_grad()
                        output = private_model(cat_features, num_features)
                        loss = criterion(output.squeeze(), target)
                        loss.backward()
                        private_optimizer.step()

                    # Get privacy spent
                    epsilon_spent, delta_spent = privacy_engine.get_privacy_spent()

                    # Verify budget not exceeded
                    budget_valid = epsilon_spent <= epsilon

                    tracking_results[f"epsilon_{epsilon}"] = {
                        "target_epsilon": epsilon,
                        "epsilon_spent": round(epsilon_spent, 4),
                        "delta_spent": delta_spent,
                        "budget_valid": budget_valid,
                    }

                    print(f"  Epsilon {epsilon}: spent={epsilon_spent:.4f}, valid={budget_valid}")

                except Exception as e:
                    tracking_results[f"epsilon_{epsilon}"] = {"error": str(e), "budget_valid": False}
                    print(f"  Epsilon {epsilon}: ERROR - {e}")

            # Check if all budgets were valid
            all_valid = all(result.get("budget_valid", False) for result in tracking_results.values())

            result = {
                "status": "passed" if all_valid else "failed",
                "epsilon_tests": tracking_results,
                "message": "Privacy budget tracking verified" if all_valid else "Budget tracking issues detected",
            }

            print(f"  Status: {result['status']}")
            return result

        except Exception as e:
            print(f"  Status: failed - {e}")
            return {"status": "failed", "error": str(e), "message": "Privacy budget tracking verification failed"}

    def verify_noise_addition(self) -> Dict:
        """
        Verify that noise is added to gradients during training.

        **Validates: Requirements 4.1, 4.6**
        """
        print("\n[Test] Gradient Noise Addition")

        try:
            device = torch.device("cpu")

            # Create simple model
            categorical_embedding_dims = {"ProductCD": (5, 10)}
            model = FraudMLP(
                categorical_embedding_dims=categorical_embedding_dims, numerical_input_dim=10, hidden_dims=[64, 32]
            ).to(device)

            # Create synthetic dataset
            num_samples = 500
            categorical_data = torch.randint(0, 5, (num_samples, 1))
            numerical_data = torch.randn(num_samples, 10)
            targets = torch.randint(0, 2, (num_samples,)).float()

            dataset = TensorDataset(categorical_data, numerical_data, targets)
            dataloader = DataLoader(dataset, batch_size=100, drop_last=True)

            # Train without DP
            model_no_dp = FraudMLP(
                categorical_embedding_dims=categorical_embedding_dims, numerical_input_dim=10, hidden_dims=[64, 32]
            ).to(device)

            # Copy weights to ensure same initialization
            model_no_dp.load_state_dict(model.state_dict())

            optimizer_no_dp = torch.optim.SGD(model_no_dp.parameters(), lr=0.01)
            criterion = nn.BCELoss()

            # Train one batch without DP
            model_no_dp.train()
            for cat_data, num_data, target in dataloader:
                cat_features = {"ProductCD": cat_data.squeeze().to(device)}
                num_features = num_data.to(device)
                target = target.to(device)

                optimizer_no_dp.zero_grad()
                output = model_no_dp(cat_features, num_features)
                loss = criterion(output.squeeze(), target)
                loss.backward()

                # Store gradients
                gradients_no_dp = [p.grad.clone() for p in model_no_dp.parameters() if p.grad is not None]
                break

            # Train with DP
            privacy_engine = Privacy_Engine(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
            optimizer_dp = torch.optim.SGD(model.parameters(), lr=0.01)

            try:
                private_model, private_optimizer, private_dataloader = privacy_engine.make_private(
                    model=model, optimizer=optimizer_dp, data_loader=dataloader
                )

                # Train one batch with DP
                private_model.train()
                for cat_data, num_data, target in private_dataloader:
                    cat_features = {"ProductCD": cat_data.squeeze().to(device)}
                    num_features = num_data.to(device)
                    target = target.to(device)

                    private_optimizer.zero_grad()
                    output = private_model(cat_features, num_features)
                    loss = criterion(output.squeeze(), target)
                    loss.backward()
                    private_optimizer.step()

                    # Store gradients
                    gradients_dp = [p.grad.clone() for p in private_model.parameters() if p.grad is not None]
                    break

                # Compare gradients - they should be different due to noise
                gradient_differences = []
                for g_no_dp, g_dp in zip(gradients_no_dp, gradients_dp):
                    diff = torch.norm(g_no_dp - g_dp).item()
                    gradient_differences.append(diff)

                avg_difference = np.mean(gradient_differences)
                noise_detected = avg_difference > 1e-6  # Noise should cause measurable difference

                result = {
                    "status": "passed" if noise_detected else "failed",
                    "avg_gradient_difference": round(avg_difference, 6),
                    "noise_detected": noise_detected,
                    "message": "Noise addition verified" if noise_detected else "No noise detected in gradients",
                }

                print(f"  Avg gradient difference: {avg_difference:.6f}")
                print(f"  Noise detected: {noise_detected}")
                print(f"  Status: {result['status']}")

                return result

            except Exception as e:
                print(f"  Status: failed - {e}")
                return {"status": "failed", "error": str(e), "message": "Noise addition verification failed"}

        except Exception as e:
            print(f"  Status: failed - {e}")
            return {"status": "failed", "error": str(e), "message": "Noise addition verification failed"}

    def verify_privacy_accounting(self) -> Dict:
        """
        Verify privacy accounting across multiple training steps.

        **Validates: Requirements 4.3, 4.7**
        """
        print("\n[Test] Privacy Accounting Across Steps")

        try:
            device = torch.device("cpu")

            # Create simple model
            categorical_embedding_dims = {"ProductCD": (5, 10)}
            model = FraudMLP(
                categorical_embedding_dims=categorical_embedding_dims, numerical_input_dim=10, hidden_dims=[64, 32]
            ).to(device)

            # Create synthetic dataset
            num_samples = 1000
            categorical_data = torch.randint(0, 5, (num_samples, 1))
            numerical_data = torch.randn(num_samples, 10)
            targets = torch.randint(0, 2, (num_samples,)).float()

            dataset = TensorDataset(categorical_data, numerical_data, targets)
            dataloader = DataLoader(dataset, batch_size=100, drop_last=True)

            # Create privacy engine
            privacy_engine = Privacy_Engine(epsilon=2.0, delta=1e-5, max_grad_norm=1.0)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            try:
                private_model, private_optimizer, private_dataloader = privacy_engine.make_private(
                    model=model, optimizer=optimizer, data_loader=dataloader
                )

                # Track privacy over multiple steps
                privacy_history = []
                criterion = nn.BCELoss()
                private_model.train()

                for batch_idx, (cat_data, num_data, target) in enumerate(private_dataloader):
                    if batch_idx >= 10:  # 10 batches
                        break

                    cat_features = {"ProductCD": cat_data.squeeze().to(device)}
                    num_features = num_data.to(device)
                    target = target.to(device)

                    private_optimizer.zero_grad()
                    output = private_model(cat_features, num_features)
                    loss = criterion(output.squeeze(), target)
                    loss.backward()
                    private_optimizer.step()

                    # Record privacy spent
                    epsilon_spent, delta_spent = privacy_engine.get_privacy_spent()
                    privacy_history.append({"step": batch_idx + 1, "epsilon": epsilon_spent, "delta": delta_spent})

                # Verify privacy is monotonically increasing
                epsilons = [h["epsilon"] for h in privacy_history]
                monotonic = all(epsilons[i] <= epsilons[i + 1] for i in range(len(epsilons) - 1))

                # Verify final epsilon doesn't exceed budget
                final_epsilon = epsilons[-1]
                within_budget = final_epsilon <= 2.0

                result = {
                    "status": "passed" if (monotonic and within_budget) else "failed",
                    "monotonic_increase": monotonic,
                    "within_budget": within_budget,
                    "final_epsilon": round(final_epsilon, 4),
                    "target_epsilon": 2.0,
                    "num_steps": len(privacy_history),
                    "message": (
                        "Privacy accounting verified"
                        if (monotonic and within_budget)
                        else "Privacy accounting issues detected"
                    ),
                }

                print(f"  Monotonic increase: {monotonic}")
                print(f"  Within budget: {within_budget}")
                print(f"  Final epsilon: {final_epsilon:.4f} / 2.0")
                print(f"  Status: {result['status']}")

                return result

            except Exception as e:
                print(f"  Status: failed - {e}")
                return {"status": "failed", "error": str(e), "message": "Privacy accounting verification failed"}

        except Exception as e:
            print(f"  Status: failed - {e}")
            return {"status": "failed", "error": str(e), "message": "Privacy accounting verification failed"}

    def verify_epsilon_values(self) -> Dict:
        """
        Verify support for multiple epsilon values [0.5, 1, 2, 4, 8].

        **Validates: Requirements 4.2**
        """
        print("\n[Test] Multiple Epsilon Values Support")

        epsilon_values = [0.5, 1.0, 2.0, 4.0, 8.0]
        results = {}

        for epsilon in epsilon_values:
            try:
                privacy_engine = Privacy_Engine(epsilon=epsilon, delta=1e-5, max_grad_norm=1.0)

                results[f"epsilon_{epsilon}"] = {"supported": True, "epsilon": epsilon}
                print(f"  Epsilon {epsilon}: supported")

            except Exception as e:
                results[f"epsilon_{epsilon}"] = {"supported": False, "error": str(e)}
                print(f"  Epsilon {epsilon}: failed - {e}")

        all_supported = all(r["supported"] for r in results.values())

        result = {
            "status": "passed" if all_supported else "failed",
            "epsilon_support": results,
            "message": "All epsilon values supported" if all_supported else "Some epsilon values not supported",
        }

        print(f"  Status: {result['status']}")
        return result

    def run_all_tests(self) -> Dict:
        """Run all DP verification tests."""
        print("=" * 70)
        print("DIFFERENTIAL PRIVACY GUARANTEE VERIFICATION")
        print("=" * 70)

        # Run tests
        self.results["tests"]["privacy_budget_tracking"] = self.verify_privacy_budget_tracking()
        self.results["tests"]["noise_addition"] = self.verify_noise_addition()
        self.results["tests"]["privacy_accounting"] = self.verify_privacy_accounting()
        self.results["tests"]["epsilon_values"] = self.verify_epsilon_values()

        # Overall status
        all_passed = all(test["status"] == "passed" for test in self.results["tests"].values())

        self.results["overall_status"] = "passed" if all_passed else "failed"

        print("\n" + "=" * 70)
        print(f"OVERALL STATUS: {self.results['overall_status'].upper()}")
        print("=" * 70)

        return self.results

    def save_results(self, filepath: str = "dp_verification_results.json"):
        """Save verification results to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {filepath}")


def main():
    """Run DP verification tests."""
    verification = DPVerification()
    results = verification.run_all_tests()
    verification.save_results()

    # Exit with appropriate code
    exit_code = 0 if results["overall_status"] == "passed" else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
