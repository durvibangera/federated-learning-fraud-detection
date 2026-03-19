"""
Test script for Privacy_Utility_Analyzer.

Demonstrates how to use the analyzer to track privacy-utility tradeoffs.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from privacy import Privacy_Utility_Analyzer

print("=" * 70)
print("PRIVACY-UTILITY ANALYZER TEST")
print("=" * 70)

# Initialize analyzer
print("\n[1/4] Initializing Privacy_Utility_Analyzer...")
analyzer = Privacy_Utility_Analyzer(target_epsilons=[0.5, 1.0, 2.0, 4.0, 8.0], delta=1e-5, max_grad_norm=1.0)
print("✓ Analyzer initialized")

# Simulate experimental results
print("\n[2/4] Adding simulated experimental results...")
# Simulating: lower epsilon (stronger privacy) = lower performance
experimental_results = [
    (0.5, 0.25, 0.75, 0.85, 0.52),  # epsilon, auprc, auroc, loss, epsilon_spent
    (1.0, 0.30, 0.80, 0.75, 1.05),
    (2.0, 0.35, 0.83, 0.65, 2.10),
    (4.0, 0.38, 0.85, 0.55, 4.15),
    (8.0, 0.40, 0.86, 0.50, 8.20),
]

for epsilon, auprc, auroc, loss, eps_spent in experimental_results:
    analyzer.add_result(epsilon=epsilon, auprc=auprc, auroc=auroc, loss=loss, epsilon_spent=eps_spent)
print(f"✓ Added {len(experimental_results)} experimental results")

# Get privacy-utility curve
print("\n[3/4] Generating privacy-utility curve...")
curve_data = analyzer.get_privacy_utility_curve()
print("✓ Privacy-utility curve generated")
print(f"\nCurve Data:")
print(f"  Epsilon values: {curve_data['epsilon']}")
print(f"  AUPRC values: {[f'{x:.3f}' for x in curve_data['auprc']]}")
print(f"  AUROC values: {[f'{x:.3f}' for x in curve_data['auroc']]}")

# Get summary statistics
print("\n[4/4] Computing summary statistics...")
summary = analyzer.get_summary_statistics()
print("✓ Summary statistics computed")
print(f"\nSummary:")
print(f"  Number of experiments: {summary['num_experiments']}")
print(f"  Epsilon range: {summary['epsilon_range']}")
print(f"  AUPRC range: ({summary['auprc_range'][0]:.3f}, {summary['auprc_range'][1]:.3f})")
print(f"  AUROC range: ({summary['auroc_range'][0]:.3f}, {summary['auroc_range'][1]:.3f})")
print(f"  AUPRC degradation: {summary['auprc_degradation']:.3f}")
print(f"  AUROC degradation: {summary['auroc_degradation']:.3f}")
print(f"  Best epsilon for AUPRC: {summary['best_epsilon_auprc']}")
print(f"  Best epsilon for AUROC: {summary['best_epsilon_auroc']}")

# Export results
print("\n[Bonus] Exporting results to JSON...")
analyzer.export_results("results/privacy_utility_analysis.json")
print("✓ Results exported to results/privacy_utility_analysis.json")

print("\n" + "=" * 70)
print("✅ PRIVACY-UTILITY ANALYZER TEST SUCCESSFUL!")
print("=" * 70)
print("\nKey Features Verified:")
print("✓ Privacy_Utility_Analyzer initialization")
print("✓ Adding experimental results")
print("✓ Generating privacy-utility curves")
print("✓ Computing summary statistics")
print("✓ Exporting results to JSON")
