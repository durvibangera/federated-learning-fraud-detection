"""
Verification Script for Prometheus_Exporter

This script verifies that the Prometheus_Exporter is correctly implemented
and can record all required metrics.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.monitoring import Prometheus_Exporter, AlertConfig


def verify_prometheus_exporter():
    """Verify Prometheus_Exporter functionality."""

    print("=" * 80)
    print("Prometheus_Exporter Verification")
    print("=" * 80)

    # Test 1: Initialization
    print("\n[Test 1] Initializing Prometheus_Exporter...")
    try:
        exporter = Prometheus_Exporter(port=8001, enable_alerts=True)  # Use different port to avoid conflicts
        print("✓ Initialization successful")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False

    # Test 2: Alert Configuration
    print("\n[Test 2] Adding alert configurations...")
    try:
        exporter.add_alert_config(
            AlertConfig(
                metric_name="test_metric", threshold=0.5, comparison="lt", severity="warning", message="Test alert"
            )
        )
        print(f"✓ Alert configuration added (total: {len(exporter.alert_configs)})")
    except Exception as e:
        print(f"✗ Alert configuration failed: {e}")
        return False

    # Test 3: FL Round Metrics
    print("\n[Test 3] Recording FL round metrics...")
    try:
        exporter.record_fl_round_start(1, 3)
        exporter.record_fl_round_complete(1, 45.5, "success")
        exporter.record_training_duration(30.2, "bank1")
        exporter.record_aggregation_duration(5.3)
        print("✓ FL round metrics recorded")
    except Exception as e:
        print(f"✗ FL round metrics failed: {e}")
        return False

    # Test 4: Performance Metrics
    print("\n[Test 4] Recording performance metrics...")
    try:
        exporter.record_performance_metrics(auprc=0.85, auroc=0.88, loss=0.25, model_type="global", client_id="global")
        print("✓ Performance metrics recorded")
    except Exception as e:
        print(f"✗ Performance metrics failed: {e}")
        return False

    # Test 5: Privacy Budget Metrics
    print("\n[Test 5] Recording privacy budget metrics...")
    try:
        exporter.record_privacy_budget(epsilon_spent=0.5, epsilon_total=2.0, delta=1e-5, client_id="bank1")
        print("✓ Privacy budget metrics recorded")
    except Exception as e:
        print(f"✗ Privacy budget metrics failed: {e}")
        return False

    # Test 6: System Metrics
    print("\n[Test 6] Recording system metrics...")
    try:
        exporter.record_system_metrics(
            {
                "memory_usage_bytes": 1_500_000_000,
                "memory_available_bytes": 6_000_000_000,
                "cpu_usage_percent": 45.5,
                "disk_usage_bytes": 15_000_000_000,
                "health_status": "healthy",
            }
        )
        print("✓ System metrics recorded")
    except Exception as e:
        print(f"✗ System metrics failed: {e}")
        return False

    # Test 7: Convergence Metrics
    print("\n[Test 7] Recording convergence metrics...")
    try:
        exporter.record_convergence_metrics(convergence_score=0.85, is_converged=False, weight_change=0.15)
        print("✓ Convergence metrics recorded")
    except Exception as e:
        print(f"✗ Convergence metrics failed: {e}")
        return False

    # Test 8: Client Failure
    print("\n[Test 8] Recording client failure...")
    try:
        exporter.record_client_failure("bank2", "timeout")
        print("✓ Client failure recorded")
    except Exception as e:
        print(f"✗ Client failure recording failed: {e}")
        return False

    # Test 9: Training Samples
    print("\n[Test 9] Recording training samples...")
    try:
        exporter.record_training_samples(10000, "bank1")
        print("✓ Training samples recorded")
    except Exception as e:
        print(f"✗ Training samples recording failed: {e}")
        return False

    # Test 10: Network Traffic
    print("\n[Test 10] Recording network traffic...")
    try:
        exporter.record_network_traffic(bytes_sent=1024000, bytes_received=2048000, component="client")
        print("✓ Network traffic recorded")
    except Exception as e:
        print(f"✗ Network traffic recording failed: {e}")
        return False

    # Test 11: Metrics Summary
    print("\n[Test 11] Getting metrics summary...")
    try:
        summary = exporter.get_metrics_summary()
        print(f"✓ Metrics summary retrieved:")
        for key, value in summary.items():
            print(f"  - {key}: {value}")
    except Exception as e:
        print(f"✗ Metrics summary failed: {e}")
        return False

    # Test 12: HTTP Server (optional - commented out to avoid blocking)
    print("\n[Test 12] HTTP server functionality...")
    print("  (Skipping server start to avoid blocking)")
    print("  To test manually, run: examples/prometheus_monitoring_example.py")

    print("\n" + "=" * 80)
    print("All Tests Passed! ✓")
    print("=" * 80)
    print("\nPrometheus_Exporter is ready for use.")
    print("\nNext steps:")
    print("  1. Run example: python examples/prometheus_monitoring_example.py")
    print("  2. View metrics: http://localhost:8000/metrics")
    print("  3. Set up Prometheus and Grafana (see monitoring/README.md)")

    return True


if __name__ == "__main__":
    success = verify_prometheus_exporter()
    sys.exit(0 if success else 1)
