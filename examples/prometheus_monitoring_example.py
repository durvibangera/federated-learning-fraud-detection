"""
Example: Prometheus Monitoring for Federated Learning

This example demonstrates how to use the Prometheus_Exporter for real-time
monitoring of federated learning experiments.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring import Prometheus_Exporter, AlertConfig
import time
import random


def simulate_federated_learning_with_monitoring():
    """Simulate federated learning with Prometheus monitoring."""
    
    print("=" * 80)
    print("Prometheus Monitoring Example for Federated Learning")
    print("=" * 80)
    
    # Initialize Prometheus exporter
    exporter = Prometheus_Exporter(
        port=8000,
        enable_alerts=True
    )
    
    # Add alert configurations
    exporter.add_alert_config(AlertConfig(
        metric_name="low_auprc",
        threshold=0.5,
        comparison='lt',
        severity='warning',
        message="AUPRC below acceptable threshold"
    ))
    
    exporter.add_alert_config(AlertConfig(
        metric_name="privacy_budget_exhausted_bank1",
        threshold=1.0,
        comparison='eq',
        severity='critical',
        message="Privacy budget exhausted for bank1"
    ))
    
    # Start HTTP server for metrics endpoint
    print(f"\nStarting Prometheus HTTP server on port {exporter.port}...")
    exporter.start_http_server()
    print(f"✓ Metrics available at http://localhost:{exporter.port}/metrics")
    
    # Simulate 10 FL rounds
    num_rounds = 10
    num_clients = 3
    epsilon_total = 2.0
    epsilon_per_round = epsilon_total / num_rounds
    
    print(f"\nSimulating {num_rounds} FL rounds with {num_clients} clients...")
    print("-" * 80)
    
    for round_num in range(1, num_rounds + 1):
        print(f"\nRound {round_num}/{num_rounds}")
        
        # Record round start
        exporter.record_fl_round_start(round_num, num_clients)
        round_start_time = time.time()
        
        # Simulate training for each client
        for client_id in range(1, num_clients + 1):
            client_name = f"bank{client_id}"
            
            # Simulate training
            training_start = time.time()
            time.sleep(random.uniform(0.1, 0.3))  # Simulate training time
            training_duration = time.time() - training_start
            
            exporter.record_training_duration(training_duration, client_name)
            
            # Record training samples
            num_samples = random.randint(5000, 15000)
            exporter.record_training_samples(num_samples, client_name)
            
            # Simulate performance metrics (improving over rounds)
            base_auprc = 0.6 + (round_num / num_rounds) * 0.25
            base_auroc = 0.65 + (round_num / num_rounds) * 0.25
            
            auprc = base_auprc + random.uniform(-0.05, 0.05)
            auroc = base_auroc + random.uniform(-0.05, 0.05)
            loss = 0.5 - (round_num / num_rounds) * 0.2 + random.uniform(-0.05, 0.05)
            
            exporter.record_performance_metrics(
                auprc=auprc,
                auroc=auroc,
                loss=loss,
                model_type='local',
                client_id=client_name
            )
            
            # Record privacy budget
            epsilon_spent = round_num * epsilon_per_round
            exporter.record_privacy_budget(
                epsilon_spent=epsilon_spent,
                epsilon_total=epsilon_total,
                delta=1e-5,
                client_id=client_name
            )
            
            print(f"  {client_name}: AUPRC={auprc:.4f}, AUROC={auroc:.4f}, "
                  f"ε_spent={epsilon_spent:.4f}/{epsilon_total:.2f}")
        
        # Simulate aggregation
        aggregation_start = time.time()
        time.sleep(random.uniform(0.05, 0.15))
        aggregation_duration = time.time() - aggregation_start
        exporter.record_aggregation_duration(aggregation_duration)
        
        # Record global model performance
        global_auprc = 0.65 + (round_num / num_rounds) * 0.25 + random.uniform(-0.03, 0.03)
        global_auroc = 0.70 + (round_num / num_rounds) * 0.25 + random.uniform(-0.03, 0.03)
        global_loss = 0.45 - (round_num / num_rounds) * 0.2 + random.uniform(-0.03, 0.03)
        
        exporter.record_performance_metrics(
            auprc=global_auprc,
            auroc=global_auroc,
            loss=global_loss,
            model_type='global',
            client_id='global'
        )
        
        # Record convergence metrics
        convergence_score = min(0.95, 0.3 + (round_num / num_rounds) * 0.7)
        is_converged = convergence_score > 0.9
        weight_change = max(0.01, 1.0 - (round_num / num_rounds) * 0.9)
        
        exporter.record_convergence_metrics(
            convergence_score=convergence_score,
            is_converged=is_converged,
            weight_change=weight_change
        )
        
        # Record system metrics
        exporter.record_system_metrics({
            'memory_usage_bytes': random.randint(1_000_000_000, 2_000_000_000),
            'memory_available_bytes': random.randint(4_000_000_000, 8_000_000_000),
            'cpu_usage_percent': random.uniform(30, 80),
            'disk_usage_bytes': random.randint(10_000_000_000, 20_000_000_000),
            'health_status': 'healthy'
        })
        
        # Simulate occasional client failure
        if random.random() < 0.1:  # 10% chance of failure
            failed_client = f"bank{random.randint(1, num_clients)}"
            exporter.record_client_failure(failed_client, 'timeout')
        
        # Record round completion
        round_duration = time.time() - round_start_time
        exporter.record_fl_round_complete(round_num, round_duration, 'success')
        
        print(f"  Global: AUPRC={global_auprc:.4f}, AUROC={global_auroc:.4f}, "
              f"Convergence={convergence_score:.4f}")
        print(f"  Round duration: {round_duration:.2f}s")
    
    # Get metrics summary
    print("\n" + "=" * 80)
    print("Final Metrics Summary")
    print("=" * 80)
    summary = exporter.get_metrics_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("Monitoring Complete!")
    print("=" * 80)
    print(f"\nMetrics are available at: http://localhost:{exporter.port}/metrics")
    print("You can scrape these metrics with Prometheus or view them directly.")
    print("\nPress Ctrl+C to stop the metrics server...")
    
    try:
        # Keep server running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        exporter.shutdown()
        print("✓ Exporter shutdown complete")


if __name__ == "__main__":
    simulate_federated_learning_with_monitoring()
