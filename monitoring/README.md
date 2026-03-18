# Federated Learning Monitoring Infrastructure

This directory contains the monitoring infrastructure for the Federated Fraud Detection system, including Prometheus metrics and Grafana dashboards.

## Overview

The monitoring system provides real-time visibility into:
- Federated learning round progress and duration
- Model performance metrics (AUPRC, AUROC, loss)
- Privacy budget tracking and consumption
- System health and resource usage
- Convergence tracking
- Client participation and failures

## Components

### Prometheus Exporter

The `Prometheus_Exporter` class (in `src/monitoring/prometheus_exporter.py`) exposes metrics via HTTP endpoint for Prometheus scraping.

**Key Features:**
- Custom FL round metrics (duration, success/failure counts)
- Model performance tracking (AUPRC, AUROC, loss)
- Privacy budget monitoring (epsilon spent/remaining)
- System health metrics (CPU, memory, disk, network)
- Convergence tracking
- Configurable alerting

**Usage:**
```python
from src.monitoring import Prometheus_Exporter, AlertConfig

# Initialize exporter
exporter = Prometheus_Exporter(port=8000, enable_alerts=True)

# Add alert configurations
exporter.add_alert_config(AlertConfig(
    metric_name="low_auprc",
    threshold=0.5,
    comparison='lt',
    severity='warning',
    message="AUPRC below acceptable threshold"
))

# Start HTTP server
exporter.start_http_server()

# Record metrics during FL rounds
exporter.record_fl_round_start(round_num=1, num_clients=3)
exporter.record_performance_metrics(auprc=0.85, auroc=0.88, loss=0.25)
exporter.record_privacy_budget(epsilon_spent=0.5, epsilon_total=2.0, delta=1e-5)
exporter.record_fl_round_complete(round_num=1, duration_seconds=45.2, status="success")
```

### Grafana Dashboard

The `federated_learning_dashboard.json` provides comprehensive visualization of all metrics.

**Dashboard Panels:**
1. FL Rounds Progress - Track completed rounds
2. FL Round Duration - Monitor training/aggregation time
3. Model Performance (AUPRC) - Global and local model performance
4. Model Performance (AUROC) - Secondary performance metric
5. Privacy Budget Spent - Track epsilon consumption
6. Privacy Budget Remaining - Monitor remaining budget
7. Convergence Score - Track model convergence
8. Model Weight Change - Monitor weight updates
9. Client Participation - Active clients per round
10. Client Failures - Track client disconnections
11. System Memory Usage - Memory consumption
12. System CPU Usage - CPU utilization
13. Network Traffic - Bytes sent/received
14. System Health Status - Overall health indicator
15. FL Round Success Rate - Success percentage

**Alerts Configured:**
- Low AUPRC (< 0.5) - Warning
- Privacy Budget Low (< 0.1 remaining) - Critical
- Client Failures - Warning

## Setup Instructions

### 1. Install Dependencies

```bash
pip install prometheus-client
```

### 2. Run Example

Test the Prometheus exporter with the example script:

```bash
python examples/prometheus_monitoring_example.py
```

This will:
- Start Prometheus HTTP server on port 8000
- Simulate 10 FL rounds with metrics
- Expose metrics at `http://localhost:8000/metrics`

### 3. Configure Prometheus

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 5s
  evaluation_interval: 5s

scrape_configs:
  - job_name: 'federated_fraud_detection'
    static_configs:
      - targets: ['localhost:8000']
        labels:
          component: 'fl_system'
```

Run Prometheus:

```bash
docker run -p 9090:9090 -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
```

Access Prometheus UI at `http://localhost:9090`

### 4. Configure Grafana

Run Grafana:

```bash
docker run -d -p 3000:3000 grafana/grafana
```

Access Grafana at `http://localhost:3000` (default credentials: admin/admin)

**Import Dashboard:**
1. Go to Dashboards → Import
2. Upload `monitoring/grafana/federated_learning_dashboard.json`
3. Select Prometheus data source
4. Click Import

### 5. Docker Compose Setup (Recommended)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false

volumes:
  prometheus_data:
  grafana_data:
```

Start services:

```bash
docker-compose up -d
```

## Metrics Reference

### FL Round Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `fl_rounds_total` | Counter | Total FL rounds by status |
| `fl_rounds_completed` | Gauge | Number of completed rounds |
| `fl_round_duration_seconds` | Histogram | Round duration distribution |
| `fl_clients_participating` | Gauge | Active clients in current round |
| `fl_client_failures_total` | Counter | Client failures by type |
| `fl_training_samples` | Gauge | Training samples per client |

### Performance Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `model_auprc` | Gauge | Area Under Precision-Recall Curve |
| `model_auroc` | Gauge | Area Under ROC Curve |
| `model_loss` | Gauge | Model loss value |
| `predictions_total` | Counter | Total predictions made |
| `performance_degradation_percent` | Gauge | Performance vs baseline |

### Privacy Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `privacy_epsilon_spent` | Gauge | Cumulative epsilon spent |
| `privacy_epsilon_remaining` | Gauge | Remaining epsilon budget |
| `privacy_delta` | Gauge | Delta parameter |
| `privacy_budget_exhausted` | Gauge | Budget exhaustion flag |

### System Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `system_memory_usage_bytes` | Gauge | Memory usage |
| `system_memory_available_bytes` | Gauge | Available memory |
| `system_cpu_usage_percent` | Gauge | CPU utilization |
| `network_bytes_sent_total` | Counter | Bytes sent |
| `network_bytes_received_total` | Counter | Bytes received |
| `disk_usage_bytes` | Gauge | Disk usage |
| `system_health_status` | Gauge | Health indicator (1=healthy) |

### Convergence Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `convergence_score` | Gauge | Convergence score |
| `is_converged` | Gauge | Convergence flag (1=converged) |
| `rounds_until_convergence` | Gauge | Estimated rounds remaining |
| `model_weight_change` | Gauge | L2 norm of weight changes |

## Alerting

### Alert Rules

Configure Prometheus alert rules in `alert_rules.yml`:

```yaml
groups:
  - name: federated_learning_alerts
    interval: 30s
    rules:
      - alert: LowAUPRC
        expr: model_auprc{model_type="global"} < 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low AUPRC detected"
          description: "Global model AUPRC is {{ $value }}"

      - alert: PrivacyBudgetLow
        expr: privacy_epsilon_remaining < 0.1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Privacy budget nearly exhausted"
          description: "Client {{ $labels.client_id }} has {{ $value }} epsilon remaining"

      - alert: ClientFailureRate
        expr: rate(fl_client_failures_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High client failure rate"
          description: "Client failure rate is {{ $value }} failures/sec"

      - alert: SystemUnhealthy
        expr: system_health_status == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "System health check failed"
          description: "System is reporting unhealthy status"
```

### Notification Channels

Configure notification channels in Grafana:
1. Go to Alerting → Notification channels
2. Add channels (Email, Slack, PagerDuty, etc.)
3. Link channels to dashboard alerts

## Best Practices

1. **Scrape Interval**: Use 5-15 second intervals for real-time monitoring
2. **Retention**: Configure Prometheus retention based on storage capacity
3. **Alerting**: Set appropriate thresholds based on baseline performance
4. **Dashboards**: Customize panels based on specific monitoring needs
5. **Security**: Secure Prometheus/Grafana endpoints in production
6. **Backup**: Regularly backup Grafana dashboards and Prometheus data

## Troubleshooting

### Metrics Not Appearing

1. Check if HTTP server is running: `curl http://localhost:8000/metrics`
2. Verify Prometheus is scraping: Check Targets in Prometheus UI
3. Check Prometheus logs for scrape errors

### High Memory Usage

1. Reduce scrape interval
2. Decrease metric retention period
3. Use recording rules for complex queries

### Missing Data Points

1. Check network connectivity
2. Verify client is recording metrics
3. Check Prometheus scrape timeout settings

## Integration with Federated Learning

The Prometheus exporter integrates seamlessly with the federated learning pipeline:

```python
from src.monitoring import Prometheus_Exporter, MLflow_Logger

# Initialize both monitoring systems
prometheus = Prometheus_Exporter(port=8000)
mlflow = MLflow_Logger()

prometheus.start_http_server()
mlflow.start_run()

# During FL rounds, record to both systems
for round_num in range(num_rounds):
    # Prometheus: Real-time metrics
    prometheus.record_fl_round_start(round_num, num_clients)
    
    # MLflow: Experiment tracking
    mlflow.log_fl_round_metrics(round_num, metrics)
    
    # Both systems complement each other:
    # - Prometheus: Real-time monitoring, alerting
    # - MLflow: Historical tracking, reproducibility
```

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Client Python](https://github.com/prometheus/client_python)
- [Grafana Alerting](https://grafana.com/docs/grafana/latest/alerting/)
