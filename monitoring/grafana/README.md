# Grafana Dashboard Configuration

## Overview

This directory contains Grafana dashboard configurations and provisioning files for the Federated Fraud Detection system.

## Directory Structure

```
monitoring/grafana/
├── provisioning/
│   ├── datasources/
│   │   └── prometheus.yaml          # Prometheus datasource config
│   ├── dashboards/
│   │   └── dashboard.yaml           # Dashboard provisioning config
│   └── alerting/
│       └── alerts.yaml              # Alert rules configuration
├── federated_learning_dashboard.json  # Main FL dashboard
└── README.md                        # This file
```

## Dashboards

### 1. Federated Learning Dashboard

**File:** `federated_learning_dashboard.json`

**Panels (15 total):**

1. **FL Rounds Progress** - Track completed rounds over time
2. **FL Round Duration** - Monitor training/aggregation time
3. **Model Performance (AUPRC)** - Global and local model AUPRC
4. **Model Performance (AUROC)** - Global and local model AUROC
5. **Privacy Budget Spent** - Track epsilon consumption per client
6. **Privacy Budget Remaining** - Monitor remaining budget
7. **Convergence Score** - Track model convergence
8. **Model Weight Change** - Monitor weight updates (L2 norm)
9. **Client Participation** - Active clients per round
10. **Client Failures** - Track client disconnections
11. **System Memory Usage** - Memory consumption
12. **System CPU Usage** - CPU utilization
13. **Network Traffic** - Bytes sent/received
14. **System Health Status** - Overall health indicator
15. **FL Round Success Rate** - Success percentage

**Alerts Configured:**
- Low AUPRC (< 0.5) - Warning
- Privacy Budget Low (< 0.1 remaining) - Critical

## Provisioning

### Datasources

**File:** `provisioning/datasources/prometheus.yaml`

Automatically configures Prometheus as the default datasource:
- URL: http://prometheus:9090
- Scrape interval: 5s
- Query timeout: 60s
- HTTP method: POST

### Dashboards

**File:** `provisioning/dashboards/dashboard.yaml`

Automatically loads dashboards from this directory:
- Folder: "Federated Fraud Detection"
- Auto-update: Every 10 seconds
- UI updates: Allowed

### Alerting

**File:** `provisioning/alerting/alerts.yaml`

Pre-configured alert rules:

1. **Low AUPRC Alert**
   - Condition: AUPRC < 0.5
   - Duration: 5 minutes
   - Severity: Warning

2. **Privacy Budget Low**
   - Condition: Remaining epsilon < 0.1
   - Duration: 1 minute
   - Severity: Critical

3. **System Unhealthy**
   - Condition: Health status == 0
   - Duration: 2 minutes
   - Severity: Critical

4. **High Client Failure Rate**
   - Condition: Failure rate > 0.1/sec
   - Duration: 5 minutes
   - Severity: Warning

## Setup

### Docker Compose Integration

The Grafana service is configured in `docker/docker-compose.yml`:

```yaml
grafana:
  image: grafana/grafana:latest
  ports:
    - "3000:3000"
  volumes:
    - grafana_data:/var/lib/grafana
    - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=admin
```

### Manual Setup

1. **Start Grafana:**
   ```bash
   docker compose up -d grafana
   ```

2. **Access UI:**
   - URL: http://localhost:3000
   - Username: admin
   - Password: admin (change on first login)

3. **Verify Datasource:**
   - Go to Configuration → Data Sources
   - Prometheus should be listed and working

4. **View Dashboards:**
   - Go to Dashboards → Browse
   - Open "Federated Fraud Detection" folder
   - Select "Federated Learning Dashboard"

## Dashboard Customization

### Adding New Panels

1. **Edit Dashboard:**
   - Click "Dashboard settings" (gear icon)
   - Click "Add panel"

2. **Configure Query:**
   ```promql
   # Example: Average AUPRC across all clients
   avg(model_auprc{model_type="local"})
   ```

3. **Set Visualization:**
   - Choose panel type (Graph, Stat, Gauge, etc.)
   - Configure axes, legends, colors

4. **Save Dashboard:**
   - Click "Save dashboard"
   - Export JSON if needed

### Modifying Existing Panels

1. **Edit Panel:**
   - Hover over panel title
   - Click "Edit" (pencil icon)

2. **Update Query:**
   - Modify PromQL expression
   - Adjust time range
   - Change aggregation

3. **Update Visualization:**
   - Change panel type
   - Modify colors/thresholds
   - Update legends

### Exporting Dashboards

```bash
# Export dashboard JSON
curl -u admin:admin http://localhost:3000/api/dashboards/uid/<dashboard-uid> > dashboard.json

# Or use Grafana UI:
# Dashboard settings → JSON Model → Copy to clipboard
```

## Alert Configuration

### Adding New Alerts

1. **Create Alert Rule:**
   - Go to Alerting → Alert rules
   - Click "New alert rule"

2. **Configure Query:**
   ```promql
   # Example: High memory usage
   (system_memory_usage_bytes / system_memory_available_bytes) > 0.9
   ```

3. **Set Conditions:**
   - Threshold: 0.9
   - Duration: 5m
   - Severity: Warning

4. **Configure Notifications:**
   - Add notification channel
   - Set message template

### Notification Channels

**Supported channels:**
- Email
- Slack
- PagerDuty
- Webhook
- Microsoft Teams
- Discord

**Setup Example (Slack):**

1. Go to Alerting → Notification channels
2. Click "New channel"
3. Select "Slack"
4. Enter webhook URL
5. Test notification
6. Save

## Metrics Reference

### FL Round Metrics

```promql
# Completed rounds
fl_rounds_completed

# Round duration (average)
rate(fl_round_duration_seconds_sum[5m]) / rate(fl_round_duration_seconds_count[5m])

# Client participation
fl_clients_participating

# Client failures
rate(fl_client_failures_total[5m])
```

### Performance Metrics

```promql
# Global model AUPRC
model_auprc{model_type="global"}

# Local model AUPRC (all clients)
model_auprc{model_type="local"}

# Specific client AUPRC
model_auprc{model_type="local",client_id="bank1"}

# Model loss
model_loss{loss_type="test"}
```

### Privacy Metrics

```promql
# Epsilon spent
privacy_epsilon_spent{client_id="bank1"}

# Epsilon remaining
privacy_epsilon_remaining{client_id="bank1"}

# Budget exhausted flag
privacy_budget_exhausted{client_id="bank1"}
```

### System Metrics

```promql
# Memory usage
system_memory_usage_bytes{component="total"}

# CPU usage
system_cpu_usage_percent{component="total"}

# Network traffic (sent)
rate(network_bytes_sent_total[5m])

# Network traffic (received)
rate(network_bytes_received_total[5m])

# System health
system_health_status
```

### Convergence Metrics

```promql
# Convergence score
convergence_score

# Is converged
is_converged

# Model weight change
model_weight_change
```

## Troubleshooting

### Dashboard Not Loading

```bash
# Check Grafana logs
docker compose logs grafana

# Verify Prometheus connection
curl http://localhost:3000/api/datasources/proxy/1/api/v1/query?query=up

# Restart Grafana
docker compose restart grafana
```

### No Data in Panels

```bash
# Check Prometheus is scraping
curl http://localhost:9090/api/v1/targets

# Verify metrics are being exported
curl http://localhost:8000/metrics  # Bank client
curl http://localhost:8001/metrics  # Aggregation server

# Check time range in dashboard
# Ensure it matches when FL training is running
```

### Alerts Not Firing

```bash
# Check alert rules
curl http://localhost:3000/api/v1/provisioning/alert-rules

# Verify notification channels
curl -u admin:admin http://localhost:3000/api/alert-notifications

# Check Grafana alerting logs
docker compose logs grafana | grep -i alert
```

### Permission Issues

```bash
# Fix volume permissions
docker compose down
sudo chown -R 472:472 monitoring/grafana/
docker compose up -d grafana
```

## Best Practices

### 1. Dashboard Organization

- Group related panels together
- Use rows to organize sections
- Add panel descriptions
- Use consistent colors/themes

### 2. Query Optimization

- Use recording rules for complex queries
- Limit time ranges appropriately
- Use rate() for counters
- Aggregate before visualizing

### 3. Alert Configuration

- Set appropriate thresholds
- Use reasonable evaluation intervals
- Configure notification channels
- Test alerts before deploying

### 4. Performance

- Limit number of panels per dashboard
- Use appropriate refresh intervals
- Cache query results when possible
- Use variables for dynamic queries

## Advanced Features

### Variables

Create dashboard variables for dynamic filtering:

```
# Variable: bank_id
Query: label_values(model_auprc, client_id)
Type: Query
Multi-value: Yes

# Usage in panel:
model_auprc{client_id=~"$bank_id"}
```

### Annotations

Add annotations for FL round events:

```promql
# Query for round completions
changes(fl_rounds_completed[1m]) > 0
```

### Templating

Use template variables in panel titles:

```
Panel Title: AUPRC - $bank_id
```

### Links

Add links between dashboards:

```json
{
  "title": "View System Health",
  "url": "/d/system-health/system-health-dashboard"
}
```

## Requirements Satisfied

### Requirement 7.4: System Health and Performance Dashboards
✅ Comprehensive dashboards for FL progress, performance, and system health

### Requirement 7.6: Alerting for Failures and Anomalies
✅ Pre-configured alert rules for critical conditions

## Summary

The Grafana configuration provides:
- ✅ Automated provisioning of datasources
- ✅ Pre-built FL monitoring dashboard
- ✅ Alert rules for critical conditions
- ✅ System health visualization
- ✅ Privacy budget tracking
- ✅ Performance monitoring

Access Grafana at http://localhost:3000 after starting the Docker stack.
