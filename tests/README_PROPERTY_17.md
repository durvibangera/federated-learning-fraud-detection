# Property 17: Real-time Monitoring Integration

## Overview

This test suite validates Property 17 from the Federated Fraud Detection design document using property-based testing with Hypothesis.

**Property Statement:**
> For any federated learning execution, Prometheus should expose system metrics including round duration and convergence, Grafana should visualize progress, and alerts should trigger on failures or performance degradation.

**Validates:** Requirements 7.3, 7.4, 7.6

## Test Coverage

The property-based tests verify the following aspects of real-time monitoring:

### 1. FL Round Metrics Exposure
- **Test:** `test_fl_round_metrics_exposed`
- **Property:** FL round metrics should be correctly recorded and accessible
- **Validates:** Round completion tracking, client participation recording
- **Iterations:** 100

### 2. Performance Metrics Exposure
- **Test:** `test_performance_metrics_exposed`
- **Property:** Performance metrics should be within valid ranges and accessible
- **Validates:** AUPRC/AUROC in [0,1], loss non-negative, metrics queryable
- **Iterations:** 100

### 3. Privacy Budget Tracking
- **Test:** `test_privacy_budget_tracking`
- **Property:** Privacy budget should be tracked correctly
- **Validates:** Epsilon spent ≤ total, remaining budget non-negative, exhaustion detection
- **Iterations:** 100

### 4. System Health Metrics
- **Test:** `test_system_health_metrics`
- **Property:** System health metrics should be tracked correctly
- **Validates:** Memory/CPU/disk metrics positive, CPU in [0,100], health status valid
- **Iterations:** 100

### 5. Convergence Tracking
- **Test:** `test_convergence_tracking`
- **Property:** Convergence metrics should be tracked correctly
- **Validates:** Convergence score in [0,1], status boolean, weight change non-negative
- **Iterations:** 100

### 6. Alert Configuration
- **Test:** `test_alert_configuration`
- **Property:** Alert configurations should be accepted and stored
- **Validates:** Alert storage, valid comparison operators, valid severity levels
- **Iterations:** 50

### 7. Multiple Rounds Tracking
- **Test:** `test_multiple_rounds_tracking`
- **Property:** Multiple FL rounds should be tracked correctly
- **Validates:** All rounds recorded, monotonic round numbers, metric accumulation
- **Iterations:** 50

### 8. Client Failure Tracking
- **Test:** `test_client_failure_tracking`
- **Property:** Client failures should be tracked correctly
- **Validates:** Failure recording, valid failure types, client ID preservation
- **Iterations:** 50

### 9. Training Samples Tracking
- **Test:** `test_training_samples_tracking`
- **Property:** Training samples should be tracked per client
- **Validates:** Positive sample counts, per-client tracking
- **Iterations:** 50

### 10. Duration Tracking
- **Test:** `test_duration_tracking`
- **Property:** Round durations should be tracked correctly
- **Validates:** Positive durations, tracking by type (training/aggregation/evaluation)
- **Iterations:** 50

### 11. Metrics Summary Completeness
- **Test:** `test_metrics_summary_completeness`
- **Property:** Metrics summary should contain all required fields
- **Validates:** Required fields present, correct data types
- **Iterations:** 1 (deterministic test)

## Hypothesis Strategies

The tests use custom Hypothesis strategies to generate random test data:

### `fl_round_config()`
Generates random FL round configurations:
- `round_num`: 1-100
- `num_clients`: 1-10
- `duration`: 0.1-300.0 seconds
- `status`: success/failed/skipped

### `performance_metrics()`
Generates random performance metrics:
- `auprc`: 0.0-1.0
- `auroc`: 0.0-1.0
- `loss`: 0.0-10.0
- `model_type`: global/local
- `client_id`: global/bank1/bank2/bank3

### `privacy_budget_config()`
Generates random privacy budget configurations:
- `epsilon_total`: 0.5-10.0
- `epsilon_spent`: 0.0-epsilon_total
- `delta`: 1e-7 to 1e-3
- `client_id`: bank1/bank2/bank3

### `system_metrics_config()`
Generates random system metrics:
- `memory_usage_bytes`: 100MB-10GB
- `memory_available_bytes`: 1GB-32GB
- `cpu_usage_percent`: 0.0-100.0
- `disk_usage_bytes`: 1GB-1TB
- `health_status`: healthy/unhealthy

### `convergence_metrics_config()`
Generates random convergence metrics:
- `convergence_score`: 0.0-1.0
- `is_converged`: boolean (true if score > 0.9)
- `weight_change`: 0.0-10.0

### `alert_config_strategy()`
Generates random alert configurations:
- `metric_name`: random string
- `threshold`: 0.0-1.0
- `comparison`: gt/lt/eq
- `severity`: warning/critical
- `message`: random string

## Running the Tests

### Prerequisites

Install required dependencies:
```bash
pip install pytest hypothesis prometheus-client
```

### Run All Property Tests

```bash
pytest tests/test_property_17_monitoring.py -v
```

### Run with Statistics

```bash
pytest tests/test_property_17_monitoring.py -v --hypothesis-show-statistics
```

### Run Specific Test

```bash
pytest tests/test_property_17_monitoring.py::TestProperty17_RealTimeMonitoring::test_fl_round_metrics_exposed -v
```

### Run with Custom Examples

```bash
pytest tests/test_property_17_monitoring.py -v --hypothesis-seed=12345
```

### Run as Python Script

```bash
python tests/test_property_17_monitoring.py
```

## Test Configuration

The tests use the following Hypothesis settings:

```python
@settings(
    max_examples=100,  # or 50 for some tests
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
)
```

- **max_examples:** 50-100 iterations per test (exceeds minimum requirement of 100 total)
- **deadline:** Disabled to allow for slower operations
- **suppress_health_check:** Allows function-scoped fixtures and slower tests

## Expected Output

Successful test run should show:

```
tests/test_property_17_monitoring.py::TestProperty17_RealTimeMonitoring::test_fl_round_metrics_exposed PASSED
tests/test_property_17_monitoring.py::TestProperty17_RealTimeMonitoring::test_performance_metrics_exposed PASSED
tests/test_property_17_monitoring.py::TestProperty17_RealTimeMonitoring::test_privacy_budget_tracking PASSED
tests/test_property_17_monitoring.py::TestProperty17_RealTimeMonitoring::test_system_health_metrics PASSED
tests/test_property_17_monitoring.py::TestProperty17_RealTimeMonitoring::test_convergence_tracking PASSED
tests/test_property_17_monitoring.py::TestProperty17_RealTimeMonitoring::test_alert_configuration PASSED
tests/test_property_17_monitoring.py::TestProperty17_RealTimeMonitoring::test_multiple_rounds_tracking PASSED
tests/test_property_17_monitoring.py::TestProperty17_RealTimeMonitoring::test_client_failure_tracking PASSED
tests/test_property_17_monitoring.py::TestProperty17_RealTimeMonitoring::test_training_samples_tracking PASSED
tests/test_property_17_monitoring.py::TestProperty17_RealTimeMonitoring::test_duration_tracking PASSED
tests/test_property_17_monitoring.py::TestProperty17_RealTimeMonitoring::test_metrics_summary_completeness PASSED

======================== 11 passed in X.XXs ========================
```

## Verification

To verify the test implementation:

```bash
python verify_property_17_tests.py
```

This checks:
- File existence and syntax
- Test class and methods
- Hypothesis imports and decorators
- Strategy definitions
- Property assertions
- Requirements validation
- Feature tags

## Integration with CI/CD

Add to GitHub Actions workflow:

```yaml
- name: Run Property 17 Tests
  run: |
    pytest tests/test_property_17_monitoring.py -v --hypothesis-show-statistics
```

## Troubleshooting

### Port Conflicts

If you see port binding errors, the tests use ports 8100-9100. Ensure these are available or modify the test to use different ports.

### Slow Tests

Some tests may be slow due to 100 iterations. This is expected for thorough property testing. Use `--hypothesis-profile=dev` for faster development testing:

```bash
pytest tests/test_property_17_monitoring.py --hypothesis-profile=dev
```

### Import Errors

If you see import errors for `src.monitoring`, ensure you're running from the project root and the src directory is in your Python path.

## Property Validation Summary

This test suite validates that the Prometheus_Exporter:

✓ Correctly exposes FL round metrics (duration, success/failure)  
✓ Tracks model performance metrics (AUPRC, AUROC, loss)  
✓ Monitors privacy budget consumption (epsilon, delta)  
✓ Records system health metrics (CPU, memory, disk, network)  
✓ Tracks convergence metrics (score, status, weight changes)  
✓ Supports configurable alerting with thresholds  
✓ Handles multiple FL rounds correctly  
✓ Tracks client failures and recovery  
✓ Records training samples per client  
✓ Tracks durations by operation type  
✓ Provides complete metrics summaries  

All properties hold across randomized inputs, ensuring robust real-time monitoring for the federated fraud detection system.
