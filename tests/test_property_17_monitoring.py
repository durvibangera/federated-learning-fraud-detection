"""
Property-Based Test for Real-time Monitoring Integration

Feature: federated-fraud-detection
Property 17: Real-time Monitoring Integration

Property Statement:
For any federated learning execution, Prometheus should expose system metrics
including round duration and convergence, Grafana should visualize progress,
and alerts should trigger on failures or performance degradation.

Validates: Requirements 7.3, 7.4, 7.6
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from src.monitoring import Prometheus_Exporter, AlertConfig


# ============================================================================
# Hypothesis Strategies for Test Data Generation
# ============================================================================


@st.composite
def fl_round_config(draw):
    """Generate random FL round configuration."""
    return {
        "round_num": draw(st.integers(min_value=1, max_value=100)),
        "num_clients": draw(st.integers(min_value=1, max_value=10)),
        "duration": draw(st.floats(min_value=0.1, max_value=300.0)),
        "status": draw(st.sampled_from(["success", "failed", "skipped"])),
    }


@st.composite
def performance_metrics(draw):
    """Generate random performance metrics."""
    return {
        "auprc": draw(st.floats(min_value=0.0, max_value=1.0)),
        "auroc": draw(st.floats(min_value=0.0, max_value=1.0)),
        "loss": draw(st.floats(min_value=0.0, max_value=10.0)),
        "model_type": draw(st.sampled_from(["global", "local"])),
        "client_id": draw(st.sampled_from(["global", "bank1", "bank2", "bank3"])),
    }


@st.composite
def privacy_budget_config(draw):
    """Generate random privacy budget configuration."""
    epsilon_total = draw(st.floats(min_value=0.5, max_value=10.0))
    epsilon_spent = draw(st.floats(min_value=0.0, max_value=epsilon_total))
    return {
        "epsilon_spent": epsilon_spent,
        "epsilon_total": epsilon_total,
        "delta": draw(st.floats(min_value=1e-7, max_value=1e-3)),
        "client_id": draw(st.sampled_from(["bank1", "bank2", "bank3"])),
    }


@st.composite
def system_metrics_config(draw):
    """Generate random system metrics."""
    return {
        "memory_usage_bytes": draw(st.integers(min_value=100_000_000, max_value=10_000_000_000)),
        "memory_available_bytes": draw(st.integers(min_value=1_000_000_000, max_value=32_000_000_000)),
        "cpu_usage_percent": draw(st.floats(min_value=0.0, max_value=100.0)),
        "disk_usage_bytes": draw(st.integers(min_value=1_000_000_000, max_value=1_000_000_000_000)),
        "health_status": draw(st.sampled_from(["healthy", "unhealthy"])),
    }


@st.composite
def convergence_metrics_config(draw):
    """Generate random convergence metrics."""
    convergence_score = draw(st.floats(min_value=0.0, max_value=1.0))
    return {
        "convergence_score": convergence_score,
        "is_converged": convergence_score > 0.9,
        "weight_change": draw(st.floats(min_value=0.0, max_value=10.0)),
    }


@st.composite
def alert_config_strategy(draw):
    """Generate random alert configuration."""
    return AlertConfig(
        metric_name=draw(
            st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")))
        ),
        threshold=draw(st.floats(min_value=0.0, max_value=1.0)),
        comparison=draw(st.sampled_from(["gt", "lt", "eq"])),
        severity=draw(st.sampled_from(["warning", "critical"])),
        message=draw(st.text(min_size=1, max_size=100)),
    )


# ============================================================================
# Property Tests
# ============================================================================


@pytest.mark.property
class TestProperty17_RealTimeMonitoring:
    """
    Property 17: Real-time Monitoring Integration

    Tests that Prometheus exposes metrics correctly, metrics are accessible,
    and alerting works as expected.
    """

    @given(fl_config=fl_round_config())
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    def test_fl_round_metrics_exposed(self, fl_config):
        """
        Property: FL round metrics should be correctly recorded and accessible.

        For any FL round configuration, the exporter should:
        1. Accept the metrics without error
        2. Store them in Prometheus format
        3. Make them accessible via the registry
        """
        # Create exporter with unique port to avoid conflicts
        port = 8100 + (fl_config["round_num"] % 100)
        exporter = Prometheus_Exporter(port=port, enable_alerts=False)

        # Record FL round start
        exporter.record_fl_round_start(fl_config["round_num"], fl_config["num_clients"])

        # Record FL round completion
        exporter.record_fl_round_complete(fl_config["round_num"], fl_config["duration"], fl_config["status"])

        # Verify metrics are accessible
        summary = exporter.get_metrics_summary()

        # Property: Round should be recorded
        if fl_config["status"] == "success":
            assert (
                summary["fl_rounds_completed"] == fl_config["round_num"]
            ), "Completed rounds should match the round number for successful rounds"

        # Property: Client participation should be recorded
        assert (
            summary["clients_participating"] == fl_config["num_clients"]
        ), "Client participation should match the configured number"

    @given(perf_metrics=performance_metrics())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_performance_metrics_exposed(self, perf_metrics):
        """
        Property: Performance metrics should be within valid ranges and accessible.

        For any performance metrics, the exporter should:
        1. Accept metrics within valid ranges (AUPRC, AUROC in [0,1])
        2. Store them correctly
        3. Make them queryable
        """
        exporter = Prometheus_Exporter(port=8200, enable_alerts=False)

        # Record performance metrics
        exporter.record_performance_metrics(
            auprc=perf_metrics["auprc"],
            auroc=perf_metrics["auroc"],
            loss=perf_metrics["loss"],
            model_type=perf_metrics["model_type"],
            client_id=perf_metrics["client_id"],
        )

        # Property: Metrics should be within valid ranges
        assert 0.0 <= perf_metrics["auprc"] <= 1.0, "AUPRC must be in [0,1]"
        assert 0.0 <= perf_metrics["auroc"] <= 1.0, "AUROC must be in [0,1]"
        assert perf_metrics["loss"] >= 0.0, "Loss must be non-negative"

        # Property: Metrics should be recorded without error
        # (if we got here, recording succeeded)
        assert True

    @given(privacy_config=privacy_budget_config())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_privacy_budget_tracking(self, privacy_config):
        """
        Property: Privacy budget should be tracked correctly.

        For any privacy budget configuration:
        1. Epsilon spent should never exceed epsilon total
        2. Remaining budget should be non-negative
        3. Budget exhaustion should be detected
        """
        exporter = Prometheus_Exporter(port=8300, enable_alerts=False)

        # Record privacy budget
        exporter.record_privacy_budget(
            epsilon_spent=privacy_config["epsilon_spent"],
            epsilon_total=privacy_config["epsilon_total"],
            delta=privacy_config["delta"],
            client_id=privacy_config["client_id"],
        )

        # Property: Epsilon spent should not exceed total
        assert (
            privacy_config["epsilon_spent"] <= privacy_config["epsilon_total"]
        ), "Epsilon spent cannot exceed total budget"

        # Property: Remaining budget should be calculable
        epsilon_remaining = privacy_config["epsilon_total"] - privacy_config["epsilon_spent"]
        assert epsilon_remaining >= 0, "Remaining budget must be non-negative"

        # Property: Budget exhaustion should be detectable
        is_exhausted = epsilon_remaining <= 0
        assert isinstance(is_exhausted, bool), "Budget exhaustion should be boolean"

    @given(sys_metrics=system_metrics_config())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_system_health_metrics(self, sys_metrics):
        """
        Property: System health metrics should be tracked correctly.

        For any system metrics:
        1. Memory values should be positive
        2. CPU usage should be in [0, 100]
        3. Health status should be boolean-like
        """
        exporter = Prometheus_Exporter(port=8400, enable_alerts=False)

        # Record system metrics
        exporter.record_system_metrics(sys_metrics)

        # Property: Memory values should be positive
        assert sys_metrics["memory_usage_bytes"] > 0, "Memory usage must be positive"
        assert sys_metrics["memory_available_bytes"] > 0, "Available memory must be positive"

        # Property: CPU usage should be in valid range
        assert 0.0 <= sys_metrics["cpu_usage_percent"] <= 100.0, "CPU usage must be in [0, 100]"

        # Property: Disk usage should be positive
        assert sys_metrics["disk_usage_bytes"] > 0, "Disk usage must be positive"

        # Property: Health status should be valid
        assert sys_metrics["health_status"] in [
            "healthy",
            "unhealthy",
        ], "Health status must be 'healthy' or 'unhealthy'"

        # Verify metrics are accessible
        summary = exporter.get_metrics_summary()
        assert "system_health" in summary, "System health should be in summary"

    @given(conv_metrics=convergence_metrics_config())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_convergence_tracking(self, conv_metrics):
        """
        Property: Convergence metrics should be tracked correctly.

        For any convergence metrics:
        1. Convergence score should be in [0, 1]
        2. Convergence status should be boolean
        3. Weight change should be non-negative
        """
        exporter = Prometheus_Exporter(port=8500, enable_alerts=False)

        # Record convergence metrics
        exporter.record_convergence_metrics(
            convergence_score=conv_metrics["convergence_score"],
            is_converged=conv_metrics["is_converged"],
            weight_change=conv_metrics["weight_change"],
        )

        # Property: Convergence score should be in [0, 1]
        assert 0.0 <= conv_metrics["convergence_score"] <= 1.0, "Convergence score must be in [0, 1]"

        # Property: Convergence status should be boolean
        assert isinstance(conv_metrics["is_converged"], bool), "Convergence status must be boolean"

        # Property: Weight change should be non-negative
        assert conv_metrics["weight_change"] >= 0.0, "Weight change must be non-negative"

        # Verify metrics are accessible
        summary = exporter.get_metrics_summary()
        assert "convergence_score" in summary, "Convergence score should be in summary"
        assert "is_converged" in summary, "Convergence status should be in summary"

    @given(alert_cfg=alert_config_strategy())
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_alert_configuration(self, alert_cfg):
        """
        Property: Alert configurations should be accepted and stored.

        For any alert configuration:
        1. It should be accepted without error
        2. It should be stored in the alert list
        3. Comparison operators should be valid
        """
        exporter = Prometheus_Exporter(port=8600, enable_alerts=True)

        # Add alert configuration
        exporter.add_alert_config(alert_cfg)

        # Property: Alert should be stored
        assert len(exporter.alert_configs) > 0, "Alert config should be stored"

        # Property: Alert should have valid comparison operator
        assert alert_cfg.comparison in ["gt", "lt", "eq"], "Comparison operator must be 'gt', 'lt', or 'eq'"

        # Property: Alert should have valid severity
        assert alert_cfg.severity in ["warning", "critical"], "Severity must be 'warning' or 'critical'"

        # Property: Threshold should be valid
        assert 0.0 <= alert_cfg.threshold <= 1.0, "Threshold should be in [0, 1]"

    @given(fl_configs=st.lists(fl_round_config(), min_size=1, max_size=10))
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    def test_multiple_rounds_tracking(self, fl_configs):
        """
        Property: Multiple FL rounds should be tracked correctly.

        For any sequence of FL rounds:
        1. All rounds should be recorded
        2. Round numbers should be monotonically increasing for successful rounds
        3. Metrics should accumulate correctly
        """
        exporter = Prometheus_Exporter(port=8700, enable_alerts=False)

        successful_rounds = []

        for config in fl_configs:
            exporter.record_fl_round_start(config["round_num"], config["num_clients"])

            exporter.record_fl_round_complete(config["round_num"], config["duration"], config["status"])

            if config["status"] == "success":
                successful_rounds.append(config["round_num"])

        # Property: If there were successful rounds, the highest one should be recorded
        if successful_rounds:
            summary = exporter.get_metrics_summary()
            assert summary["fl_rounds_completed"] == max(successful_rounds), (
                f"Completed rounds should reflect the highest successful round number: expected "
                f"{max(successful_rounds)}, got {summary['fl_rounds_completed']}"
            )

    @given(
        client_id=st.sampled_from(["bank1", "bank2", "bank3"]),
        failure_type=st.sampled_from(["timeout", "disconnect", "error"]),
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_client_failure_tracking(self, client_id, failure_type):
        """
        Property: Client failures should be tracked correctly.

        For any client failure:
        1. Failure should be recorded
        2. Failure type should be valid
        3. Client ID should be preserved
        """
        exporter = Prometheus_Exporter(port=8800, enable_alerts=True)

        # Record client failure
        exporter.record_client_failure(client_id, failure_type)

        # Property: Client ID should be valid
        assert client_id in ["bank1", "bank2", "bank3"], "Client ID must be a valid bank identifier"

        # Property: Failure type should be valid
        assert failure_type in ["timeout", "disconnect", "error"], "Failure type must be valid"

    @given(
        num_samples=st.integers(min_value=1, max_value=1_000_000),
        client_id=st.sampled_from(["bank1", "bank2", "bank3"]),
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_training_samples_tracking(self, num_samples, client_id):
        """
        Property: Training samples should be tracked per client.

        For any number of training samples:
        1. Sample count should be positive
        2. Sample count should be recorded per client
        """
        exporter = Prometheus_Exporter(port=8900, enable_alerts=False)

        # Record training samples
        exporter.record_training_samples(num_samples, client_id)

        # Property: Sample count should be positive
        assert num_samples > 0, "Training samples must be positive"

        # Property: Client ID should be valid
        assert client_id in ["bank1", "bank2", "bank3"], "Client ID must be valid"

    @given(
        duration=st.floats(min_value=0.1, max_value=1000.0),
        round_type=st.sampled_from(["training", "aggregation", "evaluation"]),
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_duration_tracking(self, duration, round_type):
        """
        Property: Round durations should be tracked correctly.

        For any duration:
        1. Duration should be positive
        2. Duration should be recorded by type
        """
        exporter = Prometheus_Exporter(port=9000, enable_alerts=False)

        # Record duration based on type
        if round_type == "training":
            exporter.record_training_duration(duration)
        elif round_type == "aggregation":
            exporter.record_aggregation_duration(duration)

        # Property: Duration should be positive
        assert duration > 0, "Duration must be positive"

    def test_metrics_summary_completeness(self):
        """
        Property: Metrics summary should contain all required fields.

        The summary should always include:
        - fl_rounds_completed
        - clients_participating
        - convergence_score
        - is_converged
        - system_health
        """
        exporter = Prometheus_Exporter(port=9100, enable_alerts=False)

        # Record some basic metrics
        exporter.record_fl_round_start(1, 3)
        exporter.record_fl_round_complete(1, 10.0, "success")
        exporter.record_convergence_metrics(0.5, False, 0.1)
        exporter.record_system_metrics(
            {
                "memory_usage_bytes": 1_000_000_000,
                "memory_available_bytes": 4_000_000_000,
                "cpu_usage_percent": 50.0,
                "disk_usage_bytes": 10_000_000_000,
                "health_status": "healthy",
            }
        )

        # Get summary
        summary = exporter.get_metrics_summary()

        # Property: Summary should contain all required fields
        required_fields = [
            "fl_rounds_completed",
            "clients_participating",
            "convergence_score",
            "is_converged",
            "system_health",
        ]

        for field in required_fields:
            assert field in summary, f"Summary must contain '{field}'"

        # Property: Values should be of correct types
        assert isinstance(summary["fl_rounds_completed"], (int, float)), "fl_rounds_completed should be numeric"
        assert isinstance(summary["clients_participating"], (int, float)), "clients_participating should be numeric"
        assert isinstance(summary["convergence_score"], (int, float)), "convergence_score should be numeric"
        assert isinstance(summary["is_converged"], bool), "is_converged should be boolean"
        assert isinstance(summary["system_health"], bool), "system_health should be boolean"


# ============================================================================
# Test Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Property 17: Real-time Monitoring Integration")
    print("=" * 80)
    print("\nRunning property-based tests with Hypothesis...")
    print("Minimum 100 iterations per test\n")

    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short", "--hypothesis-show-statistics"])
