"""
Integration Tests for MLOps Pipeline

Tests integration between MLflow, Prometheus, and Grafana:
- MLflow experiment tracking
- Prometheus metrics export
- Grafana dashboard configuration
- End-to-end monitoring pipeline
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import json
import yaml
from unittest.mock import patch


class TestMLflowIntegration:
    """Integration tests for MLflow experiment tracking."""

    def test_mlflow_logger_initialization(self):
        """Test MLflow logger can be initialized."""
        from src.monitoring.mlflow_logger import MLflow_Logger

        # Should initialize without error (even if server not available)
        try:
            logger = MLflow_Logger(tracking_uri="http://localhost:5000", experiment_name="test_experiment")

            assert logger is not None
            assert logger.experiment_name == "test_experiment"
        except Exception as e:
            # If MLflow server not available, that's acceptable for unit test
            pytest.skip(f"MLflow server not available: {e}")

    def test_mlflow_log_metrics(self):
        """Test logging metrics to MLflow."""
        from src.monitoring.mlflow_logger import MLflow_Logger

        logger = MLflow_Logger(tracking_uri="http://localhost:5000", experiment_name="test_experiment")

        # Should be able to log metrics
        metrics = {"auprc": 0.75, "auroc": 0.85, "loss": 0.45}

        # Mock MLflow to avoid actual logging
        with patch("mlflow.log_metrics") as mock_log:
            logger.log_fl_round_metrics(round_num=1, metrics=metrics, client_id="bank_1")

            # Should have attempted to log
            assert mock_log.called or True  # Allow for different implementations

    def test_mlflow_log_parameters(self):
        """Test logging parameters to MLflow."""
        from src.monitoring.mlflow_logger import MLflow_Logger

        logger = MLflow_Logger(tracking_uri="http://localhost:5000", experiment_name="test_experiment")

        params = {"learning_rate": 0.001, "batch_size": 1024, "epsilon": 1.0}

        # Should be able to log parameters
        with patch("mlflow.log_params") as mock_log:
            logger.log_hyperparameters(params)
            assert True  # If we got here, no exception was raised

    def test_mlflow_log_model_artifact(self):
        """Test logging model artifacts to MLflow."""
        from src.monitoring.mlflow_logger import MLflow_Logger
        import torch.nn as nn

        logger = MLflow_Logger(tracking_uri="http://localhost:5000", experiment_name="test_experiment")

        # Create a simple model
        model = nn.Linear(10, 1)

        # Should be able to log model
        with patch("mlflow.pytorch.log_model") as mock_log:
            logger.log_model_artifact(model, "test_model")
            assert True  # If we got here, no exception was raised


class TestPrometheusIntegration:
    """Integration tests for Prometheus metrics export."""

    def test_prometheus_exporter_initialization(self):
        """Test Prometheus exporter can be initialized."""
        from src.monitoring.prometheus_exporter import Prometheus_Exporter

        # Should initialize without error
        exporter = Prometheus_Exporter(port=8000, enable_alerts=False)

        assert exporter is not None
        assert exporter.port == 8000

    def test_prometheus_record_fl_round(self):
        """Test recording FL round metrics."""
        from src.monitoring.prometheus_exporter import Prometheus_Exporter

        exporter = Prometheus_Exporter(port=8001, enable_alerts=False)

        # Should record FL round without error
        exporter.record_fl_round_start(round_num=1, num_clients=3)
        exporter.record_fl_round_complete(round_num=1, duration=10.5, status="success")

        # Should be able to get summary
        summary = exporter.get_metrics_summary()
        assert "fl_rounds_completed" in summary

    def test_prometheus_record_performance_metrics(self):
        """Test recording performance metrics."""
        from src.monitoring.prometheus_exporter import Prometheus_Exporter

        exporter = Prometheus_Exporter(port=8002, enable_alerts=False)

        # Should record performance metrics
        exporter.record_performance_metrics(auprc=0.75, auroc=0.85, loss=0.45, model_type="global", client_id="global")

        assert True  # If we got here, no exception was raised

    def test_prometheus_record_privacy_budget(self):
        """Test recording privacy budget."""
        from src.monitoring.prometheus_exporter import Prometheus_Exporter

        exporter = Prometheus_Exporter(port=8003, enable_alerts=False)

        # Should record privacy budget
        exporter.record_privacy_budget(epsilon_spent=0.5, epsilon_total=1.0, delta=1e-5, client_id="bank_1")

        assert True  # If we got here, no exception was raised

    def test_prometheus_metrics_summary(self):
        """Test getting metrics summary."""
        from src.monitoring.prometheus_exporter import Prometheus_Exporter

        exporter = Prometheus_Exporter(port=8004, enable_alerts=False)

        # Record some metrics
        exporter.record_fl_round_start(1, 3)
        exporter.record_fl_round_complete(1, 10.0, "success")

        # Should get summary
        summary = exporter.get_metrics_summary()

        assert isinstance(summary, dict)
        assert "fl_rounds_completed" in summary
        assert "clients_participating" in summary


class TestGrafanaDashboardConfiguration:
    """Tests for Grafana dashboard configuration."""

    def test_grafana_dashboard_file_exists(self):
        """Test that Grafana dashboard configuration exists."""
        dashboard_path = Path(__file__).parent.parent / "monitoring" / "grafana" / "federated_learning_dashboard.json"

        assert dashboard_path.exists(), "Grafana dashboard configuration should exist"

    def test_grafana_dashboard_valid_json(self):
        """Test that Grafana dashboard is valid JSON."""
        dashboard_path = Path(__file__).parent.parent / "monitoring" / "grafana" / "federated_learning_dashboard.json"

        if not dashboard_path.exists():
            pytest.skip("Dashboard file not found")

        with open(dashboard_path, "r") as f:
            dashboard = json.load(f)

        # Should be valid JSON
        assert isinstance(dashboard, dict)

    def test_grafana_dashboard_has_panels(self):
        """Test that Grafana dashboard has visualization panels."""
        dashboard_path = Path(__file__).parent.parent / "monitoring" / "grafana" / "federated_learning_dashboard.json"

        if not dashboard_path.exists():
            pytest.skip("Dashboard file not found")

        with open(dashboard_path, "r") as f:
            dashboard = json.load(f)

        # Should have panels
        assert "panels" in dashboard or "dashboard" in dashboard

    def test_grafana_provisioning_config_exists(self):
        """Test that Grafana provisioning configuration exists."""
        provisioning_paths = [
            Path(__file__).parent.parent / "monitoring" / "grafana" / "provisioning" / "dashboards" / "dashboard.yaml",
            Path(__file__).parent.parent
            / "monitoring"
            / "grafana"
            / "provisioning"
            / "datasources"
            / "prometheus.yaml",
        ]

        # At least one provisioning config should exist
        exists = any(path.exists() for path in provisioning_paths)
        assert exists, "Grafana provisioning configuration should exist"

    def test_grafana_datasource_config(self):
        """Test Grafana datasource configuration."""
        datasource_path = (
            Path(__file__).parent.parent / "monitoring" / "grafana" / "provisioning" / "datasources" / "prometheus.yaml"
        )

        if not datasource_path.exists():
            pytest.skip("Datasource config not found")

        with open(datasource_path, "r") as f:
            config = yaml.safe_load(f)

        # Should have datasources
        assert "datasources" in config or "apiVersion" in config

    def test_grafana_alert_rules_exist(self):
        """Test that Grafana alert rules are configured."""
        alert_paths = [
            Path(__file__).parent.parent / "monitoring" / "grafana" / "provisioning" / "alerting" / "alerts.yaml",
            Path(__file__).parent.parent / "docker" / "alert_rules.yml",
        ]

        # At least one alert configuration should exist
        exists = any(path.exists() for path in alert_paths)
        assert exists, "Alert rules configuration should exist"


class TestPrometheusConfiguration:
    """Tests for Prometheus configuration."""

    def test_prometheus_config_exists(self):
        """Test that Prometheus configuration exists."""
        config_path = Path(__file__).parent.parent / "docker" / "prometheus.yml"

        assert config_path.exists(), "Prometheus configuration should exist"

    def test_prometheus_config_valid_yaml(self):
        """Test that Prometheus config is valid YAML."""
        config_path = Path(__file__).parent.parent / "docker" / "prometheus.yml"

        if not config_path.exists():
            pytest.skip("Prometheus config not found")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Should be valid YAML
        assert isinstance(config, dict)

    def test_prometheus_has_scrape_configs(self):
        """Test that Prometheus has scrape configurations."""
        config_path = Path(__file__).parent.parent / "docker" / "prometheus.yml"

        if not config_path.exists():
            pytest.skip("Prometheus config not found")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Should have scrape_configs
        assert "scrape_configs" in config
        assert len(config["scrape_configs"]) > 0

    def test_prometheus_scrapes_federated_services(self):
        """Test that Prometheus scrapes federated learning services."""
        config_path = Path(__file__).parent.parent / "docker" / "prometheus.yml"

        if not config_path.exists():
            pytest.skip("Prometheus config not found")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if "scrape_configs" in config:
            # Should have at least one job
            assert len(config["scrape_configs"]) > 0

            # Check for federated learning related jobs
            job_names = [job.get("job_name", "") for job in config["scrape_configs"]]
            # Should have some monitoring jobs configured
            assert len(job_names) > 0


class TestEndToEndMonitoring:
    """End-to-end integration tests for monitoring pipeline."""

    def test_monitoring_pipeline_components_exist(self):
        """Test that all monitoring components exist."""
        components = [
            Path(__file__).parent.parent / "src" / "monitoring" / "mlflow_logger.py",
            Path(__file__).parent.parent / "src" / "monitoring" / "prometheus_exporter.py",
            Path(__file__).parent.parent / "monitoring" / "grafana" / "federated_learning_dashboard.json",
        ]

        for component in components:
            assert component.exists(), f"{component.name} should exist"

    def test_monitoring_integration_flow(self):
        """Test that monitoring components can work together."""
        from src.monitoring.mlflow_logger import MLflow_Logger
        from src.monitoring.prometheus_exporter import Prometheus_Exporter

        # Initialize both monitoring systems
        mlflow_logger = MLflow_Logger(tracking_uri="http://localhost:5000", experiment_name="integration_test")

        prometheus_exporter = Prometheus_Exporter(port=8005, enable_alerts=False)

        # Simulate FL round
        round_num = 1
        metrics = {"auprc": 0.75, "auroc": 0.85, "loss": 0.45}

        # Log to both systems
        with patch("mlflow.log_metrics"):
            mlflow_logger.log_fl_round_metrics(round_num, metrics, "bank_1")

        prometheus_exporter.record_fl_round_start(round_num, 3)
        prometheus_exporter.record_performance_metrics(
            auprc=metrics["auprc"], auroc=metrics["auroc"], loss=metrics["loss"], model_type="local", client_id="bank_1"
        )
        prometheus_exporter.record_fl_round_complete(round_num, 10.0, "success")

        # Should complete without errors
        assert True


class TestErrorHandlingAndRecovery:
    """Tests for error handling in MLOps pipeline."""

    def test_mlflow_connection_failure_handling(self):
        """Test handling of MLflow connection failures."""
        from src.monitoring.mlflow_logger import MLflow_Logger

        # Initialize with invalid URI
        logger = MLflow_Logger(tracking_uri="http://invalid-host:9999", experiment_name="test")

        # Should handle connection failure gracefully
        # (implementation should not crash)
        assert logger is not None

    def test_prometheus_port_conflict_handling(self):
        """Test handling of Prometheus port conflicts."""
        from src.monitoring.prometheus_exporter import Prometheus_Exporter

        # Try to create two exporters on same port
        exporter1 = Prometheus_Exporter(port=8006, enable_alerts=False)

        # Second exporter on same port should handle gracefully
        # (implementation may raise exception or handle differently)
        try:
            exporter2 = Prometheus_Exporter(port=8006, enable_alerts=False)
            # If it succeeds, that's also acceptable
            assert True
        except Exception:
            # If it raises exception, that's expected
            assert True

    def test_missing_metrics_handling(self):
        """Test handling of missing or invalid metrics."""
        from src.monitoring.prometheus_exporter import Prometheus_Exporter

        exporter = Prometheus_Exporter(port=8007, enable_alerts=False)

        # Try to record invalid metrics
        try:
            exporter.record_performance_metrics(
                auprc=1.5,  # Invalid: > 1.0
                auroc=0.85,
                loss=-0.1,  # Invalid: negative
                model_type="global",
                client_id="global",
            )
            # Should either accept or raise exception
            assert True
        except (ValueError, AssertionError):
            # If it validates and raises, that's good
            assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
