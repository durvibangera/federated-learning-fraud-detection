"""
Prometheus Exporter Implementation for Real-Time Metrics

This module implements Prometheus-based real-time monitoring for federated learning,
including custom metrics for FL rounds, system health, and performance tracking.
"""

from prometheus_client import (
    Counter, Gauge, Histogram, Summary, Info,
    CollectorRegistry, push_to_gateway, start_http_server
)
from typing import Dict, Any, Optional, List
from loguru import logger
import time
import threading
from dataclasses import dataclass
from enum import Enum


class MetricType(Enum):
    """Enumeration of metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class AlertConfig:
    """Configuration for alerting thresholds."""
    metric_name: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    severity: str  # 'warning', 'critical'
    message: str


class Prometheus_Exporter:
    """
    Prometheus-based real-time monitoring for federated learning.
    
    This class provides comprehensive real-time metrics including:
    - FL round metrics (duration, success/failure counts)
    - Model performance metrics (AUPRC, AUROC, loss)
    - Privacy budget tracking
    - System health metrics (memory, CPU, network)
    - Convergence tracking
    - Alerting for failures and performance issues
    
    Attributes:
        registry: Prometheus collector registry
        port: HTTP server port for metrics endpoint
        pushgateway_url: Optional Pushgateway URL for batch jobs
        alert_configs: List of alert configurations
    """
    
    def __init__(
        self,
        port: int = 8000,
        pushgateway_url: Optional[str] = None,
        job_name: str = "federated_fraud_detection",
        enable_alerts: bool = True
    ):
        """
        Initialize Prometheus_Exporter.
        
        Args:
            port: Port for Prometheus HTTP server
            pushgateway_url: Optional Pushgateway URL (e.g., 'localhost:9091')
            job_name: Job name for Pushgateway
            enable_alerts: Whether to enable alerting
        """
        self.port = port
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name
        self.enable_alerts = enable_alerts
        self.alert_configs: List[AlertConfig] = []
        
        # Create custom registry
        self.registry = CollectorRegistry()
        
        # Initialize FL round metrics
        self._init_fl_metrics()
        
        # Initialize model performance metrics
        self._init_performance_metrics()
        
        # Initialize privacy metrics
        self._init_privacy_metrics()
        
        # Initialize system health metrics
        self._init_system_metrics()
        
        # Initialize convergence metrics
        self._init_convergence_metrics()
        
        # Start HTTP server in background thread
        self._server_thread = None
        self._server_started = False
        
        logger.info(f"Prometheus_Exporter initialized on port {port}")
    
    def _init_fl_metrics(self) -> None:
        """Initialize federated learning round metrics."""
        # FL round counters
        self.fl_rounds_total = Counter(
            'fl_rounds_total',
            'Total number of FL rounds executed',
            ['status'],  # success, failed, skipped
            registry=self.registry
        )
        
        self.fl_rounds_completed = Gauge(
            'fl_rounds_completed',
            'Number of completed FL rounds',
            registry=self.registry
        )
        
        # FL round duration
        self.fl_round_duration_seconds = Histogram(
            'fl_round_duration_seconds',
            'Duration of FL rounds in seconds',
            ['round_type'],  # training, evaluation, aggregation
            buckets=[1, 5, 10, 30, 60, 120, 300, 600],
            registry=self.registry
        )
        
        # Client participation
        self.fl_clients_participating = Gauge(
            'fl_clients_participating',
            'Number of clients participating in current round',
            registry=self.registry
        )
        
        self.fl_client_failures = Counter(
            'fl_client_failures_total',
            'Total number of client failures',
            ['client_id', 'failure_type'],
            registry=self.registry
        )
        
        # Training metrics per round
        self.fl_training_samples = Gauge(
            'fl_training_samples',
            'Number of training samples in current round',
            ['client_id'],
            registry=self.registry
        )
    
    def _init_performance_metrics(self) -> None:
        """Initialize model performance metrics."""
        # AUPRC metrics
        self.model_auprc = Gauge(
            'model_auprc',
            'Area Under Precision-Recall Curve',
            ['model_type', 'client_id'],  # global, local
            registry=self.registry
        )
        
        # AUROC metrics
        self.model_auroc = Gauge(
            'model_auroc',
            'Area Under ROC Curve',
            ['model_type', 'client_id'],
            registry=self.registry
        )
        
        # Loss metrics
        self.model_loss = Gauge(
            'model_loss',
            'Model loss value',
            ['loss_type', 'client_id'],  # train, val, test
            registry=self.registry
        )
        
        # Prediction metrics
        self.predictions_total = Counter(
            'predictions_total',
            'Total number of predictions made',
            ['client_id', 'prediction_class'],  # fraud, legitimate
            registry=self.registry
        )
        
        # Performance degradation
        self.performance_degradation = Gauge(
            'performance_degradation_percent',
            'Performance degradation compared to baseline',
            ['metric_name'],
            registry=self.registry
        )
    
    def _init_privacy_metrics(self) -> None:
        """Initialize differential privacy metrics."""
        # Privacy budget tracking
        self.privacy_epsilon_spent = Gauge(
            'privacy_epsilon_spent',
            'Cumulative epsilon privacy budget spent',
            ['client_id'],
            registry=self.registry
        )
        
        self.privacy_epsilon_remaining = Gauge(
            'privacy_epsilon_remaining',
            'Remaining epsilon privacy budget',
            ['client_id'],
            registry=self.registry
        )
        
        self.privacy_delta = Gauge(
            'privacy_delta',
            'Delta parameter for differential privacy',
            ['client_id'],
            registry=self.registry
        )
        
        # Privacy budget exhaustion alerts
        self.privacy_budget_exhausted = Gauge(
            'privacy_budget_exhausted',
            'Whether privacy budget is exhausted (1=yes, 0=no)',
            ['client_id'],
            registry=self.registry
        )
    
    def _init_system_metrics(self) -> None:
        """Initialize system health and performance metrics."""
        # Memory metrics
        self.system_memory_usage_bytes = Gauge(
            'system_memory_usage_bytes',
            'Memory usage in bytes',
            ['component'],  # client, server, total
            registry=self.registry
        )
        
        self.system_memory_available_bytes = Gauge(
            'system_memory_available_bytes',
            'Available memory in bytes',
            registry=self.registry
        )
        
        # CPU metrics
        self.system_cpu_usage_percent = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            ['component'],
            registry=self.registry
        )
        
        # Network metrics
        self.network_bytes_sent = Counter(
            'network_bytes_sent_total',
            'Total bytes sent over network',
            ['component'],
            registry=self.registry
        )
        
        self.network_bytes_received = Counter(
            'network_bytes_received_total',
            'Total bytes received over network',
            ['component'],
            registry=self.registry
        )
        
        # Disk metrics
        self.disk_usage_bytes = Gauge(
            'disk_usage_bytes',
            'Disk usage in bytes',
            ['path'],
            registry=self.registry
        )
        
        # System health status
        self.system_health_status = Gauge(
            'system_health_status',
            'Overall system health (1=healthy, 0=unhealthy)',
            registry=self.registry
        )
    
    def _init_convergence_metrics(self) -> None:
        """Initialize convergence tracking metrics."""
        # Convergence score
        self.convergence_score = Gauge(
            'convergence_score',
            'Model convergence score',
            registry=self.registry
        )
        
        # Convergence status
        self.is_converged = Gauge(
            'is_converged',
            'Whether model has converged (1=yes, 0=no)',
            registry=self.registry
        )
        
        # Rounds until convergence
        self.rounds_until_convergence = Gauge(
            'rounds_until_convergence',
            'Estimated rounds until convergence',
            registry=self.registry
        )
        
        # Model weight change
        self.model_weight_change = Gauge(
            'model_weight_change',
            'L2 norm of model weight changes between rounds',
            registry=self.registry
        )
    
    def start_http_server(self) -> None:
        """Start Prometheus HTTP server for metrics endpoint."""
        if self._server_started:
            logger.warning("Prometheus HTTP server already started")
            return
        
        try:
            def _start_server():
                start_http_server(self.port, registry=self.registry)
                logger.info(f"Prometheus HTTP server started on port {self.port}")
            
            self._server_thread = threading.Thread(target=_start_server, daemon=True)
            self._server_thread.start()
            self._server_started = True
            
            # Give server time to start
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Failed to start Prometheus HTTP server: {e}")
            raise
    
    def record_fl_round_start(self, round_num: int, num_clients: int) -> None:
        """
        Record the start of a FL round.
        
        Args:
            round_num: FL round number
            num_clients: Number of participating clients
        """
        self.fl_clients_participating.set(num_clients)
        logger.debug(f"FL round {round_num} started with {num_clients} clients")
    
    def record_fl_round_complete(
        self,
        round_num: int,
        duration_seconds: float,
        status: str = "success"
    ) -> None:
        """
        Record completion of a FL round.
        
        Args:
            round_num: FL round number
            duration_seconds: Duration of the round
            status: Status (success, failed, skipped)
        """
        self.fl_rounds_total.labels(status=status).inc()
        
        if status == "success":
            self.fl_rounds_completed.set(round_num)
        
        self.fl_round_duration_seconds.labels(round_type='full_round').observe(duration_seconds)
        
        logger.debug(f"FL round {round_num} completed: {status} ({duration_seconds:.2f}s)")
    
    def record_training_duration(self, duration_seconds: float, client_id: str = "global") -> None:
        """
        Record training duration.
        
        Args:
            duration_seconds: Training duration
            client_id: Client identifier
        """
        self.fl_round_duration_seconds.labels(round_type='training').observe(duration_seconds)
    
    def record_aggregation_duration(self, duration_seconds: float) -> None:
        """
        Record aggregation duration.
        
        Args:
            duration_seconds: Aggregation duration
        """
        self.fl_round_duration_seconds.labels(round_type='aggregation').observe(duration_seconds)
    
    def record_client_failure(self, client_id: str, failure_type: str) -> None:
        """
        Record a client failure.
        
        Args:
            client_id: Client identifier
            failure_type: Type of failure (disconnect, timeout, error)
        """
        self.fl_client_failures.labels(client_id=client_id, failure_type=failure_type).inc()
        logger.warning(f"Client failure recorded: {client_id} - {failure_type}")
        
        # Trigger alert if enabled
        if self.enable_alerts:
            self._check_alert(
                f"client_failure_{client_id}",
                1.0,
                f"Client {client_id} failed: {failure_type}"
            )
    
    def record_performance_metrics(
        self,
        auprc: float,
        auroc: float,
        loss: float,
        model_type: str = "global",
        client_id: str = "global"
    ) -> None:
        """
        Record model performance metrics.
        
        Args:
            auprc: AUPRC score
            auroc: AUROC score
            loss: Loss value
            model_type: Type of model (global, local)
            client_id: Client identifier
        """
        self.model_auprc.labels(model_type=model_type, client_id=client_id).set(auprc)
        self.model_auroc.labels(model_type=model_type, client_id=client_id).set(auroc)
        self.model_loss.labels(loss_type='test', client_id=client_id).set(loss)
        
        logger.debug(f"Performance metrics recorded: AUPRC={auprc:.4f}, AUROC={auroc:.4f}")
        
        # Check for performance degradation alerts
        if self.enable_alerts and auprc < 0.5:
            self._check_alert(
                "low_auprc",
                auprc,
                f"Low AUPRC detected: {auprc:.4f} for {client_id}"
            )
    
    def record_privacy_budget(
        self,
        epsilon_spent: float,
        epsilon_total: float,
        delta: float,
        client_id: str = "global"
    ) -> None:
        """
        Record privacy budget metrics.
        
        Args:
            epsilon_spent: Epsilon spent so far
            epsilon_total: Total epsilon budget
            delta: Delta parameter
            client_id: Client identifier
        """
        epsilon_remaining = max(0, epsilon_total - epsilon_spent)
        
        self.privacy_epsilon_spent.labels(client_id=client_id).set(epsilon_spent)
        self.privacy_epsilon_remaining.labels(client_id=client_id).set(epsilon_remaining)
        self.privacy_delta.labels(client_id=client_id).set(delta)
        
        # Check if budget exhausted
        is_exhausted = epsilon_remaining <= 0
        self.privacy_budget_exhausted.labels(client_id=client_id).set(1 if is_exhausted else 0)
        
        if is_exhausted and self.enable_alerts:
            self._check_alert(
                f"privacy_budget_exhausted_{client_id}",
                1.0,
                f"Privacy budget exhausted for {client_id}"
            )
        
        logger.debug(f"Privacy budget: ε_spent={epsilon_spent:.4f}, ε_remaining={epsilon_remaining:.4f}")
    
    def record_system_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Record system health metrics.
        
        Args:
            metrics: Dictionary of system metrics
        """
        # Memory metrics
        if 'memory_usage_bytes' in metrics:
            self.system_memory_usage_bytes.labels(component='total').set(
                metrics['memory_usage_bytes']
            )
        
        if 'memory_available_bytes' in metrics:
            self.system_memory_available_bytes.set(metrics['memory_available_bytes'])
        
        # CPU metrics
        if 'cpu_usage_percent' in metrics:
            self.system_cpu_usage_percent.labels(component='total').set(
                metrics['cpu_usage_percent']
            )
        
        # Disk metrics
        if 'disk_usage_bytes' in metrics:
            self.system_disk_usage_bytes.labels(path='/').set(metrics['disk_usage_bytes'])
        
        # Health status
        if 'health_status' in metrics:
            self.system_health_status.set(1 if metrics['health_status'] == 'healthy' else 0)
        
        logger.debug(f"System metrics recorded: {len(metrics)} metrics")
    
    def record_convergence_metrics(
        self,
        convergence_score: float,
        is_converged: bool,
        weight_change: Optional[float] = None
    ) -> None:
        """
        Record convergence tracking metrics.
        
        Args:
            convergence_score: Convergence score
            is_converged: Whether model has converged
            weight_change: L2 norm of weight changes
        """
        self.convergence_score.set(convergence_score)
        self.is_converged.set(1 if is_converged else 0)
        
        if weight_change is not None:
            self.model_weight_change.set(weight_change)
        
        logger.debug(f"Convergence: score={convergence_score:.4f}, converged={is_converged}")
    
    def record_training_samples(self, num_samples: int, client_id: str) -> None:
        """
        Record number of training samples.
        
        Args:
            num_samples: Number of training samples
            client_id: Client identifier
        """
        self.fl_training_samples.labels(client_id=client_id).set(num_samples)
    
    def record_prediction(self, client_id: str, is_fraud: bool) -> None:
        """
        Record a prediction.
        
        Args:
            client_id: Client identifier
            is_fraud: Whether prediction was fraud
        """
        prediction_class = "fraud" if is_fraud else "legitimate"
        self.predictions_total.labels(
            client_id=client_id,
            prediction_class=prediction_class
        ).inc()
    
    def record_network_traffic(
        self,
        bytes_sent: int,
        bytes_received: int,
        component: str = "client"
    ) -> None:
        """
        Record network traffic.
        
        Args:
            bytes_sent: Bytes sent
            bytes_received: Bytes received
            component: Component identifier
        """
        self.network_bytes_sent.labels(component=component).inc(bytes_sent)
        self.network_bytes_received.labels(component=component).inc(bytes_received)
    
    def add_alert_config(self, alert_config: AlertConfig) -> None:
        """
        Add an alert configuration.
        
        Args:
            alert_config: Alert configuration
        """
        self.alert_configs.append(alert_config)
        logger.info(f"Added alert config: {alert_config.metric_name}")
    
    def _check_alert(self, metric_name: str, value: float, message: str) -> None:
        """
        Check if alert should be triggered.
        
        Args:
            metric_name: Metric name
            value: Metric value
            message: Alert message
        """
        for config in self.alert_configs:
            if config.metric_name == metric_name:
                should_alert = False
                
                if config.comparison == 'gt' and value > config.threshold:
                    should_alert = True
                elif config.comparison == 'lt' and value < config.threshold:
                    should_alert = True
                elif config.comparison == 'eq' and value == config.threshold:
                    should_alert = True
                
                if should_alert:
                    logger.warning(f"[{config.severity.upper()}] ALERT: {message}")
    
    def push_to_gateway(self) -> None:
        """Push metrics to Pushgateway (for batch jobs)."""
        if not self.pushgateway_url:
            logger.warning("Pushgateway URL not configured")
            return
        
        try:
            push_to_gateway(
                self.pushgateway_url,
                job=self.job_name,
                registry=self.registry
            )
            logger.debug(f"Pushed metrics to Pushgateway: {self.pushgateway_url}")
        except Exception as e:
            logger.error(f"Failed to push metrics to Pushgateway: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of current metrics.
        
        Returns:
            Dictionary with metric summaries
        """
        return {
            'fl_rounds_completed': self.fl_rounds_completed._value.get(),
            'clients_participating': self.fl_clients_participating._value.get(),
            'convergence_score': self.convergence_score._value.get(),
            'is_converged': bool(self.is_converged._value.get()),
            'system_health': bool(self.system_health_status._value.get())
        }
    
    def shutdown(self) -> None:
        """Shutdown the exporter and cleanup resources."""
        logger.info("Shutting down Prometheus_Exporter")
        
        # Push final metrics if using Pushgateway
        if self.pushgateway_url:
            self.push_to_gateway()
        
        self._server_started = False
