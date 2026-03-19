"""
Adaptive Resource Management System

This module implements adaptive resource constraint handling including:
- Memory pressure detection and batch size adaptation
- Corrupted data quarantine and processing continuation
- Disk space monitoring with automatic cleanup
"""

import psutil
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
import torch
import gc


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""

    memory_used_gb: float
    memory_available_gb: float
    memory_percent: float
    cpu_percent: float
    disk_used_gb: float
    disk_available_gb: float
    disk_percent: float


@dataclass
class AdaptiveConfig:
    """Adaptive resource management configuration."""

    # Memory thresholds
    memory_warning_threshold: float = 0.75  # 75%
    memory_critical_threshold: float = 0.90  # 90%

    # Disk thresholds
    disk_warning_threshold: float = 0.80  # 80%
    disk_critical_threshold: float = 0.95  # 95%

    # Batch size adaptation
    min_batch_size: int = 32
    max_batch_size: int = 2048
    batch_size_reduction_factor: float = 0.5
    batch_size_increase_factor: float = 1.5

    # Cleanup settings
    checkpoint_retention_days: int = 7
    log_retention_days: int = 30
    auto_cleanup_enabled: bool = True

    # Quarantine settings
    quarantine_dir: str = "data/quarantine"
    max_quarantine_size_gb: float = 1.0


class Resource_Manager:
    """
    Adaptive resource management system.

    Features:
    - Memory pressure detection
    - Automatic batch size adaptation
    - Corrupted data quarantine
    - Disk space monitoring
    - Automatic cleanup of old files

    Attributes:
        config: Adaptive configuration
        current_batch_size: Current batch size
        quarantine_count: Number of quarantined records
    """

    def __init__(self, config: Optional[AdaptiveConfig] = None):
        """
        Initialize Resource_Manager.

        Args:
            config: Adaptive configuration (uses defaults if None)
        """
        self.config = config or AdaptiveConfig()
        self.current_batch_size: Optional[int] = None
        self.quarantine_count: int = 0
        self._last_cleanup: Optional[datetime] = None

        # Create quarantine directory
        Path(self.config.quarantine_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Resource_Manager initialized")

    def get_resource_metrics(self) -> ResourceMetrics:
        """
        Get current resource usage metrics.

        Returns:
            Resource metrics
        """
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_used_gb = (memory.total - memory.available) / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        memory_percent = memory.percent / 100.0

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1) / 100.0

        # Disk metrics
        disk = psutil.disk_usage("/")
        disk_used_gb = disk.used / (1024**3)
        disk_available_gb = disk.free / (1024**3)
        disk_percent = disk.percent / 100.0

        return ResourceMetrics(
            memory_used_gb=memory_used_gb,
            memory_available_gb=memory_available_gb,
            memory_percent=memory_percent,
            cpu_percent=cpu_percent,
            disk_used_gb=disk_used_gb,
            disk_available_gb=disk_available_gb,
            disk_percent=disk_percent,
        )

    def detect_memory_pressure(self) -> Tuple[bool, str]:
        """
        Detect memory pressure.

        Returns:
            Tuple of (is_under_pressure, severity_level)
            Severity levels: 'normal', 'warning', 'critical'
        """
        metrics = self.get_resource_metrics()

        if metrics.memory_percent >= self.config.memory_critical_threshold:
            logger.warning(f"Critical memory pressure: {metrics.memory_percent:.1%}")
            return True, "critical"
        elif metrics.memory_percent >= self.config.memory_warning_threshold:
            logger.warning(f"Memory pressure warning: {metrics.memory_percent:.1%}")
            return True, "warning"
        else:
            return False, "normal"

    def adapt_batch_size(self, current_batch_size: int, force_reduction: bool = False) -> int:
        """
        Adapt batch size based on memory pressure.

        Args:
            current_batch_size: Current batch size
            force_reduction: Force batch size reduction

        Returns:
            Adapted batch size
        """
        self.current_batch_size = current_batch_size

        # Check memory pressure
        under_pressure, severity = self.detect_memory_pressure()

        if force_reduction or (under_pressure and severity == "critical"):
            # Reduce batch size
            new_batch_size = max(
                self.config.min_batch_size, int(current_batch_size * self.config.batch_size_reduction_factor)
            )

            if new_batch_size != current_batch_size:
                logger.info(f"Reducing batch size: {current_batch_size} → {new_batch_size}")

                # Force garbage collection
                self._force_garbage_collection()

                self.current_batch_size = new_batch_size
                return new_batch_size

        elif not under_pressure and current_batch_size < self.config.max_batch_size:
            # Try to increase batch size if memory is available
            new_batch_size = min(
                self.config.max_batch_size, int(current_batch_size * self.config.batch_size_increase_factor)
            )

            if new_batch_size != current_batch_size:
                logger.info(f"Increasing batch size: {current_batch_size} → {new_batch_size}")
                self.current_batch_size = new_batch_size
                return new_batch_size

        return current_batch_size

    def _force_garbage_collection(self) -> None:
        """Force garbage collection to free memory."""
        gc.collect()

        # Clear PyTorch cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug("Forced garbage collection")

    def quarantine_corrupted_data(self, data: Any, reason: str, identifier: Optional[str] = None) -> str:
        """
        Quarantine corrupted data for later inspection.

        Args:
            data: Corrupted data to quarantine
            reason: Reason for quarantine
            identifier: Optional identifier for the data

        Returns:
            Path to quarantined file
        """
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        identifier = identifier or f"record_{self.quarantine_count}"
        filename = f"{timestamp}_{identifier}.txt"
        filepath = Path(self.config.quarantine_dir) / filename

        # Write quarantine file
        try:
            with open(filepath, "w") as f:
                f.write(f"Quarantine Reason: {reason}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Identifier: {identifier}\n")
                f.write(f"\nData:\n{data}\n")

            self.quarantine_count += 1
            logger.warning(f"Data quarantined: {filepath} (Reason: {reason})")

            # Check quarantine size
            self._check_quarantine_size()

            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to quarantine data: {e}")
            return ""

    def _check_quarantine_size(self) -> None:
        """Check and manage quarantine directory size."""
        quarantine_path = Path(self.config.quarantine_dir)

        if not quarantine_path.exists():
            return

        # Calculate total size
        total_size = sum(f.stat().st_size for f in quarantine_path.rglob("*") if f.is_file())
        total_size_gb = total_size / (1024**3)

        if total_size_gb > self.config.max_quarantine_size_gb:
            logger.warning(
                f"Quarantine directory exceeds limit: {total_size_gb:.2f}GB > "
                f"{self.config.max_quarantine_size_gb}GB"
            )

            # Remove oldest files
            files = sorted(quarantine_path.rglob("*"), key=lambda f: f.stat().st_mtime if f.is_file() else 0)

            removed_count = 0
            for file in files:
                if not file.is_file():
                    continue

                file.unlink()
                removed_count += 1

                # Recalculate size
                total_size = sum(f.stat().st_size for f in quarantine_path.rglob("*") if f.is_file())
                total_size_gb = total_size / (1024**3)

                if total_size_gb <= self.config.max_quarantine_size_gb:
                    break

            logger.info(f"Removed {removed_count} old quarantine files")

    def monitor_disk_space(self, path: str = "/") -> Tuple[bool, str]:
        """
        Monitor disk space and return status.

        Args:
            path: Path to monitor (default: root)

        Returns:
            Tuple of (needs_cleanup, severity_level)
            Severity levels: 'normal', 'warning', 'critical'
        """
        disk = psutil.disk_usage(path)
        disk_percent = disk.percent / 100.0

        if disk_percent >= self.config.disk_critical_threshold:
            logger.error(f"Critical disk space: {disk_percent:.1%} used")
            return True, "critical"
        elif disk_percent >= self.config.disk_warning_threshold:
            logger.warning(f"Low disk space: {disk_percent:.1%} used")
            return True, "warning"
        else:
            return False, "normal"

    def cleanup_old_files(self, directories: Optional[List[str]] = None, force: bool = False) -> Dict[str, int]:
        """
        Clean up old files based on retention policies.

        Args:
            directories: List of directories to clean (uses defaults if None)
            force: Force cleanup even if not needed

        Returns:
            Dictionary with cleanup statistics
        """
        if not self.config.auto_cleanup_enabled and not force:
            logger.info("Auto-cleanup disabled")
            return {}

        # Check if cleanup is needed
        needs_cleanup, severity = self.monitor_disk_space()

        if not needs_cleanup and not force:
            logger.debug("Disk space sufficient, skipping cleanup")
            return {}

        # Default directories
        if directories is None:
            directories = ["models", "logs", "results"]

        stats = {}
        total_freed_bytes = 0

        for directory in directories:
            dir_path = Path(directory)

            if not dir_path.exists():
                continue

            # Determine retention days based on directory
            if "checkpoint" in directory or "model" in directory:
                retention_days = self.config.checkpoint_retention_days
            else:
                retention_days = self.config.log_retention_days

            cutoff_date = datetime.now() - timedelta(days=retention_days)

            # Find and remove old files
            removed_count = 0
            freed_bytes = 0

            for file in dir_path.rglob("*"):
                if not file.is_file():
                    continue

                # Check file age
                file_mtime = datetime.fromtimestamp(file.stat().st_mtime)

                if file_mtime < cutoff_date:
                    file_size = file.stat().st_size

                    try:
                        file.unlink()
                        removed_count += 1
                        freed_bytes += file_size
                        logger.debug(f"Removed old file: {file}")
                    except Exception as e:
                        logger.error(f"Failed to remove {file}: {e}")

            if removed_count > 0:
                freed_mb = freed_bytes / (1024**2)
                logger.info(f"Cleaned {directory}: removed {removed_count} files, " f"freed {freed_mb:.2f}MB")

            stats[directory] = removed_count
            total_freed_bytes += freed_bytes

        total_freed_mb = total_freed_bytes / (1024**2)
        logger.info(f"Total cleanup: freed {total_freed_mb:.2f}MB")

        self._last_cleanup = datetime.now()

        return stats

    def handle_out_of_memory(self) -> bool:
        """
        Handle out-of-memory situation.

        Returns:
            True if recovery successful, False otherwise
        """
        logger.error("Out of memory detected, attempting recovery...")

        # Force garbage collection
        self._force_garbage_collection()

        # Reduce batch size if set
        if self.current_batch_size:
            self.current_batch_size = self.adapt_batch_size(self.current_batch_size, force_reduction=True)

        # Clean up old files
        self.cleanup_old_files(force=True)

        # Check if memory is now available
        metrics = self.get_resource_metrics()

        if metrics.memory_percent < self.config.memory_critical_threshold:
            logger.info("Memory recovery successful")
            return True
        else:
            logger.error("Memory recovery failed")
            return False

    def get_recommended_batch_size(self, dataset_size: int, model_size_mb: float) -> int:
        """
        Get recommended batch size based on available resources.

        Args:
            dataset_size: Size of dataset
            model_size_mb: Approximate model size in MB

        Returns:
            Recommended batch size
        """
        metrics = self.get_resource_metrics()

        # Estimate memory per sample (rough heuristic)
        available_memory_mb = metrics.memory_available_gb * 1024

        # Reserve 20% for overhead
        usable_memory_mb = available_memory_mb * 0.8

        # Estimate batch size
        # Assume each sample takes ~1MB (very rough estimate)
        estimated_batch_size = int(usable_memory_mb / (model_size_mb + 1))

        # Clamp to configured limits
        recommended_batch_size = max(self.config.min_batch_size, min(self.config.max_batch_size, estimated_batch_size))

        logger.info(
            f"Recommended batch size: {recommended_batch_size} " f"(available memory: {available_memory_mb:.0f}MB)"
        )

        return recommended_batch_size

    def get_status(self) -> Dict[str, Any]:
        """
        Get resource manager status.

        Returns:
            Status dictionary
        """
        metrics = self.get_resource_metrics()
        under_pressure, mem_severity = self.detect_memory_pressure()
        needs_cleanup, disk_severity = self.monitor_disk_space()

        return {
            "memory": {
                "used_gb": metrics.memory_used_gb,
                "available_gb": metrics.memory_available_gb,
                "percent": metrics.memory_percent,
                "under_pressure": under_pressure,
                "severity": mem_severity,
            },
            "disk": {
                "used_gb": metrics.disk_used_gb,
                "available_gb": metrics.disk_available_gb,
                "percent": metrics.disk_percent,
                "needs_cleanup": needs_cleanup,
                "severity": disk_severity,
            },
            "cpu": {"percent": metrics.cpu_percent},
            "batch_size": {
                "current": self.current_batch_size,
                "min": self.config.min_batch_size,
                "max": self.config.max_batch_size,
            },
            "quarantine": {"count": self.quarantine_count, "directory": self.config.quarantine_dir},
            "last_cleanup": self._last_cleanup.isoformat() if self._last_cleanup else None,
        }
