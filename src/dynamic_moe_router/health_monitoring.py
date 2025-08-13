"""Comprehensive health monitoring and observability for MoE routing."""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HealthMetric:
    """Individual health metric data structure."""
    name: str
    value: float
    unit: str
    timestamp: float
    status: str  # "healthy", "warning", "critical"
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    tags: Optional[Dict[str, str]] = None


@dataclass
class PerformanceSnapshot:
    """Performance snapshot for trend analysis."""
    timestamp: float
    throughput_tokens_per_sec: float
    avg_experts_per_token: float
    flop_reduction: float
    memory_usage_mb: float
    error_rate: float
    response_time_ms: float


class HealthMonitor:
    """Comprehensive health monitoring for MoE routing systems."""
    
    def __init__(
        self,
        max_history_size: int = 1000,
        metric_retention_hours: int = 24,
        alert_cooldown_seconds: int = 300,  # 5 minutes
        enable_trend_analysis: bool = True
    ):
        self.max_history_size = max_history_size
        self.metric_retention_seconds = metric_retention_hours * 3600
        self.alert_cooldown_seconds = alert_cooldown_seconds
        self.enable_trend_analysis = enable_trend_analysis
        
        # Metric storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=max_history_size))
        self.performance_snapshots = deque(maxlen=max_history_size)
        
        # Alert management
        self.last_alert_times = {}
        self.alert_callbacks = []
        
        # Performance counters
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.start_time = time.time()
        
        # Health thresholds
        self.thresholds = {
            "throughput_tokens_per_sec": {"warning": 100, "critical": 50},
            "error_rate": {"warning": 0.05, "critical": 0.1},
            "memory_usage_mb": {"warning": 800, "critical": 1000},
            "response_time_ms": {"warning": 1000, "critical": 2000},
            "flop_reduction": {"warning": 0.1, "critical": 0.0}
        }
        
        logger.info("Health monitor initialized with comprehensive tracking")
    
    def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a health metric."""
        
        timestamp = time.time()
        
        # Determine status based on thresholds
        status = "healthy"
        threshold_warning = None
        threshold_critical = None
        
        if name in self.thresholds:
            thresholds = self.thresholds[name]
            threshold_warning = thresholds.get("warning")
            threshold_critical = thresholds.get("critical")
            
            if threshold_critical is not None:
                if (name in ["error_rate", "response_time_ms", "memory_usage_mb"] and value >= threshold_critical) or \
                   (name in ["throughput_tokens_per_sec", "flop_reduction"] and value <= threshold_critical):
                    status = "critical"
                elif threshold_warning is not None:
                    if (name in ["error_rate", "response_time_ms", "memory_usage_mb"] and value >= threshold_warning) or \
                       (name in ["throughput_tokens_per_sec", "flop_reduction"] and value <= threshold_warning):
                        status = "warning"
        
        metric = HealthMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=timestamp,
            status=status,
            threshold_warning=threshold_warning,
            threshold_critical=threshold_critical,
            tags=tags or {}
        )
        
        # Store metric
        self.metrics_history[name].append(metric)
        
        # Trigger alerts if needed
        if status in ["warning", "critical"]:
            self._check_and_trigger_alert(metric)
        
        # Clean old metrics
        self._cleanup_old_metrics()
    
    def record_performance_snapshot(
        self,
        throughput: float,
        avg_experts: float,
        flop_reduction: float,
        memory_usage: float,
        error_rate: float,
        response_time: float
    ) -> None:
        """Record a complete performance snapshot."""
        
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            throughput_tokens_per_sec=throughput,
            avg_experts_per_token=avg_experts,
            flop_reduction=flop_reduction,
            memory_usage_mb=memory_usage,
            error_rate=error_rate,
            response_time_ms=response_time
        )
        
        self.performance_snapshots.append(snapshot)
        
        # Record individual metrics
        self.record_metric("throughput_tokens_per_sec", throughput, "tokens/sec")
        self.record_metric("avg_experts_per_token", avg_experts, "experts")
        self.record_metric("flop_reduction", flop_reduction, "ratio")
        self.record_metric("memory_usage_mb", memory_usage, "MB")
        self.record_metric("error_rate", error_rate, "ratio")
        self.record_metric("response_time_ms", response_time, "ms")
    
    def record_request(self, response_time_ms: float, success: bool = True) -> None:
        """Record a routing request for performance tracking."""
        
        self.request_count += 1
        self.total_response_time += response_time_ms
        
        if not success:
            self.error_count += 1
        
        # Calculate current error rate
        error_rate = self.error_count / max(self.request_count, 1)
        
        # Record metrics
        self.record_metric("response_time_ms", response_time_ms, "ms")
        self.record_metric("error_rate", error_rate, "ratio")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        
        current_time = time.time()
        uptime_seconds = current_time - self.start_time
        
        # Calculate aggregate metrics
        avg_response_time = (
            self.total_response_time / max(self.request_count, 1)
        )
        error_rate = self.error_count / max(self.request_count, 1)
        
        # Get latest metrics
        latest_metrics = {}
        overall_status = "healthy"
        
        for metric_name, metric_history in self.metrics_history.items():
            if metric_history:
                latest_metric = metric_history[-1]
                latest_metrics[metric_name] = {
                    "value": latest_metric.value,
                    "unit": latest_metric.unit,
                    "status": latest_metric.status,
                    "timestamp": latest_metric.timestamp
                }
                
                # Update overall status
                if latest_metric.status == "critical":
                    overall_status = "critical"
                elif latest_metric.status == "warning" and overall_status != "critical":
                    overall_status = "warning"
        
        return {
            "overall_status": overall_status,
            "uptime_seconds": uptime_seconds,
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": error_rate,
            "avg_response_time_ms": avg_response_time,
            "latest_metrics": latest_metrics,
            "alert_count": len(self.last_alert_times),
            "timestamp": current_time
        }
    
    def get_performance_trends(self, hours: int = 1) -> Dict[str, Any]:
        """Analyze performance trends over specified time period."""
        
        if not self.enable_trend_analysis:
            return {"trend_analysis_disabled": True}
        
        cutoff_time = time.time() - (hours * 3600)
        
        # Filter recent snapshots
        recent_snapshots = [
            s for s in self.performance_snapshots
            if s.timestamp >= cutoff_time
        ]
        
        if len(recent_snapshots) < 2:
            return {"insufficient_data": True, "snapshots_count": len(recent_snapshots)}
        
        # Calculate trends
        trends = {}
        
        for field in ["throughput_tokens_per_sec", "avg_experts_per_token", 
                     "flop_reduction", "memory_usage_mb", "error_rate", "response_time_ms"]:
            
            values = [getattr(s, field) for s in recent_snapshots]
            timestamps = [s.timestamp for s in recent_snapshots]
            
            if len(values) >= 2:
                # Simple linear trend (positive = increasing, negative = decreasing)
                trend_slope = np.polyfit(timestamps, values, 1)[0]
                
                # Trend direction
                if abs(trend_slope) < 0.001:
                    direction = "stable"
                elif trend_slope > 0:
                    direction = "increasing"
                else:
                    direction = "decreasing"
                
                trends[field] = {
                    "slope": float(trend_slope),
                    "direction": direction,
                    "current_value": values[-1],
                    "min_value": min(values),
                    "max_value": max(values),
                    "avg_value": np.mean(values)
                }
        
        return {
            "time_period_hours": hours,
            "snapshots_analyzed": len(recent_snapshots),
            "trends": trends
        }
    
    def add_alert_callback(self, callback: Callable[[HealthMetric], None]) -> None:
        """Add callback function for alerts."""
        self.alert_callbacks.append(callback)
    
    def _check_and_trigger_alert(self, metric: HealthMetric) -> None:
        """Check if alert should be triggered and call callbacks."""
        
        # Check cooldown
        alert_key = f"{metric.name}_{metric.status}"
        last_alert = self.last_alert_times.get(alert_key, 0)
        
        if time.time() - last_alert < self.alert_cooldown_seconds:
            return  # Still in cooldown
        
        # Update last alert time
        self.last_alert_times[alert_key] = time.time()
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(metric)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        # Log alert
        logger.warning(
            f"Health alert: {metric.name} = {metric.value} {metric.unit} "
            f"(status: {metric.status})"
        )
    
    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        
        cutoff_time = time.time() - self.metric_retention_seconds
        
        for metric_name, metric_history in self.metrics_history.items():
            # Remove old metrics
            while metric_history and metric_history[0].timestamp < cutoff_time:
                metric_history.popleft()
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        
        if format == "json":
            return self._export_json()
        elif format == "prometheus":
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self) -> str:
        """Export metrics as JSON."""
        
        export_data = {
            "health_status": self.get_health_status(),
            "performance_trends": self.get_performance_trends(),
            "thresholds": self.thresholds,
            "metrics_history": {}
        }
        
        # Include recent metrics history
        for metric_name, metric_history in self.metrics_history.items():
            export_data["metrics_history"][metric_name] = [
                asdict(metric) for metric in list(metric_history)[-100:]  # Last 100 points
            ]
        
        return json.dumps(export_data, indent=2)
    
    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        
        lines = []
        
        for metric_name, metric_history in self.metrics_history.items():
            if not metric_history:
                continue
                
            latest_metric = metric_history[-1]
            
            # Sanitize metric name for Prometheus
            prom_name = f"moe_router_{metric_name.replace('-', '_')}"
            
            # Add help comment
            lines.append(f"# HELP {prom_name} {metric_name} in {latest_metric.unit}")
            lines.append(f"# TYPE {prom_name} gauge")
            
            # Add metric with labels
            tags_str = ""
            if latest_metric.tags:
                tag_pairs = [f'{k}="{v}"' for k, v in latest_metric.tags.items()]
                tags_str = "{" + ",".join(tag_pairs) + "}"
            
            lines.append(f"{prom_name}{tags_str} {latest_metric.value}")
        
        return "\n".join(lines)
    
    def reset_monitoring(self) -> None:
        """Reset all monitoring data."""
        
        self.metrics_history.clear()
        self.performance_snapshots.clear()
        self.last_alert_times.clear()
        
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.start_time = time.time()
        
        logger.info("Health monitoring data reset")


# Default global health monitor instance
default_health_monitor = HealthMonitor()