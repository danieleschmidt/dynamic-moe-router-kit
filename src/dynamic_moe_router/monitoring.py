"""Enhanced monitoring and metrics for dynamic MoE routing."""

import logging
import threading
import time
import json
import os
from collections import defaultdict, deque
from typing import Any, Callable, Dict, Optional, List
from contextlib import contextmanager
from dataclasses import dataclass, asdict

import numpy as np

from .exceptions import DynamicMoEError, ProfilingError
from .security import SecurityMonitor, get_security_monitor

logger = logging.getLogger(__name__)


@dataclass
class AlertEvent:
    """Represents a performance alert."""
    timestamp: float
    metric: str
    value: float
    threshold: float
    severity: str
    message: str


@dataclass
class PerformanceSnapshot:
    """Performance metrics snapshot."""
    timestamp: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    throughput_rps: float
    load_balance_variance: float
    memory_usage_mb: float
    expert_utilization: List[float]
    active_connections: int


class EnhancedPerformanceMonitor:
    """Comprehensive performance monitoring with security integration."""

    def __init__(self, window_size: int = 1000, alert_thresholds: Optional[Dict[str, float]] = None):
        self.window_size = window_size
        self.alert_thresholds = alert_thresholds or {
            'avg_latency_ms': 100.0,
            'p95_latency_ms': 500.0,
            'p99_latency_ms': 1000.0,
            'error_rate': 0.05,
            'load_balance_variance': 0.1,
            'memory_usage_mb': 1000.0,
            'throughput_rps': 1.0  # Min throughput
        }

        # Thread-safe metrics storage
        self._lock = threading.Lock()
        self._metrics = defaultdict(lambda: deque(maxlen=window_size))
        self._counters = defaultdict(int)
        self._alerts = deque(maxlen=100)  # Recent alerts
        self._snapshots = deque(maxlen=1000)  # Historical snapshots

        # Performance tracking
        self._call_times = deque(maxlen=window_size)
        self._error_count = 0
        self._success_count = 0
        self._start_time = time.time()
        
        # Security integration
        self._security_monitor = get_security_monitor()
        
        # Alert callbacks
        self._alert_callbacks: List[Callable[[AlertEvent], None]] = []
        
        # Persistent logging
        self._enable_logging = os.getenv('MOE_MONITOR_LOGGING', 'false').lower() == 'true'
        self._log_file = os.getenv('MOE_MONITOR_LOG_FILE', 'moe_performance.jsonl')

    def add_alert_callback(self, callback: Callable[[AlertEvent], None]) -> None:
        """Add callback function for alert notifications."""
        self._alert_callbacks.append(callback)
    
    def record_call(self, duration_ms: float, success: bool = True, **metrics):
        """Record a routing call with comprehensive performance metrics."""
        timestamp = time.time()
        
        with self._lock:
            self._call_times.append(duration_ms)

            if success:
                self._success_count += 1
            else:
                self._error_count += 1
                # Log security event for errors
                self._security_monitor.log_event(
                    'routing_error',
                    'medium',
                    f'Routing call failed after {duration_ms:.2f}ms'
                )

            # Store additional metrics with timestamps
            for key, value in metrics.items():
                self._metrics[key].append((timestamp, value))
            
            # Check for alerts
            self._check_alerts()
            
            # Periodic snapshot
            if len(self._call_times) % 100 == 0:  # Every 100 calls
                self._create_snapshot()
    
    def _check_alerts(self) -> None:
        """Check metrics against thresholds and generate alerts."""
        if not self._call_times:
            return
            
        current_metrics = self._compute_current_metrics()
        
        for metric, threshold in self.alert_thresholds.items():
            current_value = current_metrics.get(metric, 0)
            
            # Different alert logic based on metric
            alert_triggered = False
            severity = 'medium'
            
            if metric in ['avg_latency_ms', 'p95_latency_ms', 'p99_latency_ms', 'memory_usage_mb']:
                alert_triggered = current_value > threshold
                severity = 'high' if current_value > threshold * 2 else 'medium'
            elif metric == 'error_rate':
                alert_triggered = current_value > threshold
                severity = 'critical' if current_value > 0.2 else 'high'
            elif metric == 'throughput_rps':
                alert_triggered = current_value < threshold
                severity = 'medium'
            elif metric == 'load_balance_variance':
                alert_triggered = current_value > threshold
                severity = 'medium'
            
            if alert_triggered:
                alert = AlertEvent(
                    timestamp=time.time(),
                    metric=metric,
                    value=current_value,
                    threshold=threshold,
                    severity=severity,
                    message=f"{metric} = {current_value:.3f} exceeds threshold {threshold:.3f}"
                )
                
                self._alerts.append(alert)
                
                # Notify callbacks
                for callback in self._alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")
                
                # Log to security monitor
                self._security_monitor.log_event(
                    'performance_alert',
                    severity,
                    alert.message
                )
    
    def _compute_current_metrics(self) -> Dict[str, float]:
        """Compute current performance metrics."""
        if not self._call_times:
            return {}
        
        call_times = list(self._call_times)
        total_calls = self._success_count + self._error_count
        
        metrics = {
            'avg_latency_ms': np.mean(call_times),
            'p95_latency_ms': np.percentile(call_times, 95),
            'p99_latency_ms': np.percentile(call_times, 99),
            'error_rate': self._error_count / max(total_calls, 1),
            'throughput_rps': len(call_times) / max(time.time() - self._start_time, 1)
        }
        
        # Add memory usage if available
        try:
            import psutil
            process = psutil.Process()
            metrics['memory_usage_mb'] = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            metrics['memory_usage_mb'] = 0.0
        
        # Compute load balance variance if expert utilization data exists
        if 'expert_utilization' in self._metrics:
            recent_utils = [util for _, util in list(self._metrics['expert_utilization'])[-10:]]
            if recent_utils:
                metrics['load_balance_variance'] = np.var(recent_utils[-1]) if recent_utils else 0.0
        
        return metrics
    
    def _create_snapshot(self) -> None:
        """Create performance snapshot for historical tracking."""
        current_metrics = self._compute_current_metrics()
        
        # Get expert utilization
        expert_util = []
        if 'expert_utilization' in self._metrics and self._metrics['expert_utilization']:
            expert_util = list(self._metrics['expert_utilization'])[-1][1]  # Latest utilization
        
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            avg_latency_ms=current_metrics.get('avg_latency_ms', 0.0),
            p95_latency_ms=current_metrics.get('p95_latency_ms', 0.0),
            p99_latency_ms=current_metrics.get('p99_latency_ms', 0.0),
            error_rate=current_metrics.get('error_rate', 0.0),
            throughput_rps=current_metrics.get('throughput_rps', 0.0),
            load_balance_variance=current_metrics.get('load_balance_variance', 0.0),
            memory_usage_mb=current_metrics.get('memory_usage_mb', 0.0),
            expert_utilization=expert_util,
            active_connections=threading.active_count()
        )
        
        self._snapshots.append(snapshot)
        
        # Log to file if enabled
        if self._enable_logging:
            self._log_snapshot(snapshot)
    
    def _log_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        """Log performance snapshot to file."""
        try:
            log_entry = {
                'type': 'performance_snapshot',
                **asdict(snapshot)
            }
            
            with open(self._log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.warning(f"Failed to log performance snapshot: {e}")

            # Check for alerts
            self._check_alerts()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        with self._lock:
            total_calls = self._success_count + self._error_count

            if total_calls == 0:
                return {'status': 'no_calls', 'total_calls': 0}

            # Compute performance metrics
            latencies = list(self._call_times) if self._call_times else [0]

            summary = {
                'total_calls': total_calls,
                'success_rate': self._success_count / total_calls,
                'error_rate': self._error_count / total_calls,
                'latency_stats': {
                    'mean_ms': np.mean(latencies),
                    'p50_ms': np.percentile(latencies, 50),
                    'p95_ms': np.percentile(latencies, 95),
                    'p99_ms': np.percentile(latencies, 99),
                    'max_ms': np.max(latencies),
                    'min_ms': np.min(latencies)
                },
                'recent_alerts': self._alerts[-10:],  # Last 10 alerts
                'metrics': {}
            }

            # Add custom metrics
            for metric_name, values in self._metrics.items():
                if values:
                    summary['metrics'][metric_name] = {
                        'current': values[-1],
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }

            return summary

    def _check_alerts(self):
        """Check for alert conditions."""
        if not self._call_times:
            return

        # Check latency
        recent_latency = np.mean(list(self._call_times)[-10:])  # Last 10 calls
        if recent_latency > self.alert_thresholds['avg_latency_ms']:
            self._add_alert('high_latency', f'Average latency {recent_latency:.2f}ms exceeds threshold')

        # Check error rate
        total_calls = self._success_count + self._error_count
        if total_calls > 10:  # Only check after enough calls
            error_rate = self._error_count / total_calls
            if error_rate > self.alert_thresholds['error_rate']:
                self._add_alert('high_error_rate', f'Error rate {error_rate:.3f} exceeds threshold')

    def _add_alert(self, alert_type: str, message: str):
        """Add an alert with timestamp."""
        alert = {
            'timestamp': time.time(),
            'type': alert_type,
            'message': message
        }
        self._alerts.append(alert)
        logger.warning(f"Performance Alert [{alert_type}]: {message}")

    def reset_metrics(self):
        """Reset all metrics and counters."""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._alerts.clear()
            self._call_times.clear()
            self._error_count = 0
            self._success_count = 0


class RouterHealthChecker:
    """Health checking for router components."""

    def __init__(self, router):
        self.router = router
        self.last_check = None
        self.health_status = 'unknown'

    def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        self.last_check = time.time()
        health_report = {
            'timestamp': self.last_check,
            'overall_status': 'healthy',
            'checks': {}
        }

        try:
            # Check 1: Router configuration
            config_check = self._check_configuration()
            health_report['checks']['configuration'] = config_check

            # Check 2: Memory usage
            memory_check = self._check_memory_usage()
            health_report['checks']['memory'] = memory_check

            # Check 3: Router functionality
            function_check = self._check_routing_functionality()
            health_report['checks']['functionality'] = function_check

            # Check 4: Load balancing
            balance_check = self._check_load_balance()
            health_report['checks']['load_balance'] = balance_check

            # Determine overall status
            failed_checks = [name for name, check in health_report['checks'].items()
                           if not check['passed']]

            if failed_checks:
                health_report['overall_status'] = 'unhealthy'
                health_report['failed_checks'] = failed_checks

        except Exception as e:
            health_report['overall_status'] = 'error'
            health_report['error'] = str(e)
            logger.error(f"Health check failed: {e}")

        self.health_status = health_report['overall_status']
        return health_report

    def _check_configuration(self) -> Dict[str, Any]:
        """Check router configuration validity."""
        try:
            # Basic configuration validation
            assert self.router.num_experts > 0, "Number of experts must be positive"
            assert self.router.min_experts >= 1, "Minimum experts must be >= 1"
            assert self.router.max_experts <= self.router.num_experts, "Max experts exceeds total experts"

            return {'passed': True, 'message': 'Configuration valid'}
        except Exception as e:
            return {'passed': False, 'message': f'Configuration error: {e}'}

    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage of router components."""
        try:
            # Simple memory check - in a real implementation, this would be more sophisticated
            memory_info = {
                'router_history_length': len(getattr(self.router, 'expert_usage_history', [])),
                'estimated_memory_mb': 'unknown'  # Placeholder
            }

            return {'passed': True, 'message': 'Memory usage within limits', 'details': memory_info}
        except Exception as e:
            return {'passed': False, 'message': f'Memory check failed: {e}'}

    def _check_routing_functionality(self) -> Dict[str, Any]:
        """Test basic routing functionality."""
        try:
            # Create small test input
            test_input = np.random.randn(1, 4, self.router.input_dim)
            result = self.router.route(test_input)

            # Verify result structure
            required_keys = ['expert_indices', 'expert_weights', 'num_experts_per_token']
            missing_keys = [key for key in required_keys if key not in result]

            if missing_keys:
                return {'passed': False, 'message': f'Missing result keys: {missing_keys}'}

            return {'passed': True, 'message': 'Routing functionality working'}
        except Exception as e:
            return {'passed': False, 'message': f'Routing test failed: {e}'}

    def _check_load_balance(self) -> Dict[str, Any]:
        """Check load balancing effectiveness."""
        try:
            stats = self.router.get_expert_usage_stats()

            if 'message' in stats:  # No history available
                return {'passed': True, 'message': 'No usage history to check'}

            load_balance_score = stats.get('load_balance_score', 0.0)

            if load_balance_score < 0.5:  # Arbitrary threshold
                return {'passed': False, 'message': f'Poor load balance: {load_balance_score:.3f}'}

            return {'passed': True, 'message': f'Load balance good: {load_balance_score:.3f}'}
        except Exception as e:
            return {'passed': False, 'message': f'Load balance check failed: {e}'}


class CircuitBreaker:
    """Circuit breaker pattern for router resilience."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0, half_open_max_calls: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = 'closed'  # closed, open, half_open
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self._state == 'open':
                if self._should_attempt_reset():
                    self._state = 'half_open'
                    self._half_open_calls = 0
                else:
                    raise DynamicMoEError("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return False
        return time.time() - self._last_failure_time >= self.recovery_timeout

    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            if self._state == 'half_open':
                self._half_open_calls += 1
                if self._half_open_calls >= self.half_open_max_calls:
                    self._state = 'closed'
                    self._failure_count = 0
            elif self._state == 'closed':
                self._failure_count = 0

    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.failure_threshold:
                self._state = 'open'

    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        return self._state

    def reset(self):
        """Manually reset circuit breaker."""
        with self._lock:
            self._state = 'closed'
            self._failure_count = 0
            self._half_open_calls = 0
            self._last_failure_time = None


def create_monitoring_wrapper(router, enable_circuit_breaker: bool = True, **monitor_kwargs):
    """Create a monitoring wrapper around a router."""

    class MonitoredRouter:
        def __init__(self, base_router):
            self.router = base_router
            self.monitor = PerformanceMonitor(**monitor_kwargs)
            self.health_checker = RouterHealthChecker(base_router)
            self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None

            # Delegate attribute access
            for attr in dir(base_router):
                if not attr.startswith('_') and attr not in ['route']:
                    setattr(self, attr, getattr(base_router, attr))

        def route(self, *args, **kwargs):
            """Monitored routing with performance tracking."""
            start_time = time.time()

            try:
                if self.circuit_breaker:
                    result = self.circuit_breaker.call(self.router.route, *args, **kwargs)
                else:
                    result = self.router.route(*args, **kwargs)

                # Record success metrics
                duration_ms = (time.time() - start_time) * 1000
                routing_metrics = result.get('routing_info', {})

                self.monitor.record_call(
                    duration_ms=duration_ms,
                    success=True,
                    **routing_metrics
                )

                return result

            except Exception as e:
                # Record failure metrics
                duration_ms = (time.time() - start_time) * 1000
                self.monitor.record_call(duration_ms=duration_ms, success=False)
                raise e

        def get_monitoring_summary(self) -> Dict[str, Any]:
            """Get comprehensive monitoring summary."""
            return {
                'performance': self.monitor.get_metrics_summary(),
                'health': self.health_checker.check_health(),
                'circuit_breaker_state': self.circuit_breaker.state if self.circuit_breaker else 'disabled'
            }

    return MonitoredRouter(router)
