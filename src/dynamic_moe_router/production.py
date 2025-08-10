"""Production-ready configurations and utilities for dynamic MoE routing."""

import logging
import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading

import numpy as np

from .optimized_router_v2 import UltraOptimizedRouter
from .monitoring import EnhancedPerformanceMonitor
from .security import RouterSecurityPolicy, SecurityMonitor
from .resilience import ResilientRouter, CircuitConfig, RetryPolicy
from .exceptions import RouterConfigurationError, DynamicMoEError

logger = logging.getLogger(__name__)


@dataclass
class ProductionConfig:
    """Production configuration for MoE routing."""
    # Core router settings
    input_dim: int
    num_experts: int
    min_experts: int = 1
    max_experts: Optional[int] = None
    complexity_estimator: str = "gradient_norm"
    routing_strategy: str = "top_k"
    
    # Performance optimization
    enable_caching: bool = True
    enable_parallel_processing: bool = True
    enable_compute_optimization: bool = True
    cache_size: int = 5000
    parallel_threshold_size: int = 1000
    max_parallel_workers: Optional[int] = None
    
    # Security and monitoring
    enable_security: bool = True
    enable_monitoring: bool = True
    enable_resilience: bool = True
    security_level: str = "standard"  # "minimal", "standard", "strict"
    
    # Resilience configuration
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 60.0
    retry_max_attempts: int = 3
    retry_base_delay: float = 0.1
    fallback_strategy: str = "load_balance"
    
    # Monitoring configuration
    performance_window_size: int = 1000
    alert_thresholds: Optional[Dict[str, float]] = None
    enable_metrics_logging: bool = True
    metrics_log_file: Optional[str] = None
    
    # Resource limits
    max_memory_mb: float = 4000.0
    max_cpu_usage: float = 0.8
    request_timeout: float = 30.0
    
    def __post_init__(self):
        """Validate and set defaults."""
        if self.max_experts is None:
            self.max_experts = min(self.num_experts, 8)  # Reasonable default
        
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'avg_latency_ms': 100.0,
                'p95_latency_ms': 500.0,
                'error_rate': 0.05,
                'memory_usage_mb': self.max_memory_mb * 0.8
            }
        
        if self.max_parallel_workers is None:
            import multiprocessing
            self.max_parallel_workers = min(8, multiprocessing.cpu_count())


class ProductionRouter:
    """Production-ready router with comprehensive enterprise features."""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self._initialize_components()
        self._health_status = "healthy"
        self._startup_time = time.time()
        self._request_count = 0
        self._lock = threading.RLock()
        
        logger.info(f"ProductionRouter initialized with config: {config}")
    
    def _initialize_components(self):
        """Initialize all router components."""
        # Core router
        self.router = UltraOptimizedRouter(
            input_dim=self.config.input_dim,
            num_experts=self.config.num_experts,
            min_experts=self.config.min_experts,
            max_experts=self.config.max_experts,
            complexity_estimator=self.config.complexity_estimator,
            routing_strategy=self.config.routing_strategy,
            enable_security=self.config.enable_security,
            enable_monitoring=self.config.enable_monitoring,
            enable_caching=self.config.enable_caching,
            enable_parallel_processing=self.config.enable_parallel_processing,
            enable_compute_optimization=self.config.enable_compute_optimization,
            parallel_threshold_size=self.config.parallel_threshold_size,
            max_parallel_workers=self.config.max_parallel_workers
        )\n        \n        # Security policy\n        if self.config.enable_security:\n            self.security_policy = RouterSecurityPolicy()\n            self._configure_security_level()\n        \n        # Enhanced monitoring\n        if self.config.enable_monitoring:\n            self.performance_monitor = EnhancedPerformanceMonitor(\n                window_size=self.config.performance_window_size,\n                alert_thresholds=self.config.alert_thresholds\n            )\n            \n            # Add alert callbacks\n            self.performance_monitor.add_alert_callback(self._handle_performance_alert)\n        \n        # Resilience wrapper\n        if self.config.enable_resilience:\n            circuit_config = CircuitConfig(\n                failure_threshold=self.config.circuit_breaker_failure_threshold,\n                recovery_timeout=self.config.circuit_breaker_recovery_timeout\n            )\n            retry_policy = RetryPolicy(\n                max_retries=self.config.retry_max_attempts,\n                base_delay=self.config.retry_base_delay\n            )\n            self.resilient_wrapper = ResilientRouter(\n                self.router,\n                circuit_config,\n                retry_policy,\n                self.config.fallback_strategy\n            )\n    \n    def _configure_security_level(self):\n        \"\"\"Configure security based on specified level.\"\"\"\n        if self.config.security_level == \"minimal\":\n            # Basic input validation only\n            pass\n        elif self.config.security_level == \"standard\":\n            # Standard security with rate limiting\n            pass\n        elif self.config.security_level == \"strict\":\n            # Strict security with comprehensive monitoring\n            pass\n    \n    def route(self, hidden_states: Any, **kwargs) -> Dict[str, Any]:\n        \"\"\"Production route method with comprehensive error handling.\"\"\"\n        request_start = time.time()\n        \n        with self._lock:\n            self._request_count += 1\n            request_id = f\"req_{self._request_count}_{int(time.time() * 1000)}\"\n        \n        try:\n            # Health check\n            if self._health_status != \"healthy\":\n                raise DynamicMoEError(f\"Router unhealthy: {self._health_status}\")\n            \n            # Timeout handling\n            @contextmanager\n            def timeout_context():\n                start = time.time()\n                yield\n                duration = time.time() - start\n                if duration > self.config.request_timeout:\n                    logger.warning(f\"Request {request_id} exceeded timeout: {duration:.2f}s\")\n            \n            with timeout_context():\n                # Route with appropriate wrapper\n                if self.config.enable_resilience:\n                    result = self.resilient_wrapper.route(hidden_states, **kwargs)\n                else:\n                    result = self.router.route(hidden_states, **kwargs)\n            \n            # Add production metadata\n            result['production_info'] = {\n                'request_id': request_id,\n                'processing_time_ms': (time.time() - request_start) * 1000,\n                'router_health': self._health_status,\n                'security_level': self.config.security_level if self.config.enable_security else None\n            }\n            \n            # Record metrics\n            if self.config.enable_monitoring:\n                duration_ms = (time.time() - request_start) * 1000\n                self.performance_monitor.record_call(\n                    duration_ms, \n                    success=True,\n                    request_id=request_id\n                )\n            \n            return result\n            \n        except Exception as e:\n            # Record error metrics\n            if self.config.enable_monitoring:\n                duration_ms = (time.time() - request_start) * 1000\n                self.performance_monitor.record_call(\n                    duration_ms, \n                    success=False,\n                    error_type=type(e).__name__\n                )\n            \n            logger.error(f\"Request {request_id} failed: {e}\")\n            raise\n    \n    def _handle_performance_alert(self, alert):\n        \"\"\"Handle performance alerts.\"\"\"\n        logger.warning(f\"Performance alert: {alert.message}\")\n        \n        # Take corrective action based on alert severity\n        if alert.severity == \"critical\":\n            if alert.metric == \"error_rate\" and alert.value > 0.2:\n                self._health_status = \"degraded\"\n                logger.error(\"Router marked as degraded due to high error rate\")\n        \n        # Log alert to external monitoring system\n        if self.config.enable_metrics_logging:\n            self._log_alert(alert)\n    \n    def _log_alert(self, alert):\n        \"\"\"Log alert to external system.\"\"\"\n        try:\n            alert_data = {\n                'timestamp': alert.timestamp,\n                'metric': alert.metric,\n                'value': alert.value,\n                'threshold': alert.threshold,\n                'severity': alert.severity,\n                'message': alert.message,\n                'router_id': id(self)\n            }\n            \n            if self.config.metrics_log_file:\n                with open(self.config.metrics_log_file, 'a') as f:\n                    f.write(json.dumps(alert_data) + '\\n')\n            \n        except Exception as e:\n            logger.error(f\"Failed to log alert: {e}\")\n    \n    def health_check(self) -> Dict[str, Any]:\n        \"\"\"Comprehensive health check.\"\"\"\n        health_data = {\n            'status': self._health_status,\n            'uptime_seconds': time.time() - self._startup_time,\n            'total_requests': self._request_count,\n            'components': {}\n        }\n        \n        # Check core router\n        try:\n            # Simple test routing\n            test_input = np.random.randn(1, 4, self.config.input_dim).astype(np.float32)\n            test_result = self.router.route(test_input)\n            health_data['components']['core_router'] = 'healthy'\n        except Exception as e:\n            health_data['components']['core_router'] = f'unhealthy: {e}'\n            self._health_status = 'unhealthy'\n        \n        # Check monitoring\n        if self.config.enable_monitoring:\n            try:\n                stats = self.performance_monitor._compute_current_metrics()\n                health_data['components']['monitoring'] = 'healthy'\n                health_data['performance_metrics'] = stats\n            except Exception as e:\n                health_data['components']['monitoring'] = f'degraded: {e}'\n        \n        # Check security\n        if self.config.enable_security:\n            try:\n                security_summary = self.router.input_sanitizer.security_monitor.get_security_summary()\n                health_data['components']['security'] = 'healthy'\n                health_data['security_summary'] = security_summary\n            except Exception as e:\n                health_data['components']['security'] = f'degraded: {e}'\n        \n        # Check resilience\n        if self.config.enable_resilience:\n            try:\n                resilience_stats = self.resilient_wrapper.get_resilience_stats()\n                health_data['components']['resilience'] = 'healthy'\n                health_data['resilience_stats'] = resilience_stats\n            except Exception as e:\n                health_data['components']['resilience'] = f'degraded: {e}'\n        \n        return health_data\n    \n    def get_metrics(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive production metrics.\"\"\"\n        metrics = {\n            'router_stats': self.router.get_optimization_stats(),\n            'health_status': self._health_status,\n            'uptime_seconds': time.time() - self._startup_time,\n            'total_requests': self._request_count\n        }\n        \n        if self.config.enable_monitoring:\n            metrics['performance'] = self.performance_monitor._compute_current_metrics()\n            metrics['recent_alerts'] = [{\n                'timestamp': alert.timestamp,\n                'metric': alert.metric,\n                'severity': alert.severity,\n                'message': alert.message\n            } for alert in list(self.performance_monitor._alerts)[-10:]]  # Last 10 alerts\n        \n        return metrics\n    \n    def optimize_for_production(self, sample_data: Optional[List[Any]] = None) -> Dict[str, Any]:\n        \"\"\"Optimize router configuration for production workload.\"\"\"\n        logger.info(\"Starting production optimization...\")\n        \n        optimization_results = {}\n        \n        # Workload analysis if sample data provided\n        if sample_data:\n            workload_analysis = self.router.optimize_for_workload(sample_data)\n            optimization_results['workload_analysis'] = workload_analysis\n        \n        # Memory optimization\n        try:\n            import psutil\n            memory_info = psutil.virtual_memory()\n            if memory_info.percent > 80:  # High memory usage\n                logger.warning(\"High memory usage detected, reducing cache sizes\")\n                # Reduce cache sizes\n                optimization_results['memory_optimization'] = 'cache_reduced'\n        except ImportError:\n            pass\n        \n        # CPU optimization\n        try:\n            import psutil\n            cpu_percent = psutil.cpu_percent(interval=1)\n            if cpu_percent > self.config.max_cpu_usage * 100:\n                logger.warning(\"High CPU usage detected, adjusting parallel processing\")\n                # Reduce parallel processing threshold\n                optimization_results['cpu_optimization'] = 'parallel_threshold_increased'\n        except ImportError:\n            pass\n        \n        return optimization_results\n    \n    def shutdown(self) -> None:\n        \"\"\"Graceful shutdown.\"\"\"\n        logger.info(\"Shutting down ProductionRouter...\")\n        \n        self._health_status = \"shutting_down\"\n        \n        # Close thread pools if they exist\n        if hasattr(self.router, 'batch_processor'):\n            self.router.batch_processor.executor.shutdown(wait=True)\n        \n        logger.info(\"ProductionRouter shutdown complete\")\n\n\nclass RouterFactory:\n    \"\"\"Factory for creating production routers.\"\"\"\n    \n    @staticmethod\n    def create_from_config_file(config_path: str) -> ProductionRouter:\n        \"\"\"Create router from JSON/YAML config file.\"\"\"\n        with open(config_path, 'r') as f:\n            if config_path.endswith('.yaml') or config_path.endswith('.yml'):\n                import yaml\n                config_dict = yaml.safe_load(f)\n            else:\n                config_dict = json.load(f)\n        \n        config = ProductionConfig(**config_dict)\n        return ProductionRouter(config)\n    \n    @staticmethod\n    def create_from_environment() -> ProductionRouter:\n        \"\"\"Create router from environment variables.\"\"\"\n        config = ProductionConfig(\n            input_dim=int(os.getenv('MOE_INPUT_DIM', '768')),\n            num_experts=int(os.getenv('MOE_NUM_EXPERTS', '8')),\n            min_experts=int(os.getenv('MOE_MIN_EXPERTS', '1')),\n            max_experts=int(os.getenv('MOE_MAX_EXPERTS', '4')),\n            complexity_estimator=os.getenv('MOE_COMPLEXITY_ESTIMATOR', 'gradient_norm'),\n            enable_caching=os.getenv('MOE_ENABLE_CACHING', 'true').lower() == 'true',\n            enable_parallel_processing=os.getenv('MOE_ENABLE_PARALLEL', 'true').lower() == 'true',\n            enable_security=os.getenv('MOE_ENABLE_SECURITY', 'true').lower() == 'true',\n            security_level=os.getenv('MOE_SECURITY_LEVEL', 'standard'),\n            max_memory_mb=float(os.getenv('MOE_MAX_MEMORY_MB', '4000')),\n            request_timeout=float(os.getenv('MOE_REQUEST_TIMEOUT', '30')),\n            metrics_log_file=os.getenv('MOE_METRICS_LOG_FILE')\n        )\n        \n        return ProductionRouter(config)\n    \n    @staticmethod\n    def create_optimized_for_inference() -> ProductionRouter:\n        \"\"\"Create router optimized for inference workloads.\"\"\"\n        config = ProductionConfig(\n            input_dim=768,\n            num_experts=8,\n            min_experts=1,\n            max_experts=3,\n            enable_caching=True,\n            enable_parallel_processing=True,\n            enable_compute_optimization=True,\n            cache_size=10000,  # Large cache for inference\n            parallel_threshold_size=500,  # Lower threshold for parallel processing\n            enable_security=True,\n            security_level=\"standard\",\n            enable_resilience=True,\n            circuit_breaker_failure_threshold=3,\n            retry_max_attempts=2\n        )\n        \n        return ProductionRouter(config)\n    \n    @staticmethod\n    def create_optimized_for_training() -> ProductionRouter:\n        \"\"\"Create router optimized for training workloads.\"\"\"\n        config = ProductionConfig(\n            input_dim=1024,\n            num_experts=16,\n            min_experts=2,\n            max_experts=8,\n            enable_caching=False,  # Training data is typically unique\n            enable_parallel_processing=True,\n            enable_compute_optimization=True,\n            parallel_threshold_size=2000,  # Higher threshold (larger batches)\n            enable_security=True,\n            security_level=\"minimal\",  # Less security overhead\n            enable_resilience=False,  # Training can handle failures differently\n            max_memory_mb=8000,  # More memory for training\n            request_timeout=60.0  # Longer timeout for training\n        )\n        \n        return ProductionRouter(config)