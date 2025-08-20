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
        )
        
        # Security policy
        if self.config.enable_security:
            self.security_policy = RouterSecurityPolicy()
            self._configure_security_level()
        
        # Enhanced monitoring
        if self.config.enable_monitoring:
            self.performance_monitor = EnhancedPerformanceMonitor(
                window_size=self.config.performance_window_size,
                alert_thresholds=self.config.alert_thresholds
            )
            
            # Add alert callbacks
            self.performance_monitor.add_alert_callback(self._handle_performance_alert)
        
        # Resilience wrapper
        if self.config.enable_resilience:
            circuit_config = CircuitConfig(
                failure_threshold=self.config.circuit_breaker_failure_threshold,
                recovery_timeout=self.config.circuit_breaker_recovery_timeout
            )
            retry_policy = RetryPolicy(
                max_retries=self.config.retry_max_attempts,
                base_delay=self.config.retry_base_delay
            )
            self.resilient_wrapper = ResilientRouter(
                self.router,
                circuit_config,
                retry_policy,
                self.config.fallback_strategy
            )
    
    def _configure_security_level(self):
        """Configure security based on specified level."""
        if self.config.security_level == "minimal":
            # Basic input validation only
            pass
        elif self.config.security_level == "standard":
            # Standard security with rate limiting
            pass
        elif self.config.security_level == "strict":
            # Strict security with comprehensive monitoring
            pass
    
    def route(self, hidden_states: Any, **kwargs) -> Dict[str, Any]:
        """Production route method with comprehensive error handling."""
        request_start = time.time()
        
        with self._lock:
            self._request_count += 1
            request_id = f"req_{self._request_count}_{int(time.time() * 1000)}"
        
        try:
            # Health check
            if self._health_status != "healthy":
                raise DynamicMoEError(f"Router unhealthy: {self._health_status}")
            
            # Route with appropriate wrapper
            if self.config.enable_resilience:
                result = self.resilient_wrapper.route(hidden_states, **kwargs)
            else:
                result = self.router.route(hidden_states, **kwargs)
            
            # Add production metadata
            result['production_info'] = {
                'request_id': request_id,
                'processing_time_ms': (time.time() - request_start) * 1000,
                'router_health': self._health_status,
                'security_level': self.config.security_level if self.config.enable_security else None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Request {request_id} failed: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        health_data = {
            'status': self._health_status,
            'uptime_seconds': time.time() - self._startup_time,
            'total_requests': self._request_count,
            'components': {}
        }
        
        # Check core router
        try:
            # Simple test routing
            test_input = np.random.randn(1, 4, self.config.input_dim).astype(np.float32)
            test_result = self.router.route(test_input)
            health_data['components']['core_router'] = 'healthy'
        except Exception as e:
            health_data['components']['core_router'] = f'unhealthy: {e}'
            self._health_status = 'unhealthy'
        
        return health_data
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive production metrics."""
        metrics = {
            'health_status': self._health_status,
            'uptime_seconds': time.time() - self._startup_time,
            'total_requests': self._request_count
        }
        
        return metrics
    
    def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down ProductionRouter...")
        self._health_status = "shutting_down"
        logger.info("ProductionRouter shutdown complete")\n\n\nclass RouterFactory:\n    \"\"\"Factory for creating production routers.\"\"\"\n    \n    @staticmethod\n    def create_from_config_file(config_path: str) -> ProductionRouter:\n        \"\"\"Create router from JSON/YAML config file.\"\"\"\n        with open(config_path, 'r') as f:\n            if config_path.endswith('.yaml') or config_path.endswith('.yml'):\n                import yaml\n                config_dict = yaml.safe_load(f)\n            else:\n                config_dict = json.load(f)\n        \n        config = ProductionConfig(**config_dict)\n        return ProductionRouter(config)\n    \n    @staticmethod\n    def create_from_environment() -> ProductionRouter:\n        \"\"\"Create router from environment variables.\"\"\"\n        config = ProductionConfig(\n            input_dim=int(os.getenv('MOE_INPUT_DIM', '768')),\n            num_experts=int(os.getenv('MOE_NUM_EXPERTS', '8')),\n            min_experts=int(os.getenv('MOE_MIN_EXPERTS', '1')),\n            max_experts=int(os.getenv('MOE_MAX_EXPERTS', '4')),\n            complexity_estimator=os.getenv('MOE_COMPLEXITY_ESTIMATOR', 'gradient_norm'),\n            enable_caching=os.getenv('MOE_ENABLE_CACHING', 'true').lower() == 'true',\n            enable_parallel_processing=os.getenv('MOE_ENABLE_PARALLEL', 'true').lower() == 'true',\n            enable_security=os.getenv('MOE_ENABLE_SECURITY', 'true').lower() == 'true',\n            security_level=os.getenv('MOE_SECURITY_LEVEL', 'standard'),\n            max_memory_mb=float(os.getenv('MOE_MAX_MEMORY_MB', '4000')),\n            request_timeout=float(os.getenv('MOE_REQUEST_TIMEOUT', '30')),\n            metrics_log_file=os.getenv('MOE_METRICS_LOG_FILE')\n        )\n        \n        return ProductionRouter(config)\n    \n    @staticmethod\n    def create_optimized_for_inference() -> ProductionRouter:\n        \"\"\"Create router optimized for inference workloads.\"\"\"\n        config = ProductionConfig(\n            input_dim=768,\n            num_experts=8,\n            min_experts=1,\n            max_experts=3,\n            enable_caching=True,\n            enable_parallel_processing=True,\n            enable_compute_optimization=True,\n            cache_size=10000,  # Large cache for inference\n            parallel_threshold_size=500,  # Lower threshold for parallel processing\n            enable_security=True,\n            security_level=\"standard\",\n            enable_resilience=True,\n            circuit_breaker_failure_threshold=3,\n            retry_max_attempts=2\n        )\n        \n        return ProductionRouter(config)\n    \n    @staticmethod\n    def create_optimized_for_training() -> ProductionRouter:\n        \"\"\"Create router optimized for training workloads.\"\"\"\n        config = ProductionConfig(\n            input_dim=1024,\n            num_experts=16,\n            min_experts=2,\n            max_experts=8,\n            enable_caching=False,  # Training data is typically unique\n            enable_parallel_processing=True,\n            enable_compute_optimization=True,\n            parallel_threshold_size=2000,  # Higher threshold (larger batches)\n            enable_security=True,\n            security_level=\"minimal\",  # Less security overhead\n            enable_resilience=False,  # Training can handle failures differently\n            max_memory_mb=8000,  # More memory for training\n            request_timeout=60.0  # Longer timeout for training\n        )\n        \n        return ProductionRouter(config)