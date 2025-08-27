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
        logger.info("ProductionRouter shutdown complete")


class RouterFactory:
    """Factory for creating production routers."""
    
    @staticmethod
    def create_from_config_file(config_path: str) -> ProductionRouter:
        """Create router from JSON/YAML config file."""
        import json
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                try:
                    import yaml
                    config_dict = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required for YAML config files")
            else:
                config_dict = json.load(f)
        
        config = ProductionConfig(**config_dict)
        return ProductionRouter(config)
    
    @staticmethod
    def create_from_environment() -> ProductionRouter:
        """Create router from environment variables."""
        import os
        config = ProductionConfig(
            input_dim=int(os.getenv('MOE_INPUT_DIM', '768')),
            num_experts=int(os.getenv('MOE_NUM_EXPERTS', '8')),
            min_experts=int(os.getenv('MOE_MIN_EXPERTS', '1')),
            max_experts=int(os.getenv('MOE_MAX_EXPERTS', '4')),
            complexity_estimator=os.getenv('MOE_COMPLEXITY_ESTIMATOR', 'gradient_norm'),
            enable_caching=os.getenv('MOE_ENABLE_CACHING', 'true').lower() == 'true',
            enable_parallel_processing=os.getenv('MOE_ENABLE_PARALLEL', 'true').lower() == 'true',
            enable_security=os.getenv('MOE_ENABLE_SECURITY', 'true').lower() == 'true',
            security_level=os.getenv('MOE_SECURITY_LEVEL', 'standard'),
            max_memory_mb=float(os.getenv('MOE_MAX_MEMORY_MB', '4000')),
            request_timeout=float(os.getenv('MOE_REQUEST_TIMEOUT', '30')),
            metrics_log_file=os.getenv('MOE_METRICS_LOG_FILE')
        )
        
        return ProductionRouter(config)
    
    @staticmethod
    def create_optimized_for_inference() -> ProductionRouter:
        """Create router optimized for inference workloads."""
        config = ProductionConfig(
            input_dim=768,
            num_experts=8,
            min_experts=1,
            max_experts=3,
            enable_caching=True,
            enable_parallel_processing=True,
            enable_compute_optimization=True,
            cache_size=10000,
            parallel_threshold_size=500,
            enable_security=True,
            security_level="standard",
            enable_resilience=True,
            circuit_breaker_failure_threshold=3,
            retry_max_attempts=2
        )
        
        return ProductionRouter(config)
    
    @staticmethod
    def create_optimized_for_training() -> ProductionRouter:
        """Create router optimized for training workloads."""
        config = ProductionConfig(
            input_dim=1024,
            num_experts=16,
            min_experts=2,
            max_experts=8,
            enable_caching=False,
            enable_parallel_processing=True,
            enable_compute_optimization=True,
            parallel_threshold_size=2000,
            enable_security=True,
            security_level="minimal",
            enable_resilience=False,
            max_memory_mb=8000,
            request_timeout=60.0
        )
        
        return ProductionRouter(config)