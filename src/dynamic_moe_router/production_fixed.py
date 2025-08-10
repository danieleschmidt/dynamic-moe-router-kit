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

from .router import DynamicRouter
from .monitoring import EnhancedPerformanceMonitor
from .security import RouterSecurityPolicy
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
    
    # Security and monitoring
    enable_security: bool = True
    enable_monitoring: bool = True
    security_level: str = "standard"  # "minimal", "standard", "strict"
    
    # Resource limits
    max_memory_mb: float = 4000.0
    max_cpu_usage: float = 0.8
    request_timeout: float = 30.0
    
    def __post_init__(self):
        """Validate and set defaults."""
        if self.max_experts is None:
            self.max_experts = min(self.num_experts, 8)  # Reasonable default


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
        self.router = DynamicRouter(
            input_dim=self.config.input_dim,
            num_experts=self.config.num_experts,
            min_experts=self.config.min_experts,
            max_experts=self.config.max_experts,
            complexity_estimator=self.config.complexity_estimator,
            routing_strategy=self.config.routing_strategy,
            enable_security=self.config.enable_security,
            enable_monitoring=self.config.enable_monitoring,
        )
        
        # Security policy
        if self.config.enable_security:
            self.security_policy = RouterSecurityPolicy()
        
        # Enhanced monitoring
        if self.config.enable_monitoring:
            self.performance_monitor = EnhancedPerformanceMonitor()
    
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
            
            # Route with core router
            result = self.router.route(hidden_states, **kwargs)
            
            # Add production metadata
            result['production_info'] = {
                'request_id': request_id,
                'processing_time_ms': (time.time() - request_start) * 1000,
                'router_health': self._health_status,
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
            'router_stats': self.router.get_expert_usage_stats(),
            'health_status': self._health_status,
            'uptime_seconds': time.time() - self._startup_time,
            'total_requests': self._request_count
        }
        
        return metrics


class RouterFactory:
    """Factory for creating production routers."""
    
    @staticmethod
    def create_optimized_for_inference() -> ProductionRouter:
        """Create router optimized for inference workloads."""
        config = ProductionConfig(
            input_dim=768,
            num_experts=8,
            min_experts=1,
            max_experts=3,
            enable_caching=True,
            enable_security=True,
            security_level="standard"
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
            enable_caching=False,  # Training data is typically unique
            enable_security=True,
            security_level="minimal",  # Less security overhead
            max_memory_mb=8000,  # More memory for training
            request_timeout=60.0  # Longer timeout for training
        )
        
        return ProductionRouter(config)