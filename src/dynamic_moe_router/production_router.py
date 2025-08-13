"""Production-ready router with all optimizations and enterprise features."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .secure_router import SecureEnhancedRouter
from .high_performance import PerformanceOptimizer, ConcurrentRouter
from .auto_scaling import AutoScaler, LoadBalancer, ResourceAllocation
from .health_monitoring import HealthMonitor, PerformanceSnapshot
from .robust_security import SecurityValidator, RobustErrorHandler, ResourceMonitor

logger = logging.getLogger(__name__)


class ProductionMoERouter:
    """Enterprise-grade production router with comprehensive features."""
    
    def __init__(
        self,
        # Core router parameters
        input_dim: int,
        num_experts: int,
        min_experts: int = 1,
        max_experts: Optional[int] = None,
        complexity_estimator: Union[str, Any] = "gradient_norm",
        
        # Performance optimization
        enable_performance_optimization: bool = True,
        cache_size: int = 1000,
        max_batch_size: int = 128,
        thread_pool_size: int = 4,
        
        # Security and robustness
        enable_security: bool = True,
        max_concurrent_requests: int = 100,
        request_timeout_seconds: float = 30.0,
        max_memory_mb: float = 1024,
        
        # Auto-scaling
        enable_auto_scaling: bool = True,
        initial_instance_count: int = 1,
        max_instances: int = 10,
        
        # Monitoring
        enable_monitoring: bool = True,
        metric_retention_hours: int = 24,
        
        # Load balancing
        enable_load_balancing: bool = True,
        balancing_strategy: str = "least_connections",
        
        **router_kwargs
    ):
        
        # Initialize health monitoring first
        if enable_monitoring:
            self.health_monitor = HealthMonitor(
                max_history_size=1000,
                metric_retention_hours=metric_retention_hours,
                enable_trend_analysis=True
            )
        else:
            self.health_monitor = None
        
        # Initialize core secure router
        self.core_router = SecureEnhancedRouter(
            input_dim=input_dim,
            num_experts=num_experts,
            min_experts=min_experts,
            max_experts=max_experts,
            complexity_estimator=complexity_estimator,
            enable_security=enable_security,
            max_memory_mb=max_memory_mb,
            **router_kwargs
        )
        
        # Performance optimization
        if enable_performance_optimization:
            self.performance_optimizer = PerformanceOptimizer(
                cache_size=cache_size,
                max_batch_size=max_batch_size,
                thread_pool_size=thread_pool_size,
                enable_caching=True,
                enable_vectorization=True,
                enable_batching=True,
                enable_async=True
            )
            
            # Apply performance decorators
            self.core_router.route = self.performance_optimizer.optimized_routing_cache(
                self.core_router.route
            )
        else:
            self.performance_optimizer = None
        
        # Concurrent request handling
        self.concurrent_router = ConcurrentRouter(
            base_router=self.core_router,
            max_concurrent_requests=max_concurrent_requests,
            request_timeout_seconds=request_timeout_seconds,
            enable_request_pooling=True
        )
        
        # Auto-scaling
        if enable_auto_scaling and self.health_monitor:
            initial_allocation = ResourceAllocation(
                cpu_cores=4,
                memory_mb=int(max_memory_mb),
                instance_count=initial_instance_count
            )
            
            self.auto_scaler = AutoScaler(
                health_monitor=self.health_monitor,
                initial_allocation=initial_allocation,
                enable_predictive_scaling=True,
                enable_load_balancing=enable_load_balancing
            )
            
            # Add scaling callback
            self.auto_scaler.add_scaling_callback(self._handle_scaling_event)
        else:
            self.auto_scaler = None
        
        # Load balancing for multiple instances
        if enable_load_balancing:
            self.load_balancer = LoadBalancer(
                balancing_strategy=balancing_strategy,
                health_check_interval=30,
                enable_circuit_breaker=True
            )
            
            # Register initial instance
            self.load_balancer.register_instance("primary", self.concurrent_router, weight=1.0)
        else:
            self.load_balancer = None
        
        # Production state
        self.is_started = False
        self.start_time = time.time()
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        logger.info("Production MoE router initialized with enterprise features")
    
    async def start(self):
        """Start all production services."""
        if self.is_started:
            logger.warning("Production router already started")
            return
        
        # Start auto-scaling
        if self.auto_scaler:
            self.auto_scaler.start_auto_scaling(check_interval_seconds=60)
        
        # Setup health monitoring alerts
        if self.health_monitor:
            self.health_monitor.add_alert_callback(self._handle_health_alert)
        
        self.is_started = True
        self.start_time = time.time()
        
        logger.info("Production MoE router started successfully")
    
    async def stop(self):
        """Stop all production services gracefully."""
        if not self.is_started:
            return
        
        # Stop auto-scaling
        if self.auto_scaler:
            self.auto_scaler.stop_auto_scaling()
        
        # Cleanup performance optimizer
        if self.performance_optimizer:
            self.performance_optimizer.shutdown()
        
        # Cleanup concurrent router
        self.concurrent_router.shutdown()
        
        self.is_started = False
        
        logger.info("Production MoE router stopped gracefully")
    
    def route(
        self,
        hidden_states: Any,
        return_router_logits: bool = False,
        return_load_balancing_loss: bool = False,
        client_id: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """Production routing with comprehensive monitoring and optimization."""
        
        if not self.is_started:
            raise RuntimeError("Production router not started. Call start() first.")
        
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Route through load balancer if available, otherwise direct
            if self.load_balancer and len(self.load_balancer.instances) > 1:
                result = self.load_balancer.route_request(
                    hidden_states,
                    return_router_logits=return_router_logits,
                    return_load_balancing_loss=return_load_balancing_loss,
                    client_id=client_id,
                    **kwargs
                )
            else:
                result = self.concurrent_router.route(
                    hidden_states,
                    return_router_logits=return_router_logits,
                    return_load_balancing_loss=return_load_balancing_loss,
                    client_id=client_id,
                    **kwargs
                )
            
            # Record successful request
            response_time = time.time() - start_time
            self.successful_requests += 1
            
            # Update monitoring
            if self.health_monitor:
                self._record_performance_metrics(result, response_time, success=True)
            
            return result
            
        except Exception as e:
            # Record failed request
            response_time = time.time() - start_time
            self.failed_requests += 1
            
            # Update monitoring
            if self.health_monitor:
                self._record_performance_metrics({}, response_time, success=False)
            
            logger.error(f"Production routing failed: {e}")
            raise
    
    async def async_route(self, *args, **kwargs) -> Dict[str, Any]:
        """Asynchronous production routing."""
        
        if self.performance_optimizer and self.performance_optimizer.enable_async:
            return await self.performance_optimizer.async_route(self.route, *args, **kwargs)
        else:
            # Fallback to sync in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.route, *args, **kwargs)
    
    def _record_performance_metrics(
        self,
        result: Dict[str, Any],
        response_time: float,
        success: bool
    ):
        """Record comprehensive performance metrics."""
        
        if not self.health_monitor:
            return
        
        # Calculate performance metrics
        error_rate = self.failed_requests / max(self.total_requests, 1)
        
        # Extract routing info
        routing_info = result.get("routing_info", {})
        avg_experts = routing_info.get("avg_experts_per_token", 0)
        flop_reduction = routing_info.get("flop_reduction", 0)
        
        # Get resource usage
        resource_stats = {}
        if self.core_router.resource_monitor:
            resource_stats = self.core_router.resource_monitor.check_resources()
        
        memory_usage = resource_stats.get("memory_mb", 0)
        
        # Calculate throughput (approximate)
        if hasattr(result.get("expert_indices", []), 'size'):
            tokens_processed = result["expert_indices"].size
        else:
            tokens_processed = 100  # Fallback estimate
        
        throughput = tokens_processed / max(response_time, 0.001)
        
        # Record performance snapshot
        self.health_monitor.record_performance_snapshot(
            throughput=throughput,
            avg_experts=avg_experts,
            flop_reduction=flop_reduction,
            memory_usage=memory_usage,
            error_rate=error_rate,
            response_time=response_time * 1000  # Convert to ms
        )
        
        # Record individual request
        self.health_monitor.record_request(
            response_time_ms=response_time * 1000,
            success=success
        )
    
    def _handle_scaling_event(self, scaling_event: Dict[str, Any]):
        """Handle auto-scaling events."""
        
        action = scaling_event["action"]
        new_instances = scaling_event["new_instances"]
        old_instances = scaling_event["old_instances"]
        
        if self.load_balancer:
            if new_instances > old_instances:
                # Scale up: add new instances
                for i in range(old_instances, new_instances):
                    instance_id = f"instance_{i}"
                    
                    # Create new router instance (simplified)
                    new_router = ConcurrentRouter(
                        base_router=self.core_router,
                        max_concurrent_requests=100
                    )
                    
                    self.load_balancer.register_instance(instance_id, new_router)
                    logger.info(f"Added new router instance: {instance_id}")
            
            elif new_instances < old_instances:
                # Scale down: remove instances
                for i in range(new_instances, old_instances):
                    instance_id = f"instance_{i}"
                    if instance_id in self.load_balancer.instances:
                        self.load_balancer.remove_instance(instance_id)
                        logger.info(f"Removed router instance: {instance_id}")
    
    def _handle_health_alert(self, metric):
        """Handle health monitoring alerts."""
        
        logger.warning(f"Health alert: {metric.name} = {metric.value} {metric.unit} "
                      f"(status: {metric.status})")
        
        # Could trigger additional actions like notifications, emergency scaling, etc.
        if metric.status == "critical":
            # Emergency actions for critical alerts
            if metric.name == "error_rate" and metric.value > 0.1:  # >10% error rate
                logger.critical("Critical error rate detected - consider emergency scaling")
            elif metric.name == "memory_usage_mb" and metric.value > 1000:
                logger.critical("Critical memory usage - consider immediate scaling")
    
    def get_production_status(self) -> Dict[str, Any]:
        """Get comprehensive production status."""
        
        uptime = time.time() - self.start_time
        success_rate = self.successful_requests / max(self.total_requests, 1)
        
        status = {
            "production_ready": self.is_started,
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "features": {
                "security_enabled": self.core_router.enable_security,
                "performance_optimization": self.performance_optimizer is not None,
                "auto_scaling": self.auto_scaler is not None,
                "load_balancing": self.load_balancer is not None,
                "health_monitoring": self.health_monitor is not None
            }
        }
        
        # Add component statuses
        if self.health_monitor:
            status["health_status"] = self.health_monitor.get_health_status()
        
        if self.auto_scaler:
            status["scaling_status"] = self.auto_scaler.get_scaling_status()
        
        if self.load_balancer:
            status["load_balancing"] = self.load_balancer.get_load_balancing_stats()
        
        if self.performance_optimizer:
            status["performance_stats"] = self.performance_optimizer.get_performance_stats()
        
        if hasattr(self.concurrent_router, 'get_concurrency_stats'):
            status["concurrency_stats"] = self.concurrent_router.get_concurrency_stats()
        
        # Security summary
        status["security_summary"] = self.core_router.get_security_summary()
        
        return status
    
    def export_production_metrics(self, format: str = "json") -> str:
        """Export production metrics for monitoring systems."""
        
        if not self.health_monitor:
            raise RuntimeError("Health monitoring not enabled")
        
        return self.health_monitor.export_metrics(format=format)
    
    def reset_production_stats(self):
        """Reset all production statistics."""
        
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        if self.health_monitor:
            self.health_monitor.reset_monitoring()
        
        if self.auto_scaler:
            self.auto_scaler.reset_scaling_history()
        
        if self.performance_optimizer:
            self.performance_optimizer.reset_stats()
        
        if hasattr(self.concurrent_router, 'reset_stats'):
            self.concurrent_router.reset_stats()
        
        self.core_router.reset_security_state()
        
        logger.info("Production statistics reset")
    
    def validate_production_readiness(self) -> Dict[str, Any]:
        """Validate that the router is production-ready."""
        
        checks = {
            "core_router_initialized": self.core_router is not None,
            "security_enabled": self.core_router.enable_security if self.core_router else False,
            "monitoring_enabled": self.health_monitor is not None,
            "concurrency_handling": self.concurrent_router is not None,
            "performance_optimization": self.performance_optimizer is not None,
            "auto_scaling_available": self.auto_scaler is not None,
            "load_balancing_available": self.load_balancer is not None,
            "production_started": self.is_started
        }
        
        # Check for any failures
        failed_checks = [check for check, passed in checks.items() if not passed]
        
        # Overall readiness
        production_ready = len(failed_checks) == 0
        
        return {
            "production_ready": production_ready,
            "individual_checks": checks,
            "failed_checks": failed_checks,
            "readiness_score": (len(checks) - len(failed_checks)) / len(checks)
        }
    
    def __enter__(self):
        """Context manager entry."""
        asyncio.create_task(self.start())
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        asyncio.create_task(self.stop())