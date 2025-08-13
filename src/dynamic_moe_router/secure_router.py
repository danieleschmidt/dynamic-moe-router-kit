"""Secure and robust router implementation with comprehensive protections."""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .adaptive_router import EnhancedDynamicRouter
from .robust_security import (
    SecurityValidator,
    RobustErrorHandler,
    ResourceMonitor,
    secure_router_decorator
)
from .exceptions import (
    SecurityValidationError,
    ResourceExhaustionError,
    RouterConfigurationError
)

logger = logging.getLogger(__name__)


class SecureEnhancedRouter(EnhancedDynamicRouter):
    """Production-ready secure router with comprehensive protections."""
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        min_experts: int = 1,
        max_experts: Optional[int] = None,
        complexity_estimator: Union[str, Any] = "gradient_norm",
        routing_strategy: str = "adaptive_top_k",
        load_balancing: bool = True,
        noise_factor: float = 0.1,
        temperature: float = 1.0,
        expert_capacity_factor: float = 1.25,
        adaptive_threshold: bool = True,
        # Security parameters
        enable_security: bool = True,
        max_input_size: int = 1024 * 1024 * 10,  # 10MB
        max_batch_size: int = 1024,
        max_sequence_length: int = 8192,
        enable_rate_limiting: bool = True,
        rate_limit_requests: int = 1000,
        rate_limit_window: int = 3600,
        # Robustness parameters
        max_retries: int = 3,
        enable_fallback: bool = True,
        circuit_breaker_threshold: int = 10,
        # Resource monitoring
        max_memory_mb: float = 1024,
        enable_resource_monitoring: bool = True,
        **estimator_kwargs
    ):
        # Validate security configuration
        if enable_security:
            self._validate_security_config({
                "input_dim": input_dim,
                "num_experts": num_experts,
                "min_experts": min_experts,
                "max_experts": max_experts,
                "noise_factor": noise_factor,
                "temperature": temperature
            })
        
        # Initialize parent router
        super().__init__(
            input_dim=input_dim,
            num_experts=num_experts,
            min_experts=min_experts,
            max_experts=max_experts,
            complexity_estimator=complexity_estimator,
            routing_strategy=routing_strategy,
            load_balancing=load_balancing,
            noise_factor=noise_factor,
            temperature=temperature,
            expert_capacity_factor=expert_capacity_factor,
            adaptive_threshold=adaptive_threshold,
            **estimator_kwargs
        )
        
        # Security components
        self.enable_security = enable_security
        if enable_security:
            self.security_validator = SecurityValidator(
                max_input_size=max_input_size,
                max_batch_size=max_batch_size,
                max_sequence_length=max_sequence_length,
                enable_rate_limiting=enable_rate_limiting,
                rate_limit_requests=rate_limit_requests,
                rate_limit_window=rate_limit_window
            )
        else:
            self.security_validator = None
        
        # Error handling
        self.error_handler = RobustErrorHandler(
            max_retries=max_retries,
            enable_fallback=enable_fallback,
            circuit_breaker_threshold=circuit_breaker_threshold
        )
        
        # Resource monitoring
        if enable_resource_monitoring:
            self.resource_monitor = ResourceMonitor(
                max_memory_mb=max_memory_mb,
                monitoring_enabled=True
            )
        else:
            self.resource_monitor = None
        
        # Security audit log
        self.security_events = []
        self.max_security_events = 1000
        
        logger.info("Secure enhanced router initialized with comprehensive protections")
    
    def _validate_security_config(self, config: Dict[str, Any]) -> None:
        """Validate router configuration for security."""
        # Basic validation
        if config["num_experts"] <= 0 or config["num_experts"] > 1000:
            raise RouterConfigurationError("num_experts must be in range [1, 1000]")
        
        if config["input_dim"] <= 0 or config["input_dim"] > 100000:
            raise RouterConfigurationError("input_dim must be in range [1, 100000]")
        
        if config["noise_factor"] < 0 or config["noise_factor"] > 10:
            raise RouterConfigurationError("noise_factor must be in range [0, 10]")
        
        if config["temperature"] <= 0 or config["temperature"] > 10:
            raise RouterConfigurationError("temperature must be in range (0, 10]")
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security events for audit trail."""
        import time
        
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "details": details
        }
        
        self.security_events.append(event)
        
        # Maintain event log size
        if len(self.security_events) > self.max_security_events:
            self.security_events = self.security_events[-self.max_security_events:]
        
        logger.info(f"Security event logged: {event_type}")
    
    @secure_router_decorator()
    def route(
        self,
        hidden_states: Any,
        return_router_logits: bool = False,
        return_load_balancing_loss: bool = False,
        client_id: str = "default",
        **complexity_kwargs
    ) -> Dict[str, Any]:
        """Secure routing with comprehensive protections."""
        
        # Pre-execution security checks
        if self.enable_security:
            try:
                self.security_validator.validate_input_tensor(hidden_states, "hidden_states")
                self.security_validator.check_rate_limit(client_id)
            except (SecurityValidationError, Exception) as e:
                self._log_security_event("validation_failure", {
                    "client_id": client_id,
                    "error": str(e),
                    "input_shape": getattr(hidden_states, 'shape', 'unknown')
                })
                raise
        
        # Resource monitoring
        if self.resource_monitor:
            resource_stats = self.resource_monitor.check_resources()
            if resource_stats.get("memory_mb", 0) > self.resource_monitor.max_memory_mb * 0.9:
                logger.warning("High memory usage detected")
        
        # Execute routing with error handling
        try:
            result = super().route(
                hidden_states=hidden_states,
                return_router_logits=return_router_logits,
                return_load_balancing_loss=return_load_balancing_loss,
                **complexity_kwargs
            )
            
            # Post-execution validation
            if self.enable_security:
                self._validate_routing_result(result)
            
            # Log successful routing
            self._log_security_event("successful_routing", {
                "client_id": client_id,
                "input_shape": getattr(hidden_states, 'shape', 'unknown'),
                "avg_experts": result.get("routing_info", {}).get("avg_experts_per_token", 0)
            })
            
            return result
            
        except Exception as e:
            self._log_security_event("routing_failure", {
                "client_id": client_id,
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise
    
    def _validate_routing_result(self, result: Dict[str, Any]) -> None:
        """Validate routing result for security and correctness."""
        
        required_keys = ["expert_indices", "expert_weights", "routing_info"]
        for key in required_keys:
            if key not in result:
                raise SecurityValidationError(f"Missing required result key: {key}")
        
        # Validate expert indices
        expert_indices = result["expert_indices"]
        if hasattr(expert_indices, '__len__') and len(expert_indices) > 0:
            max_index = np.max(expert_indices) if hasattr(expert_indices, 'max') else max(expert_indices.flatten() if hasattr(expert_indices, 'flatten') else expert_indices)
            if max_index >= self.num_experts:
                raise SecurityValidationError(f"Expert index {max_index} >= num_experts {self.num_experts}")
        
        # Validate weights
        expert_weights = result["expert_weights"]
        if hasattr(expert_weights, '__len__') and len(expert_weights) > 0:
            # Check for negative weights
            min_weight = np.min(expert_weights) if hasattr(expert_weights, 'min') else min(expert_weights.flatten() if hasattr(expert_weights, 'flatten') else expert_weights)
            if min_weight < 0:
                raise SecurityValidationError(f"Negative expert weight: {min_weight}")
            
            # Check for extreme weights
            max_weight = np.max(expert_weights) if hasattr(expert_weights, 'max') else max(expert_weights.flatten() if hasattr(expert_weights, 'flatten') else expert_weights)
            if max_weight > 10:
                logger.warning(f"Very large expert weight detected: {max_weight}")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get comprehensive security and performance summary."""
        
        # Count security events by type
        event_counts = {}
        for event in self.security_events:
            event_type = event["type"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Get routing statistics
        routing_stats = self.get_routing_stats()
        
        # Resource information
        resource_info = {}
        if self.resource_monitor:
            resource_info = self.resource_monitor.check_resources()
        
        # Circuit breaker status
        circuit_info = {
            "failure_count": self.error_handler.failure_count,
            "circuit_open": self.error_handler.circuit_open,
            "last_failure_time": self.error_handler.last_failure_time
        }
        
        return {
            "security_enabled": self.enable_security,
            "total_security_events": len(self.security_events),
            "security_event_counts": event_counts,
            "routing_statistics": routing_stats,
            "resource_usage": resource_info,
            "circuit_breaker": circuit_info,
            "configuration": {
                "num_experts": self.num_experts,
                "routing_strategy": self.routing_strategy,
                "load_balancing": self.load_balancer is not None,
                "adaptive_threshold": self.adaptive_threshold
            }
        }
    
    def reset_security_state(self) -> None:
        """Reset security state and statistics."""
        self.security_events = []
        self.error_handler.failure_count = 0
        self.error_handler.circuit_open = False
        
        if self.security_validator and self.security_validator.enable_rate_limiting:
            self.security_validator.request_history = []
        
        self.reset_statistics()
        
        logger.info("Security state and statistics reset")
    
    def export_security_audit(self) -> Dict[str, Any]:
        """Export security audit log for compliance."""
        
        return {
            "router_id": id(self),
            "audit_timestamp": __import__('time').time(),
            "configuration": {
                "security_enabled": self.enable_security,
                "rate_limiting_enabled": (
                    self.security_validator.enable_rate_limiting 
                    if self.security_validator else False
                ),
                "resource_monitoring_enabled": self.resource_monitor is not None,
                "fallback_enabled": self.error_handler.enable_fallback
            },
            "security_events": self.security_events,
            "performance_summary": self.get_routing_stats(),
            "total_events": len(self.security_events)
        }