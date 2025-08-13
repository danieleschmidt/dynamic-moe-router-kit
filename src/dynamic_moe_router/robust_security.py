"""Enhanced security and robustness features for dynamic MoE routing."""

import hashlib
import logging
import os
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Union
import numpy as np

from .exceptions import (
    SecurityValidationError,
    ResourceExhaustionError,
    InputValidationError
)

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Comprehensive security validation for MoE routing."""
    
    def __init__(
        self,
        max_input_size: int = 1024 * 1024 * 10,  # 10MB
        max_batch_size: int = 1024,
        max_sequence_length: int = 8192,
        allowed_dtypes: Optional[Set[str]] = None,
        enable_input_sanitization: bool = True,
        enable_rate_limiting: bool = True,
        rate_limit_requests: int = 1000,
        rate_limit_window: int = 3600  # 1 hour
    ):
        self.max_input_size = max_input_size
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.allowed_dtypes = allowed_dtypes or {"float32", "float64", "float16"}
        self.enable_input_sanitization = enable_input_sanitization
        self.enable_rate_limiting = enable_rate_limiting
        
        # Rate limiting
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_window = rate_limit_window
        self.request_history = []
        
        logger.info("Security validator initialized with comprehensive protections")
    
    def validate_input_tensor(self, tensor: Any, name: str = "input") -> None:
        """Validate input tensor for security and safety."""
        
        # Basic existence check
        if tensor is None:
            raise InputValidationError(f"{name} cannot be None")
        
        # Check if tensor-like
        if not hasattr(tensor, 'shape') and not hasattr(tensor, '__len__'):
            raise InputValidationError(f"{name} must be tensor-like with shape attribute")
        
        # Handle different tensor types
        if hasattr(tensor, 'numpy'):  # PyTorch tensor
            tensor_np = tensor.detach().numpy()
        elif hasattr(tensor, '__array__'):  # NumPy-like
            tensor_np = np.asarray(tensor)
        else:
            tensor_np = np.array(tensor)
        
        # Validate shape constraints
        if len(tensor_np.shape) > 4:
            raise InputValidationError(f"{name} has too many dimensions: {len(tensor_np.shape)} > 4")
        
        # Validate size constraints
        total_elements = np.prod(tensor_np.shape)
        if total_elements > self.max_input_size:
            raise InputValidationError(
                f"{name} size {total_elements} exceeds maximum {self.max_input_size}"
            )
        
        # Validate batch size
        if len(tensor_np.shape) >= 1 and tensor_np.shape[0] > self.max_batch_size:
            raise InputValidationError(
                f"{name} batch size {tensor_np.shape[0]} exceeds maximum {self.max_batch_size}"
            )
        
        # Validate sequence length
        if len(tensor_np.shape) >= 2 and tensor_np.shape[1] > self.max_sequence_length:
            raise InputValidationError(
                f"{name} sequence length {tensor_np.shape[1]} exceeds maximum {self.max_sequence_length}"
            )
        
        # Validate data type
        dtype_str = str(tensor_np.dtype)
        if dtype_str not in self.allowed_dtypes:
            raise InputValidationError(
                f"{name} dtype {dtype_str} not in allowed types: {self.allowed_dtypes}"
            )
        
        # Check for NaN/Inf values
        if np.any(np.isnan(tensor_np)):
            raise InputValidationError(f"{name} contains NaN values")
        
        if np.any(np.isinf(tensor_np)):
            raise InputValidationError(f"{name} contains infinite values")
        
        # Check for extremely large values that could cause overflow
        max_val = np.max(np.abs(tensor_np))
        if max_val > 1e6:
            logger.warning(f"{name} contains very large values (max: {max_val})")
        
        # Input sanitization
        if self.enable_input_sanitization:
            self._sanitize_tensor(tensor_np, name)
    
    def _sanitize_tensor(self, tensor: np.ndarray, name: str) -> None:
        """Apply input sanitization to prevent adversarial inputs."""
        
        # Check for potential adversarial patterns
        mean_val = np.mean(tensor)
        std_val = np.std(tensor)
        
        # Detect suspiciously uniform or extreme distributions
        if std_val < 1e-8:
            logger.warning(f"{name} has suspiciously low variance: {std_val}")
        
        if abs(mean_val) > 100:
            logger.warning(f"{name} has extreme mean value: {mean_val}")
        
        # Check for repeating patterns (possible adversarial)
        flattened = tensor.flatten()
        if len(flattened) > 100:
            # Sample check for repeating patterns
            sample_size = min(100, len(flattened))
            sample = flattened[:sample_size]
            unique_ratio = len(np.unique(sample)) / len(sample)
            if unique_ratio < 0.1:
                logger.warning(f"{name} has low unique value ratio: {unique_ratio}")
    
    def check_rate_limit(self, client_id: str = "default") -> None:
        """Check rate limiting for API abuse protection."""
        if not self.enable_rate_limiting:
            return
        
        current_time = time.time()
        
        # Clean old requests
        self.request_history = [
            (timestamp, cid) for timestamp, cid in self.request_history
            if current_time - timestamp < self.rate_limit_window
        ]
        
        # Count requests from this client
        client_requests = sum(
            1 for _, cid in self.request_history if cid == client_id
        )
        
        if client_requests >= self.rate_limit_requests:
            raise SecurityValidationError(
                f"Rate limit exceeded for client {client_id}: "
                f"{client_requests} requests in {self.rate_limit_window} seconds"
            )
        
        # Record this request
        self.request_history.append((current_time, client_id))
    
    def validate_config_security(self, config: Dict[str, Any]) -> None:
        """Validate router configuration for security issues."""
        
        # Check for dangerous parameters
        dangerous_keys = {"__class__", "__module__", "eval", "exec", "import"}
        for key in config.keys():
            if any(dangerous in str(key).lower() for dangerous in dangerous_keys):
                raise SecurityValidationError(f"Dangerous configuration key detected: {key}")
        
        # Validate numeric ranges
        if "num_experts" in config:
            if not isinstance(config["num_experts"], int) or config["num_experts"] <= 0:
                raise InputValidationError("num_experts must be positive integer")
            if config["num_experts"] > 1000:
                raise SecurityValidationError("num_experts too large (>1000)")
        
        if "noise_factor" in config:
            if not isinstance(config["noise_factor"], (int, float)):
                raise InputValidationError("noise_factor must be numeric")
            if config["noise_factor"] < 0 or config["noise_factor"] > 10:
                raise SecurityValidationError("noise_factor out of safe range [0, 10]")
        
        # Check for suspicious string values
        for key, value in config.items():
            if isinstance(value, str):
                if len(value) > 1000:
                    raise SecurityValidationError(f"Configuration value too long: {key}")
                if any(dangerous in value.lower() for dangerous in ["eval(", "exec(", "__"]):
                    raise SecurityValidationError(f"Suspicious configuration value: {key}")


class RobustErrorHandler:
    """Comprehensive error handling and recovery."""
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_fallback: bool = True,
        circuit_breaker_threshold: int = 10,
        circuit_breaker_timeout: float = 60.0
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_fallback = enable_fallback
        
        # Circuit breaker
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.circuit_open = False
        
        logger.info("Robust error handler initialized")
    
    def with_retry_and_fallback(self, operation_name: str = "operation"):
        """Decorator for retry logic and fallback handling."""
        
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Check circuit breaker
                if self._is_circuit_open():
                    raise ResourceExhaustionError(
                        f"Circuit breaker open for {operation_name}"
                    )
                
                last_exception = None
                
                for attempt in range(self.max_retries + 1):
                    try:
                        result = func(*args, **kwargs)
                        # Reset circuit breaker on success
                        self._reset_circuit_breaker()
                        return result
                        
                    except Exception as e:
                        last_exception = e
                        self._record_failure()
                        
                        logger.warning(
                            f"{operation_name} attempt {attempt + 1} failed: {e}"
                        )
                        
                        # Don't retry on certain errors
                        if isinstance(e, (SecurityValidationError, InputValidationError)):
                            break
                        
                        # Wait before retry
                        if attempt < self.max_retries:
                            time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                
                # All retries failed
                if self.enable_fallback:
                    try:
                        return self._fallback_operation(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed: {fallback_error}")
                
                # Circuit breaker check
                if self.failure_count >= self.circuit_breaker_threshold:
                    self.circuit_open = True
                    self.last_failure_time = time.time()
                
                raise last_exception
            
            return wrapper
        return decorator
    
    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if not self.circuit_open:
            return False
        
        # Check if timeout has elapsed
        if time.time() - self.last_failure_time > self.circuit_breaker_timeout:
            self.circuit_open = False
            self.failure_count = 0
            logger.info("Circuit breaker reset")
            return False
        
        return True
    
    def _record_failure(self) -> None:
        """Record a failure for circuit breaker."""
        self.failure_count += 1
        self.last_failure_time = time.time()
    
    def _reset_circuit_breaker(self) -> None:
        """Reset circuit breaker on successful operation."""
        self.failure_count = 0
        self.circuit_open = False
    
    def _fallback_operation(self, *args, **kwargs) -> Any:
        """Fallback operation when primary fails."""
        logger.warning("Using fallback operation")
        
        # Return a safe default result structure
        return {
            "expert_indices": np.array([]),
            "expert_weights": np.array([]),
            "num_experts_per_token": np.array([]),
            "complexity_scores": np.array([]),
            "routing_info": {
                "fallback_used": True,
                "avg_experts_per_token": 0,
                "flop_reduction": 0
            }
        }


class ResourceMonitor:
    """Monitor and limit resource usage."""
    
    def __init__(
        self,
        max_memory_mb: float = 1024,  # 1GB
        max_cpu_percent: float = 80.0,
        monitoring_enabled: bool = True
    ):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.monitoring_enabled = monitoring_enabled
        
        if monitoring_enabled:
            logger.info(f"Resource monitoring enabled: {max_memory_mb}MB memory limit")
    
    def check_resources(self) -> Dict[str, float]:
        """Check current resource usage."""
        if not self.monitoring_enabled:
            return {}
        
        try:
            import psutil
            
            # Memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            
            # Check limits
            if memory_mb > self.max_memory_mb:
                raise ResourceExhaustionError(
                    f"Memory usage {memory_mb:.1f}MB exceeds limit {self.max_memory_mb}MB"
                )
            
            if cpu_percent > self.max_cpu_percent:
                logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            
            return {
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent
            }
            
        except ImportError:
            logger.warning("psutil not available, resource monitoring disabled")
            return {}
        except Exception as e:
            logger.warning(f"Resource monitoring failed: {e}")
            return {}


def secure_router_decorator(
    security_validator: Optional[SecurityValidator] = None,
    error_handler: Optional[RobustErrorHandler] = None,
    resource_monitor: Optional[ResourceMonitor] = None
):
    """Decorator to add security and robustness to router methods."""
    
    # Initialize default components if not provided
    if security_validator is None:
        security_validator = SecurityValidator()
    if error_handler is None:
        error_handler = RobustErrorHandler()
    if resource_monitor is None:
        resource_monitor = ResourceMonitor()
    
    def decorator(func: Callable) -> Callable:
        @error_handler.with_retry_and_fallback(operation_name=func.__name__)
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Security validation
            if len(args) > 1:  # Assume first arg is self, second is input
                security_validator.validate_input_tensor(args[1], "input_tensor")
            
            # Rate limiting check
            client_id = kwargs.get("client_id", "default")
            security_validator.check_rate_limit(client_id)
            
            # Resource monitoring
            resource_stats = resource_monitor.check_resources()
            if resource_stats:
                logger.debug(f"Resource usage: {resource_stats}")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Post-execution validation
            if isinstance(result, dict) and "expert_indices" in result:
                security_validator.validate_input_tensor(
                    result["expert_indices"], "expert_indices"
                )
            
            return result
        
        return wrapper
    return decorator


# Global instances for easy use
default_security_validator = SecurityValidator()
default_error_handler = RobustErrorHandler()
default_resource_monitor = ResourceMonitor()