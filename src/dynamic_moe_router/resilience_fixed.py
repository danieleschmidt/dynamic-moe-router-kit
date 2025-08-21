"""Resilience patterns for dynamic MoE routing."""

import logging
import time
import threading
from enum import Enum
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass

import numpy as np

from .exceptions import ExpertDispatchError

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5           # Failures to trigger open
    recovery_timeout: float = 60.0       # Seconds before trying half-open
    success_threshold: int = 3           # Successes to close from half-open
    timeout_threshold: float = 5.0       # Request timeout threshold


class CircuitBreaker:
    """Circuit breaker for expert routing resilience."""
    
    def __init__(self, name: str, config: Optional[CircuitConfig] = None):
        self.name = name
        self.config = config or CircuitConfig()
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()
        
        # Metrics
        self._total_requests = 0
        self._blocked_requests = 0
        self._successful_requests = 0
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        with self._lock:
            self._total_requests += 1
            
            # Check if circuit is open
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if time.time() - self._last_failure_time > self.config.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(f"Circuit breaker {self.name} entering half-open state")
                else:
                    self._blocked_requests += 1
                    raise ExpertDispatchError(f"Circuit breaker {self.name} is open")
        
        # Execute function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Check for timeout
            if execution_time > self.config.timeout_threshold:
                self._record_failure()
                raise ExpertDispatchError(f"Function execution timeout: {execution_time:.2f}s")
            
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure()
            raise
    
    def _record_success(self) -> None:
        """Record successful execution."""
        with self._lock:
            self._successful_requests += 1
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info(f"Circuit breaker {self.name} closed after recovery")
            
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
    
    def _record_failure(self) -> None:
        """Record failed execution."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker {self.name} opened due to failures")
                    
            elif self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._success_count = 0
                logger.warning(f"Circuit breaker {self.name} reopened after half-open failure")
    
    def _reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                'name': self.name,
                'state': self._state.value,
                'total_requests': self._total_requests,
                'failure_count': self._failure_count,
                'success_count': self._success_count,
            }


class RetryPolicy:
    """Retry policy with exponential backoff."""
    
    def __init__(self,
                 max_retries: int = 3,
                 base_delay: float = 0.1,
                 max_delay: float = 10.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    break
                
                # Calculate delay
                delay = min(
                    self.base_delay * (self.exponential_base ** attempt),
                    self.max_delay
                )
                
                # Add jitter to prevent thundering herd
                if self.jitter:
                    delay *= (0.5 + 0.5 * np.random.random())
                
                logger.debug(f"Retry attempt {attempt + 1}/{self.max_retries} after {delay:.3f}s")
                time.sleep(delay)
        
        raise last_exception


class ResilientRouter:
    """Router with built-in resilience patterns."""
    
    def __init__(self,
                 base_router: Any,
                 circuit_config: Optional[CircuitConfig] = None,
                 retry_policy: Optional[RetryPolicy] = None):
        self.base_router = base_router
        self.circuit_breaker = CircuitBreaker("router", circuit_config)
        self.retry_policy = retry_policy or RetryPolicy()
    
    def route(self, *args, **kwargs) -> Dict[str, Any]:
        """Route with full resilience patterns."""
        def resilient_route():
            return self.circuit_breaker.call(
                self.retry_policy.execute,
                self.base_router.route,
                *args, **kwargs
            )
        
        return resilient_route()
    
    def get_resilience_stats(self) -> Dict[str, Any]:
        """Get comprehensive resilience statistics."""
        return {
            'circuit_breaker': self.circuit_breaker.get_stats(),
        }