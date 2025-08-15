"""Enhanced resilience patterns for dynamic MoE routing."""

import logging
import time
import threading
from enum import Enum
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass
import numpy as np

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
        """Execute function with circuit breaker protection."""
        with self._lock:
            self._total_requests += 1
            
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                else:
                    self._blocked_requests += 1
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        return time.time() - self._last_failure_time > self.config.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            self._successful_requests += 1
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._reset()
                    logger.info(f"Circuit breaker {self.name} reset to CLOSED")
            
            # Reset failure count on successful call in CLOSED state
            if self._state == CircuitState.CLOSED:
                self._failure_count = 0
    
    def _on_failure(self, error: Exception) -> None:
        """Handle failed call."""
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
                'successful_requests': self._successful_requests,
                'blocked_requests': self._blocked_requests,
                'failure_count': self._failure_count,
                'success_count': self._success_count,
                'success_rate': self._successful_requests / max(self._total_requests, 1)
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


class FallbackManager:
    """Manages fallback strategies for failed routing."""
    
    def __init__(self):
        self.fallback_strategies = {
            'random_expert': self._random_expert_fallback,
            'single_expert': self._single_expert_fallback,
            'load_balance': self._load_balance_fallback
        }
    
    def execute_with_fallback(self,
                            primary_func: Callable,
                            fallback_strategy: str,
                            *args, **kwargs) -> Any:
        """Execute primary function with fallback on failure."""
        try:
            return primary_func(*args, **kwargs)
        except Exception as primary_error:
            logger.warning(f"Primary routing failed: {primary_error}, using fallback")
            
            if fallback_strategy not in self.fallback_strategies:
                raise ValueError(f"Unknown fallback strategy: {fallback_strategy}")
            
            try:
                return self.fallback_strategies[fallback_strategy](*args, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise Exception(
                    f"Both primary and fallback routing failed: "
                    f"primary={primary_error}, fallback={fallback_error}"
                )
    
    def _random_expert_fallback(self, hidden_states, num_experts: int = 8, **kwargs) -> Dict[str, Any]:
        """Fallback to random expert selection."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Select 1-2 random experts per token
        expert_indices = np.random.randint(0, num_experts, size=(batch_size, seq_len, 2))
        expert_weights = np.random.dirichlet([1, 1], size=(batch_size, seq_len))
        
        return {
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'num_experts_per_token': np.full((batch_size, seq_len), 2),
            'complexity_scores': np.full((batch_size, seq_len), 0.5),
            'routing_info': {
                'avg_experts_per_token': 2.0,
                'flop_reduction': 0.75,  # Using 2/8 experts
                'expert_utilization': [0.125] * num_experts,
                'fallback_used': True
            }
        }
    
    def _single_expert_fallback(self, hidden_states, num_experts: int = 8, **kwargs) -> Dict[str, Any]:
        """Fallback to single expert (expert 0)."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        expert_indices = np.zeros((batch_size, seq_len, 1), dtype=int)
        expert_weights = np.ones((batch_size, seq_len, 1))
        
        return {
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'num_experts_per_token': np.ones((batch_size, seq_len)),
            'complexity_scores': np.zeros((batch_size, seq_len)),
            'routing_info': {
                'avg_experts_per_token': 1.0,
                'flop_reduction': 0.875,  # Using 1/8 experts
                'expert_utilization': [1.0] + [0.0] * (num_experts - 1),
                'fallback_used': True
            }
        }
    
    def _load_balance_fallback(self, hidden_states, num_experts: int = 8, **kwargs) -> Dict[str, Any]:
        """Fallback to round-robin load balancing."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Round-robin assignment
        expert_indices = np.zeros((batch_size, seq_len, 1), dtype=int)
        for b in range(batch_size):
            for s in range(seq_len):
                token_idx = b * seq_len + s
                expert_indices[b, s, 0] = token_idx % num_experts
        
        expert_weights = np.ones((batch_size, seq_len, 1))
        
        return {
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'num_experts_per_token': np.ones((batch_size, seq_len)),
            'complexity_scores': np.full((batch_size, seq_len), 0.5),
            'routing_info': {
                'avg_experts_per_token': 1.0,
                'flop_reduction': 0.875,  # Using 1/8 experts
                'expert_utilization': [1.0 / num_experts] * num_experts,
                'fallback_used': True
            }
        }


class ResilientRouter:
    """Router with built-in resilience patterns."""
    
    def __init__(self,
                 base_router: Any,
                 circuit_config: Optional[CircuitConfig] = None,
                 retry_policy: Optional[RetryPolicy] = None,
                 fallback_strategy: str = "load_balance"):
        self.base_router = base_router
        self.circuit_breaker = CircuitBreaker("router", circuit_config)
        self.retry_policy = retry_policy or RetryPolicy()
        self.fallback_manager = FallbackManager()
        self.fallback_strategy = fallback_strategy
    
    def route(self, *args, **kwargs) -> Dict[str, Any]:
        """Route with full resilience patterns."""
        def resilient_route():
            return self.circuit_breaker.call(
                self.retry_policy.execute,
                self.base_router.route,
                *args, **kwargs
            )
        
        return self.fallback_manager.execute_with_fallback(
            resilient_route,
            self.fallback_strategy,
            *args, **kwargs
        )
    
    def get_resilience_stats(self) -> Dict[str, Any]:
        """Get comprehensive resilience statistics."""
        return {
            'circuit_breaker': self.circuit_breaker.get_stats(),
            'fallback_strategy': self.fallback_strategy
        }