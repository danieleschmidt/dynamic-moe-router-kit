"""Resilience patterns for dynamic MoE routing."""

import logging
import time
import threading
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np

from .exceptions import ExpertDispatchError, BackendError, ProfilingError
from .security import get_security_monitor

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
        
        # Security integration
        self._security_monitor = get_security_monitor()
    
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
                    self._security_monitor.log_event(
                        'circuit_breaker_block',
                        'medium',
                        f'Circuit breaker {self.name} blocked request'
                    )
                    raise ExpertDispatchError(f"Circuit breaker {self.name} is OPEN")
        
        # Execute the function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            self._on_success(duration)
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self._on_failure(e, duration)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        return time.time() - self._last_failure_time > self.config.recovery_timeout
    
    def _on_success(self, duration: float) -> None:
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
    
    def _on_failure(self, error: Exception, duration: float) -> None:
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            # Log security event
            self._security_monitor.log_event(
                'circuit_breaker_failure',
                'medium',
                f'Circuit breaker {self.name} recorded failure: {str(error)}'
            )
            
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
            }\n\n\nclass RetryPolicy:\n    \"\"\"Retry policy with exponential backoff.\"\"\"\n    \n    def __init__(self,\n                 max_retries: int = 3,\n                 base_delay: float = 0.1,\n                 max_delay: float = 10.0,\n                 exponential_base: float = 2.0,\n                 jitter: bool = True):\n        self.max_retries = max_retries\n        self.base_delay = base_delay\n        self.max_delay = max_delay\n        self.exponential_base = exponential_base\n        self.jitter = jitter\n    \n    def execute(self, func: Callable, *args, **kwargs) -> Any:\n        \"\"\"Execute function with retry logic.\"\"\"\n        last_exception = None\n        \n        for attempt in range(self.max_retries + 1):\n            try:\n                return func(*args, **kwargs)\n            except Exception as e:\n                last_exception = e\n                \n                if attempt == self.max_retries:\n                    break\n                \n                # Calculate delay\n                delay = min(\n                    self.base_delay * (self.exponential_base ** attempt),\n                    self.max_delay\n                )\n                \n                # Add jitter to prevent thundering herd\n                if self.jitter:\n                    delay *= (0.5 + 0.5 * np.random.random())\n                \n                logger.debug(f\"Retry attempt {attempt + 1}/{self.max_retries} after {delay:.3f}s\")\n                time.sleep(delay)\n        \n        raise last_exception\n\n\nclass FallbackManager:\n    \"\"\"Manages fallback strategies for failed routing.\"\"\"\n    \n    def __init__(self):\n        self.fallback_strategies = {\n            'random_expert': self._random_expert_fallback,\n            'single_expert': self._single_expert_fallback,\n            'load_balance': self._load_balance_fallback\n        }\n        self._security_monitor = get_security_monitor()\n    \n    def execute_with_fallback(self,\n                            primary_func: Callable,\n                            fallback_strategy: str,\n                            *args, **kwargs) -> Any:\n        \"\"\"Execute primary function with fallback on failure.\"\"\"\n        try:\n            return primary_func(*args, **kwargs)\n        except Exception as primary_error:\n            logger.warning(f\"Primary routing failed: {primary_error}, using fallback\")\n            \n            self._security_monitor.log_event(\n                'fallback_activated',\n                'medium',\n                f'Primary routing failed, using {fallback_strategy} fallback'\n            )\n            \n            if fallback_strategy not in self.fallback_strategies:\n                raise ValueError(f\"Unknown fallback strategy: {fallback_strategy}\")\n            \n            try:\n                return self.fallback_strategies[fallback_strategy](*args, **kwargs)\n            except Exception as fallback_error:\n                logger.error(f\"Fallback also failed: {fallback_error}\")\n                raise ExpertDispatchError(\n                    f\"Both primary and fallback routing failed: \"\n                    f\"primary={primary_error}, fallback={fallback_error}\"\n                )\n    \n    def _random_expert_fallback(self, hidden_states, num_experts: int = 8, **kwargs) -> Dict[str, Any]:\n        \"\"\"Fallback to random expert selection.\"\"\"\n        batch_size, seq_len, hidden_dim = hidden_states.shape\n        \n        # Select 1-2 random experts per token\n        expert_indices = np.random.randint(0, num_experts, size=(batch_size, seq_len, 2))\n        expert_weights = np.random.dirichlet([1, 1], size=(batch_size, seq_len))\n        \n        return {\n            'expert_indices': expert_indices,\n            'expert_weights': expert_weights,\n            'num_experts_per_token': np.full((batch_size, seq_len), 2),\n            'complexity_scores': np.full((batch_size, seq_len), 0.5),\n            'routing_info': {\n                'avg_experts_per_token': 2.0,\n                'flop_reduction': 0.75,  # Using 2/8 experts\n                'expert_utilization': [0.125] * num_experts,\n                'fallback_used': True\n            }\n        }\n    \n    def _single_expert_fallback(self, hidden_states, num_experts: int = 8, **kwargs) -> Dict[str, Any]:\n        \"\"\"Fallback to single expert (expert 0).\"\"\"\n        batch_size, seq_len, hidden_dim = hidden_states.shape\n        \n        expert_indices = np.zeros((batch_size, seq_len, 1), dtype=int)\n        expert_weights = np.ones((batch_size, seq_len, 1))\n        \n        return {\n            'expert_indices': expert_indices,\n            'expert_weights': expert_weights,\n            'num_experts_per_token': np.ones((batch_size, seq_len)),\n            'complexity_scores': np.zeros((batch_size, seq_len)),\n            'routing_info': {\n                'avg_experts_per_token': 1.0,\n                'flop_reduction': 0.875,  # Using 1/8 experts\n                'expert_utilization': [1.0] + [0.0] * (num_experts - 1),\n                'fallback_used': True\n            }\n        }\n    \n    def _load_balance_fallback(self, hidden_states, num_experts: int = 8, **kwargs) -> Dict[str, Any]:\n        \"\"\"Fallback to round-robin load balancing.\"\"\"\n        batch_size, seq_len, hidden_dim = hidden_states.shape\n        total_tokens = batch_size * seq_len\n        \n        # Round-robin assignment\n        expert_indices = np.zeros((batch_size, seq_len, 1), dtype=int)\n        for b in range(batch_size):\n            for s in range(seq_len):\n                token_idx = b * seq_len + s\n                expert_indices[b, s, 0] = token_idx % num_experts\n        \n        expert_weights = np.ones((batch_size, seq_len, 1))\n        \n        return {\n            'expert_indices': expert_indices,\n            'expert_weights': expert_weights,\n            'num_experts_per_token': np.ones((batch_size, seq_len)),\n            'complexity_scores': np.full((batch_size, seq_len), 0.5),\n            'routing_info': {\n                'avg_experts_per_token': 1.0,\n                'flop_reduction': 0.875,  # Using 1/8 experts\n                'expert_utilization': [1.0 / num_experts] * num_experts,\n                'fallback_used': True\n            }\n        }\n\n\nclass ResilientRouter:\n    \"\"\"Router with built-in resilience patterns.\"\"\"\n    \n    def __init__(self,\n                 base_router: Any,\n                 circuit_config: Optional[CircuitConfig] = None,\n                 retry_policy: Optional[RetryPolicy] = None,\n                 fallback_strategy: str = \"load_balance\"):\n        self.base_router = base_router\n        self.circuit_breaker = CircuitBreaker(\"router\", circuit_config)\n        self.retry_policy = retry_policy or RetryPolicy()\n        self.fallback_manager = FallbackManager()\n        self.fallback_strategy = fallback_strategy\n    \n    def route(self, *args, **kwargs) -> Dict[str, Any]:\n        \"\"\"Route with full resilience patterns.\"\"\"\n        def resilient_route():\n            return self.circuit_breaker.call(\n                self.retry_policy.execute,\n                self.base_router.route,\n                *args, **kwargs\n            )\n        \n        return self.fallback_manager.execute_with_fallback(\n            resilient_route,\n            self.fallback_strategy,\n            *args, **kwargs\n        )\n    \n    def get_resilience_stats(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive resilience statistics.\"\"\"\n        return {\n            'circuit_breaker': self.circuit_breaker.get_stats(),\n            'fallback_strategy': self.fallback_strategy,\n            'security_summary': get_security_monitor().get_security_summary()\n        }\n