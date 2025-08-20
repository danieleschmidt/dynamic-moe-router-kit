"""Robust router with comprehensive error handling and validation."""

import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np

from .router import DynamicRouter
from .resilience_fixed import ResilientRouter, CircuitConfig, RetryPolicy
from .exceptions import (
    DynamicMoEError, RouterConfigurationError, ExpertDispatchError,
    ValidationError, PerformanceError
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for input validation."""
    max_batch_size: int = 1024
    max_sequence_length: int = 8192
    min_input_dim: int = 1
    max_input_dim: int = 16384
    validate_numerical_stability: bool = True
    check_inf_nan: bool = True
    memory_limit_mb: float = 8000.0


class InputValidator:
    """Validates and sanitizes router inputs."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
    def validate_input(self, hidden_states: Any) -> Tuple[bool, Optional[str]]:
        """Validate input tensor. Returns (is_valid, error_message)."""
        try:
            if not isinstance(hidden_states, np.ndarray):
                return False, f"Expected numpy array, got {type(hidden_states)}"
            
            # Check shape
            if len(hidden_states.shape) != 3:
                return False, f"Expected 3D tensor [batch, seq, dim], got shape {hidden_states.shape}"
            
            batch_size, seq_len, input_dim = hidden_states.shape
            
            # Validate dimensions
            if batch_size > self.config.max_batch_size:
                return False, f"Batch size {batch_size} exceeds limit {self.config.max_batch_size}"
            
            if seq_len > self.config.max_sequence_length:
                return False, f"Sequence length {seq_len} exceeds limit {self.config.max_sequence_length}"
            
            if input_dim < self.config.min_input_dim or input_dim > self.config.max_input_dim:
                return False, f"Input dim {input_dim} outside valid range [{self.config.min_input_dim}, {self.config.max_input_dim}]"
            
            # Check memory usage
            memory_mb = hidden_states.nbytes / (1024 * 1024)
            if memory_mb > self.config.memory_limit_mb:
                return False, f"Input tensor requires {memory_mb:.1f}MB, exceeds limit {self.config.memory_limit_mb}MB"
            
            # Check for NaN/Inf
            if self.config.check_inf_nan:
                if np.any(np.isnan(hidden_states)):
                    return False, "Input contains NaN values"
                if np.any(np.isinf(hidden_states)):
                    return False, "Input contains infinite values"
            
            # Numerical stability checks
            if self.config.validate_numerical_stability:
                # Check for extreme values
                abs_max = np.abs(hidden_states).max()
                if abs_max > 1e6:
                    return False, f"Input contains extreme values (max abs value: {abs_max})"
                
                # Check for very small values that might cause underflow
                non_zero_min = np.abs(hidden_states[hidden_states != 0]).min() if np.any(hidden_states != 0) else 1.0
                if non_zero_min < 1e-8:
                    return False, f"Input contains very small values that may cause numerical instability"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def sanitize_input(self, hidden_states: np.ndarray) -> np.ndarray:
        """Sanitize input by clipping extreme values."""
        # Clip extreme values
        clipped = np.clip(hidden_states, -1e6, 1e6)
        
        # Replace NaN/Inf with zeros
        clipped = np.where(np.isfinite(clipped), clipped, 0.0)
        
        return clipped


class ErrorRecoveryHandler:
    """Handles error recovery and graceful degradation."""
    
    def __init__(self, num_experts: int = 8):
        self.num_experts = num_experts
        self.fallback_strategies = {
            'simple_uniform': self._uniform_fallback,
            'single_expert': self._single_expert_fallback,
            'random_selection': self._random_fallback
        }
    
    def recover_from_error(self, 
                          hidden_states: np.ndarray, 
                          error: Exception,
                          strategy: str = 'simple_uniform') -> Dict[str, Any]:
        """Attempt to recover from routing error with fallback strategy."""
        batch_size, seq_len, input_dim = hidden_states.shape
        
        logger.warning(f"Recovering from error: {error}. Using fallback strategy: {strategy}")
        
        if strategy not in self.fallback_strategies:
            strategy = 'simple_uniform'
        
        try:
            return self.fallback_strategies[strategy](batch_size, seq_len)
        except Exception as fallback_error:
            logger.error(f"Fallback strategy {strategy} also failed: {fallback_error}")
            # Ultimate fallback - single expert
            return self._single_expert_fallback(batch_size, seq_len)
    
    def _uniform_fallback(self, batch_size: int, seq_len: int) -> Dict[str, Any]:
        """Uniform distribution across experts."""
        experts_per_token = 2  # Use 2 experts per token
        expert_indices = np.random.choice(
            self.num_experts, 
            size=(batch_size, seq_len, experts_per_token),
            replace=True
        )
        expert_weights = np.ones((batch_size, seq_len, experts_per_token)) / experts_per_token
        
        return {
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'num_experts_per_token': np.full((batch_size, seq_len), experts_per_token),
            'complexity_scores': np.full((batch_size, seq_len), 0.5),
            'routing_info': {
                'avg_experts_per_token': float(experts_per_token),
                'flop_reduction': 1.0 - (experts_per_token / self.num_experts),
                'expert_utilization': [1.0 / self.num_experts] * self.num_experts,
                'fallback_used': True,
                'fallback_strategy': 'simple_uniform'
            }
        }
    
    def _single_expert_fallback(self, batch_size: int, seq_len: int) -> Dict[str, Any]:
        """Route everything to expert 0."""
        expert_indices = np.zeros((batch_size, seq_len, 1), dtype=int)
        expert_weights = np.ones((batch_size, seq_len, 1))
        
        return {
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'num_experts_per_token': np.ones((batch_size, seq_len)),
            'complexity_scores': np.zeros((batch_size, seq_len)),
            'routing_info': {
                'avg_experts_per_token': 1.0,
                'flop_reduction': 1.0 - (1.0 / self.num_experts),
                'expert_utilization': [1.0] + [0.0] * (self.num_experts - 1),
                'fallback_used': True,
                'fallback_strategy': 'single_expert'
            }
        }
    
    def _random_fallback(self, batch_size: int, seq_len: int) -> Dict[str, Any]:
        """Random expert selection."""
        experts_per_token = np.random.randint(1, min(4, self.num_experts) + 1, size=(batch_size, seq_len))
        max_experts = experts_per_token.max()
        
        expert_indices = np.zeros((batch_size, seq_len, max_experts), dtype=int)
        expert_weights = np.zeros((batch_size, seq_len, max_experts))
        
        for b in range(batch_size):
            for s in range(seq_len):
                k = experts_per_token[b, s]
                indices = np.random.choice(self.num_experts, size=k, replace=False)
                weights = np.random.dirichlet([1.0] * k)
                
                expert_indices[b, s, :k] = indices
                expert_weights[b, s, :k] = weights
        
        return {
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'num_experts_per_token': experts_per_token,
            'complexity_scores': np.random.uniform(0.3, 0.7, size=(batch_size, seq_len)),
            'routing_info': {
                'avg_experts_per_token': float(experts_per_token.mean()),
                'flop_reduction': 1.0 - (experts_per_token.mean() / self.num_experts),
                'expert_utilization': [1.0 / self.num_experts] * self.num_experts,
                'fallback_used': True,
                'fallback_strategy': 'random_selection'
            }
        }


class RobustRouter:
    """Production-grade robust router with comprehensive error handling."""
    
    def __init__(self,
                 input_dim: int,
                 num_experts: int,
                 min_experts: int = 1,
                 max_experts: int = 4,
                 complexity_estimator: str = "gradient_norm",
                 validation_config: Optional[ValidationConfig] = None,
                 enable_resilience: bool = True,
                 fallback_strategy: str = 'simple_uniform'):
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.min_experts = min_experts
        self.max_experts = max_experts
        self.fallback_strategy = fallback_strategy
        
        # Validation
        self.validation_config = validation_config or ValidationConfig()
        self.input_validator = InputValidator(self.validation_config)
        
        # Error recovery
        self.error_handler = ErrorRecoveryHandler(num_experts)
        
        # Core router
        try:
            self.core_router = DynamicRouter(
                input_dim=input_dim,
                num_experts=num_experts,
                min_experts=min_experts,
                max_experts=max_experts,
                complexity_estimator=complexity_estimator
            )
            
            # Resilience wrapper
            if enable_resilience:
                circuit_config = CircuitConfig(
                    failure_threshold=3,
                    recovery_timeout=30.0,
                    success_threshold=2
                )
                retry_policy = RetryPolicy(max_retries=2, base_delay=0.1)
                
                self.resilient_router = ResilientRouter(
                    self.core_router,
                    circuit_config=circuit_config,
                    retry_policy=retry_policy
                )
            else:
                self.resilient_router = None
                
        except Exception as e:
            logger.error(f"Failed to initialize router: {e}")
            raise RouterConfigurationError(f"Router initialization failed: {e}")
        
        # Metrics
        self._total_requests = 0
        self._error_count = 0
        self._fallback_count = 0
        self._validation_errors = 0
    
    def route(self, hidden_states: Any, **kwargs) -> Dict[str, Any]:
        """Robust routing with comprehensive error handling."""
        request_start = time.time()
        self._total_requests += 1
        
        try:
            # Input validation
            is_valid, error_msg = self.input_validator.validate_input(hidden_states)
            if not is_valid:
                self._validation_errors += 1
                logger.warning(f"Input validation failed: {error_msg}")
                
                # Try to sanitize and continue
                try:
                    hidden_states = self.input_validator.sanitize_input(hidden_states)
                    logger.info("Input sanitized successfully")
                except Exception as sanitize_error:
                    raise ValidationError(f"Input validation failed and sanitization unsuccessful: {error_msg}")
            
            # Attempt routing
            try:
                if self.resilient_router:
                    result = self.resilient_router.route(hidden_states, **kwargs)
                else:
                    result = self.core_router.route(hidden_states, **kwargs)
                
                # Add robustness metadata
                result['robustness_info'] = {
                    'validation_passed': is_valid,
                    'sanitization_applied': not is_valid,
                    'processing_time_ms': (time.time() - request_start) * 1000,
                    'resilience_used': self.resilient_router is not None,
                    'total_requests': self._total_requests,
                    'error_rate': self._error_count / self._total_requests,
                    'fallback_rate': self._fallback_count / self._total_requests
                }
                
                return result
                
            except Exception as routing_error:
                # Routing failed, use error recovery
                self._error_count += 1
                self._fallback_count += 1
                
                logger.error(f"Routing failed: {routing_error}")
                logger.debug(f"Routing error traceback: {traceback.format_exc()}")
                
                result = self.error_handler.recover_from_error(
                    hidden_states, routing_error, self.fallback_strategy
                )
                
                # Add error recovery metadata
                result['robustness_info'] = {
                    'validation_passed': is_valid,
                    'sanitization_applied': not is_valid,
                    'processing_time_ms': (time.time() - request_start) * 1000,
                    'error_recovery_used': True,
                    'original_error': str(routing_error),
                    'fallback_strategy': self.fallback_strategy,
                    'total_requests': self._total_requests,
                    'error_rate': self._error_count / self._total_requests,
                    'fallback_rate': self._fallback_count / self._total_requests
                }
                
                return result
                
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Unexpected error - log and re-raise
            self._error_count += 1
            logger.error(f"Unexpected error in robust router: {e}")
            logger.debug(f"Unexpected error traceback: {traceback.format_exc()}")
            raise DynamicMoEError(f"Robust router failed: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        if self._total_requests == 0:
            return {
                'status': 'ready',
                'total_requests': 0,
                'error_rate': 0.0,
                'fallback_rate': 0.0,
                'validation_error_rate': 0.0
            }
        
        error_rate = self._error_count / self._total_requests
        fallback_rate = self._fallback_count / self._total_requests
        validation_error_rate = self._validation_errors / self._total_requests
        
        # Determine health status
        if error_rate > 0.5:
            status = 'critical'
        elif error_rate > 0.2 or fallback_rate > 0.3:
            status = 'degraded'
        elif validation_error_rate > 0.1:
            status = 'warning'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'total_requests': self._total_requests,
            'error_count': self._error_count,
            'error_rate': error_rate,
            'fallback_count': self._fallback_count,
            'fallback_rate': fallback_rate,
            'validation_errors': self._validation_errors,
            'validation_error_rate': validation_error_rate,
            'resilience_enabled': self.resilient_router is not None
        }
    
    def reset_metrics(self):
        """Reset internal metrics."""
        self._total_requests = 0
        self._error_count = 0
        self._fallback_count = 0
        self._validation_errors = 0
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get router configuration."""
        return {
            'input_dim': self.input_dim,
            'num_experts': self.num_experts,
            'min_experts': self.min_experts,
            'max_experts': self.max_experts,
            'fallback_strategy': self.fallback_strategy,
            'validation_config': {
                'max_batch_size': self.validation_config.max_batch_size,
                'max_sequence_length': self.validation_config.max_sequence_length,
                'memory_limit_mb': self.validation_config.memory_limit_mb,
                'validate_numerical_stability': self.validation_config.validate_numerical_stability
            },
            'resilience_enabled': self.resilient_router is not None
        }