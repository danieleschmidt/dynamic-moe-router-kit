"""
Advanced Validation and Error Handling for MoE Routing Systems

This module provides comprehensive validation, error handling, and resilience
mechanisms for advanced MoE routing algorithms to ensure production reliability.

Features:
- Input validation with detailed error messages
- Gradient flow validation for backpropagation
- Numerical stability checks
- Resource monitoring and limits
- Graceful degradation strategies
- Comprehensive logging and debugging

Author: Terry (Terragon Labs)
Research Period: 2024 Advanced MoE Validation Framework
"""

import logging
import time
import traceback
import warnings
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from dataclasses import dataclass

from .exceptions import (
    ComplexityEstimationError,
    ExpertDispatchError,
    RouterConfigurationError,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation and error handling."""
    
    # Input validation
    check_input_shapes: bool = True
    check_input_ranges: bool = True
    check_nan_inf: bool = True
    max_input_magnitude: float = 1e6
    min_input_magnitude: float = 1e-6
    
    # Numerical validation
    check_gradient_flow: bool = True
    gradient_clip_threshold: float = 10.0
    numerical_stability_eps: float = 1e-8
    
    # Resource validation
    max_memory_mb: float = 1000.0
    max_computation_time_sec: float = 30.0
    check_expert_load_balance: bool = True
    min_expert_utilization: float = 0.01
    
    # Error handling
    enable_graceful_degradation: bool = True
    fallback_routing_strategy: str = "uniform"
    max_retries: int = 3
    
    # Debugging
    verbose_validation: bool = False
    save_debug_info: bool = False
    debug_output_dir: str = "debug_output"


class InputValidator:
    """Comprehensive input validation for MoE routing."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
    def validate_input_tensor(
        self,
        inputs: np.ndarray,
        expected_shape: Optional[Tuple[int, ...]] = None,
        tensor_name: str = "input"
    ) -> Dict[str, Any]:
        """
        Validate input tensor with comprehensive checks.
        
        Args:
            inputs: Input tensor to validate
            expected_shape: Expected tensor shape (optional)
            tensor_name: Name for debugging
            
        Returns:
            validation_info: Detailed validation information
            
        Raises:
            ValueError: If validation fails
        """
        validation_info = {
            'tensor_name': tensor_name,
            'shape': inputs.shape,
            'dtype': str(inputs.dtype),
            'validation_passed': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Basic shape validation
            if self.config.check_input_shapes:
                self._validate_shape(inputs, expected_shape, validation_info)
                
            # Data type validation
            self._validate_dtype(inputs, validation_info)
            
            # NaN/Inf validation
            if self.config.check_nan_inf:
                self._validate_nan_inf(inputs, validation_info)
                
            # Range validation
            if self.config.check_input_ranges:
                self._validate_ranges(inputs, validation_info)
                
            # Statistical validation
            self._validate_statistics(inputs, validation_info)
            
        except Exception as e:
            validation_info['validation_passed'] = False
            validation_info['errors'].append(f"Validation failed: {str(e)}")
            
            if self.config.verbose_validation:
                logger.error(f"Input validation failed for {tensor_name}: {e}")
                logger.error(traceback.format_exc())
                
            raise ValueError(f"Input validation failed for {tensor_name}: {e}")
            
        # Log warnings
        if validation_info['warnings'] and self.config.verbose_validation:
            for warning in validation_info['warnings']:
                logger.warning(f"{tensor_name}: {warning}")
                
        return validation_info
        
    def _validate_shape(
        self,
        inputs: np.ndarray,
        expected_shape: Optional[Tuple[int, ...]],
        validation_info: Dict[str, Any]
    ):
        """Validate tensor shape."""
        if len(inputs.shape) < 2:
            raise ValueError(f"Input must have at least 2 dimensions, got {len(inputs.shape)}")
            
        if len(inputs.shape) > 4:
            validation_info['warnings'].append(
                f"Input has {len(inputs.shape)} dimensions, which is unusual for MoE routing"
            )
            
        if expected_shape is not None:
            if inputs.shape != expected_shape:
                # Allow flexible batch/sequence dimensions
                if len(inputs.shape) == len(expected_shape):
                    # Check if only batch/sequence dimensions differ
                    flexible_dims = [0, 1] if len(inputs.shape) >= 3 else [0]
                    fixed_dims_match = all(
                        inputs.shape[i] == expected_shape[i] 
                        for i in range(len(inputs.shape)) 
                        if i not in flexible_dims
                    )
                    
                    if not fixed_dims_match:
                        raise ValueError(
                            f"Shape mismatch: expected {expected_shape}, got {inputs.shape}"
                        )
                else:
                    raise ValueError(
                        f"Shape mismatch: expected {expected_shape}, got {inputs.shape}"
                    )
                    
        # Check for reasonable dimensions
        total_elements = np.prod(inputs.shape)
        if total_elements > 1e8:  # 100M elements
            validation_info['warnings'].append(
                f"Large tensor with {total_elements:,} elements may cause memory issues"
            )
            
    def _validate_dtype(self, inputs: np.ndarray, validation_info: Dict[str, Any]):
        """Validate data type."""
        if not np.issubdtype(inputs.dtype, np.floating):
            validation_info['warnings'].append(
                f"Non-floating point dtype {inputs.dtype} may cause precision issues"
            )
            
        if inputs.dtype == np.float16:
            validation_info['warnings'].append(
                "Float16 precision may be insufficient for stable routing"
            )
            
    def _validate_nan_inf(self, inputs: np.ndarray, validation_info: Dict[str, Any]):
        """Validate for NaN and Inf values."""
        nan_count = np.sum(np.isnan(inputs))
        inf_count = np.sum(np.isinf(inputs))
        
        if nan_count > 0:
            raise ValueError(f"Input contains {nan_count} NaN values")
            
        if inf_count > 0:
            raise ValueError(f"Input contains {inf_count} infinite values")
            
    def _validate_ranges(self, inputs: np.ndarray, validation_info: Dict[str, Any]):
        """Validate input value ranges."""
        min_val = np.min(inputs)
        max_val = np.max(inputs)
        
        if abs(max_val) > self.config.max_input_magnitude:
            validation_info['warnings'].append(
                f"Maximum value {max_val:.2e} exceeds recommended threshold "
                f"{self.config.max_input_magnitude:.2e}"
            )
            
        if abs(min_val) > self.config.max_input_magnitude:
            validation_info['warnings'].append(
                f"Minimum value {min_val:.2e} exceeds recommended threshold "
                f"{self.config.max_input_magnitude:.2e}"
            )
            
        # Check for very small values that might cause numerical issues
        non_zero_mask = inputs != 0
        if np.any(non_zero_mask):
            min_non_zero = np.min(np.abs(inputs[non_zero_mask]))
            if min_non_zero < self.config.min_input_magnitude:
                validation_info['warnings'].append(
                    f"Minimum non-zero magnitude {min_non_zero:.2e} may cause numerical instability"
                )
                
    def _validate_statistics(self, inputs: np.ndarray, validation_info: Dict[str, Any]):
        """Validate statistical properties."""
        mean_val = np.mean(inputs)
        std_val = np.std(inputs)
        
        validation_info['statistics'] = {
            'mean': float(mean_val),
            'std': float(std_val),
            'min': float(np.min(inputs)),
            'max': float(np.max(inputs)),
            'zeros_fraction': float(np.mean(inputs == 0))
        }
        
        # Check for suspicious statistics
        if std_val == 0:
            validation_info['warnings'].append("Input has zero variance (constant values)")
            
        if abs(mean_val) > 10 * std_val:
            validation_info['warnings'].append(
                f"Large mean/std ratio ({abs(mean_val)/std_val:.2f}) may indicate scaling issues"
            )
            
        zeros_fraction = np.mean(inputs == 0)
        if zeros_fraction > 0.9:
            validation_info['warnings'].append(
                f"Input is {zeros_fraction:.1%} zeros, which may be too sparse"
            )


class NumericalStabilityChecker:
    """Check numerical stability of routing computations."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
    def check_routing_stability(
        self,
        routing_logits: np.ndarray,
        routing_weights: np.ndarray,
        expert_indices: np.ndarray
    ) -> Dict[str, Any]:
        """
        Check numerical stability of routing computations.
        
        Args:
            routing_logits: Raw routing logits
            routing_weights: Normalized routing weights
            expert_indices: Selected expert indices
            
        Returns:
            stability_info: Detailed stability analysis
        """
        stability_info = {
            'stable': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check logits stability
        self._check_logits_stability(routing_logits, stability_info)
        
        # Check weights normalization
        self._check_weights_normalization(routing_weights, stability_info)
        
        # Check expert selection consistency
        self._check_expert_selection(expert_indices, stability_info)
        
        # Check for gradient flow issues
        if self.config.check_gradient_flow:
            self._check_gradient_flow_potential(routing_logits, routing_weights, stability_info)
            
        return stability_info
        
    def _check_logits_stability(self, logits: np.ndarray, stability_info: Dict[str, Any]):
        """Check stability of routing logits."""
        # Check for extreme values
        max_logit = np.max(logits)
        min_logit = np.min(logits)
        logit_range = max_logit - min_logit
        
        if logit_range > 50:  # exp(50) is very large
            stability_info['issues'].append(
                f"Large logit range ({logit_range:.2f}) may cause softmax overflow"
            )
            stability_info['recommendations'].append(
                "Consider logit clipping or temperature scaling"
            )
            
        # Check for saturated softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        exp_sum = np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Check if any dimension has near-zero sum (underflow)
        min_exp_sum = np.min(exp_sum)
        if min_exp_sum < 1e-30:
            stability_info['issues'].append(
                f"Softmax underflow detected (min sum: {min_exp_sum:.2e})"
            )
            stability_info['stable'] = False
            
    def _check_weights_normalization(self, weights: np.ndarray, stability_info: Dict[str, Any]):
        """Check if routing weights are properly normalized."""
        weight_sums = np.sum(weights, axis=-1)
        
        # Check normalization
        normalization_errors = np.abs(weight_sums - 1.0)
        max_error = np.max(normalization_errors)
        
        if max_error > 1e-4:
            stability_info['issues'].append(
                f"Poor weight normalization (max error: {max_error:.2e})"
            )
            
        # Check for negative weights
        negative_weights = np.sum(weights < 0)
        if negative_weights > 0:
            stability_info['issues'].append(
                f"Negative weights detected ({negative_weights} elements)"
            )
            stability_info['stable'] = False
            
    def _check_expert_selection(self, expert_indices: np.ndarray, stability_info: Dict[str, Any]):
        """Check expert selection consistency."""
        valid_mask = expert_indices >= 0
        
        if not np.any(valid_mask):
            stability_info['issues'].append("No valid experts selected")
            stability_info['stable'] = False
            return
            
        # Check for out-of-range indices
        if np.any(expert_indices[valid_mask] < 0):
            stability_info['issues'].append("Negative expert indices detected")
            stability_info['stable'] = False
            
        # Check expert utilization
        valid_indices = expert_indices[valid_mask]
        unique_experts = len(np.unique(valid_indices))
        total_experts = np.max(valid_indices) + 1 if len(valid_indices) > 0 else 0
        
        if total_experts > 0:
            utilization_ratio = unique_experts / total_experts
            if utilization_ratio < self.config.min_expert_utilization:
                stability_info['issues'].append(
                    f"Low expert utilization ({utilization_ratio:.2%})"
                )
                stability_info['recommendations'].append(
                    "Consider adjusting routing strategy for better load balancing"
                )
                
    def _check_gradient_flow_potential(
        self,
        logits: np.ndarray,
        weights: np.ndarray,
        stability_info: Dict[str, Any]
    ):
        """Check potential gradient flow issues."""
        # Check for saturated softmax (poor gradients)
        max_weights = np.max(weights, axis=-1)
        saturated_fraction = np.mean(max_weights > 0.99)
        
        if saturated_fraction > 0.5:
            stability_info['issues'].append(
                f"High softmax saturation ({saturated_fraction:.1%}) may impede gradient flow"
            )
            stability_info['recommendations'].append(
                "Consider temperature scaling or entropy regularization"
            )
            
        # Check weight entropy (diversity)
        eps = self.config.numerical_stability_eps
        entropies = -np.sum(weights * np.log(weights + eps), axis=-1)
        mean_entropy = np.mean(entropies)
        max_entropy = np.log(weights.shape[-1])
        
        normalized_entropy = mean_entropy / max_entropy
        if normalized_entropy < 0.1:
            stability_info['issues'].append(
                f"Low routing entropy ({normalized_entropy:.2f}) indicates poor exploration"
            )


class ResourceMonitor:
    """Monitor computational resources during MoE routing."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.start_time = None
        self.start_memory = None
        
    @contextmanager
    def monitor_resources(self, operation_name: str = "routing"):
        """Context manager for resource monitoring."""
        self.start_time = time.time()
        
        try:
            # Approximate memory tracking
            import psutil
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            self.start_memory = None
            
        resource_info = {'operation': operation_name}
        
        try:
            yield resource_info
        finally:
            # Compute resource usage
            end_time = time.time()
            elapsed_time = end_time - self.start_time
            
            resource_info['elapsed_time_sec'] = elapsed_time
            
            if self.start_memory is not None:
                try:
                    import psutil
                    process = psutil.Process()
                    end_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_delta = end_memory - self.start_memory
                    resource_info['memory_delta_mb'] = memory_delta
                    resource_info['peak_memory_mb'] = end_memory
                except ImportError:
                    pass
                    
            # Check limits
            self._check_resource_limits(resource_info)
            
    def _check_resource_limits(self, resource_info: Dict[str, Any]):
        """Check if resource usage exceeds limits."""
        elapsed_time = resource_info.get('elapsed_time_sec', 0)
        memory_delta = resource_info.get('memory_delta_mb', 0)
        
        if elapsed_time > self.config.max_computation_time_sec:
            warnings.warn(
                f"Operation '{resource_info['operation']}' took {elapsed_time:.2f}s, "
                f"exceeding limit of {self.config.max_computation_time_sec}s"
            )
            
        if memory_delta > self.config.max_memory_mb:
            warnings.warn(
                f"Operation '{resource_info['operation']}' used {memory_delta:.2f}MB, "
                f"exceeding limit of {self.config.max_memory_mb}MB"
            )


class GracefulDegradationHandler:
    """Handle graceful degradation when routing fails."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
    def handle_routing_failure(
        self,
        inputs: np.ndarray,
        num_experts: int,
        k: int = 2,
        error: Optional[Exception] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Handle routing failure with graceful degradation.
        
        Args:
            inputs: Input tensor
            num_experts: Total number of experts
            k: Number of experts to select
            error: Original error that caused failure
            
        Returns:
            expert_indices: Fallback expert selection
            expert_weights: Fallback expert weights
            fallback_info: Information about fallback strategy
        """
        batch_size, seq_len = inputs.shape[:2]
        
        fallback_info = {
            'fallback_triggered': True,
            'original_error': str(error) if error else None,
            'fallback_strategy': self.config.fallback_routing_strategy,
            'degraded_performance_expected': True
        }
        
        logger.warning(f"Routing failure handled with {self.config.fallback_routing_strategy} fallback")
        
        if self.config.fallback_routing_strategy == "uniform":
            # Uniform random selection
            expert_indices = np.random.randint(
                0, num_experts, (batch_size, seq_len, k)
            )
            expert_weights = np.full((batch_size, seq_len, k), 1.0/k, dtype=np.float32)
            
        elif self.config.fallback_routing_strategy == "round_robin":
            # Round-robin selection
            expert_indices = np.zeros((batch_size, seq_len, k), dtype=int)
            for b in range(batch_size):
                for s in range(seq_len):
                    start_expert = (b * seq_len + s) % num_experts
                    for i in range(k):
                        expert_indices[b, s, i] = (start_expert + i) % num_experts
                        
            expert_weights = np.full((batch_size, seq_len, k), 1.0/k, dtype=np.float32)
            
        elif self.config.fallback_routing_strategy == "first_k":
            # Always use first k experts
            expert_indices = np.zeros((batch_size, seq_len, k), dtype=int)
            for i in range(k):
                expert_indices[:, :, i] = i
                
            expert_weights = np.full((batch_size, seq_len, k), 1.0/k, dtype=np.float32)
            
        else:
            raise ValueError(f"Unknown fallback strategy: {self.config.fallback_routing_strategy}")
            
        return expert_indices, expert_weights, fallback_info


class AdvancedValidator:
    """Main advanced validation system for MoE routing."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.input_validator = InputValidator(config)
        self.stability_checker = NumericalStabilityChecker(config)
        self.resource_monitor = ResourceMonitor(config)
        self.degradation_handler = GracefulDegradationHandler(config)
        
    def validate_and_route(
        self,
        router: Any,
        inputs: np.ndarray,
        expected_input_shape: Optional[Tuple[int, ...]] = None,
        num_experts: Optional[int] = None,
        k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Validate inputs and perform robust routing with error handling.
        
        Args:
            router: Routing algorithm instance
            inputs: Input tensor
            expected_input_shape: Expected input shape
            num_experts: Number of experts (for fallback)
            k: Number of experts to select (for fallback)
            
        Returns:
            expert_indices: Selected expert indices
            expert_weights: Expert routing weights
            validation_info: Comprehensive validation information
        """
        validation_info = {
            'validation_passed': True,
            'routing_successful': True,
            'fallback_used': False,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Input validation
            with self.resource_monitor.monitor_resources("input_validation") as resource_info:
                input_validation = self.input_validator.validate_input_tensor(
                    inputs, expected_input_shape, "routing_input"
                )
                validation_info['input_validation'] = input_validation
                validation_info['input_validation_resources'] = resource_info
                
            # Routing with monitoring
            for attempt in range(self.config.max_retries + 1):
                try:
                    with self.resource_monitor.monitor_resources("routing") as routing_resources:
                        # Perform routing
                        if hasattr(router, 'route'):
                            result = router.route(inputs, return_routing_info=True)
                            if len(result) == 3:
                                expert_indices, expert_weights, routing_info = result
                            else:
                                expert_indices, expert_weights = result
                                routing_info = {}
                        else:
                            # Fallback for different router interfaces
                            expert_indices, expert_weights, routing_info = router(inputs)
                            
                    validation_info['routing_resources'] = routing_resources
                    validation_info['routing_info'] = routing_info
                    
                    # Validate routing results
                    if hasattr(router, 'num_experts'):
                        num_experts = router.num_experts
                    
                    stability_check = self.stability_checker.check_routing_stability(
                        routing_info.get('routing_logits', expert_weights),
                        expert_weights,
                        expert_indices
                    )
                    validation_info['stability_check'] = stability_check
                    
                    if not stability_check['stable']:
                        if attempt < self.config.max_retries:
                            validation_info['warnings'].append(
                                f"Routing unstable on attempt {attempt + 1}, retrying..."
                            )
                            continue
                        else:
                            raise RuntimeError("Routing failed stability check after all retries")
                            
                    # Success
                    break
                    
                except Exception as routing_error:
                    if attempt < self.config.max_retries:
                        validation_info['warnings'].append(
                            f"Routing attempt {attempt + 1} failed: {routing_error}"
                        )
                        continue
                    else:
                        # Use graceful degradation
                        if self.config.enable_graceful_degradation:
                            validation_info['routing_successful'] = False
                            validation_info['fallback_used'] = True
                            
                            # Determine fallback parameters
                            if num_experts is None:
                                num_experts = getattr(router, 'num_experts', 8)
                            if k is None:
                                k = getattr(router, 'max_experts', 2)
                                
                            expert_indices, expert_weights, fallback_info = (
                                self.degradation_handler.handle_routing_failure(
                                    inputs, num_experts, k, routing_error
                                )
                            )
                            validation_info['fallback_info'] = fallback_info
                            break
                        else:
                            raise routing_error
                            
        except Exception as e:
            validation_info['validation_passed'] = False
            validation_info['errors'].append(str(e))
            
            if self.config.verbose_validation:
                logger.error(f"Validation failed: {e}")
                logger.error(traceback.format_exc())
                
            raise
            
        # Final validation summary
        self._generate_validation_summary(validation_info)
        
        return expert_indices, expert_weights, validation_info
        
    def _generate_validation_summary(self, validation_info: Dict[str, Any]):
        """Generate comprehensive validation summary."""
        summary = {
            'overall_health': 'healthy',
            'performance_grade': 'A',
            'recommendation': 'No action required'
        }
        
        # Assess overall health
        if validation_info['fallback_used']:
            summary['overall_health'] = 'degraded'
            summary['performance_grade'] = 'C'
            summary['recommendation'] = 'Investigate routing failures'
        elif validation_info.get('stability_check', {}).get('issues'):
            summary['overall_health'] = 'unstable'
            summary['performance_grade'] = 'B'
            summary['recommendation'] = 'Address stability issues'
        elif validation_info['warnings']:
            summary['overall_health'] = 'cautionary'
            summary['performance_grade'] = 'B+'
            summary['recommendation'] = 'Monitor warnings'
            
        validation_info['summary'] = summary
        
        # Log summary if verbose
        if self.config.verbose_validation:
            logger.info(f"Validation Summary: {summary['overall_health']} "
                       f"(Grade: {summary['performance_grade']})")
            if summary['recommendation'] != 'No action required':
                logger.info(f"Recommendation: {summary['recommendation']}")


# Export main classes
__all__ = [
    'ValidationConfig',
    'InputValidator',
    'NumericalStabilityChecker',
    'ResourceMonitor',
    'GracefulDegradationHandler',
    'AdvancedValidator'
]