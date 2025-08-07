"""Input validation utilities for dynamic MoE routing."""

import logging
from typing import Any, Dict, Tuple, Optional, Union
import warnings

import numpy as np

from .exceptions import ValidationError, RouterConfigurationError

logger = logging.getLogger(__name__)


def validate_tensor_shape(
    tensor: Any, 
    expected_dims: int, 
    min_shape: Optional[Tuple[int, ...]] = None,
    max_shape: Optional[Tuple[int, ...]] = None,
    name: str = "tensor"
) -> None:
    """Validate tensor shape and dimensions.
    
    Args:
        tensor: Input tensor to validate
        expected_dims: Expected number of dimensions
        min_shape: Minimum allowed shape (optional)
        max_shape: Maximum allowed shape (optional)
        name: Name for error messages
        
    Raises:
        ValidationError: If tensor shape is invalid
    """
    if not hasattr(tensor, 'shape'):
        raise ValidationError(f"{name} must have a 'shape' attribute")
    
    if len(tensor.shape) != expected_dims:
        raise ValidationError(
            f"{name} must have {expected_dims} dimensions, got {len(tensor.shape)}"
        )
    
    if min_shape is not None:
        for i, (actual, min_val) in enumerate(zip(tensor.shape, min_shape)):
            if actual < min_val:
                raise ValidationError(
                    f"{name} dimension {i} must be >= {min_val}, got {actual}"
                )
    
    if max_shape is not None:
        for i, (actual, max_val) in enumerate(zip(tensor.shape, max_shape)):
            if max_val > 0 and actual > max_val:
                raise ValidationError(
                    f"{name} dimension {i} must be <= {max_val}, got {actual}"
                )


def validate_router_config(
    input_dim: int,
    num_experts: int,
    min_experts: int,
    max_experts: int,
    **kwargs
) -> None:
    """Validate router configuration parameters.
    
    Args:
        input_dim: Input dimension
        num_experts: Number of experts
        min_experts: Minimum experts per token
        max_experts: Maximum experts per token
        **kwargs: Additional configuration parameters
        
    Raises:
        RouterConfigurationError: If configuration is invalid
    """
    # Validate basic parameters
    if input_dim <= 0:
        raise RouterConfigurationError(f"input_dim must be positive, got {input_dim}")
    
    if num_experts <= 0:
        raise RouterConfigurationError(f"num_experts must be positive, got {num_experts}")
    
    if min_experts < 1:
        raise RouterConfigurationError(f"min_experts must be >= 1, got {min_experts}")
    
    if min_experts > num_experts:
        raise RouterConfigurationError(
            f"min_experts ({min_experts}) cannot be greater than num_experts ({num_experts})"
        )
    
    if max_experts < min_experts:
        raise RouterConfigurationError(
            f"max_experts ({max_experts}) cannot be less than min_experts ({min_experts})"
        )
    
    if max_experts > num_experts:
        raise RouterConfigurationError(
            f"max_experts ({max_experts}) cannot be greater than num_experts ({num_experts})"
        )
    
    # Validate optional parameters
    if 'noise_factor' in kwargs:
        noise_factor = kwargs['noise_factor']
        if noise_factor < 0:
            raise RouterConfigurationError(f"noise_factor must be >= 0, got {noise_factor}")
    
    if 'expert_capacity_factor' in kwargs:
        capacity_factor = kwargs['expert_capacity_factor']
        if capacity_factor <= 0:
            raise RouterConfigurationError(
                f"expert_capacity_factor must be positive, got {capacity_factor}"
            )
    
    if 'dropout_rate' in kwargs:
        dropout_rate = kwargs['dropout_rate']
        if not (0 <= dropout_rate < 1):
            raise RouterConfigurationError(
                f"dropout_rate must be in [0, 1), got {dropout_rate}"
            )


def validate_complexity_scores(complexity_scores: Any, name: str = "complexity_scores") -> None:
    """Validate complexity scores.
    
    Args:
        complexity_scores: Complexity scores to validate
        name: Name for error messages
        
    Raises:
        ValidationError: If complexity scores are invalid
    """
    validate_tensor_shape(complexity_scores, expected_dims=2, name=name)
    
    # Check for NaN or infinite values
    if hasattr(complexity_scores, 'isnan'):
        # Handle PyTorch tensors
        if complexity_scores.isnan().any():
            raise ValidationError(f"{name} contains NaN values")
        if complexity_scores.isinf().any():
            raise ValidationError(f"{name} contains infinite values")
    elif hasattr(complexity_scores, 'dtype') and np.issubdtype(complexity_scores.dtype, np.floating):
        # Handle NumPy arrays
        if np.isnan(complexity_scores).any():
            raise ValidationError(f"{name} contains NaN values")
        if np.isinf(complexity_scores).any():
            raise ValidationError(f"{name} contains infinite values")
    
    # Check value range (should be in [0, 1] for normalized complexity)
    if hasattr(complexity_scores, 'min') and hasattr(complexity_scores, 'max'):
        min_val = float(complexity_scores.min())
        max_val = float(complexity_scores.max())
        
        if min_val < 0:
            warnings.warn(f"{name} contains negative values (min: {min_val})")
        
        if max_val > 1:
            warnings.warn(f"{name} contains values > 1 (max: {max_val})")


def validate_expert_indices(
    expert_indices: Any,
    num_experts: int,
    name: str = "expert_indices"
) -> None:
    """Validate expert indices.
    
    Args:
        expert_indices: Expert indices to validate
        num_experts: Total number of experts
        name: Name for error messages
        
    Raises:
        ValidationError: If expert indices are invalid
    """
    validate_tensor_shape(expert_indices, expected_dims=3, name=name)
    
    # Check index range
    if hasattr(expert_indices, 'min') and hasattr(expert_indices, 'max'):
        min_idx = int(expert_indices.min())
        max_idx = int(expert_indices.max())
        
        if min_idx < -1:  # -1 is allowed for padding
            raise ValidationError(
                f"{name} contains invalid indices < -1 (min: {min_idx})"
            )
        
        if max_idx >= num_experts:
            raise ValidationError(
                f"{name} contains indices >= num_experts ({num_experts}) (max: {max_idx})"
            )


def validate_expert_weights(
    expert_weights: Any,
    expert_indices: Optional[Any] = None,
    name: str = "expert_weights"
) -> None:
    """Validate expert weights.
    
    Args:
        expert_weights: Expert weights to validate
        expert_indices: Corresponding expert indices (optional)
        name: Name for error messages
        
    Raises:
        ValidationError: If expert weights are invalid
    """
    validate_tensor_shape(expert_weights, expected_dims=3, name=name)
    
    # Check weight range [0, 1]
    if hasattr(expert_weights, 'min') and hasattr(expert_weights, 'max'):
        min_weight = float(expert_weights.min())
        max_weight = float(expert_weights.max())
        
        if min_weight < 0:
            raise ValidationError(
                f"{name} contains negative weights (min: {min_weight})"
            )
        
        if max_weight > 1:
            raise ValidationError(
                f"{name} contains weights > 1 (max: {max_weight})"
            )
    
    # Check for NaN values
    if hasattr(expert_weights, 'isnan'):
        if expert_weights.isnan().any():
            raise ValidationError(f"{name} contains NaN values")
    elif hasattr(expert_weights, 'dtype') and np.issubdtype(expert_weights.dtype, np.floating):
        if np.isnan(expert_weights).any():
            raise ValidationError(f"{name} contains NaN values")
    
    # Validate normalization (weights should sum to ~1 for each token)
    if expert_indices is not None:
        _validate_weight_normalization(expert_weights, expert_indices, name)


def _validate_weight_normalization(
    expert_weights: Any,
    expert_indices: Any,
    name: str,
    tolerance: float = 1e-5
) -> None:
    """Validate that expert weights are properly normalized."""
    try:
        if hasattr(expert_weights, 'numpy'):
            # PyTorch tensor
            weights_np = expert_weights.detach().cpu().numpy()
            indices_np = expert_indices.detach().cpu().numpy()
        else:
            # NumPy array
            weights_np = expert_weights
            indices_np = expert_indices
        
        batch_size, seq_len, max_k = weights_np.shape
        
        for b in range(batch_size):
            for s in range(seq_len):
                # Get valid weights for this token
                token_indices = indices_np[b, s]
                token_weights = weights_np[b, s]
                
                # Only consider non-padding entries
                valid_mask = token_indices >= 0
                if valid_mask.any():
                    valid_weights = token_weights[valid_mask]
                    weight_sum = np.sum(valid_weights)
                    
                    if abs(weight_sum - 1.0) > tolerance and weight_sum > 0:
                        warnings.warn(
                            f"{name} not properly normalized at position [{b}, {s}]: "
                            f"sum = {weight_sum:.6f}"
                        )
    except Exception as e:
        logger.warning(f"Could not validate weight normalization: {e}")


def validate_profiler_inputs(
    hidden_states_shape: Tuple[int, ...],
    num_experts: int,
    routing_result: Dict[str, Any]
) -> None:
    """Validate inputs for profiler operations.
    
    Args:
        hidden_states_shape: Shape of hidden states
        num_experts: Number of experts
        routing_result: Routing result dictionary
        
    Raises:
        ValidationError: If inputs are invalid
    """
    if len(hidden_states_shape) != 3:
        raise ValidationError(
            f"hidden_states_shape must have 3 elements, got {len(hidden_states_shape)}"
        )
    
    batch_size, seq_len, hidden_dim = hidden_states_shape
    
    if batch_size <= 0 or seq_len <= 0 or hidden_dim <= 0:
        raise ValidationError(
            f"All dimensions must be positive: {hidden_states_shape}"
        )
    
    if num_experts <= 0:
        raise ValidationError(f"num_experts must be positive, got {num_experts}")
    
    # Validate routing result structure
    required_keys = ['expert_indices', 'expert_weights', 'num_experts_per_token']
    for key in required_keys:
        if key not in routing_result:
            raise ValidationError(f"routing_result missing required key: {key}")


def validate_moe_layer_config(
    expert_capacity_factor: float,
    dropout_rate: float,
    **kwargs
) -> None:
    """Validate MoE layer configuration.
    
    Args:
        expert_capacity_factor: Capacity factor for experts
        dropout_rate: Dropout rate
        **kwargs: Additional configuration
        
    Raises:
        RouterConfigurationError: If configuration is invalid
    """
    if expert_capacity_factor <= 0:
        raise RouterConfigurationError(
            f"expert_capacity_factor must be positive, got {expert_capacity_factor}"
        )
    
    if not (0 <= dropout_rate < 1):
        raise RouterConfigurationError(
            f"dropout_rate must be in [0, 1), got {dropout_rate}"
        )


def validate_adaptation_parameters(
    adaptation_rate: float,
    performance_score: Optional[float] = None
) -> None:
    """Validate adaptive routing parameters.
    
    Args:
        adaptation_rate: Learning rate for adaptation
        performance_score: Performance score (optional)
        
    Raises:
        RouterConfigurationError: If parameters are invalid
    """
    if not (0 < adaptation_rate <= 1):
        raise RouterConfigurationError(
            f"adaptation_rate must be in (0, 1], got {adaptation_rate}"
        )
    
    if performance_score is not None and not (0 <= performance_score <= 1):
        warnings.warn(
            f"performance_score typically should be in [0, 1], got {performance_score}"
        )


def check_memory_usage(tensor: Any, name: str, max_size_mb: float = 1000) -> None:
    """Check if tensor memory usage is reasonable.
    
    Args:
        tensor: Tensor to check
        name: Name for warnings
        max_size_mb: Maximum size in MB before warning
    """
    try:
        if hasattr(tensor, 'numel') and hasattr(tensor, 'element_size'):
            # PyTorch tensor
            size_bytes = tensor.numel() * tensor.element_size()
        elif hasattr(tensor, 'nbytes'):
            # NumPy array
            size_bytes = tensor.nbytes
        else:
            return  # Can't determine size
        
        size_mb = size_bytes / (1024 * 1024)
        
        if size_mb > max_size_mb:
            warnings.warn(
                f"{name} is using {size_mb:.1f} MB of memory, "
                f"which exceeds recommended limit of {max_size_mb} MB"
            )
    
    except Exception:
        # Silently continue if we can't determine memory usage
        pass


def sanitize_routing_kwargs(**kwargs) -> Dict[str, Any]:
    """Sanitize and validate routing keyword arguments.
    
    Args:
        **kwargs: Routing arguments to sanitize
        
    Returns:
        Sanitized keyword arguments
        
    Raises:
        ValidationError: If arguments are invalid
    """
    sanitized = {}
    
    for key, value in kwargs.items():
        if key == 'return_router_logits':
            if not isinstance(value, bool):
                raise ValidationError(f"return_router_logits must be bool, got {type(value)}")
            sanitized[key] = value
        
        elif key in ['attention_weights', 'gradients', 'logits']:
            # These are optional tensor inputs
            if value is not None:
                check_memory_usage(value, key)
            sanitized[key] = value
        
        elif key in ['temperature', 'threshold']:
            if not isinstance(value, (int, float)):
                raise ValidationError(f"{key} must be numeric, got {type(value)}")
            if value <= 0:
                raise ValidationError(f"{key} must be positive, got {value}")
            sanitized[key] = float(value)
        
        else:
            # Pass through unknown arguments with a warning
            warnings.warn(f"Unknown routing argument: {key}")
            sanitized[key] = value
    
    return sanitized