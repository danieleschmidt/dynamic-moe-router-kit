"""Input validation utilities for dynamic MoE routing."""

import logging
import warnings
import hashlib
import time
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np

from .exceptions import RouterConfigurationError, ValidationError, BackendError

logger = logging.getLogger(__name__)

# Security constants
MAX_TENSOR_SIZE = 1_000_000_000  # 1B elements max
MAX_BATCH_SIZE = 1024
MAX_SEQUENCE_LENGTH = 8192
MAX_HIDDEN_DIM = 16384
MAX_NUM_EXPERTS = 128
MIN_EXPERTS = 1
TRUSTED_DTYPES = {np.float32, np.float64, np.int32, np.int64}
MAX_COMPLEXITY_SCORE = 10.0
MIN_COMPLEXITY_SCORE = 0.0


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


def validate_dtype_security(tensor: Any, name: str = "tensor") -> None:
    """Validate tensor data type for security.
    
    Args:
        tensor: Input tensor to validate
        name: Name for error messages
        
    Raises:
        ValidationError: If dtype is not trusted
    """
    if hasattr(tensor, 'dtype'):
        if tensor.dtype not in TRUSTED_DTYPES:
            raise ValidationError(
                f"{name} dtype {tensor.dtype} not in trusted types: {TRUSTED_DTYPES}"
            )
    
    # Check for unusual values that might indicate adversarial inputs
    if hasattr(tensor, 'min') and hasattr(tensor, 'max'):
        min_val, max_val = float(tensor.min()), float(tensor.max())
        
        # Detect extreme values
        if abs(min_val) > 1e6 or abs(max_val) > 1e6:
            warnings.warn(
                f"{name} contains extreme values: min={min_val:.2e}, max={max_val:.2e}", 
                UserWarning
            )
            
        # Detect NaN/Inf values
        if np.isnan(tensor).any():
            raise ValidationError(f"{name} contains NaN values")
        if np.isinf(tensor).any():
            raise ValidationError(f"{name} contains infinite values")


def validate_tensor_size_limits(tensor: Any, name: str = "tensor") -> None:
    """Validate tensor doesn't exceed security size limits.
    
    Args:
        tensor: Input tensor to validate
        name: Name for error messages
        
    Raises:
        ValidationError: If tensor exceeds size limits
    """
    if hasattr(tensor, 'size'):
        total_size = int(tensor.size)
        if total_size > MAX_TENSOR_SIZE:
            raise ValidationError(
                f"{name} size {total_size} exceeds maximum {MAX_TENSOR_SIZE}"
            )
    
    # Validate specific dimensions for MoE context
    if hasattr(tensor, 'shape') and len(tensor.shape) >= 3:
        batch_size, seq_len, hidden_dim = tensor.shape[:3]
        
        if batch_size > MAX_BATCH_SIZE:
            raise ValidationError(
                f"Batch size {batch_size} exceeds maximum {MAX_BATCH_SIZE}"
            )
        if seq_len > MAX_SEQUENCE_LENGTH:
            raise ValidationError(
                f"Sequence length {seq_len} exceeds maximum {MAX_SEQUENCE_LENGTH}"
            )
        if hidden_dim > MAX_HIDDEN_DIM:
            raise ValidationError(
                f"Hidden dimension {hidden_dim} exceeds maximum {MAX_HIDDEN_DIM}"
            )


def validate_input_integrity(tensor: Any, name: str = "tensor") -> str:
    """Validate input tensor integrity using hash verification.
    
    Args:
        tensor: Input tensor to validate
        name: Name for error messages
        
    Returns:
        SHA-256 hash of tensor for integrity checking
        
    Raises:
        ValidationError: If tensor integrity check fails
    """
    try:
        # Convert to bytes for hashing
        if hasattr(tensor, 'tobytes'):
            tensor_bytes = tensor.tobytes()
        else:
            tensor_bytes = str(tensor).encode('utf-8')
            
        # Compute hash
        hash_obj = hashlib.sha256(tensor_bytes)
        tensor_hash = hash_obj.hexdigest()
        
        logger.debug(f"Computed integrity hash for {name}: {tensor_hash[:16]}...")
        return tensor_hash
        
    except Exception as e:
        raise ValidationError(f"Failed to compute integrity hash for {name}: {e}")


def sanitize_routing_kwargs(**kwargs) -> Dict[str, Any]:
    """Sanitize routing keyword arguments for security.
    
    Args:
        **kwargs: Routing arguments to sanitize
        
    Returns:
        Sanitized arguments dictionary
        
    Raises:
        ValidationError: If arguments contain unsafe values
    """
    sanitized = {}
    
    for key, value in kwargs.items():
        # Validate key names (prevent injection)
        if not key.isalnum() and '_' not in key:
            raise ValidationError(f"Invalid argument name: {key}")
        
        # Sanitize values
        if isinstance(value, (int, float)):
            if abs(value) > 1e6:
                raise ValidationError(f"Argument {key}={value} exceeds safe range")
            sanitized[key] = float(value)
        elif isinstance(value, str):
            if len(value) > 256:  # Prevent buffer overflow attacks
                raise ValidationError(f"String argument {key} exceeds length limit")
            sanitized[key] = value.strip()
        elif isinstance(value, bool):
            sanitized[key] = bool(value)
        elif value is None:
            sanitized[key] = None
        else:
            # Log and skip unknown types
            logger.warning(f"Skipping unknown argument type: {key}={type(value)}")
    
    return sanitized


def check_memory_usage(tensor: Any, name: str = "tensor") -> Dict[str, float]:
    """Check memory usage of tensor and system resources.
    
    Args:
        tensor: Input tensor to analyze
        name: Name for logging
        
    Returns:
        Memory usage statistics
        
    Raises:
        ValidationError: If memory usage is excessive
    """
    try:
        import psutil
        
        # Get tensor memory usage
        tensor_memory_mb = 0.0
        if hasattr(tensor, 'nbytes'):
            tensor_memory_mb = tensor.nbytes / (1024 * 1024)
        
        # Get system memory
        memory = psutil.virtual_memory()
        available_memory_mb = memory.available / (1024 * 1024)
        
        # Check if tensor would consume too much memory
        if tensor_memory_mb > available_memory_mb * 0.5:  # 50% limit
            raise ValidationError(
                f"{name} requires {tensor_memory_mb:.1f}MB but only "
                f"{available_memory_mb:.1f}MB available"
            )
        
        stats = {
            'tensor_memory_mb': tensor_memory_mb,
            'available_memory_mb': available_memory_mb,
            'memory_usage_percent': memory.percent
        }
        
        logger.debug(f"Memory check for {name}: {stats}")
        return stats
        
    except ImportError:
        logger.warning("psutil not available, skipping memory check")
        return {'tensor_memory_mb': 0.0, 'available_memory_mb': float('inf'), 'memory_usage_percent': 0.0}
    except Exception as e:
        logger.warning(f"Memory check failed: {e}")
        return {'tensor_memory_mb': 0.0, 'available_memory_mb': float('inf'), 'memory_usage_percent': 0.0}


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
