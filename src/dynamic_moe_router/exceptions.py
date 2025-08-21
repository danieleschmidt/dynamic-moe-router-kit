"""Custom exceptions for dynamic MoE router kit."""


class DynamicMoEError(Exception):
    """Base exception for dynamic MoE routing errors."""
    pass


class RouterConfigurationError(DynamicMoEError):
    """Raised when router configuration is invalid."""
    pass


class ComplexityEstimationError(DynamicMoEError):
    """Raised when complexity estimation fails."""
    pass


class ExpertDispatchError(DynamicMoEError):
    """Raised when expert dispatching fails."""
    pass


class LoadBalancingError(DynamicMoEError):
    """Raised when load balancing encounters issues."""
    pass


class ModelPatchingError(DynamicMoEError):
    """Raised when model patching fails."""
    pass


class ProfilingError(DynamicMoEError):
    """Raised when profiling operations fail."""
    pass


class ValidationError(DynamicMoEError):
    """Raised when input validation fails."""
    pass


class BackendError(DynamicMoEError):
    """Raised when backend-specific operations fail."""
    pass


class ConvergenceError(DynamicMoEError):
    """Raised when adaptive algorithms fail to converge."""
    pass


class SecurityValidationError(DynamicMoEError):
    """Raised when security validation fails."""
    pass


class ResourceExhaustionError(DynamicMoEError):
    """Raised when system resources are exhausted."""
    pass


class InputValidationError(DynamicMoEError):
    """Raised when input validation fails."""
    pass


class PerformanceError(DynamicMoEError):
    """Raised when performance issues are detected."""
    pass


class ProfilingError(DynamicMoEError):
    """Raised when profiling operations fail."""
    pass
