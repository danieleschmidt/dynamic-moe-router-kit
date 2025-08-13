"""Dynamic MoE Router Kit - Adaptive expert routing for Mixture-of-Experts models."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "author@example.com"

# Core components
from .estimator import (
    ESTIMATOR_REGISTRY,
    AttentionEntropyEstimator,
    ComplexityEstimator,
    GradientNormEstimator,
    PerplexityProxyEstimator,
    ThresholdEstimator,
    get_estimator,
)
from .moe import LayerNormMoE, MoELayer, SparseMoELayer
from .profiler import ComparisonProfiler, FLOPProfiler
from .router import DynamicRouter
from .adaptive_router import EnhancedDynamicRouter, AdaptiveLoadBalancer
from .secure_router import SecureEnhancedRouter
from .robust_security import SecurityValidator, RobustErrorHandler, ResourceMonitor
from .health_monitoring import HealthMonitor, HealthMetric, PerformanceSnapshot
from .high_performance import PerformanceOptimizer, ConcurrentRouter
from .auto_scaling import AutoScaler, LoadBalancer, ResourceAllocation, ScalingPolicy
from .production_router import ProductionMoERouter

# Framework-specific implementations (optional imports)
try:
    import torch
    _torch_available = True
except ImportError:
    _torch_available = False

if _torch_available:
    try:
        from .torch import (
            TorchAttentionEntropyEstimator,
            TorchDynamicRouter,
            TorchGradientNormEstimator,
            TorchMoELayer,
            TorchPerplexityProxyEstimator,
            patch_model_with_dynamic_routing,
        )
        TORCH_AVAILABLE = True
    except ImportError as e:
        TORCH_AVAILABLE = False
        import warnings
        warnings.warn(f"PyTorch detected but torch backend failed to load: {e}")
else:
    TORCH_AVAILABLE = False

__all__ = [
    "__version__",
    # Core components
    "DynamicRouter",
    "EnhancedDynamicRouter",
    "SecureEnhancedRouter",
    "AdaptiveLoadBalancer",
    "SecurityValidator",
    "RobustErrorHandler",
    "ResourceMonitor",
    "HealthMonitor",
    "HealthMetric",
    "PerformanceSnapshot",
    "PerformanceOptimizer",
    "ConcurrentRouter",
    "AutoScaler",
    "LoadBalancer",
    "ResourceAllocation",
    "ScalingPolicy",
    "ProductionMoERouter",
    "ComplexityEstimator",
    "GradientNormEstimator",
    "AttentionEntropyEstimator",
    "PerplexityProxyEstimator",
    "ThresholdEstimator",
    "get_estimator",
    "ESTIMATOR_REGISTRY",
    "MoELayer",
    "SparseMoELayer",
    "LayerNormMoE",
    "FLOPProfiler",
    "ComparisonProfiler",
]

# Add PyTorch components if available
if TORCH_AVAILABLE:
    __all__.extend([
        "TorchDynamicRouter",
        "TorchMoELayer",
        "TorchGradientNormEstimator",
        "TorchAttentionEntropyEstimator",
        "TorchPerplexityProxyEstimator",
        "patch_model_with_dynamic_routing"
    ])
