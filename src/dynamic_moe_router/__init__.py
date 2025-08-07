"""Dynamic MoE Router Kit - Adaptive expert routing for Mixture-of-Experts models."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "author@example.com"

# Core components
from .router import DynamicRouter, AdaptiveRouter
from .estimator import (
    ComplexityEstimator,
    GradientNormEstimator,
    AttentionEntropyEstimator,
    PerplexityProxyEstimator,
    ThresholdEstimator,
    get_estimator,
    ESTIMATOR_REGISTRY
)
from .moe import MoELayer, SparseMoELayer, LayerNormMoE
from .profiler import FLOPProfiler, ComparisonProfiler

# Framework-specific implementations (optional imports)
try:
    import torch
    _torch_available = True
except ImportError:
    _torch_available = False

if _torch_available:
    try:
        from .torch import (
            TorchDynamicRouter,
            TorchMoELayer,
            TorchGradientNormEstimator,
            TorchAttentionEntropyEstimator,
            TorchPerplexityProxyEstimator,
            patch_model_with_dynamic_routing
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
    "AdaptiveRouter",
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