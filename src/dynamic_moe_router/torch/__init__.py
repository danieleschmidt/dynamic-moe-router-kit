"""PyTorch backend for dynamic MoE routing."""

from .router import TorchDynamicRouter
from .moe import TorchMoELayer
from .estimator import (
    TorchGradientNormEstimator,
    TorchAttentionEntropyEstimator,
    TorchPerplexityProxyEstimator
)
from .integration import patch_model_with_dynamic_routing

__all__ = [
    "TorchDynamicRouter",
    "TorchMoELayer", 
    "TorchGradientNormEstimator",
    "TorchAttentionEntropyEstimator",
    "TorchPerplexityProxyEstimator",
    "patch_model_with_dynamic_routing"
]