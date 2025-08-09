"""PyTorch backend for dynamic MoE routing."""

from .estimator import (
    TorchAttentionEntropyEstimator,
    TorchGradientNormEstimator,
    TorchPerplexityProxyEstimator,
)
from .integration import patch_model_with_dynamic_routing
from .moe import TorchMoELayer
from .router import TorchDynamicRouter

__all__ = [
    "TorchDynamicRouter",
    "TorchMoELayer",
    "TorchGradientNormEstimator",
    "TorchAttentionEntropyEstimator",
    "TorchPerplexityProxyEstimator",
    "patch_model_with_dynamic_routing"
]
