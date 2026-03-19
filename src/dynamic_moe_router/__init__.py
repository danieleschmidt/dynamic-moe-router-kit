"""dynamic-moe-router-kit: dynamic Mixture-of-Experts routing for PyTorch."""

from .analytics import RouterAnalytics
from .experts import Expert
from .moe import DynamicMoE
from .router import DynamicRouter, RouterOutput

__all__ = [
    "Expert",
    "DynamicRouter",
    "RouterOutput",
    "DynamicMoE",
    "RouterAnalytics",
]

__version__ = "0.2.0"
