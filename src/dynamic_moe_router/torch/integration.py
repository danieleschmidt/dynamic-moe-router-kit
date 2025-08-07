"""Integration utilities for patching existing models with dynamic routing."""

from typing import Dict, Any, Optional, Union, List
import warnings

try:
    import torch
    import torch.nn as nn
    from transformers import PreTrainedModel
    TORCH_AVAILABLE = True
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    TRANSFORMERS_AVAILABLE = False

from .router import TorchDynamicRouter
from .moe import TorchMoELayer, LinearExpert
from .estimator import TorchGradientNormEstimator


class ModelPatcher:
    """Utility class for patching models with dynamic routing."""
    
    def __init__(self):
        self.original_modules = {}
    
    def patch_model_with_dynamic_routing(
        self,
        model: nn.Module,
        target_layers: Optional[List[str]] = None,
        min_experts_ratio: float = 0.125,
        max_experts_ratio: float = 0.5,
        complexity_metric: str = "gradient_norm",
        expert_hidden_dim_ratio: float = 4.0,
        **router_kwargs
    ) -> nn.Module:
        """Patch a model with dynamic MoE routing.
        
        Args:
            model: The model to patch
            target_layers: List of layer names to replace (if None, auto-detect MoE layers)
            min_experts_ratio: Minimum fraction of experts to use
            max_experts_ratio: Maximum fraction of experts to use
            complexity_metric: Complexity estimation method
            expert_hidden_dim_ratio: Hidden dimension expansion ratio for experts
            **router_kwargs: Additional router configuration
            
        Returns:
            Modified model with dynamic routing
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for model patching")
        
        # Auto-detect MoE layers if not specified
        if target_layers is None:
            target_layers = self._detect_moe_layers(model)
        
        if not target_layers:
            warnings.warn("No MoE layers detected. Creating new MoE layers in FFN positions.")
            target_layers = self._detect_ffn_layers(model)
        
        # Patch each target layer
        for layer_name in target_layers:
            self._patch_single_layer(
                model, layer_name, min_experts_ratio, max_experts_ratio,
                complexity_metric, expert_hidden_dim_ratio, **router_kwargs
            )
        
        return model
    
    def _detect_moe_layers(self, model: nn.Module) -> List[str]:
        """Detect existing MoE layers in the model."""
        moe_layers = []
        for name, module in model.named_modules():
            # Common MoE layer names and types
            if any(moe_indicator in name.lower() for moe_indicator in ['moe', 'expert', 'mixture']):
                moe_layers.append(name)
            # Check for specific MoE module types
            elif hasattr(module, 'experts') or hasattr(module, 'gate'):
                moe_layers.append(name)
        
        return moe_layers
    
    def _detect_ffn_layers(self, model: nn.Module) -> List[str]:
        """Detect FFN layers that could be replaced with MoE."""
        ffn_layers = []
        
        for name, module in model.named_modules():
            # Common FFN layer names
            if any(ffn_indicator in name.lower() for ffn_indicator in ['ffn', 'feed_forward', 'mlp']):
                # Check if it's a complete FFN block (not just a component)
                if hasattr(module, 'forward') and not name.endswith(('weight', 'bias')):
                    ffn_layers.append(name)
        
        return ffn_layers
    
    def _patch_single_layer(
        self,
        model: nn.Module,
        layer_name: str,
        min_experts_ratio: float,
        max_experts_ratio: float,
        complexity_metric: str,
        expert_hidden_dim_ratio: float,
        **router_kwargs
    ):
        """Patch a single layer with dynamic MoE routing."""
        # Get the target module
        parent_module, attr_name = self._get_parent_and_attr(model, layer_name)
        original_module = getattr(parent_module, attr_name)
        
        # Store original module for potential restoration
        self.original_modules[layer_name] = original_module
        
        # Determine input/output dimensions
        input_dim = self._get_input_dim(original_module)
        if input_dim is None:
            warnings.warn(f"Could not determine input dimension for layer {layer_name}. Skipping.")
            return
        
        # Create dynamic MoE replacement
        moe_layer = self._create_dynamic_moe_layer(
            original_module, input_dim, min_experts_ratio, max_experts_ratio,
            complexity_metric, expert_hidden_dim_ratio, **router_kwargs
        )
        
        # Replace the original module
        setattr(parent_module, attr_name, moe_layer)
    
    def _get_parent_and_attr(self, model: nn.Module, layer_name: str):
        """Get parent module and attribute name for a given layer path."""
        parts = layer_name.split('.')
        parent = model
        
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        return parent, parts[-1]
    
    def _get_input_dim(self, module: nn.Module) -> Optional[int]:
        """Extract input dimension from a module."""
        if hasattr(module, 'in_features'):
            return module.in_features
        elif hasattr(module, 'hidden_size'):
            return module.hidden_size
        elif hasattr(module, 'config') and hasattr(module.config, 'hidden_size'):
            return module.config.hidden_size
        
        # Try to infer from the first linear layer
        for child in module.children():
            if isinstance(child, nn.Linear):
                return child.in_features
        
        return None
    
    def _create_dynamic_moe_layer(
        self,
        original_module: nn.Module,
        input_dim: int,
        min_experts_ratio: float,
        max_experts_ratio: float,
        complexity_metric: str,
        expert_hidden_dim_ratio: float,
        **router_kwargs
    ) -> TorchMoELayer:
        """Create a dynamic MoE layer to replace the original module."""
        
        # Default configuration
        num_experts = router_kwargs.get('num_experts', 8)
        min_experts = max(1, int(num_experts * min_experts_ratio))
        max_experts = min(num_experts, int(num_experts * max_experts_ratio))
        
        # Create complexity estimator
        if complexity_metric == "gradient_norm":
            estimator = TorchGradientNormEstimator()
        else:
            # Fallback to gradient norm
            estimator = TorchGradientNormEstimator()
            warnings.warn(f"Unknown complexity metric {complexity_metric}, using gradient_norm")
        
        # Create router
        router = TorchDynamicRouter(
            input_dim=input_dim,
            num_experts=num_experts,
            min_experts=min_experts,
            max_experts=max_experts,
            complexity_estimator=estimator,
            **{k: v for k, v in router_kwargs.items() if k != 'num_experts'}
        )
        
        # Expert factory function
        def expert_fn():
            hidden_dim = int(input_dim * expert_hidden_dim_ratio)
            return LinearExpert(input_dim, hidden_dim)
        
        # Create MoE layer
        moe_layer = TorchMoELayer(
            router=router,
            expert_fn=expert_fn,
            num_experts=num_experts
        )
        
        return moe_layer
    
    def restore_original_layer(self, model: nn.Module, layer_name: str):
        """Restore an original layer that was replaced with MoE."""
        if layer_name not in self.original_modules:
            raise ValueError(f"No original module stored for layer {layer_name}")
        
        parent_module, attr_name = self._get_parent_and_attr(model, layer_name)
        setattr(parent_module, attr_name, self.original_modules[layer_name])
        del self.original_modules[layer_name]
    
    def restore_all_layers(self, model: nn.Module):
        """Restore all original layers."""
        layer_names = list(self.original_modules.keys())
        for layer_name in layer_names:
            self.restore_original_layer(model, layer_name)


# Global patcher instance
_global_patcher = ModelPatcher()


def patch_model_with_dynamic_routing(
    model: Union[nn.Module, PreTrainedModel],
    target_layers: Optional[List[str]] = None,
    min_experts_ratio: float = 0.125,
    max_experts_ratio: float = 0.5,
    complexity_metric: str = "gradient_norm",
    **router_kwargs
) -> Union[nn.Module, PreTrainedModel]:
    """Convenience function to patch a model with dynamic routing.
    
    Args:
        model: The model to patch (PyTorch nn.Module or HuggingFace model)
        target_layers: Specific layers to replace (None for auto-detection)
        min_experts_ratio: Minimum fraction of experts to activate
        max_experts_ratio: Maximum fraction of experts to activate
        complexity_metric: Method for estimating input complexity
        **router_kwargs: Additional router configuration
        
    Returns:
        Modified model with dynamic MoE routing
        
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
        >>> model = patch_model_with_dynamic_routing(
        ...     model,
        ...     min_experts_ratio=0.125,
        ...     max_experts_ratio=0.5,
        ...     complexity_metric="gradient_norm"
        ... )
    """
    return _global_patcher.patch_model_with_dynamic_routing(
        model, target_layers, min_experts_ratio, max_experts_ratio,
        complexity_metric, **router_kwargs
    )


def restore_original_layers(model: nn.Module, layer_names: Optional[List[str]] = None):
    """Restore original layers in a patched model.
    
    Args:
        model: The patched model
        layer_names: Specific layers to restore (None for all)
    """
    if layer_names is None:
        _global_patcher.restore_all_layers(model)
    else:
        for layer_name in layer_names:
            _global_patcher.restore_original_layer(model, layer_name)


class HuggingFaceIntegration:
    """Specialized integration for HuggingFace transformers."""
    
    @staticmethod
    def patch_mixtral_model(
        model,
        min_experts_ratio: float = 0.125,
        max_experts_ratio: float = 0.5,
        **kwargs
    ):
        """Patch a Mixtral model with improved dynamic routing."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required for HuggingFace integration")
        
        # Mixtral-specific layer detection
        moe_layer_pattern = "model.layers.*.block_sparse_moe"
        target_layers = []
        
        for name, _ in model.named_modules():
            if "block_sparse_moe" in name:
                target_layers.append(name)
        
        return patch_model_with_dynamic_routing(
            model,
            target_layers=target_layers,
            min_experts_ratio=min_experts_ratio,
            max_experts_ratio=max_experts_ratio,
            **kwargs
        )
    
    @staticmethod
    def patch_switch_transformer(
        model,
        min_experts_ratio: float = 0.125,
        max_experts_ratio: float = 0.5,
        **kwargs
    ):
        """Patch a Switch Transformer model with dynamic routing."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required for HuggingFace integration")
        
        # Switch Transformer layer detection
        target_layers = []
        for name, _ in model.named_modules():
            if "mlp" in name and "router" not in name:
                target_layers.append(name)
        
        return patch_model_with_dynamic_routing(
            model,
            target_layers=target_layers,
            min_experts_ratio=min_experts_ratio,
            max_experts_ratio=max_experts_ratio,
            **kwargs
        )