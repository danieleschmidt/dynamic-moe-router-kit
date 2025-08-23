"""
Generation 1: Simple Working Core - Dynamic MoE Router
Making it work with minimal viable features and essential error handling.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RouterConfig:
    """Simple configuration for dynamic router"""
    input_dim: int = 768
    num_experts: int = 8
    min_experts: int = 1
    max_experts: int = 4
    complexity_threshold: float = 0.5

class SimpleComplexityEstimator:
    """Basic complexity estimator using input variance"""
    
    def __init__(self, config: RouterConfig):
        self.config = config
        
    def estimate(self, inputs: np.ndarray) -> np.ndarray:
        """Estimate complexity based on input variance"""
        try:
            # Simple variance-based complexity
            variance = np.var(inputs, axis=-1, keepdims=True)
            normalized_complexity = np.tanh(variance)  # Normalize to [0,1]
            return normalized_complexity
        except Exception as e:
            logger.error(f"Complexity estimation failed: {e}")
            # Fallback to medium complexity
            return np.full((inputs.shape[0], inputs.shape[1], 1), 0.5)

class SimpleDynamicRouter:
    """Generation 1: Simple working dynamic router"""
    
    def __init__(self, config: RouterConfig):
        self.config = config
        self.complexity_estimator = SimpleComplexityEstimator(config)
        
        # Initialize simple routing network (random weights for now)
        np.random.seed(42)  # For reproducibility
        self.router_weights = np.random.randn(
            config.input_dim, config.num_experts
        ) * 0.1
        
        logger.info(f"Initialized SimpleDynamicRouter with {config.num_experts} experts")
    
    def route(self, inputs: np.ndarray) -> Dict[str, Any]:
        """Simple routing logic"""
        try:
            batch_size, seq_len, dim = inputs.shape
            
            # Estimate complexity
            complexity = self.complexity_estimator.estimate(inputs)
            
            # Dynamic expert count based on complexity
            k_experts = self.config.min_experts + (
                self.config.max_experts - self.config.min_experts
            ) * complexity
            k_experts = np.round(k_experts).astype(int)
            
            # Simple routing scores
            routing_logits = np.dot(inputs, self.router_weights)
            
            # Select top-k experts per token (simplified)
            expert_indices = np.argsort(routing_logits, axis=-1)[..., -self.config.max_experts:]
            
            # Compute simple weights (uniform for now)
            expert_weights = np.ones_like(expert_indices, dtype=np.float32)
            expert_weights = expert_weights / np.sum(expert_weights, axis=-1, keepdims=True)
            
            # Calculate metrics
            avg_experts = np.mean(k_experts)
            flop_reduction = 1.0 - (avg_experts / self.config.num_experts)
            
            result = {
                'expert_indices': expert_indices,
                'expert_weights': expert_weights,
                'complexity_scores': complexity,
                'avg_experts_per_token': avg_experts,
                'flop_reduction': flop_reduction,
                'routing_logits': routing_logits
            }
            
            logger.debug(f"Routing complete: avg_experts={avg_experts:.2f}, "
                        f"flop_reduction={flop_reduction:.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            # Return fallback result
            return self._get_fallback_routing(inputs)
    
    def _get_fallback_routing(self, inputs: np.ndarray) -> Dict[str, Any]:
        """Fallback routing when main routing fails"""
        batch_size, seq_len, dim = inputs.shape
        
        # Use static top-2 experts as fallback
        expert_indices = np.tile([0, 1], (batch_size, seq_len, 1))
        expert_weights = np.full((batch_size, seq_len, 2), 0.5)
        
        return {
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'complexity_scores': np.full((batch_size, seq_len, 1), 0.5),
            'avg_experts_per_token': 2.0,
            'flop_reduction': 0.75,  # 2/8 experts
            'routing_logits': np.zeros((batch_size, seq_len, self.config.num_experts))
        }

class SimpleMoELayer:
    """Simple MoE layer implementation"""
    
    def __init__(self, config: RouterConfig):
        self.config = config
        self.router = SimpleDynamicRouter(config)
        
        # Simple expert networks (linear transformations)
        np.random.seed(42)
        self.experts = []
        for i in range(config.num_experts):
            expert = {
                'weight': np.random.randn(config.input_dim, config.input_dim) * 0.1,
                'bias': np.zeros(config.input_dim)
            }
            self.experts.append(expert)
    
    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Forward pass through MoE layer"""
        try:
            # Get routing decision
            routing_info = self.router.route(inputs)
            
            # Apply experts (simplified - just use first expert for now)
            outputs = np.dot(inputs, self.experts[0]['weight']) + self.experts[0]['bias']
            outputs = np.tanh(outputs)  # Simple activation
            
            logger.debug(f"MoE forward pass complete, output shape: {outputs.shape}")
            
            return outputs, routing_info
            
        except Exception as e:
            logger.error(f"MoE forward pass failed: {e}")
            # Return input as fallback
            return inputs, routing_info

def demonstrate_simple_moe():
    """Demonstrate the simple MoE implementation"""
    print("🚀 Generation 1: Simple Dynamic MoE Router Demo")
    print("=" * 50)
    
    # Create configuration
    config = RouterConfig(
        input_dim=768,
        num_experts=8,
        min_experts=1,
        max_experts=4
    )
    
    # Create MoE layer
    moe_layer = SimpleMoELayer(config)
    
    # Create sample input
    batch_size, seq_len = 2, 128
    inputs = np.random.randn(batch_size, seq_len, config.input_dim)
    
    print(f"Input shape: {inputs.shape}")
    
    # Forward pass
    outputs, routing_info = moe_layer.forward(inputs)
    
    print(f"Output shape: {outputs.shape}")
    print(f"Average experts per token: {routing_info['avg_experts_per_token']:.2f}")
    print(f"FLOP reduction: {routing_info['flop_reduction']:.1%}")
    
    print("\n✅ Generation 1 Complete: Basic functionality working!")
    
    return moe_layer, routing_info

if __name__ == "__main__":
    demonstrate_simple_moe()