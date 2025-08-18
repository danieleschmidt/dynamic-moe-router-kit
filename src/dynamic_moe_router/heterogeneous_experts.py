"""
Heterogeneous Collaborative Expert Architecture for MoE Models

This module implements heterogeneous expert architectures based on 2024 research
showing that diverse expert types (deep, attention-based, focal) can collaborate
more effectively than homogeneous expert pools.

Key innovations from recent research:
- HCE-MoE: Heterogeneous Collaborative and Expansion Mixture-of-Experts
- Dynamic routing for structurally varied experts
- Multi-scale expert specialization (depth, attention, focal processing)
- Collaborative optimization between expert types

Research Foundation:
- Multi-modal Collaborative Optimization and Expansion Network (MCO-E Net)
- Heterogeneous expert collaboration with dynamic routing
- Structural diversity for improved model capacity

Author: Terry (Terragon Labs)
Research Period: 2024 Advanced Heterogeneous MoE Architectures
"""

import logging
import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .estimator import ComplexityEstimator, get_estimator
from .exceptions import (
    ExpertDispatchError,
    RouterConfigurationError,
)
from .validation import (
    validate_expert_indices,
    validate_expert_weights,
    validate_router_config,
    validate_tensor_shape,
)

logger = logging.getLogger(__name__)


class ExpertType(Enum):
    """Enumeration of different expert types."""
    DEEP = "deep"           # Deep feedforward experts
    ATTENTION = "attention" # Multi-head attention experts  
    FOCAL = "focal"         # Focal/convolutional experts
    HYBRID = "hybrid"       # Hybrid experts combining multiple types
    SPARSE = "sparse"       # Sparse connectivity experts


class ExpertCapability(Enum):
    """Expert specialization capabilities."""
    REASONING = "reasoning"         # Complex reasoning tasks
    FEATURE_EXTRACTION = "feature"  # Feature extraction and processing
    PATTERN_MATCHING = "pattern"    # Pattern recognition
    SEQUENCE_MODELING = "sequence"  # Sequence modeling
    ATTENTION_FOCUS = "attention"   # Attention-based processing
    SPARSE_PROCESSING = "sparse"    # Sparse data processing


class BaseExpert(ABC):
    """Abstract base class for heterogeneous experts."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        expert_type: ExpertType,
        capabilities: List[ExpertCapability],
        expert_id: Optional[int] = None,
        **kwargs
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.expert_type = expert_type
        self.capabilities = capabilities
        self.expert_id = expert_id
        self.parameters = {}
        
        # Performance tracking
        self.activation_count = 0
        self.total_flops = 0
        self.specialization_score = 0.0
        
        self._initialize_parameters(**kwargs)
        
    @abstractmethod
    def _initialize_parameters(self, **kwargs):
        """Initialize expert-specific parameters."""
        pass
        
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Forward pass through the expert."""
        pass
        
    @abstractmethod
    def compute_flops(self, input_shape: Tuple[int, ...]) -> int:
        """Compute FLOPs for this expert given input shape."""
        pass
        
    def get_specialization_score(self, task_type: ExpertCapability) -> float:
        """Get specialization score for a specific task type."""
        if task_type in self.capabilities:
            return 1.0 + self.specialization_score
        return 0.1  # Low score for non-specialized tasks
        
    def update_specialization(self, performance_feedback: float):
        """Update specialization score based on performance feedback."""
        # Exponential moving average
        alpha = 0.1
        self.specialization_score = alpha * performance_feedback + (1 - alpha) * self.specialization_score


class DeepExpert(BaseExpert):
    """Deep feedforward expert with multiple layers."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
        num_layers: int = 3,
        activation: str = "relu",
        dropout: float = 0.1,
        expert_id: Optional[int] = None,
        **kwargs
    ):
        self.hidden_dims = hidden_dims or [input_dim * 2, input_dim * 4, input_dim * 2]
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            expert_type=ExpertType.DEEP,
            capabilities=[ExpertCapability.REASONING, ExpertCapability.FEATURE_EXTRACTION],
            expert_id=expert_id,
            **kwargs
        )
        
    def _initialize_parameters(self, **kwargs):
        """Initialize deep network parameters."""
        # Build layer dimensions
        layer_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        
        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            
            # Xavier initialization
            scale = math.sqrt(2.0 / (in_dim + out_dim))
            weight = np.random.normal(0, scale, (in_dim, out_dim)).astype(np.float32)
            bias = np.zeros(out_dim, dtype=np.float32)
            
            self.weights.append(weight)
            self.biases.append(bias)
            
        self.parameters = {
            'weights': self.weights,
            'biases': self.biases,
            'num_layers': len(self.weights)
        }
        
    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Forward pass through deep expert."""
        x = inputs
        activations = [x]
        
        # Forward through layers
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            # Linear transformation
            x = np.dot(x, weight) + bias
            
            # Activation (except last layer)
            if i < len(self.weights) - 1:
                if self.activation == "relu":
                    x = np.maximum(0, x)
                elif self.activation == "gelu":
                    # Approximate GELU
                    x = 0.5 * x * (1 + np.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))
                elif self.activation == "swish":
                    x = x / (1 + np.exp(-x))
                    
                # Dropout simulation (in training)
                if self.dropout > 0:
                    # Simplified dropout for demonstration
                    dropout_mask = np.random.binomial(1, 1 - self.dropout, x.shape)
                    x = x * dropout_mask / (1 - self.dropout)
                    
            activations.append(x)
            
        self.activation_count += 1
        
        # Compute FLOPs
        batch_size = inputs.shape[0]
        flops = self.compute_flops(inputs.shape)
        self.total_flops += flops
        
        expert_info = {
            'expert_type': self.expert_type.value,
            'expert_id': self.expert_id,
            'activations': activations,
            'flops': flops,
            'num_parameters': sum(w.size + b.size for w, b in zip(self.weights, self.biases)),
            'depth': len(self.weights)
        }
        
        return x, expert_info
        
    def compute_flops(self, input_shape: Tuple[int, ...]) -> int:
        """Compute FLOPs for deep expert."""
        batch_size = input_shape[0] if len(input_shape) > 1 else 1
        seq_len = input_shape[1] if len(input_shape) > 2 else 1
        
        total_flops = 0
        for weight in self.weights:
            # Matrix multiplication FLOPs: batch_size * seq_len * input_dim * output_dim
            total_flops += batch_size * seq_len * weight.shape[0] * weight.shape[1]
            
        return total_flops


class AttentionExpert(BaseExpert):
    """Multi-head attention expert."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_flash_attention: bool = False,
        expert_id: Optional[int] = None,
        **kwargs
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim or input_dim // num_heads
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention
        
        # Ensure dimensions are compatible
        if self.head_dim * self.num_heads != input_dim:
            self.head_dim = input_dim // self.num_heads
            
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            expert_type=ExpertType.ATTENTION,
            capabilities=[ExpertCapability.ATTENTION_FOCUS, ExpertCapability.SEQUENCE_MODELING],
            expert_id=expert_id,
            **kwargs
        )
        
    def _initialize_parameters(self, **kwargs):
        """Initialize attention parameters."""
        # Query, Key, Value projection matrices
        scale = math.sqrt(2.0 / self.input_dim)
        
        self.W_q = np.random.normal(0, scale, (self.input_dim, self.input_dim)).astype(np.float32)
        self.W_k = np.random.normal(0, scale, (self.input_dim, self.input_dim)).astype(np.float32)
        self.W_v = np.random.normal(0, scale, (self.input_dim, self.input_dim)).astype(np.float32)
        
        # Output projection
        self.W_o = np.random.normal(0, scale, (self.input_dim, self.output_dim)).astype(np.float32)
        
        # Biases
        self.b_q = np.zeros(self.input_dim, dtype=np.float32)
        self.b_k = np.zeros(self.input_dim, dtype=np.float32)
        self.b_v = np.zeros(self.input_dim, dtype=np.float32)
        self.b_o = np.zeros(self.output_dim, dtype=np.float32)
        
        self.parameters = {
            'W_q': self.W_q, 'W_k': self.W_k, 'W_v': self.W_v, 'W_o': self.W_o,
            'b_q': self.b_q, 'b_k': self.b_k, 'b_v': self.b_v, 'b_o': self.b_o,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim
        }
        
    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Forward pass through attention expert."""
        batch_size, seq_len, _ = inputs.shape
        
        # Compute Q, K, V
        Q = np.dot(inputs, self.W_q) + self.b_q  # (B, L, D)
        K = np.dot(inputs, self.W_k) + self.b_k  # (B, L, D)
        V = np.dot(inputs, self.W_v) + self.b_v  # (B, L, D)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Compute attention scores
        attention_scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        
        # Apply softmax
        attention_weights = self._softmax(attention_scores, axis=-1)
        
        # Apply attention to values
        attention_output = np.matmul(attention_weights, V)  # (B, H, L, head_dim)
        
        # Reshape and project output
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.input_dim
        )
        
        output = np.dot(attention_output, self.W_o) + self.b_o
        
        self.activation_count += 1
        
        # Compute FLOPs
        flops = self.compute_flops(inputs.shape)
        self.total_flops += flops
        
        expert_info = {
            'expert_type': self.expert_type.value,
            'expert_id': self.expert_id,
            'attention_weights': attention_weights,
            'attention_entropy': self._compute_attention_entropy(attention_weights),
            'flops': flops,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim
        }
        
        return output, expert_info
        
    def compute_flops(self, input_shape: Tuple[int, ...]) -> int:
        """Compute FLOPs for attention expert."""
        batch_size, seq_len, input_dim = input_shape
        
        # Q, K, V projections
        qkv_flops = 3 * batch_size * seq_len * input_dim * input_dim
        
        # Attention computation
        attention_flops = batch_size * self.num_heads * seq_len * seq_len * self.head_dim
        
        # Output projection
        output_flops = batch_size * seq_len * input_dim * self.output_dim
        
        return qkv_flops + attention_flops + output_flops
        
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        
    def _compute_attention_entropy(self, attention_weights: np.ndarray) -> np.ndarray:
        """Compute entropy of attention distributions."""
        eps = 1e-10
        entropy = -np.sum(attention_weights * np.log(attention_weights + eps), axis=-1)
        return np.mean(entropy, axis=(1, 2))  # Average over heads and sequence


class FocalExpert(BaseExpert):
    """Focal/convolutional expert for local pattern processing."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_sizes: List[int] = None,
        num_filters: List[int] = None,
        focal_radius: int = 3,
        dilation: int = 1,
        expert_id: Optional[int] = None,
        **kwargs
    ):
        self.kernel_sizes = kernel_sizes or [3, 5, 7]
        self.num_filters = num_filters or [input_dim // 2] * len(self.kernel_sizes)
        self.focal_radius = focal_radius
        self.dilation = dilation
        
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            expert_type=ExpertType.FOCAL,
            capabilities=[ExpertCapability.PATTERN_MATCHING, ExpertCapability.FEATURE_EXTRACTION],
            expert_id=expert_id,
            **kwargs
        )
        
    def _initialize_parameters(self, **kwargs):
        """Initialize focal convolution parameters."""
        self.conv_weights = []
        self.conv_biases = []
        
        # Initialize convolution kernels for each kernel size
        for kernel_size, num_filter in zip(self.kernel_sizes, self.num_filters):
            # 1D convolution weights: (kernel_size, input_dim, num_filter)
            scale = math.sqrt(2.0 / (kernel_size * self.input_dim))
            weight = np.random.normal(0, scale, (kernel_size, self.input_dim, num_filter)).astype(np.float32)
            bias = np.zeros(num_filter, dtype=np.float32)
            
            self.conv_weights.append(weight)
            self.conv_biases.append(bias)
            
        # Final projection layer
        total_filters = sum(self.num_filters)
        proj_scale = math.sqrt(2.0 / total_filters)
        self.proj_weight = np.random.normal(0, proj_scale, (total_filters, self.output_dim)).astype(np.float32)
        self.proj_bias = np.zeros(self.output_dim, dtype=np.float32)
        
        self.parameters = {
            'conv_weights': self.conv_weights,
            'conv_biases': self.conv_biases,
            'proj_weight': self.proj_weight,
            'proj_bias': self.proj_bias,
            'kernel_sizes': self.kernel_sizes,
            'num_filters': self.num_filters
        }
        
    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Forward pass through focal expert."""
        batch_size, seq_len, input_dim = inputs.shape
        
        # Apply multiple focal convolutions
        conv_outputs = []
        
        for kernel_size, weight, bias in zip(self.kernel_sizes, self.conv_weights, self.conv_biases):
            # Compute 1D convolution manually
            conv_out = self._conv1d(inputs, weight, bias, kernel_size)
            conv_outputs.append(conv_out)
            
        # Concatenate multi-scale features
        concatenated = np.concatenate(conv_outputs, axis=-1)  # (B, L, sum(num_filters))
        
        # Final projection
        output = np.dot(concatenated, self.proj_weight) + self.proj_bias
        
        self.activation_count += 1
        
        # Compute FLOPs
        flops = self.compute_flops(inputs.shape)
        self.total_flops += flops
        
        expert_info = {
            'expert_type': self.expert_type.value,
            'expert_id': self.expert_id,
            'conv_outputs': conv_outputs,
            'focal_patterns': self._analyze_focal_patterns(conv_outputs),
            'flops': flops,
            'kernel_sizes': self.kernel_sizes,
            'effective_receptive_field': max(self.kernel_sizes)
        }
        
        return output, expert_info
        
    def _conv1d(self, inputs: np.ndarray, weight: np.ndarray, bias: np.ndarray, kernel_size: int) -> np.ndarray:
        """Simple 1D convolution implementation."""
        batch_size, seq_len, input_dim = inputs.shape
        num_filters = weight.shape[2]
        
        # Padding
        pad = kernel_size // 2
        padded_inputs = np.pad(inputs, ((0, 0), (pad, pad), (0, 0)), mode='constant')
        
        # Convolution output
        output = np.zeros((batch_size, seq_len, num_filters), dtype=np.float32)
        
        for i in range(seq_len):
            # Extract window
            window = padded_inputs[:, i:i+kernel_size, :]  # (B, kernel_size, input_dim)
            
            # Convolution: sum over kernel_size and input_dim
            for f in range(num_filters):
                output[:, i, f] = np.sum(window * weight[:, :, f], axis=(1, 2)) + bias[f]
                
        # ReLU activation
        output = np.maximum(0, output)
        
        return output
        
    def compute_flops(self, input_shape: Tuple[int, ...]) -> int:
        """Compute FLOPs for focal expert."""
        batch_size, seq_len, input_dim = input_shape
        
        # Convolution FLOPs
        conv_flops = 0
        for kernel_size, num_filter in zip(self.kernel_sizes, self.num_filters):
            conv_flops += batch_size * seq_len * kernel_size * input_dim * num_filter
            
        # Projection FLOPs
        total_filters = sum(self.num_filters)
        proj_flops = batch_size * seq_len * total_filters * self.output_dim
        
        return conv_flops + proj_flops
        
    def _analyze_focal_patterns(self, conv_outputs: List[np.ndarray]) -> Dict[str, float]:
        """Analyze focal pattern statistics."""
        pattern_stats = {}
        
        for i, output in enumerate(conv_outputs):
            kernel_size = self.kernel_sizes[i]
            
            # Activation statistics
            activation_mean = float(np.mean(output))
            activation_std = float(np.std(output))
            sparsity = float(np.mean(output == 0))
            
            pattern_stats[f'kernel_{kernel_size}'] = {
                'activation_mean': activation_mean,
                'activation_std': activation_std,
                'sparsity': sparsity,
                'pattern_complexity': activation_std / (activation_mean + 1e-8)
            }
            
        return pattern_stats


class HeterogeneousExpertPool:
    """Pool of heterogeneous experts with collaborative routing."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        expert_config: Dict[ExpertType, int],
        collaboration_matrix: Optional[np.ndarray] = None,
        **expert_kwargs
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.expert_config = expert_config
        self.total_experts = sum(expert_config.values())
        
        # Initialize experts
        self.experts = []
        self.expert_types = []
        self._initialize_experts(**expert_kwargs)
        
        # Initialize collaboration matrix
        if collaboration_matrix is not None:
            self.collaboration_matrix = collaboration_matrix
        else:
            self.collaboration_matrix = self._initialize_collaboration_matrix()
            
        # Performance tracking
        self.expert_performance = np.ones(self.total_experts, dtype=np.float32)
        self.collaboration_history = []
        
    def _initialize_experts(self, **expert_kwargs):
        """Initialize heterogeneous expert pool."""
        expert_id = 0
        
        for expert_type, count in self.expert_config.items():
            for _ in range(count):
                if expert_type == ExpertType.DEEP:
                    expert = DeepExpert(
                        input_dim=self.input_dim,
                        output_dim=self.output_dim,
                        expert_id=expert_id,
                        **expert_kwargs.get('deep', {})
                    )
                elif expert_type == ExpertType.ATTENTION:
                    expert = AttentionExpert(
                        input_dim=self.input_dim,
                        output_dim=self.output_dim,
                        expert_id=expert_id,
                        **expert_kwargs.get('attention', {})
                    )
                elif expert_type == ExpertType.FOCAL:
                    expert = FocalExpert(
                        input_dim=self.input_dim,
                        output_dim=self.output_dim,
                        expert_id=expert_id,
                        **expert_kwargs.get('focal', {})
                    )
                else:
                    raise ValueError(f"Unsupported expert type: {expert_type}")
                    
                self.experts.append(expert)
                self.expert_types.append(expert_type)
                expert_id += 1
                
    def _initialize_collaboration_matrix(self) -> np.ndarray:
        """Initialize collaboration matrix between expert types."""
        # Higher values indicate better collaboration potential
        collaboration_map = {
            (ExpertType.DEEP, ExpertType.ATTENTION): 0.8,
            (ExpertType.DEEP, ExpertType.FOCAL): 0.6,
            (ExpertType.ATTENTION, ExpertType.FOCAL): 0.7,
            (ExpertType.ATTENTION, ExpertType.ATTENTION): 0.9,
            (ExpertType.DEEP, ExpertType.DEEP): 0.7,
            (ExpertType.FOCAL, ExpertType.FOCAL): 0.6,
        }
        
        # Initialize symmetric collaboration matrix
        matrix = np.eye(self.total_experts, dtype=np.float32)
        
        for i, type_i in enumerate(self.expert_types):
            for j, type_j in enumerate(self.expert_types):
                if i != j:
                    # Look up collaboration strength
                    key = (type_i, type_j) if (type_i, type_j) in collaboration_map else (type_j, type_i)
                    if key in collaboration_map:
                        matrix[i, j] = collaboration_map[key]
                    else:
                        matrix[i, j] = 0.3  # Default low collaboration
                        
        return matrix
        
    def route_collaborative(
        self,
        inputs: np.ndarray,
        task_capabilities: List[ExpertCapability],
        num_experts: int = 2,
        collaboration_weight: float = 0.3
    ) -> Tuple[List[int], np.ndarray, Dict[str, Any]]:
        """
        Route to experts considering both individual capability and collaboration.
        
        Args:
            inputs: Input tensor
            task_capabilities: Required capabilities for the task
            num_experts: Number of experts to select
            collaboration_weight: Weight for collaboration in routing decision
            
        Returns:
            expert_indices: List of selected expert indices
            expert_weights: Routing weights for selected experts
            routing_info: Collaborative routing information
        """
        batch_size, seq_len, _ = inputs.shape
        
        # Compute individual expert scores
        individual_scores = np.zeros(self.total_experts)
        
        for i, expert in enumerate(self.experts):
            # Base capability score
            capability_score = sum(
                expert.get_specialization_score(cap) for cap in task_capabilities
            ) / len(task_capabilities)
            
            # Performance history
            performance_score = self.expert_performance[i]
            
            individual_scores[i] = 0.7 * capability_score + 0.3 * performance_score
            
        # Compute collaboration scores
        collaboration_scores = np.zeros((self.total_experts, self.total_experts))
        
        for i in range(self.total_experts):
            for j in range(self.total_experts):
                if i != j:
                    # Collaboration potential between experts i and j
                    collab_potential = self.collaboration_matrix[i, j]
                    
                    # Combined individual strengths
                    combined_strength = (individual_scores[i] + individual_scores[j]) / 2
                    
                    collaboration_scores[i, j] = collab_potential * combined_strength
                    
        # Select optimal expert combination
        best_combination = None
        best_score = -1
        
        # Try all combinations of num_experts
        from itertools import combinations
        
        for expert_combo in combinations(range(self.total_experts), num_experts):
            # Individual contribution
            individual_contrib = sum(individual_scores[i] for i in expert_combo) / num_experts
            
            # Collaboration contribution
            collab_contrib = 0
            if num_experts > 1:
                for i in range(len(expert_combo)):
                    for j in range(i + 1, len(expert_combo)):
                        collab_contrib += collaboration_scores[expert_combo[i], expert_combo[j]]
                collab_contrib /= (num_experts * (num_experts - 1) / 2)  # Normalize
                
            # Combined score
            total_score = (1 - collaboration_weight) * individual_contrib + collaboration_weight * collab_contrib
            
            if total_score > best_score:
                best_score = total_score
                best_combination = expert_combo
                
        # Compute routing weights based on individual scores
        expert_indices = list(best_combination)
        raw_weights = np.array([individual_scores[i] for i in expert_indices])
        expert_weights = raw_weights / np.sum(raw_weights)
        
        # Record collaboration
        self.collaboration_history.append({
            'expert_combination': expert_indices,
            'expert_types': [self.expert_types[i] for i in expert_indices],
            'collaboration_score': best_score,
            'task_capabilities': task_capabilities
        })
        
        routing_info = {
            'selected_experts': expert_indices,
            'expert_types': [self.expert_types[i] for i in expert_indices],
            'individual_scores': individual_scores,
            'collaboration_matrix': self.collaboration_matrix[np.ix_(expert_indices, expert_indices)],
            'routing_strategy': 'collaborative',
            'collaboration_weight': collaboration_weight,
            'total_routing_score': best_score
        }
        
        return expert_indices, expert_weights, routing_info
        
    def forward_collaborative(
        self,
        inputs: np.ndarray,
        expert_indices: List[int],
        expert_weights: np.ndarray,
        fusion_strategy: str = "weighted_sum"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward pass through selected experts with collaborative fusion.
        
        Args:
            inputs: Input tensor
            expert_indices: Selected expert indices
            expert_weights: Expert routing weights
            fusion_strategy: How to fuse expert outputs
            
        Returns:
            fused_output: Combined expert outputs
            expert_info: Information from all experts
        """
        expert_outputs = []
        expert_infos = []
        
        # Forward through selected experts
        for idx in expert_indices:
            output, info = self.experts[idx].forward(inputs)
            expert_outputs.append(output)
            expert_infos.append(info)
            
        # Fuse expert outputs
        if fusion_strategy == "weighted_sum":
            fused_output = np.zeros_like(expert_outputs[0])
            for output, weight in zip(expert_outputs, expert_weights):
                fused_output += weight * output
                
        elif fusion_strategy == "attention_fusion":
            # Use attention mechanism to fuse outputs
            fused_output = self._attention_fusion(expert_outputs, expert_weights)
            
        elif fusion_strategy == "gated_fusion":
            # Use gating mechanism for fusion
            fused_output = self._gated_fusion(expert_outputs, expert_weights, inputs)
            
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
            
        # Combine expert information
        combined_info = {
            'fusion_strategy': fusion_strategy,
            'num_active_experts': len(expert_indices),
            'expert_types': [info['expert_type'] for info in expert_infos],
            'total_flops': sum(info['flops'] for info in expert_infos),
            'individual_outputs': expert_outputs,
            'expert_infos': expert_infos,
            'fusion_weights': expert_weights
        }
        
        return fused_output, combined_info
        
    def _attention_fusion(self, expert_outputs: List[np.ndarray], base_weights: np.ndarray) -> np.ndarray:
        """Fuse expert outputs using attention mechanism."""
        # Stack outputs
        stacked_outputs = np.stack(expert_outputs, axis=-1)  # (B, L, D, num_experts)
        
        # Compute attention weights (simplified)
        # In practice, this would use learnable parameters
        attention_logits = np.sum(stacked_outputs**2, axis=2)  # (B, L, num_experts)
        attention_weights = self._softmax(attention_logits, axis=-1)
        
        # Combine with base routing weights
        combined_weights = 0.7 * attention_weights + 0.3 * base_weights
        combined_weights = combined_weights / np.sum(combined_weights, axis=-1, keepdims=True)
        
        # Apply attention
        fused = np.sum(stacked_outputs * combined_weights[..., None, :], axis=-1)
        
        return fused
        
    def _gated_fusion(self, expert_outputs: List[np.ndarray], base_weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Fuse expert outputs using gating mechanism."""
        # Simple input-dependent gating
        input_gate = np.tanh(np.mean(inputs, axis=-1, keepdims=True))  # (B, L, 1)
        
        # Modulate weights based on input
        modulated_weights = base_weights * (1 + 0.2 * input_gate.squeeze(-1))
        modulated_weights = modulated_weights / np.sum(modulated_weights, axis=-1, keepdims=True)
        
        # Weighted combination
        fused_output = np.zeros_like(expert_outputs[0])
        for output, weight in zip(expert_outputs, modulated_weights.T):
            fused_output += output * weight[..., None]
            
        return fused_output
        
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        
    def update_expert_performance(self, expert_indices: List[int], performance_scores: List[float]):
        """Update expert performance based on feedback."""
        for idx, score in zip(expert_indices, performance_scores):
            # Exponential moving average
            alpha = 0.1
            self.expert_performance[idx] = alpha * score + (1 - alpha) * self.expert_performance[idx]
            
            # Update expert's internal specialization
            self.experts[idx].update_specialization(score)
            
    def get_expert_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about expert pool."""
        stats = {
            'total_experts': self.total_experts,
            'expert_type_distribution': {},
            'performance_stats': {
                'mean_performance': float(np.mean(self.expert_performance)),
                'std_performance': float(np.std(self.expert_performance)),
                'best_expert': int(np.argmax(self.expert_performance)),
                'worst_expert': int(np.argmin(self.expert_performance))
            },
            'activation_stats': {},
            'collaboration_stats': {}
        }
        
        # Expert type distribution
        for expert_type in ExpertType:
            count = sum(1 for t in self.expert_types if t == expert_type)
            if count > 0:
                stats['expert_type_distribution'][expert_type.value] = count
                
        # Activation statistics
        for expert_type in ExpertType:
            type_experts = [e for e, t in zip(self.experts, self.expert_types) if t == expert_type]
            if type_experts:
                activations = [e.activation_count for e in type_experts]
                total_flops = [e.total_flops for e in type_experts]
                
                stats['activation_stats'][expert_type.value] = {
                    'total_activations': sum(activations),
                    'mean_activations': float(np.mean(activations)),
                    'total_flops': sum(total_flops),
                    'mean_flops': float(np.mean(total_flops))
                }
                
        # Collaboration statistics
        if self.collaboration_history:
            type_combinations = {}
            for collab in self.collaboration_history:
                combo_key = tuple(sorted(t.value for t in collab['expert_types']))
                if combo_key not in type_combinations:
                    type_combinations[combo_key] = 0
                type_combinations[combo_key] += 1
                
            stats['collaboration_stats'] = {
                'total_collaborations': len(self.collaboration_history),
                'type_combinations': type_combinations,
                'average_collaboration_score': float(np.mean([
                    c['collaboration_score'] for c in self.collaboration_history
                ]))
            }
            
        return stats


# Export main classes
__all__ = [
    'ExpertType',
    'ExpertCapability', 
    'BaseExpert',
    'DeepExpert',
    'AttentionExpert',
    'FocalExpert',
    'HeterogeneousExpertPool'
]