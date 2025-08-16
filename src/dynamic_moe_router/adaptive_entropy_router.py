"""Advanced Adaptive Entropy-Based Dynamic Router with Novel Confidence Algorithms.

This module implements cutting-edge research from 2024 papers on dynamic MoE routing,
including:
1. Confidence-based expert selection (Huang et al. 2024)
2. Entropy threshold routing for task complexity
3. Expert-token resonance with bidirectional selection
4. Similarity/attention-aware routing for entropy reduction

Research Foundation:
- "Harder Tasks Need More Experts: Dynamic Routing in MoE Models" (ACL 2024)
- "Expert-Token Resonance MoE: Bidirectional Routing with Efficiency Affinity-Driven Active Selection"
- Entropy reduction through similarity-aware routing mechanisms
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

logger = logging.getLogger(__name__)


class ConfidenceBasedRouter:
    """Implements confidence-based dynamic expert selection from Huang et al. 2024."""
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        min_experts: int = 1,
        max_experts: Optional[int] = None,
        confidence_threshold: float = 0.9,
        entropy_threshold: float = 0.9,
        temperature: float = 1.0,
        enable_layer_adaptive: bool = True,
    ):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.min_experts = min_experts
        self.max_experts = max_experts or num_experts
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.temperature = temperature
        self.enable_layer_adaptive = enable_layer_adaptive
        
        # Initialize router network weights (simplified for research)
        self.router_weights = np.random.normal(0, 0.02, (input_dim, num_experts))
        self.confidence_weights = np.random.normal(0, 0.02, (input_dim, 1))
        
        # Metrics tracking
        self.routing_decisions = []
        self.confidence_scores = []
        self.entropy_values = []
        
    def _compute_expert_scores(self, inputs: np.ndarray) -> np.ndarray:
        """Compute raw expert affinity scores."""
        # inputs: [batch_size, seq_len, input_dim]
        scores = np.matmul(inputs, self.router_weights)  # [batch, seq, num_experts]
        return scores / self.temperature
    
    def _compute_confidence(self, inputs: np.ndarray) -> np.ndarray:
        """Compute confidence in expert selection for each token."""
        confidence_logits = np.matmul(inputs, self.confidence_weights)  # [batch, seq, 1]
        confidence = 1.0 / (1.0 + np.exp(-confidence_logits))  # Sigmoid
        return confidence.squeeze(-1)  # [batch, seq]
    
    def _compute_entropy(self, probabilities: np.ndarray) -> np.ndarray:
        """Compute entropy of expert probability distribution."""
        # Add small epsilon to prevent log(0)
        eps = 1e-10
        prob_safe = np.clip(probabilities, eps, 1.0 - eps)
        entropy = -np.sum(prob_safe * np.log(prob_safe), axis=-1)
        return entropy
    
    def _adaptive_expert_selection(
        self, 
        expert_scores: np.ndarray, 
        confidence: np.ndarray,
        layer_depth: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Dynamic expert selection based on confidence and task complexity."""
        batch_size, seq_len, num_experts = expert_scores.shape
        
        # Convert scores to probabilities
        expert_probs = np.exp(expert_scores) / np.sum(np.exp(expert_scores), axis=-1, keepdims=True)
        
        # Compute entropy for each token
        entropy = self._compute_entropy(expert_probs)
        
        # Determine dynamic k based on multiple factors
        dynamic_k = self._compute_dynamic_k(confidence, entropy, layer_depth)
        
        # Select experts using cumulative probability threshold
        selected_experts, expert_weights = self._threshold_based_selection(
            expert_probs, dynamic_k, confidence
        )
        
        # Collect metrics
        routing_info = {
            'avg_experts_per_token': np.mean(dynamic_k),
            'confidence_scores': confidence,
            'entropy_values': entropy,
            'expert_utilization': self._compute_expert_utilization(selected_experts),
            'flop_reduction': self._estimate_flop_reduction(dynamic_k)
        }
        
        return selected_experts, expert_weights, routing_info
    
    def _compute_dynamic_k(
        self, 
        confidence: np.ndarray, 
        entropy: np.ndarray,
        layer_depth: Optional[float] = None
    ) -> np.ndarray:
        """Compute dynamic number of experts based on confidence and entropy."""
        batch_size, seq_len = confidence.shape
        
        # Base k selection using confidence
        # Higher confidence -> fewer experts needed
        # Lower confidence -> more experts needed
        confidence_factor = 1.0 - confidence  # Invert confidence
        
        # Entropy factor: high entropy indicates uncertainty, need more experts
        entropy_normalized = entropy / np.log(self.num_experts)  # Normalize to [0, 1]
        
        # Combine factors
        complexity_score = 0.6 * confidence_factor + 0.4 * entropy_normalized
        
        # Layer-adaptive scaling (deeper layers can use more experts)
        if self.enable_layer_adaptive and layer_depth is not None:
            layer_factor = 0.5 + 0.5 * layer_depth  # Scale from 0.5 to 1.0
            complexity_score *= layer_factor
        
        # Map to expert count range
        expert_range = self.max_experts - self.min_experts
        dynamic_k = self.min_experts + complexity_score * expert_range
        
        # Round and clip to valid range
        dynamic_k = np.round(dynamic_k).astype(int)
        dynamic_k = np.clip(dynamic_k, self.min_experts, self.max_experts)
        
        return dynamic_k
    
    def _threshold_based_selection(
        self, 
        expert_probs: np.ndarray, 
        dynamic_k: np.ndarray,
        confidence: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select experts using cumulative probability threshold method."""
        batch_size, seq_len, num_experts = expert_probs.shape
        
        # Sort experts by probability (descending)
        sorted_indices = np.argsort(-expert_probs, axis=-1)
        sorted_probs = np.take_along_axis(expert_probs, sorted_indices, axis=-1)
        
        # For each token, select top-k experts based on dynamic_k
        selected_experts = np.zeros((batch_size, seq_len, self.max_experts), dtype=int)
        expert_weights = np.zeros((batch_size, seq_len, self.max_experts), dtype=np.float32)
        
        for b in range(batch_size):
            for s in range(seq_len):
                k = dynamic_k[b, s]
                
                # Select top-k experts
                top_k_indices = sorted_indices[b, s, :k]
                top_k_weights = sorted_probs[b, s, :k]
                
                # Renormalize weights
                weight_sum = np.sum(top_k_weights)
                if weight_sum > 0:
                    top_k_weights = top_k_weights / weight_sum
                
                # Store results
                selected_experts[b, s, :k] = top_k_indices
                expert_weights[b, s, :k] = top_k_weights
        
        return selected_experts, expert_weights
    
    def _compute_expert_utilization(self, selected_experts: np.ndarray) -> Dict[str, float]:
        """Compute expert utilization statistics."""
        # Count how often each expert is selected
        expert_counts = np.zeros(self.num_experts)
        total_selections = 0
        
        batch_size, seq_len, max_selected = selected_experts.shape
        
        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(max_selected):
                    expert_id = selected_experts[b, s, k]
                    if expert_id < self.num_experts:  # Valid expert
                        expert_counts[expert_id] += 1
                        total_selections += 1
        
        utilization = expert_counts / max(total_selections, 1)
        
        return {
            'mean_utilization': np.mean(utilization),
            'std_utilization': np.std(utilization),
            'max_utilization': np.max(utilization),
            'min_utilization': np.min(utilization),
            'gini_coefficient': self._compute_gini(utilization)
        }
    
    def _compute_gini(self, utilization: np.ndarray) -> float:
        """Compute Gini coefficient for expert utilization inequality."""
        sorted_util = np.sort(utilization)
        n = len(sorted_util)
        cumsum = np.cumsum(sorted_util)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0
    
    def _estimate_flop_reduction(self, dynamic_k: np.ndarray) -> float:
        """Estimate FLOP reduction compared to static routing."""
        avg_experts_used = np.mean(dynamic_k)
        static_experts = self.max_experts  # Assume static uses max experts
        reduction = 1.0 - (avg_experts_used / static_experts)
        return max(0.0, reduction)
    
    def forward(
        self, 
        inputs: np.ndarray, 
        layer_depth: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Forward pass with adaptive entropy-based routing."""
        if inputs.ndim != 3:
            raise ValueError(f"Expected 3D input [batch, seq, dim], got {inputs.shape}")
        
        # Compute expert scores and confidence
        expert_scores = self._compute_expert_scores(inputs)
        confidence = self._compute_confidence(inputs)
        
        # Perform adaptive expert selection
        selected_experts, expert_weights, routing_info = self._adaptive_expert_selection(
            expert_scores, confidence, layer_depth
        )
        
        # Store metrics for analysis
        self.routing_decisions.append(routing_info)
        
        return selected_experts, expert_weights, routing_info


class ExpertTokenResonanceRouter:
    """Implements bidirectional expert-token selection with resonance mechanism."""
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        expert_capacity_factor: float = 1.25,
        resonance_threshold: float = 0.5,
        bidirectional_strength: float = 0.3,
    ):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.expert_capacity_factor = expert_capacity_factor
        self.resonance_threshold = resonance_threshold
        self.bidirectional_strength = bidirectional_strength
        
        # Initialize resonance matrices
        self.token_to_expert_weights = np.random.normal(0, 0.02, (input_dim, num_experts))
        self.expert_to_token_weights = np.random.normal(0, 0.02, (num_experts, input_dim))
        
    def _compute_resonance_scores(
        self, 
        inputs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute bidirectional resonance scores between tokens and experts."""
        batch_size, seq_len, input_dim = inputs.shape
        
        # Token-to-expert affinity
        token_expert_scores = np.matmul(inputs, self.token_to_expert_weights)
        
        # Expert-to-token affinity (transpose expert weights for broadcasting)
        expert_token_scores = np.matmul(
            inputs, 
            self.expert_to_token_weights.T
        )
        
        # Combine bidirectional scores with resonance
        resonance_scores = (
            (1 - self.bidirectional_strength) * token_expert_scores +
            self.bidirectional_strength * expert_token_scores
        )
        
        return resonance_scores, token_expert_scores
    
    def forward(
        self, 
        inputs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Forward pass with expert-token resonance routing."""
        resonance_scores, base_scores = self._compute_resonance_scores(inputs)
        
        # Convert to probabilities
        expert_probs = np.exp(resonance_scores) / np.sum(
            np.exp(resonance_scores), axis=-1, keepdims=True
        )
        
        # Apply resonance threshold for expert selection
        selected_mask = expert_probs > self.resonance_threshold
        
        # Compute expert utilization and capacity
        expert_capacity = int(
            inputs.shape[0] * inputs.shape[1] * self.expert_capacity_factor / self.num_experts
        )
        
        routing_info = {
            'resonance_scores': resonance_scores,
            'expert_probabilities': expert_probs,
            'selection_mask': selected_mask,
            'expert_capacity': expert_capacity,
            'avg_experts_per_token': np.mean(np.sum(selected_mask, axis=-1))
        }
        
        return selected_mask.astype(int), expert_probs, routing_info


class SimilarityAwareRouter:
    """Implements similarity/attention-aware routing for entropy reduction."""
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        similarity_metric: str = "cosine",
        attention_heads: int = 8,
        entropy_regularization: float = 0.1,
    ):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.similarity_metric = similarity_metric
        self.attention_heads = attention_heads
        self.entropy_regularization = entropy_regularization
        
        # Initialize expert prototypes for similarity computation
        self.expert_prototypes = np.random.normal(
            0, 0.02, (num_experts, input_dim)
        )
        
        # Multi-head attention for routing
        self.attention_weights = np.random.normal(
            0, 0.02, (attention_heads, input_dim, input_dim)
        )
        
    def _compute_similarity_scores(self, inputs: np.ndarray) -> np.ndarray:
        """Compute similarity between inputs and expert prototypes."""
        if self.similarity_metric == "cosine":
            # Normalize inputs and prototypes
            inputs_norm = inputs / np.linalg.norm(inputs, axis=-1, keepdims=True)
            prototypes_norm = self.expert_prototypes / np.linalg.norm(
                self.expert_prototypes, axis=-1, keepdims=True
            )
            
            # Compute cosine similarity
            similarity = np.matmul(inputs_norm, prototypes_norm.T)
            
        elif self.similarity_metric == "euclidean":
            # Compute negative euclidean distance
            distances = np.linalg.norm(
                inputs[..., np.newaxis, :] - self.expert_prototypes[np.newaxis, np.newaxis, :, :],
                axis=-1
            )
            similarity = -distances
            
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        return similarity
    
    def _apply_attention_routing(self, inputs: np.ndarray) -> np.ndarray:
        """Apply multi-head attention for enhanced routing."""
        batch_size, seq_len, input_dim = inputs.shape
        
        attention_outputs = []
        
        for head in range(self.attention_heads):
            # Apply attention transformation
            attended = np.matmul(inputs, self.attention_weights[head])
            attention_outputs.append(attended)
        
        # Concatenate and average attention heads
        attended_inputs = np.mean(attention_outputs, axis=0)
        
        return attended_inputs
    
    def forward(
        self, 
        inputs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Forward pass with similarity-aware routing."""
        # Apply attention routing
        attended_inputs = self._apply_attention_routing(inputs)
        
        # Compute similarity scores
        similarity_scores = self._compute_similarity_scores(attended_inputs)
        
        # Convert to probabilities with temperature scaling
        expert_probs = np.exp(similarity_scores) / np.sum(
            np.exp(similarity_scores), axis=-1, keepdims=True
        )
        
        # Compute entropy for regularization
        entropy = -np.sum(expert_probs * np.log(expert_probs + 1e-10), axis=-1)
        
        # Select experts based on similarity threshold
        threshold = np.percentile(similarity_scores, 70, axis=-1, keepdims=True)
        selected_experts = (similarity_scores >= threshold).astype(int)
        
        routing_info = {
            'similarity_scores': similarity_scores,
            'expert_probabilities': expert_probs,
            'entropy_values': entropy,
            'mean_entropy': np.mean(entropy),
            'entropy_reduction': max(0, np.log(self.num_experts) - np.mean(entropy))
        }
        
        return selected_experts, expert_probs, routing_info


class AdaptiveEntropyRouterEnsemble:
    """Ensemble of advanced routing algorithms for comprehensive evaluation."""
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        enable_confidence_routing: bool = True,
        enable_resonance_routing: bool = True,
        enable_similarity_routing: bool = True,
        ensemble_weight_confidence: float = 0.4,
        ensemble_weight_resonance: float = 0.3,
        ensemble_weight_similarity: float = 0.3,
    ):
        self.input_dim = input_dim
        self.num_experts = num_experts
        
        # Initialize routers
        self.routers = {}
        self.ensemble_weights = {}
        
        if enable_confidence_routing:
            self.routers['confidence'] = ConfidenceBasedRouter(input_dim, num_experts)
            self.ensemble_weights['confidence'] = ensemble_weight_confidence
            
        if enable_resonance_routing:
            self.routers['resonance'] = ExpertTokenResonanceRouter(input_dim, num_experts)
            self.ensemble_weights['resonance'] = ensemble_weight_resonance
            
        if enable_similarity_routing:
            self.routers['similarity'] = SimilarityAwareRouter(input_dim, num_experts)
            self.ensemble_weights['similarity'] = ensemble_weight_similarity
        
        # Normalize ensemble weights
        total_weight = sum(self.ensemble_weights.values())
        self.ensemble_weights = {
            k: v / total_weight for k, v in self.ensemble_weights.items()
        }
        
    def forward(
        self, 
        inputs: np.ndarray, 
        layer_depth: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Forward pass with ensemble routing."""
        router_outputs = {}
        
        # Run all enabled routers
        for name, router in self.routers.items():
            if name == 'confidence':
                experts, weights, info = router.forward(inputs, layer_depth)
            else:
                experts, weights, info = router.forward(inputs)
            
            router_outputs[name] = {
                'experts': experts,
                'weights': weights,
                'info': info
            }
        
        # Ensemble combination
        first_router_output = router_outputs[list(self.routers.keys())[0]]
        ensemble_weights = np.zeros_like(first_router_output['weights'], dtype=np.float32)
        ensemble_experts = np.zeros_like(first_router_output['experts'], dtype=np.float32)
        
        for name, output in router_outputs.items():
            weight = self.ensemble_weights[name]
            ensemble_weights += weight * output['weights']
            # For experts, use majority voting
            ensemble_experts += weight * output['experts'].astype(np.float32)
        
        # Convert ensemble experts to binary selection
        ensemble_experts = (ensemble_experts > 0.5).astype(int)
        
        # Combine routing info
        routing_info = {
            'ensemble_weights': self.ensemble_weights,
            'router_outputs': {name: output['info'] for name, output in router_outputs.items()},
            'avg_experts_per_token': np.mean(np.sum(ensemble_experts, axis=-1)),
            'ensemble_entropy': self._compute_ensemble_entropy(ensemble_weights)
        }
        
        return ensemble_experts, ensemble_weights, routing_info
    
    def _compute_ensemble_entropy(self, weights: np.ndarray) -> float:
        """Compute entropy of ensemble routing decisions."""
        eps = 1e-10
        weights_safe = np.clip(weights, eps, 1.0 - eps)
        entropy = -np.sum(weights_safe * np.log(weights_safe), axis=-1)
        return np.mean(entropy)


# Comparative study framework
class RouterComparativeStudy:
    """Framework for comparative evaluation of routing algorithms."""
    
    def __init__(self, input_dim: int, num_experts: int):
        self.input_dim = input_dim
        self.num_experts = num_experts
        
        # Initialize all routers for comparison
        self.routers = {
            'confidence_based': ConfidenceBasedRouter(input_dim, num_experts),
            'expert_token_resonance': ExpertTokenResonanceRouter(input_dim, num_experts),
            'similarity_aware': SimilarityAwareRouter(input_dim, num_experts),
            'ensemble': AdaptiveEntropyRouterEnsemble(input_dim, num_experts)
        }
        
        self.results = {}
    
    def run_comparative_study(
        self, 
        test_inputs: List[np.ndarray], 
        input_complexities: List[str]
    ) -> Dict[str, Any]:
        """Run comprehensive comparative study."""
        logger.info("Starting comparative study of routing algorithms")
        
        study_results = {}
        
        for router_name, router in self.routers.items():
            router_results = {
                'flop_reductions': [],
                'expert_utilizations': [],
                'entropy_values': [],
                'routing_decisions': []
            }
            
            for i, (inputs, complexity) in enumerate(zip(test_inputs, input_complexities)):
                try:
                    if router_name == 'confidence_based' or router_name == 'ensemble':
                        experts, weights, info = router.forward(inputs, layer_depth=0.5)
                    else:
                        experts, weights, info = router.forward(inputs)
                    
                    # Extract metrics
                    if 'flop_reduction' in info:
                        router_results['flop_reductions'].append(info['flop_reduction'])
                    if 'expert_utilization' in info:
                        router_results['expert_utilizations'].append(info['expert_utilization'])
                    if 'entropy_values' in info:
                        router_results['entropy_values'].append(np.mean(info['entropy_values']))
                    elif 'mean_entropy' in info:
                        router_results['entropy_values'].append(info['mean_entropy'])
                    
                    router_results['routing_decisions'].append({
                        'input_complexity': complexity,
                        'experts_selected': np.mean(np.sum(experts, axis=-1)),
                        'routing_info': info
                    })
                    
                except Exception as e:
                    logger.error(f"Error in {router_name}: {e}")
            
            study_results[router_name] = router_results
        
        # Compute comparative metrics
        comparison_summary = self._generate_comparison_summary(study_results)
        
        return {
            'detailed_results': study_results,
            'comparison_summary': comparison_summary,
            'statistical_significance': self._compute_statistical_significance(study_results)
        }
    
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary comparison across algorithms."""
        summary = {}
        
        for router_name, router_results in results.items():
            summary[router_name] = {
                'avg_flop_reduction': np.mean(router_results.get('flop_reductions', [0])),
                'avg_entropy': np.mean(router_results.get('entropy_values', [0])),
                'routing_efficiency': len([
                    d for d in router_results.get('routing_decisions', [])
                    if d.get('experts_selected', 0) < self.num_experts * 0.5
                ]) / max(len(router_results.get('routing_decisions', [])), 1)
            }
        
        return summary
    
    def _compute_statistical_significance(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Compute statistical significance of performance differences."""
        # Simplified statistical analysis
        significance_results = {}
        
        baseline_name = 'confidence_based'
        if baseline_name not in results:
            return significance_results
        
        baseline_flops = results[baseline_name].get('flop_reductions', [])
        
        for router_name, router_results in results.items():
            if router_name == baseline_name:
                continue
                
            router_flops = router_results.get('flop_reductions', [])
            
            if len(baseline_flops) > 0 and len(router_flops) > 0:
                # Simple statistical test (in practice, use proper statistical tests)
                diff_mean = np.mean(router_flops) - np.mean(baseline_flops)
                diff_std = np.sqrt(np.var(router_flops) + np.var(baseline_flops))
                
                if diff_std > 0:
                    t_statistic = diff_mean / diff_std
                    significance_results[f"{router_name}_vs_{baseline_name}"] = abs(t_statistic)
                
        return significance_results