"""Complexity estimation algorithms for dynamic expert routing."""

import math
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple

import numpy as np


class ComplexityEstimator(ABC):
    """Base class for input complexity estimation algorithms.
    
    Complexity estimators analyze input features to determine
    how many experts should process each token/sample.
    """
    
    def __init__(self, normalize: bool = True, epsilon: float = 1e-8):
        self.normalize = normalize
        self.epsilon = epsilon
        self._calibration_stats: Optional[Dict[str, float]] = None
    
    @abstractmethod
    def estimate(self, hidden_states: Any, **kwargs) -> Any:
        """Estimate complexity scores for input tokens.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            **kwargs: Additional context (attention weights, gradients, etc.)
            
        Returns:
            Complexity scores [batch, seq_len] in range [0, 1]
        """
        pass
    
    def calibrate(self, samples: list, target_range: Tuple[float, float] = (0.0, 1.0)):
        """Calibrate estimator on sample data for better score distribution."""
        scores = []
        for sample in samples:
            score = self.estimate(sample)
            scores.extend(score.flatten().tolist())
        
        scores = np.array(scores)
        self._calibration_stats = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'target_min': target_range[0],
            'target_max': target_range[1]
        }
    
    def _normalize_scores(self, scores: Any) -> Any:
        """Apply normalization to complexity scores."""
        if not self.normalize:
            return scores
        
        # Use calibration stats if available
        if self._calibration_stats:
            stats = self._calibration_stats
            normalized = (scores - stats['mean']) / (stats['std'] + self.epsilon)
            # Map to target range
            range_size = stats['target_max'] - stats['target_min']
            return stats['target_min'] + range_size * self._sigmoid(normalized)
        
        # Default normalization
        return self._sigmoid(scores)
    
    def _sigmoid(self, x: Any) -> Any:
        """Apply sigmoid activation (framework-agnostic)."""
        # This will be overridden in framework-specific implementations
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class GradientNormEstimator(ComplexityEstimator):
    """Estimates complexity based on gradient magnitude.
    
    The intuition is that tokens requiring more computation
    will have larger gradients during training.
    """
    
    def __init__(self, aggregation: str = 'l2', **kwargs):
        super().__init__(**kwargs)
        self.aggregation = aggregation
    
    def estimate(self, hidden_states: Any, gradients: Optional[Any] = None, **kwargs) -> Any:
        """Estimate complexity from gradient norms.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            gradients: Gradient tensor (if available)
            
        Returns:
            Complexity scores [batch, seq_len]
        """
        if gradients is None:
            # Fallback to activation magnitude as proxy
            gradients = hidden_states
        
        # Compute norm across hidden dimension
        if self.aggregation == 'l2':
            scores = self._compute_l2_norm(gradients)
        elif self.aggregation == 'l1':
            scores = self._compute_l1_norm(gradients)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        return self._normalize_scores(scores)
    
    def _compute_l2_norm(self, tensor: Any) -> Any:
        """Compute L2 norm across last dimension."""
        # Framework-specific implementation will override this
        return np.sqrt(np.sum(tensor**2, axis=-1))
    
    def _compute_l1_norm(self, tensor: Any) -> Any:
        """Compute L1 norm across last dimension."""
        return np.sum(np.abs(tensor), axis=-1)


class AttentionEntropyEstimator(ComplexityEstimator):
    """Estimates complexity based on attention pattern entropy.
    
    Low entropy (focused attention) suggests simple patterns,
    high entropy (diffuse attention) suggests complex patterns.
    """
    
    def __init__(self, head_aggregation: str = 'mean', **kwargs):
        super().__init__(**kwargs)
        self.head_aggregation = head_aggregation
    
    def estimate(self, hidden_states: Any, attention_weights: Optional[Any] = None, **kwargs) -> Any:
        """Estimate complexity from attention entropy.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            attention_weights: Attention weights [batch, num_heads, seq_len, seq_len]
            
        Returns:
            Complexity scores [batch, seq_len]
        """
        if attention_weights is None:
            # Fallback: compute pseudo-attention from hidden states
            attention_weights = self._compute_pseudo_attention(hidden_states)
        
        # Compute entropy for each token's attention distribution
        entropy_scores = self._compute_attention_entropy(attention_weights)
        
        return self._normalize_scores(entropy_scores)
    
    def _compute_pseudo_attention(self, hidden_states: Any) -> Any:
        """Compute pseudo-attention when real attention weights unavailable."""
        # Simple scaled dot-product attention
        d_model = hidden_states.shape[-1]
        scale = 1.0 / math.sqrt(d_model)
        
        # Q = K = V = hidden_states for self-attention
        scores = np.matmul(hidden_states, hidden_states.transpose(0, 2, 1)) * scale
        return self._softmax(scores, axis=-1)
    
    def _compute_attention_entropy(self, attention_weights: Any) -> Any:
        """Compute entropy of attention distributions."""
        # attention_weights: [batch, num_heads, seq_len, seq_len]
        log_probs = np.log(np.clip(attention_weights, self.epsilon, 1.0))
        entropy = -np.sum(attention_weights * log_probs, axis=-1)  # [batch, num_heads, seq_len]
        
        # Aggregate across heads
        if self.head_aggregation == 'mean':
            return np.mean(entropy, axis=1)  # [batch, seq_len]
        elif self.head_aggregation == 'max':
            return np.max(entropy, axis=1)
        else:
            raise ValueError(f"Unknown head aggregation: {self.head_aggregation}")
    
    def _softmax(self, x: Any, axis: int = -1) -> Any:
        """Compute softmax (framework-agnostic)."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class PerplexityProxyEstimator(ComplexityEstimator):
    """Estimates complexity based on model confidence (perplexity proxy).
    
    High confidence predictions suggest simple patterns,
    low confidence suggests complex patterns requiring more experts.
    """
    
    def __init__(self, temperature: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
    
    def estimate(self, hidden_states: Any, logits: Optional[Any] = None, **kwargs) -> Any:
        """Estimate complexity from prediction confidence.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            logits: Model predictions [batch, seq_len, vocab_size]
            
        Returns:
            Complexity scores [batch, seq_len]
        """
        if logits is None:
            # Fallback: project hidden states to vocab space
            logits = self._project_to_vocab(hidden_states)
        
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        
        # Compute prediction entropy (uncertainty)
        probs = self._softmax(scaled_logits, axis=-1)
        log_probs = np.log(np.clip(probs, self.epsilon, 1.0))
        entropy = -np.sum(probs * log_probs, axis=-1)  # [batch, seq_len]
        
        # High entropy = high complexity
        return self._normalize_scores(entropy)
    
    def _project_to_vocab(self, hidden_states: Any) -> Any:
        """Project hidden states to vocabulary space (simplified)."""
        # Simplified projection - in practice this would use the model's LM head
        vocab_size = 32000  # Typical vocab size
        d_model = hidden_states.shape[-1]
        
        # Random projection as placeholder
        np.random.seed(42)  # Deterministic for testing
        projection = np.random.randn(d_model, vocab_size) * 0.02
        return np.dot(hidden_states, projection)
    
    def _softmax(self, x: Any, axis: int = -1) -> Any:
        """Compute softmax (framework-agnostic)."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class ThresholdEstimator(ComplexityEstimator):
    """Simple threshold-based estimator for testing and baselines.
    
    Assigns fixed complexity scores based on configurable rules.
    """
    
    def __init__(self, threshold_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold_fn = threshold_fn or self._default_threshold
    
    def estimate(self, hidden_states: Any, **kwargs) -> Any:
        """Apply threshold function to determine complexity."""
        scores = self.threshold_fn(hidden_states, **kwargs)
        return self._normalize_scores(scores)
    
    def _default_threshold(self, hidden_states: Any, **kwargs) -> Any:
        """Default threshold: complexity based on sequence position."""
        batch_size, seq_len = hidden_states.shape[:2]
        # Later tokens are more complex (simple heuristic)
        position_scores = np.linspace(0.2, 0.8, seq_len)
        return np.tile(position_scores, (batch_size, 1))


# Registry for easy access to estimators
ESTIMATOR_REGISTRY = {
    'gradient_norm': GradientNormEstimator,
    'attention_entropy': AttentionEntropyEstimator,
    'perplexity_proxy': PerplexityProxyEstimator,
    'threshold': ThresholdEstimator,
}


def get_estimator(name: str, **kwargs) -> ComplexityEstimator:
    """Factory function to create complexity estimators."""
    if name not in ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown estimator: {name}. Available: {list(ESTIMATOR_REGISTRY.keys())}")
    
    return ESTIMATOR_REGISTRY[name](**kwargs)
