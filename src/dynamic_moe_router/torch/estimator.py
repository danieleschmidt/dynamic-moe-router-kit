"""PyTorch implementations of complexity estimators."""

import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..estimator import ComplexityEstimator


class TorchComplexityEstimator(ComplexityEstimator, nn.Module):
    """Base PyTorch complexity estimator with neural network capabilities."""

    def __init__(self, **kwargs):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TorchComplexityEstimator")

        ComplexityEstimator.__init__(self, **kwargs)
        nn.Module.__init__(self)

    def estimate_torch(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """PyTorch-native complexity estimation."""
        return self.estimate(hidden_states, **kwargs)

    def _sigmoid_torch(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch sigmoid activation."""
        return torch.sigmoid(x)

    def _normalize_scores_torch(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply normalization to complexity scores using PyTorch."""
        if not self.normalize:
            return scores

        # Use calibration stats if available
        if self._calibration_stats:
            stats = self._calibration_stats
            normalized = (scores - stats['mean']) / (stats['std'] + self.epsilon)
            # Map to target range
            range_size = stats['target_max'] - stats['target_min']
            return stats['target_min'] + range_size * self._sigmoid_torch(normalized)

        # Default normalization
        return self._sigmoid_torch(scores)


class TorchGradientNormEstimator(TorchComplexityEstimator):
    """PyTorch implementation of gradient norm estimator."""

    def __init__(self, aggregation: str = 'l2', **kwargs):
        super().__init__(**kwargs)
        self.aggregation = aggregation

    def estimate(self, hidden_states: torch.Tensor, gradients: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Estimate complexity from gradient norms using PyTorch."""
        if gradients is None:
            # Fallback to activation magnitude as proxy
            gradients = hidden_states

        # Compute norm across hidden dimension
        if self.aggregation == 'l2':
            scores = torch.norm(gradients, p=2, dim=-1)
        elif self.aggregation == 'l1':
            scores = torch.norm(gradients, p=1, dim=-1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return self._normalize_scores_torch(scores)

    def estimate_torch(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """PyTorch-optimized gradient norm estimation."""
        return self.estimate(hidden_states, **kwargs)


class TorchAttentionEntropyEstimator(TorchComplexityEstimator):
    """PyTorch implementation of attention entropy estimator."""

    def __init__(self, head_aggregation: str = 'mean', **kwargs):
        super().__init__(**kwargs)
        self.head_aggregation = head_aggregation

    def estimate(self, hidden_states: torch.Tensor, attention_weights: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Estimate complexity from attention entropy using PyTorch."""
        if attention_weights is None:
            # Fallback: compute pseudo-attention from hidden states
            attention_weights = self._compute_pseudo_attention_torch(hidden_states)

        # Compute entropy for each token's attention distribution
        entropy_scores = self._compute_attention_entropy_torch(attention_weights)

        return self._normalize_scores_torch(entropy_scores)

    def _compute_pseudo_attention_torch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute pseudo-attention using PyTorch operations."""
        # Simple scaled dot-product attention
        d_model = hidden_states.shape[-1]
        scale = 1.0 / math.sqrt(d_model)

        # Q = K = V = hidden_states for self-attention
        scores = torch.matmul(hidden_states, hidden_states.transpose(-2, -1)) * scale
        return F.softmax(scores, dim=-1)

    def _compute_attention_entropy_torch(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention distributions using PyTorch."""
        # attention_weights: [batch, num_heads, seq_len, seq_len] or [batch, seq_len, seq_len]

        # Ensure we have the right dimensions
        if attention_weights.dim() == 3:
            # Add head dimension if missing
            attention_weights = attention_weights.unsqueeze(1)

        # Compute entropy: -sum(p * log(p))
        log_probs = torch.log(torch.clamp(attention_weights, min=self.epsilon))
        entropy = -torch.sum(attention_weights * log_probs, dim=-1)  # [batch, num_heads, seq_len]

        # Aggregate across heads
        if self.head_aggregation == 'mean':
            return torch.mean(entropy, dim=1)  # [batch, seq_len]
        elif self.head_aggregation == 'max':
            return torch.max(entropy, dim=1)[0]
        else:
            raise ValueError(f"Unknown head aggregation: {self.head_aggregation}")

    def estimate_torch(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """PyTorch-optimized attention entropy estimation."""
        return self.estimate(hidden_states, **kwargs)


class TorchPerplexityProxyEstimator(TorchComplexityEstimator):
    """PyTorch implementation of perplexity proxy estimator."""

    def __init__(self, temperature: float = 1.0, vocab_size: int = 32000, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.vocab_size = vocab_size

        # Learnable projection to vocabulary space (optional)
        self.vocab_projection = None

    def set_vocab_projection(self, input_dim: int):
        """Set up learnable vocabulary projection."""
        if self.vocab_projection is None:
            self.vocab_projection = nn.Linear(input_dim, self.vocab_size, bias=False)

    def estimate(self, hidden_states: torch.Tensor, logits: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Estimate complexity from prediction confidence using PyTorch."""
        if logits is None:
            # Use learnable projection if available, otherwise random projection
            if self.vocab_projection is not None:
                logits = self.vocab_projection(hidden_states)
            else:
                logits = self._project_to_vocab_torch(hidden_states)

        # Apply temperature scaling
        scaled_logits = logits / self.temperature

        # Compute prediction entropy (uncertainty)
        probs = F.softmax(scaled_logits, dim=-1)
        log_probs = torch.log(torch.clamp(probs, min=self.epsilon))
        entropy = -torch.sum(probs * log_probs, dim=-1)  # [batch, seq_len]

        # High entropy = high complexity
        return self._normalize_scores_torch(entropy)

    def _project_to_vocab_torch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary space using PyTorch."""
        batch_size, seq_len, d_model = hidden_states.shape
        device = hidden_states.device

        # Create or reuse random projection matrix
        if not hasattr(self, '_projection_matrix') or self._projection_matrix.device != device:
            torch.manual_seed(42)  # Deterministic for testing
            self._projection_matrix = torch.randn(d_model, self.vocab_size, device=device) * 0.02

        return torch.matmul(hidden_states, self._projection_matrix)

    def estimate_torch(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """PyTorch-optimized perplexity proxy estimation."""
        return self.estimate(hidden_states, **kwargs)


class TorchLearnedComplexityEstimator(TorchComplexityEstimator):
    """Learned complexity estimator using a small neural network."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Small MLP for complexity prediction
        self.complexity_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def estimate(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Estimate complexity using learned network."""
        # Pass through complexity network
        complexity_scores = self.complexity_net(hidden_states)  # [batch, seq_len, 1]
        return complexity_scores.squeeze(-1)  # [batch, seq_len]

    def estimate_torch(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """PyTorch-optimized learned complexity estimation."""
        return self.estimate(hidden_states, **kwargs)

    def train_on_batch(
        self,
        hidden_states: torch.Tensor,
        target_complexity: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Train the complexity estimator on a batch of data."""
        self.train()
        optimizer.zero_grad()

        # Forward pass
        predicted_complexity = self.estimate(hidden_states)

        # Compute loss (MSE)
        loss = F.mse_loss(predicted_complexity, target_complexity)

        # Backward pass
        loss.backward()
        optimizer.step()

        return loss.item()


class TorchMultiModalComplexityEstimator(TorchComplexityEstimator):
    """Multi-modal complexity estimator combining multiple signals."""

    def __init__(self, estimators: list, combination_method: str = 'weighted_sum', **kwargs):
        super().__init__(**kwargs)
        self.estimators = nn.ModuleList(estimators)
        self.combination_method = combination_method

        if combination_method == 'weighted_sum':
            # Learnable weights for combining estimators
            self.combination_weights = nn.Parameter(torch.ones(len(estimators)) / len(estimators))
        elif combination_method == 'learned':
            # Small MLP to combine estimator outputs
            self.combination_net = nn.Sequential(
                nn.Linear(len(estimators), len(estimators) * 2),
                nn.ReLU(),
                nn.Linear(len(estimators) * 2, 1),
                nn.Sigmoid()
            )

    def estimate(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Combine multiple complexity estimators."""
        # Get estimates from all sub-estimators
        estimates = []
        for estimator in self.estimators:
            if hasattr(estimator, 'estimate_torch'):
                estimate = estimator.estimate_torch(hidden_states, **kwargs)
            else:
                estimate = estimator.estimate(hidden_states, **kwargs)
            estimates.append(estimate)

        # Stack estimates: [batch, seq_len, num_estimators]
        stacked_estimates = torch.stack(estimates, dim=-1)

        # Combine estimates
        if self.combination_method == 'mean':
            return torch.mean(stacked_estimates, dim=-1)
        elif self.combination_method == 'max':
            return torch.max(stacked_estimates, dim=-1)[0]
        elif self.combination_method == 'weighted_sum':
            weights = F.softmax(self.combination_weights, dim=0)
            return torch.sum(stacked_estimates * weights, dim=-1)
        elif self.combination_method == 'learned':
            batch_size, seq_len, num_estimators = stacked_estimates.shape
            # Reshape for MLP
            flat_estimates = stacked_estimates.view(-1, num_estimators)
            combined = self.combination_net(flat_estimates)  # [batch*seq_len, 1]
            return combined.view(batch_size, seq_len)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")

    def estimate_torch(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """PyTorch-optimized multi-modal estimation."""
        return self.estimate(hidden_states, **kwargs)
