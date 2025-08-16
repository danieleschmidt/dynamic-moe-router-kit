"""Unit tests for adaptive entropy router implementations."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dynamic_moe_router.adaptive_entropy_router import (
    ConfidenceBasedRouter,
    ExpertTokenResonanceRouter,
    SimilarityAwareRouter,
    AdaptiveEntropyRouterEnsemble,
    RouterComparativeStudy
)


class TestConfidenceBasedRouter:
    """Test confidence-based dynamic routing."""
    
    def test_initialization(self):
        """Test router initialization."""
        router = ConfidenceBasedRouter(
            input_dim=768,
            num_experts=8,
            min_experts=1,
            max_experts=4
        )
        
        assert router.input_dim == 768
        assert router.num_experts == 8
        assert router.min_experts == 1
        assert router.max_experts == 4
    
    def test_forward_pass(self):
        """Test forward pass functionality."""
        router = ConfidenceBasedRouter(input_dim=64, num_experts=4)
        
        # Create test input
        batch_size, seq_len = 2, 8
        inputs = np.random.normal(0, 1, (batch_size, seq_len, 64)).astype(np.float32)
        
        # Run forward pass
        experts, weights, info = router.forward(inputs)
        
        # Check output shapes
        assert experts.shape == (batch_size, seq_len, router.max_experts)
        assert weights.shape == (batch_size, seq_len, router.max_experts)
        
        # Check info dictionary
        assert 'avg_experts_per_token' in info
        assert 'confidence_scores' in info
        assert 'entropy_values' in info
        assert 'flop_reduction' in info
    
    def test_dynamic_k_computation(self):
        """Test dynamic expert count computation."""
        router = ConfidenceBasedRouter(input_dim=64, num_experts=8, min_experts=1, max_experts=4)
        
        # High confidence should result in fewer experts
        high_confidence = np.ones((2, 8)) * 0.9
        low_entropy = np.ones((2, 8)) * 0.1
        
        dynamic_k = router._compute_dynamic_k(high_confidence, low_entropy)
        
        # Should use fewer experts for high confidence
        assert np.all(dynamic_k >= router.min_experts)
        assert np.all(dynamic_k <= router.max_experts)
        assert np.mean(dynamic_k) < 3.0  # Should be on lower end
    
    def test_entropy_computation(self):
        """Test entropy computation."""
        router = ConfidenceBasedRouter(input_dim=64, num_experts=4)
        
        # Uniform distribution should have high entropy
        uniform_probs = np.ones((2, 8, 4)) * 0.25
        entropy = router._compute_entropy(uniform_probs)
        
        expected_entropy = -4 * 0.25 * np.log(0.25)
        assert np.allclose(entropy, expected_entropy, atol=1e-6)


class TestExpertTokenResonanceRouter:
    """Test expert-token resonance routing."""
    
    def test_initialization(self):
        """Test router initialization."""
        router = ExpertTokenResonanceRouter(
            input_dim=768,
            num_experts=8,
            resonance_threshold=0.5
        )
        
        assert router.input_dim == 768
        assert router.num_experts == 8
        assert router.resonance_threshold == 0.5
    
    def test_forward_pass(self):
        """Test forward pass functionality."""
        router = ExpertTokenResonanceRouter(input_dim=64, num_experts=4)
        
        batch_size, seq_len = 2, 8
        inputs = np.random.normal(0, 1, (batch_size, seq_len, 64)).astype(np.float32)
        
        experts, weights, info = router.forward(inputs)
        
        # Check outputs
        assert experts.shape == (batch_size, seq_len, router.num_experts)
        assert weights.shape == (batch_size, seq_len, router.num_experts)
        assert 'resonance_scores' in info
        assert 'avg_experts_per_token' in info
    
    def test_resonance_computation(self):
        """Test bidirectional resonance computation."""
        router = ExpertTokenResonanceRouter(input_dim=64, num_experts=4, bidirectional_strength=0.3)
        
        inputs = np.random.normal(0, 1, (2, 8, 64)).astype(np.float32)
        resonance_scores, base_scores = router._compute_resonance_scores(inputs)
        
        assert resonance_scores.shape == (2, 8, 4)
        assert base_scores.shape == (2, 8, 4)
        
        # Resonance scores should be different from base scores due to bidirectional component
        assert not np.allclose(resonance_scores, base_scores)


class TestSimilarityAwareRouter:
    """Test similarity-aware routing."""
    
    def test_initialization(self):
        """Test router initialization."""
        router = SimilarityAwareRouter(
            input_dim=768,
            num_experts=8,
            similarity_metric="cosine"
        )
        
        assert router.input_dim == 768
        assert router.num_experts == 8
        assert router.similarity_metric == "cosine"
    
    def test_similarity_computation(self):
        """Test similarity score computation."""
        router = SimilarityAwareRouter(input_dim=64, num_experts=4, similarity_metric="cosine")
        
        inputs = np.random.normal(0, 1, (2, 8, 64)).astype(np.float32)
        similarity_scores = router._compute_similarity_scores(inputs)
        
        assert similarity_scores.shape == (2, 8, 4)
        # Cosine similarity should be in [-1, 1]
        assert np.all(similarity_scores >= -1.0)
        assert np.all(similarity_scores <= 1.0)
    
    def test_euclidean_similarity(self):
        """Test Euclidean distance-based similarity."""
        router = SimilarityAwareRouter(input_dim=64, num_experts=4, similarity_metric="euclidean")
        
        inputs = np.random.normal(0, 1, (2, 8, 64)).astype(np.float32)
        similarity_scores = router._compute_similarity_scores(inputs)
        
        assert similarity_scores.shape == (2, 8, 4)
        # Euclidean distances are negative, so similarity should be negative
        assert np.all(similarity_scores <= 0.0)
    
    def test_attention_routing(self):
        """Test multi-head attention routing."""
        router = SimilarityAwareRouter(input_dim=64, num_experts=4, attention_heads=4)
        
        inputs = np.random.normal(0, 1, (2, 8, 64)).astype(np.float32)
        attended_inputs = router._apply_attention_routing(inputs)
        
        assert attended_inputs.shape == inputs.shape


class TestAdaptiveEntropyRouterEnsemble:
    """Test ensemble routing."""
    
    def test_initialization(self):
        """Test ensemble initialization."""
        ensemble = AdaptiveEntropyRouterEnsemble(
            input_dim=768,
            num_experts=8,
            enable_confidence_routing=True,
            enable_resonance_routing=True,
            enable_similarity_routing=True
        )
        
        assert len(ensemble.routers) == 3
        assert 'confidence' in ensemble.routers
        assert 'resonance' in ensemble.routers
        assert 'similarity' in ensemble.routers
    
    def test_ensemble_weights_normalization(self):
        """Test that ensemble weights are properly normalized."""
        ensemble = AdaptiveEntropyRouterEnsemble(
            input_dim=64,
            num_experts=4,
            ensemble_weight_confidence=0.6,
            ensemble_weight_resonance=0.3,
            ensemble_weight_similarity=0.1
        )
        
        total_weight = sum(ensemble.ensemble_weights.values())
        assert abs(total_weight - 1.0) < 1e-6
    
    def test_forward_pass(self):
        """Test ensemble forward pass."""
        ensemble = AdaptiveEntropyRouterEnsemble(input_dim=64, num_experts=4)
        
        inputs = np.random.normal(0, 1, (2, 8, 64)).astype(np.float32)
        experts, weights, info = ensemble.forward(inputs)
        
        assert experts.shape[:-1] == inputs.shape[:-1]  # Batch and sequence dimensions
        assert weights.shape[:-1] == inputs.shape[:-1]
        assert 'ensemble_weights' in info
        assert 'router_outputs' in info


class TestRouterComparativeStudy:
    """Test comparative study framework."""
    
    def test_initialization(self):
        """Test comparative study initialization."""
        study = RouterComparativeStudy(input_dim=64, num_experts=4)
        
        assert len(study.routers) == 4
        assert 'confidence_based' in study.routers
        assert 'expert_token_resonance' in study.routers
        assert 'similarity_aware' in study.routers
        assert 'ensemble' in study.routers
    
    def test_comparison_summary_generation(self):
        """Test comparison summary generation."""
        study = RouterComparativeStudy(input_dim=64, num_experts=4)
        
        # Mock results
        mock_results = {
            'router1': {
                'flop_reductions': [0.1, 0.2, 0.15],
                'entropy_values': [1.5, 1.3, 1.4],
                'routing_decisions': [
                    {'experts_selected': 1.5},
                    {'experts_selected': 2.0},
                    {'experts_selected': 1.0}
                ]
            },
            'router2': {
                'flop_reductions': [0.3, 0.25, 0.35],
                'entropy_values': [1.2, 1.0, 1.1],
                'routing_decisions': [
                    {'experts_selected': 1.0},
                    {'experts_selected': 1.5},
                    {'experts_selected': 0.5}
                ]
            }
        }
        
        summary = study._generate_comparison_summary(mock_results)
        
        assert 'router1' in summary
        assert 'router2' in summary
        assert 'avg_flop_reduction' in summary['router1']
        assert 'avg_entropy' in summary['router1']
        assert 'routing_efficiency' in summary['router1']


class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_invalid_input_dimensions(self):
        """Test handling of invalid input dimensions."""
        router = ConfidenceBasedRouter(input_dim=64, num_experts=4)
        
        # 2D input should raise error
        invalid_input = np.random.normal(0, 1, (32, 64))
        
        with pytest.raises(ValueError, match="Expected 3D input"):
            router.forward(invalid_input)
    
    def test_invalid_similarity_metric(self):
        """Test handling of invalid similarity metric."""
        with pytest.raises(ValueError, match="Unknown similarity metric"):
            router = SimilarityAwareRouter(
                input_dim=64, 
                num_experts=4, 
                similarity_metric="invalid_metric"
            )
            inputs = np.random.normal(0, 1, (2, 8, 64)).astype(np.float32)
            router._compute_similarity_scores(inputs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])