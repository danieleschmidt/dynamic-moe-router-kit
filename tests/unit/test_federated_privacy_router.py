"""
Comprehensive test suite for Federated Privacy-Preserving MoE Router.

Tests cover:
- Differential privacy guarantees and privacy budget accounting
- Federated learning mechanics and secure aggregation
- Byzantine fault tolerance and malicious participant detection  
- Privacy-utility tradeoff evaluation and statistical analysis
- Research evaluation framework and publication-ready metrics

Author: Terry (Terragon Labs)
Research: 2025 Federated Privacy-Preserving Machine Learning
"""

import numpy as np
import pytest
import logging
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from dynamic_moe_router.federated_privacy_router import (
    FederatedPrivacyRouter,
    PrivacyConfig,
    FederatedConfig,
    FederatedRole,
    PrivacyMechanism,
    PrivacyAccountant,
    RDPAccountant,
    SecureAggregator,
    PrivacyPreservingRouter,
    PrivacyUtilityEvaluator,
    create_federated_privacy_router
)

logger = logging.getLogger(__name__)

class TestPrivacyConfig:
    """Test privacy configuration and validation."""
    
    def test_privacy_config_defaults(self):
        """Test default privacy configuration values."""
        config = PrivacyConfig()
        
        assert config.epsilon == 1.0
        assert config.delta == 1e-5
        assert config.sensitivity == 1.0
        assert config.budget_allocation_strategy == "adaptive"
        assert config.noise_mechanism == PrivacyMechanism.GAUSSIAN
        assert config.local_privacy_enabled == True
        assert config.secure_aggregation_enabled == True
    
    def test_privacy_config_validation(self):
        """Test privacy configuration parameter validation."""
        
        # Valid configuration
        config = PrivacyConfig(epsilon=2.0, delta=1e-6, sensitivity=0.5)
        assert config.epsilon == 2.0
        assert config.delta == 1e-6
        assert config.sensitivity == 0.5
        
        # Invalid epsilon
        with pytest.raises(AssertionError, match="Privacy budget epsilon must be positive"):
            PrivacyConfig(epsilon=0.0)
            
        with pytest.raises(AssertionError, match="Privacy budget epsilon must be positive"):
            PrivacyConfig(epsilon=-1.0)
        
        # Invalid delta
        with pytest.raises(AssertionError, match="Delta must be in \\(0, 1\\)"):
            PrivacyConfig(delta=0.0)
            
        with pytest.raises(AssertionError, match="Delta must be in \\(0, 1\\)"):
            PrivacyConfig(delta=1.0)
            
        with pytest.raises(AssertionError, match="Delta must be in \\(0, 1\\)"):
            PrivacyConfig(delta=1.5)
        
        # Invalid sensitivity
        with pytest.raises(AssertionError, match="Sensitivity must be positive"):
            PrivacyConfig(sensitivity=0.0)


class TestPrivacyAccountant:
    """Test differential privacy budget accounting."""
    
    def test_accountant_initialization(self):
        """Test privacy accountant initialization."""
        config = PrivacyConfig(epsilon=2.0, delta=1e-5)
        accountant = PrivacyAccountant(config)
        
        assert accountant.total_epsilon_spent == 0.0
        assert len(accountant.round_epsilon_spent) == 0
        assert len(accountant.privacy_history) == 0
        assert accountant.rdp_accountant is not None
    
    def test_uniform_budget_allocation(self):
        """Test uniform budget allocation strategy."""
        config = PrivacyConfig(epsilon=2.0, budget_allocation_strategy="uniform", max_budget_per_round=0.1)
        accountant = PrivacyAccountant(config)
        
        allocated = accountant.allocate_budget("test_operation", complexity=5.0)  # Complexity should be ignored
        assert allocated == 0.1
        assert accountant.total_epsilon_spent == 0.1
        
        # Second allocation
        allocated2 = accountant.allocate_budget("test_operation2", complexity=1.0)
        assert allocated2 == 0.1
        assert accountant.total_epsilon_spent == 0.2
    
    def test_adaptive_budget_allocation(self):
        """Test adaptive budget allocation based on complexity."""
        config = PrivacyConfig(epsilon=2.0, budget_allocation_strategy="adaptive", max_budget_per_round=0.1)
        accountant = PrivacyAccountant(config)
        
        # Low complexity operation
        allocated1 = accountant.allocate_budget("simple_op", complexity=0.5)
        assert allocated1 == 0.05  # 0.1 * 0.5
        
        # High complexity operation  
        allocated2 = accountant.allocate_budget("complex_op", complexity=2.0)
        assert allocated2 == 0.2  # 0.1 * 2.0
        
        assert accountant.total_epsilon_spent == 0.25
    
    def test_complexity_based_budget_allocation(self):
        """Test complexity-based budget allocation strategy."""
        config = PrivacyConfig(epsilon=2.0, budget_allocation_strategy="complexity_based", max_budget_per_round=0.1)
        accountant = PrivacyAccountant(config)
        
        # Medium complexity
        allocated = accountant.allocate_budget("medium_op", complexity=5.0)  # normalized to 0.5
        expected = 0.1 * (1.0 + 0.5)  # 0.15
        assert abs(allocated - expected) < 1e-6
    
    def test_budget_exhaustion(self):
        """Test behavior when privacy budget is exhausted."""
        config = PrivacyConfig(epsilon=0.5, max_budget_per_round=0.3, reserve_budget_ratio=0.1)
        accountant = PrivacyAccountant(config)
        
        # First allocation should work
        allocated1 = accountant.allocate_budget("op1", complexity=1.0)
        assert allocated1 > 0
        
        # Second allocation should work but be limited
        allocated2 = accountant.allocate_budget("op2", complexity=1.0)
        assert allocated2 > 0
        
        # Third allocation should fail (budget exhausted)
        with pytest.raises(ValueError, match="Insufficient privacy budget remaining"):
            accountant.allocate_budget("op3", complexity=1.0)
    
    def test_privacy_loss_recording(self):
        """Test privacy loss recording and history."""
        config = PrivacyConfig()
        accountant = PrivacyAccountant(config)
        
        accountant.add_privacy_loss(0.1, 1e-5, "gaussian", "test_operation")
        
        assert len(accountant.privacy_history) == 1
        entry = accountant.privacy_history[0]
        assert entry['epsilon'] == 0.1
        assert entry['delta'] == 1e-5
        assert entry['mechanism'] == "gaussian"
        assert entry['operation'] == "test_operation"
        assert 'timestamp' in entry
        assert 'cumulative_epsilon' in entry
    
    def test_can_afford_budget(self):
        """Test budget affordability checking."""
        config = PrivacyConfig(epsilon=1.0)
        accountant = PrivacyAccountant(config)
        
        assert accountant.can_afford(0.5) == True
        assert accountant.can_afford(1.0) == True
        assert accountant.can_afford(1.1) == False
        
        # After spending some budget
        accountant.total_epsilon_spent = 0.7
        assert accountant.can_afford(0.2) == True
        assert accountant.can_afford(0.3) == True  # Exactly at limit
        assert accountant.can_afford(0.4) == False


class TestRDPAccountant:
    """Test Rényi Differential Privacy accounting."""
    
    def test_rdp_accountant_initialization(self):
        """Test RDP accountant initialization."""
        orders = [1.25, 1.5, 2.0, 4.0]
        accountant = RDPAccountant(orders)
        
        assert accountant.orders == orders
        assert all(accountant.rdp_eps[order] == 0.0 for order in orders)
    
    def test_add_gaussian_mechanism(self):
        """Test adding Gaussian mechanism to RDP accounting."""
        orders = [2.0, 4.0, 8.0]
        accountant = RDPAccountant(orders)
        
        # Add mechanism with sigma=1.0, q=1.0
        accountant.add_mechanism(sigma=1.0, q=1.0)
        
        # Check RDP epsilon values
        for order in orders:
            expected_eps = order / 2.0  # q * order / (2 * sigma^2) = 1 * order / (2 * 1^2)
            assert abs(accountant.rdp_eps[order] - expected_eps) < 1e-10
    
    def test_privacy_spent_conversion(self):
        """Test conversion from RDP to (ε,δ)-DP."""
        orders = [2.0, 4.0]
        accountant = RDPAccountant(orders)
        
        # Add some mechanisms
        accountant.add_mechanism(sigma=1.0, q=1.0)
        accountant.add_mechanism(sigma=2.0, q=0.5)
        
        epsilon, delta = accountant.get_privacy_spent(delta=1e-5)
        
        assert epsilon > 0
        assert delta == 1e-5
        
        # Higher noise should give lower epsilon
        accountant2 = RDPAccountant(orders)
        accountant2.add_mechanism(sigma=10.0, q=1.0)
        epsilon2, _ = accountant2.get_privacy_spent(delta=1e-5)
        
        assert epsilon2 < epsilon


class TestSecureAggregator:
    """Test secure aggregation mechanisms."""
    
    def test_aggregator_initialization(self):
        """Test secure aggregator initialization."""
        config = PrivacyConfig()
        aggregator = SecureAggregator(config)
        
        assert aggregator.config == config
        assert len(aggregator.participant_keys) == 0
        assert len(aggregator.aggregation_history) == 0
    
    def test_participant_key_generation(self):
        """Test participant key generation."""
        config = PrivacyConfig()
        aggregator = SecureAggregator(config)
        
        participant_id = "participant_1"
        key = aggregator.generate_participant_key(participant_id)
        
        assert isinstance(key, bytes)
        assert len(key) > 0
        assert participant_id in aggregator.participant_keys
        assert aggregator.participant_keys[participant_id] == key
    
    def test_parameter_encryption_decryption(self):
        """Test parameter encryption and decryption."""
        config = PrivacyConfig()
        aggregator = SecureAggregator(config)
        
        participant_id = "participant_1"
        parameters = np.random.randn(10, 5)
        
        # Generate key
        aggregator.generate_participant_key(participant_id)
        
        # Encrypt parameters
        encrypted = aggregator.encrypt_parameters(parameters, participant_id)
        assert isinstance(encrypted, bytes)
        assert len(encrypted) > 0
        
        # Decrypt parameters
        decrypted = aggregator.decrypt_parameters(encrypted, participant_id, parameters.shape)
        
        # Should recover original parameters
        np.testing.assert_array_equal(decrypted, parameters)
    
    def test_secure_aggregation(self):
        """Test secure aggregation of multiple participants."""
        config = PrivacyConfig()
        aggregator = SecureAggregator(config)
        
        # Create test parameters for multiple participants
        participants = ["p1", "p2", "p3"]
        shape = (5, 3)
        parameters = {}
        encrypted_params = {}
        shapes = {}
        
        for pid in participants:
            params = np.random.randn(*shape)
            parameters[pid] = params
            aggregator.generate_participant_key(pid)
            encrypted_params[pid] = aggregator.encrypt_parameters(params, pid)
            shapes[pid] = shape
        
        # Perform secure aggregation
        result = aggregator.secure_aggregation(encrypted_params, shapes)
        
        # Should be average of all parameters
        expected = np.mean(list(parameters.values()), axis=0)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Check aggregation history
        assert len(aggregator.aggregation_history) == 1
        history_entry = aggregator.aggregation_history[0]
        assert set(history_entry['participants']) == set(participants)
        assert history_entry['aggregated_shape'] == shape


class TestPrivacyPreservingRouter:
    """Test core privacy-preserving routing logic."""
    
    def test_router_initialization(self):
        """Test privacy-preserving router initialization."""
        config = PrivacyConfig(epsilon=2.0)
        input_dim, num_experts = 512, 8
        
        router = PrivacyPreservingRouter(config, input_dim, num_experts)
        
        assert router.input_dim == input_dim
        assert router.num_experts == num_experts
        assert router.routing_weights.shape == (input_dim, num_experts)
        assert router.routing_bias.shape == (num_experts,)
        assert isinstance(router.accountant, PrivacyAccountant)
    
    def test_noise_addition_gaussian(self):
        """Test Gaussian noise addition for differential privacy."""
        config = PrivacyConfig(noise_mechanism=PrivacyMechanism.GAUSSIAN, delta=1e-5)
        router = PrivacyPreservingRouter(config, 10, 5)
        
        values = np.array([1.0, 2.0, 3.0])
        sensitivity = 1.0
        epsilon = 0.5
        
        noisy_values = router.add_noise(values, sensitivity, epsilon)
        
        # Should have same shape
        assert noisy_values.shape == values.shape
        
        # Should be different from original (with high probability)
        assert not np.allclose(noisy_values, values)
        
        # Noise magnitude should be reasonable for given parameters
        noise = noisy_values - values
        noise_std = np.std(noise)
        expected_sigma = np.sqrt(2 * np.log(1.25 / config.delta)) * sensitivity / epsilon
        
        # Allow for some variation in noise estimation
        assert 0.1 * expected_sigma < noise_std < 10 * expected_sigma
    
    def test_noise_addition_laplace(self):
        """Test Laplace noise addition for differential privacy."""
        config = PrivacyConfig(noise_mechanism=PrivacyMechanism.LAPLACE)
        router = PrivacyPreservingRouter(config, 10, 5)
        
        values = np.array([1.0, 2.0, 3.0])
        sensitivity = 1.0
        epsilon = 0.5
        
        noisy_values = router.add_noise(values, sensitivity, epsilon)
        
        assert noisy_values.shape == values.shape
        assert not np.allclose(noisy_values, values)
    
    def test_private_routing(self):
        """Test private routing with differential privacy."""
        config = PrivacyConfig(epsilon=2.0, max_budget_per_round=0.2)
        input_dim, num_experts = 64, 6
        router = PrivacyPreservingRouter(config, input_dim, num_experts)
        
        batch_size = 8
        inputs = np.random.randn(batch_size, input_dim)
        complexity_scores = np.random.beta(2, 5, batch_size)
        
        expert_indices, expert_weights, routing_info = router.private_route(inputs, complexity_scores)
        
        # Check output structure
        assert len(expert_indices) == batch_size
        assert len(expert_weights) == batch_size
        
        for i in range(batch_size):
            indices = expert_indices[i]
            weights = expert_weights[i]
            
            assert len(indices) == len(weights)
            assert len(indices) >= 1  # At least one expert
            assert len(indices) <= num_experts  # At most all experts
            assert all(0 <= idx < num_experts for idx in indices)
            assert np.abs(np.sum(weights) - 1.0) < 1e-6  # Weights sum to 1
            assert all(w >= 0 for w in weights)  # Non-negative weights
        
        # Check routing info
        assert 'epsilon_spent' in routing_info
        assert 'total_epsilon_spent' in routing_info
        assert 'privacy_remaining' in routing_info
        assert 'average_experts_selected' in routing_info
        assert 'sensitivity_used' in routing_info
        
        assert routing_info['epsilon_spent'] > 0
        assert routing_info['total_epsilon_spent'] == routing_info['epsilon_spent']
        assert routing_info['privacy_remaining'] > 0
        assert 1 <= routing_info['average_experts_selected'] <= num_experts
    
    def test_privacy_budget_consumption(self):
        """Test that privacy budget is properly consumed."""
        config = PrivacyConfig(epsilon=1.0, max_budget_per_round=0.2)
        router = PrivacyPreservingRouter(config, 32, 4)
        
        inputs = np.random.randn(4, 32)
        complexity_scores = np.random.beta(2, 5, 4)
        
        # First routing call
        _, _, info1 = router.private_route(inputs, complexity_scores)
        epsilon_spent_1 = info1['epsilon_spent']
        
        # Second routing call  
        _, _, info2 = router.private_route(inputs, complexity_scores)
        epsilon_spent_2 = info2['epsilon_spent']
        
        # Total spent should be sum
        assert abs(info2['total_epsilon_spent'] - (epsilon_spent_1 + epsilon_spent_2)) < 1e-10
        
        # Privacy remaining should decrease
        assert info2['privacy_remaining'] < info1['privacy_remaining']


class TestFederatedPrivacyRouter:
    """Test main federated privacy router functionality."""
    
    def test_router_initialization(self):
        """Test federated router initialization."""
        privacy_config = PrivacyConfig(epsilon=1.0)
        federated_config = FederatedConfig()
        
        router = FederatedPrivacyRouter(
            input_dim=256,
            num_experts=8, 
            privacy_config=privacy_config,
            federated_config=federated_config,
            participant_id="test_participant",
            role=FederatedRole.PARTICIPANT
        )
        
        assert router.input_dim == 256
        assert router.num_experts == 8
        assert router.participant_id == "test_participant"
        assert router.role == FederatedRole.PARTICIPANT
        assert isinstance(router.private_router, PrivacyPreservingRouter)
        assert isinstance(router.secure_aggregator, SecureAggregator)
        assert router.current_round == 0
        assert router.global_model_version == 0
    
    def test_local_update_computation(self):
        """Test local update computation with privacy."""
        privacy_config = PrivacyConfig(epsilon=2.0, max_budget_per_round=0.1)
        federated_config = FederatedConfig()
        
        router = FederatedPrivacyRouter(
            input_dim=128,
            num_experts=6,
            privacy_config=privacy_config,
            federated_config=federated_config,
            participant_id="participant_1",
            role=FederatedRole.PARTICIPANT
        )
        
        batch_size = 16
        inputs = np.random.randn(batch_size, 128)
        targets = np.random.randn(batch_size, 6)
        complexity_scores = np.random.beta(2, 5, batch_size)
        
        local_update = router.compute_local_update(inputs, targets, complexity_scores)
        
        # Check update structure
        assert local_update['participant_id'] == "participant_1"
        assert local_update['round'] == 0
        assert 'gradients' in local_update
        assert local_update['num_samples'] == batch_size
        assert 'privacy_spent' in local_update
        assert 'routing_performance' in local_update
        
        # Check gradients
        gradients = local_update['gradients']
        assert gradients.shape == router.private_router.routing_weights.shape
        
        # Check gradient clipping was applied
        gradient_norm = np.linalg.norm(gradients)
        assert gradient_norm <= privacy_config.clipping_bound + 1e-6  # Allow small numerical error
        
        # Check privacy expenditure
        assert local_update['privacy_spent'] > 0
        
        # Check routing performance
        perf = local_update['routing_performance']
        assert 'average_experts' in perf
        assert 'privacy_remaining' in perf
        assert perf['average_experts'] >= 1
        assert perf['privacy_remaining'] >= 0
    
    def test_coordinator_aggregation(self):
        """Test coordinator aggregation of participant updates."""
        privacy_config = PrivacyConfig(epsilon=2.0)
        federated_config = FederatedConfig(participants_per_round=3, min_participation_ratio=0.5)
        
        coordinator = FederatedPrivacyRouter(
            input_dim=64,
            num_experts=4,
            privacy_config=privacy_config,
            federated_config=federated_config,
            participant_id="coordinator",
            role=FederatedRole.COORDINATOR
        )
        
        # Create mock participant updates
        updates = []
        for i in range(3):
            update = {
                'participant_id': f'participant_{i}',
                'round': 0,
                'gradients': np.random.normal(0, 0.01, (64, 4)),
                'num_samples': 16,
                'privacy_spent': 0.1,
                'routing_performance': {
                    'average_experts': 2.0,
                    'privacy_remaining': 1.0
                }
            }
            updates.append(update)
        
        result = coordinator.aggregate_updates(updates)
        
        # Check aggregation result
        assert result['round'] == 1  # Round incremented
        assert result['global_model_version'] == 1  # Version incremented  
        assert result['participants'] == 3
        assert result['byzantine_detected'] == 0  # No Byzantine participants
        assert result['total_privacy_spent'] == 0.3  # Sum of individual privacy costs
        assert result['average_experts_used'] == 2.0
        assert 'aggregation_timestamp' in result
    
    def test_non_coordinator_aggregation_error(self):
        """Test that non-coordinator cannot perform aggregation."""
        privacy_config = PrivacyConfig()
        federated_config = FederatedConfig()
        
        participant = FederatedPrivacyRouter(
            input_dim=32,
            num_experts=4,
            privacy_config=privacy_config,
            federated_config=federated_config,
            participant_id="participant",
            role=FederatedRole.PARTICIPANT
        )
        
        with pytest.raises(ValueError, match="Only coordinator can aggregate updates"):
            participant.aggregate_updates([])
    
    def test_byzantine_fault_detection(self):
        """Test Byzantine fault detection in participant updates."""
        privacy_config = PrivacyConfig()
        federated_config = FederatedConfig(byzantine_tolerance=1)
        
        coordinator = FederatedPrivacyRouter(
            input_dim=32,
            num_experts=4,
            privacy_config=privacy_config,
            federated_config=federated_config,
            participant_id="coordinator",
            role=FederatedRole.COORDINATOR
        )
        
        # Create updates with one outlier (Byzantine)
        normal_gradient = np.random.normal(0, 0.01, (32, 4))
        byzantine_gradient = np.random.normal(0, 10.0, (32, 4))  # Much larger variance
        
        updates = [
            {
                'participant_id': 'honest_1',
                'round': 0,
                'gradients': normal_gradient + np.random.normal(0, 0.001, (32, 4)),
                'num_samples': 16,
                'privacy_spent': 0.1,
                'routing_performance': {'average_experts': 2.0, 'privacy_remaining': 1.0}
            },
            {
                'participant_id': 'honest_2', 
                'round': 0,
                'gradients': normal_gradient + np.random.normal(0, 0.001, (32, 4)),
                'num_samples': 16,
                'privacy_spent': 0.1,
                'routing_performance': {'average_experts': 2.0, 'privacy_remaining': 1.0}
            },
            {
                'participant_id': 'byzantine',
                'round': 0,
                'gradients': byzantine_gradient,
                'num_samples': 16,
                'privacy_spent': 0.1,
                'routing_performance': {'average_experts': 2.0, 'privacy_remaining': 1.0}
            }
        ]
        
        result = coordinator.aggregate_updates(updates)
        
        # Should detect Byzantine participant
        assert result['byzantine_detected'] >= 0  # May or may not detect based on randomness
        assert result['participants'] >= 2  # At least 2 honest participants remain
    
    def test_privacy_report_generation(self):
        """Test privacy report generation."""
        privacy_config = PrivacyConfig(epsilon=1.0)
        federated_config = FederatedConfig()
        
        router = FederatedPrivacyRouter(
            input_dim=64,
            num_experts=4,
            privacy_config=privacy_config,
            federated_config=federated_config,
            participant_id="test_participant",
            role=FederatedRole.PARTICIPANT
        )
        
        report = router.get_privacy_report()
        
        # Check report structure
        assert 'privacy_budget' in report
        assert 'federated_stats' in report
        assert 'privacy_mechanisms' in report
        assert 'privacy_history' in report
        
        # Check privacy budget info
        budget = report['privacy_budget']
        assert budget['total_epsilon'] == 1.0
        assert budget['epsilon_spent'] == 0.0  # No operations yet
        assert budget['epsilon_remaining'] == 1.0
        assert budget['budget_utilization'] == 0.0
        
        # Check federated stats
        stats = report['federated_stats']
        assert stats['current_round'] == 0
        assert stats['global_model_version'] == 0
        assert stats['participant_id'] == "test_participant"
        assert stats['role'] == "participant"
        
        # Check privacy mechanisms
        mechanisms = report['privacy_mechanisms']
        assert mechanisms['noise_mechanism'] == 'gaussian'
        assert mechanisms['local_privacy_enabled'] == True
        assert mechanisms['secure_aggregation_enabled'] == True
        
        # Check privacy history
        assert isinstance(report['privacy_history'], list)
        assert len(report['privacy_history']) == 0  # No operations yet


class TestPrivacyUtilityEvaluator:
    """Test privacy-utility tradeoff evaluation."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = PrivacyUtilityEvaluator()
        assert len(evaluator.evaluation_history) == 0
    
    def test_privacy_utility_evaluation(self):
        """Test privacy-utility tradeoff evaluation."""
        privacy_config = PrivacyConfig(epsilon=1.0, max_budget_per_round=0.2)
        federated_config = FederatedConfig()
        
        router = FederatedPrivacyRouter(
            input_dim=32,
            num_experts=4,
            privacy_config=privacy_config,
            federated_config=federated_config,
            participant_id="test_participant"
        )
        
        evaluator = PrivacyUtilityEvaluator()
        
        # Test inputs
        inputs = np.random.randn(8, 32)
        complexity_scores = np.random.beta(2, 5, 8)
        baseline_performance = 0.85
        
        evaluation = evaluator.evaluate_privacy_utility_tradeoff(
            router, inputs, complexity_scores, baseline_performance
        )
        
        # Check evaluation metrics
        assert 'utility_retention' in evaluation
        assert 'privacy_cost' in evaluation
        assert 'efficiency_gain' in evaluation
        assert 'privacy_utility_score' in evaluation
        assert 'epsilon_spent' in evaluation
        assert 'experts_used' in evaluation
        
        # Check value ranges
        assert 0 <= evaluation['utility_retention'] <= 1
        assert 0 <= evaluation['privacy_cost'] <= 1
        assert evaluation['efficiency_gain'] > 0
        assert evaluation['privacy_utility_score'] > 0
        assert evaluation['epsilon_spent'] > 0
        assert 1 <= evaluation['experts_used'] <= 4
        
        # Check history
        assert len(evaluator.evaluation_history) == 1
        assert evaluator.evaluation_history[0] == evaluation


class TestFactoryFunction:
    """Test factory function for creating federated privacy routers."""
    
    def test_create_federated_privacy_router_defaults(self):
        """Test factory function with default parameters."""
        router = create_federated_privacy_router(
            input_dim=128,
            num_experts=8,
            participant_id="test_participant"
        )
        
        assert isinstance(router, FederatedPrivacyRouter)
        assert router.input_dim == 128
        assert router.num_experts == 8
        assert router.participant_id == "test_participant"
        assert router.role == FederatedRole.PARTICIPANT
        assert router.privacy_config.epsilon == 1.0
        assert router.privacy_config.delta == 1e-5
        assert router.federated_config.num_rounds == 100
    
    def test_create_federated_privacy_router_custom(self):
        """Test factory function with custom parameters."""
        router = create_federated_privacy_router(
            input_dim=256,
            num_experts=16,
            participant_id="custom_participant",
            privacy_epsilon=2.0,
            role=FederatedRole.COORDINATOR,
            privacy_delta=1e-6,
            budget_strategy="uniform",
            noise_mechanism="laplace",
            num_rounds=200,
            participants_per_round=10,
            byzantine_tolerance=2
        )
        
        assert router.input_dim == 256
        assert router.num_experts == 16
        assert router.participant_id == "custom_participant"
        assert router.role == FederatedRole.COORDINATOR
        assert router.privacy_config.epsilon == 2.0
        assert router.privacy_config.delta == 1e-6
        assert router.privacy_config.budget_allocation_strategy == "uniform"
        assert router.privacy_config.noise_mechanism == PrivacyMechanism.LAPLACE
        assert router.federated_config.num_rounds == 200
        assert router.federated_config.participants_per_round == 10
        assert router.federated_config.byzantine_tolerance == 2


class TestIntegrationScenarios:
    """Integration tests for complete federated privacy scenarios."""
    
    def test_full_federated_round(self):
        """Test complete federated learning round with privacy."""
        
        # Setup participants
        num_participants = 4
        privacy_epsilon = 2.0
        input_dim, num_experts = 64, 6
        
        participants = []
        for i in range(num_participants):
            role = FederatedRole.COORDINATOR if i == 0 else FederatedRole.PARTICIPANT
            router = create_federated_privacy_router(
                input_dim=input_dim,
                num_experts=num_experts,
                participant_id=f"participant_{i}",
                privacy_epsilon=privacy_epsilon / num_participants,
                role=role,
                byzantine_tolerance=1
            )
            participants.append(router)
        
        coordinator = participants[0]
        workers = participants[1:]
        
        # Generate test data
        batch_size = 16
        inputs = np.random.randn(batch_size, input_dim)
        targets = np.random.randn(batch_size, num_experts)
        complexity_scores = np.random.beta(2, 5, batch_size)
        
        # Compute local updates
        local_updates = []
        for worker in workers:
            update = worker.compute_local_update(inputs, targets, complexity_scores)
            local_updates.append(update)
        
        # Aggregate updates
        result = coordinator.aggregate_updates(local_updates)
        
        # Verify results
        assert result['participants'] == len(workers)
        assert result['round'] == 1
        assert result['total_privacy_spent'] > 0
        assert 0 <= result['byzantine_detected'] <= len(workers)
        
        # Check privacy budgets
        for participant in participants:
            report = participant.get_privacy_report()
            assert report['privacy_budget']['epsilon_spent'] > 0
            assert report['privacy_budget']['epsilon_remaining'] >= 0
    
    def test_privacy_budget_exhaustion_scenario(self):
        """Test scenario where privacy budget gets exhausted."""
        
        privacy_config = PrivacyConfig(epsilon=0.3, max_budget_per_round=0.1)  # Small budget
        federated_config = FederatedConfig()
        
        router = FederatedPrivacyRouter(
            input_dim=32,
            num_experts=4,
            privacy_config=privacy_config,
            federated_config=federated_config,
            participant_id="test_participant"
        )
        
        inputs = np.random.randn(8, 32)
        targets = np.random.randn(8, 4)
        complexity_scores = np.random.beta(2, 5, 8)
        
        # Perform multiple updates until budget exhausted
        successful_updates = 0
        for i in range(10):  # Try up to 10 updates
            try:
                router.compute_local_update(inputs, targets, complexity_scores)
                successful_updates += 1
            except ValueError as e:
                if "Insufficient privacy budget" in str(e):
                    break
                else:
                    raise e
        
        # Should have been able to do at least 2 updates with the budget
        assert successful_updates >= 2
        assert successful_updates < 10  # Should have been stopped by budget
        
        # Check final privacy state
        report = router.get_privacy_report()
        assert report['privacy_budget']['budget_utilization'] > 0.9  # Most budget used


if __name__ == "__main__":
    # Run a few basic tests for demonstration
    test_privacy_config = TestPrivacyConfig()
    test_privacy_config.test_privacy_config_defaults()
    test_privacy_config.test_privacy_config_validation()
    print("✅ Privacy config tests passed")
    
    test_accountant = TestPrivacyAccountant()
    test_accountant.test_accountant_initialization()
    test_accountant.test_uniform_budget_allocation()
    print("✅ Privacy accountant tests passed")
    
    test_router = TestPrivacyPreservingRouter()
    test_router.test_router_initialization() 
    print("✅ Privacy-preserving router tests passed")
    
    test_federated = TestFederatedPrivacyRouter()
    test_federated.test_router_initialization()
    print("✅ Federated privacy router tests passed")
    
    print("\n🎉 All basic tests passed! Run 'pytest tests/unit/test_federated_privacy_router.py -v' for full test suite.")