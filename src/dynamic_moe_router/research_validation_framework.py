"""
Research Validation Framework for Advanced MoE Routing Algorithms

This module provides comprehensive experimental validation for novel MoE routing
algorithms with academic-grade statistical rigor and reproducibility.

Features:
- Comparative studies with proper baselines
- Statistical significance testing
- Reproducible experimental framework
- Performance profiling and analysis
- Publication-ready result generation

Author: Terry (Terragon Labs)
Research Period: 2024 Advanced MoE Validation Framework
"""

import json
import logging
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from abc import ABC, abstractmethod

from .quadratic_attention_router import QuadraticAttentionDynamicRouter
from .heterogeneous_experts import HeterogeneousExpertPool, ExpertType, ExpertCapability
from .router import DynamicRouter
from .adaptive_router import EnhancedDynamicRouter
from .estimator import get_estimator, ComplexityEstimator

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experimental validation."""
    
    # Dataset parameters
    input_dim: int = 768
    sequence_lengths: List[int] = None
    batch_sizes: List[int] = None
    num_samples_per_config: int = 100
    
    # Model parameters
    num_experts: int = 8
    min_experts: int = 1
    max_experts: int = 4
    
    # Experimental parameters
    num_runs: int = 5
    random_seeds: List[int] = None
    significance_level: float = 0.05
    
    # Validation parameters
    validate_statistical_power: bool = True
    min_effect_size: float = 0.1
    confidence_interval: float = 0.95
    
    # Output parameters
    save_detailed_results: bool = True
    generate_plots: bool = True
    output_dir: str = "research_validation_results"
    
    def __post_init__(self):
        if self.sequence_lengths is None:
            self.sequence_lengths = [128, 256, 512, 1024]
        if self.batch_sizes is None:
            self.batch_sizes = [8, 16, 32]
        if self.random_seeds is None:
            self.random_seeds = [42, 123, 456, 789, 999][:self.num_runs]


@dataclass
class PerformanceMetrics:
    """Performance metrics for routing algorithms."""
    
    # Efficiency metrics
    average_experts_per_token: float
    flop_reduction_percentage: float
    routing_latency_ms: float
    memory_usage_mb: float
    
    # Quality metrics
    routing_entropy: float
    expert_utilization_balance: float
    complexity_correlation: float
    
    # Stability metrics
    routing_consistency: float
    expert_specialization_score: float
    
    # Statistical metrics
    confidence_interval: Tuple[float, float]
    statistical_significance: bool
    p_value: float
    effect_size: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaselineAlgorithm(ABC):
    """Abstract base class for baseline algorithms."""
    
    @abstractmethod
    def route(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Route inputs through experts."""
        pass
        
    @abstractmethod
    def get_name(self) -> str:
        """Get algorithm name."""
        pass


class StaticTopKBaseline(BaselineAlgorithm):
    """Static Top-K routing baseline."""
    
    def __init__(self, input_dim: int, num_experts: int, k: int = 2):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.k = k
        
        # Initialize routing network
        scale = np.sqrt(2.0 / input_dim)
        self.W = np.random.normal(0, scale, (input_dim, num_experts)).astype(np.float32)
        self.b = np.zeros(num_experts, dtype=np.float32)
        
    def route(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Static top-k routing."""
        # Compute logits
        logits = np.dot(inputs, self.W) + self.b
        
        # Select top-k
        top_k_indices = np.argpartition(logits, -self.k, axis=-1)[..., -self.k:]
        
        # Get weights
        batch_indices = np.arange(inputs.shape[0])[:, None, None]
        seq_indices = np.arange(inputs.shape[1])[None, :, None]
        
        top_k_logits = logits[batch_indices, seq_indices, top_k_indices]
        top_k_weights = self._softmax(top_k_logits, axis=-1)
        
        routing_info = {
            'algorithm': 'static_top_k',
            'k': self.k,
            'routing_logits': logits,
            'average_experts_per_token': float(self.k),
            'routing_entropy': self._compute_entropy(top_k_weights)
        }
        
        return top_k_indices, top_k_weights, routing_info
        
    def get_name(self) -> str:
        return f"Static_Top{self.k}"
        
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        
    def _compute_entropy(self, probs: np.ndarray) -> float:
        eps = 1e-10
        entropy = -np.sum(probs * np.log(probs + eps), axis=-1)
        return float(np.mean(entropy))


class RandomRoutingBaseline(BaselineAlgorithm):
    """Random routing baseline."""
    
    def __init__(self, input_dim: int, num_experts: int, k: int = 2):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.k = k
        
    def route(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Random routing."""
        batch_size, seq_len, _ = inputs.shape
        
        # Random expert selection
        expert_indices = np.random.randint(
            0, self.num_experts, (batch_size, seq_len, self.k)
        )
        
        # Random weights
        expert_weights = np.random.dirichlet([1]*self.k, (batch_size, seq_len))
        
        routing_info = {
            'algorithm': 'random',
            'k': self.k,
            'average_experts_per_token': float(self.k),
            'routing_entropy': float(np.log(self.k))  # Maximum entropy
        }
        
        return expert_indices, expert_weights, routing_info
        
    def get_name(self) -> str:
        return "Random"


class ExperimentalDataGenerator:
    """Generates synthetic data with controlled complexity patterns."""
    
    def __init__(self, input_dim: int = 768):
        self.input_dim = input_dim
        
    def generate_complexity_graded_data(
        self,
        batch_size: int,
        seq_len: int,
        complexity_pattern: str = "mixed",
        noise_level: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data with controlled complexity patterns.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            complexity_pattern: Type of complexity pattern
            noise_level: Amount of noise to add
            
        Returns:
            inputs: Generated input data
            complexity_labels: True complexity scores
        """
        if complexity_pattern == "linear":
            # Linear complexity increase
            complexity_scores = np.linspace(0.1, 1.0, seq_len)
            complexity_scores = np.tile(complexity_scores, (batch_size, 1))
            
        elif complexity_pattern == "sine_wave":
            # Sinusoidal complexity pattern
            positions = np.linspace(0, 4*np.pi, seq_len)
            complexity_scores = 0.5 + 0.4 * np.sin(positions)
            complexity_scores = np.tile(complexity_scores, (batch_size, 1))
            
        elif complexity_pattern == "step_function":
            # Step function complexity
            step_size = seq_len // 4
            complexity_scores = np.concatenate([
                np.full(step_size, 0.2),
                np.full(step_size, 0.5),
                np.full(step_size, 0.8),
                np.full(seq_len - 3*step_size, 1.0)
            ])
            complexity_scores = np.tile(complexity_scores, (batch_size, 1))
            
        elif complexity_pattern == "random":
            # Random complexity
            complexity_scores = np.random.uniform(0.1, 1.0, (batch_size, seq_len))
            
        elif complexity_pattern == "mixed":
            # Mixed patterns
            patterns = ["linear", "sine_wave", "step_function", "random"]
            complexity_scores = np.zeros((batch_size, seq_len))
            
            for b in range(batch_size):
                pattern = np.random.choice(patterns)
                if pattern == "linear":
                    complexity_scores[b] = np.linspace(0.1, 1.0, seq_len)
                elif pattern == "sine_wave":
                    positions = np.linspace(0, 4*np.pi, seq_len)
                    complexity_scores[b] = 0.5 + 0.4 * np.sin(positions)
                elif pattern == "step_function":
                    step_size = seq_len // 4
                    complexity_scores[b] = np.concatenate([
                        np.full(step_size, 0.2),
                        np.full(step_size, 0.5),
                        np.full(step_size, 0.8),
                        np.full(seq_len - 3*step_size, 1.0)
                    ])
                else:  # random
                    complexity_scores[b] = np.random.uniform(0.1, 1.0, seq_len)
        else:
            raise ValueError(f"Unknown complexity pattern: {complexity_pattern}")
            
        # Generate inputs based on complexity
        inputs = np.zeros((batch_size, seq_len, self.input_dim))
        
        for b in range(batch_size):
            for s in range(seq_len):
                complexity = complexity_scores[b, s]
                
                # Generate input with complexity-dependent structure
                if complexity < 0.3:
                    # Simple pattern - low rank structure
                    base_vector = np.random.normal(0, 1, self.input_dim // 4)
                    inputs[b, s] = np.tile(base_vector, 4)[:self.input_dim]
                elif complexity < 0.7:
                    # Medium pattern - structured noise
                    base_pattern = np.sin(np.linspace(0, 2*np.pi, self.input_dim))
                    noise = np.random.normal(0, 0.5, self.input_dim)
                    inputs[b, s] = complexity * base_pattern + (1-complexity) * noise
                else:
                    # Complex pattern - high dimensional structure
                    inputs[b, s] = np.random.normal(0, 1, self.input_dim)
                    
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, inputs.shape)
            inputs += noise
            
        return inputs.astype(np.float32), complexity_scores.astype(np.float32)


class StatisticalAnalyzer:
    """Statistical analysis tools for routing validation."""
    
    @staticmethod
    def welch_t_test(sample1: np.ndarray, sample2: np.ndarray) -> Tuple[float, float]:
        """Welch's t-test for unequal variances."""
        n1, n2 = len(sample1), len(sample2)
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
        
        # Welch's t-statistic
        pooled_se = np.sqrt(var1/n1 + var2/n2)
        t_stat = (mean1 - mean2) / pooled_se
        
        # Degrees of freedom (Welch-Satterthwaite equation)
        df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        
        # P-value (two-tailed)
        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        return t_stat, p_value
        
    @staticmethod
    def cohen_d(sample1: np.ndarray, sample2: np.ndarray) -> float:
        """Cohen's d effect size."""
        n1, n2 = len(sample1), len(sample2)
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        
        # Cohen's d
        d = (np.mean(sample1) - np.mean(sample2)) / pooled_std
        return d
        
    @staticmethod
    def bootstrap_confidence_interval(
        data: np.ndarray, 
        statistic_func: Callable = np.mean,
        confidence: float = 0.95,
        num_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval."""
        bootstrap_stats = []
        
        for _ in range(num_bootstrap):
            bootstrap_sample = np.random.choice(data, len(data), replace=True)
            bootstrap_stats.append(statistic_func(bootstrap_sample))
            
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, 100 * alpha/2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha/2))
        
        return lower, upper
        
    @staticmethod
    def statistical_power_analysis(
        effect_size: float,
        sample_size: int,
        alpha: float = 0.05
    ) -> float:
        """Statistical power analysis."""
        # Simplified power calculation for t-test
        from scipy import stats
        
        # Critical t-value
        df = sample_size - 1
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(sample_size)
        
        # Power
        power = 1 - stats.nct.cdf(t_critical, df, ncp) + stats.nct.cdf(-t_critical, df, ncp)
        
        return power


class RoutingAlgorithmValidator:
    """Main validation framework for routing algorithms."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_generator = ExperimentalDataGenerator(config.input_dim)
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Results storage
        self.experiment_results = {}
        self.statistical_summaries = {}
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def validate_algorithm(
        self,
        algorithm: Any,
        algorithm_name: str,
        baselines: List[BaselineAlgorithm],
        complexity_patterns: List[str] = None
    ) -> Dict[str, PerformanceMetrics]:
        """
        Comprehensive validation of a routing algorithm.
        
        Args:
            algorithm: Algorithm to validate
            algorithm_name: Name for the algorithm
            baselines: List of baseline algorithms
            complexity_patterns: Data complexity patterns to test
            
        Returns:
            results: Validation results for each comparison
        """
        if complexity_patterns is None:
            complexity_patterns = ["mixed", "linear", "sine_wave", "step_function"]
            
        logger.info(f"Starting validation for {algorithm_name}")
        
        validation_results = {}
        
        # Test against each baseline
        for baseline in baselines:
            baseline_name = baseline.get_name()
            logger.info(f"Comparing {algorithm_name} vs {baseline_name}")
            
            # Run comparative study
            comparison_results = self._run_comparative_study(
                algorithm, algorithm_name,
                baseline, baseline_name,
                complexity_patterns
            )
            
            # Perform statistical analysis
            statistical_results = self._statistical_analysis(comparison_results)
            
            # Compute performance metrics
            performance_metrics = self._compute_performance_metrics(
                comparison_results, statistical_results
            )
            
            validation_results[baseline_name] = performance_metrics
            
        # Save results
        self._save_validation_results(algorithm_name, validation_results)
        
        return validation_results
        
    def _run_comparative_study(
        self,
        algorithm: Any,
        algorithm_name: str,
        baseline: BaselineAlgorithm,
        baseline_name: str,
        complexity_patterns: List[str]
    ) -> Dict[str, Any]:
        """Run comparative study between algorithm and baseline."""
        
        results = {
            'algorithm_metrics': [],
            'baseline_metrics': [],
            'experimental_conditions': []
        }
        
        # Test across different conditions
        for pattern in complexity_patterns:
            for seq_len in self.config.sequence_lengths:
                for batch_size in self.config.batch_sizes:
                    
                    # Multiple runs for statistical significance
                    algorithm_run_metrics = []
                    baseline_run_metrics = []
                    
                    for run_idx, seed in enumerate(self.config.random_seeds):
                        np.random.seed(seed)
                        
                        # Generate test data
                        inputs, true_complexity = self.data_generator.generate_complexity_graded_data(
                            batch_size, seq_len, pattern
                        )
                        
                        # Test algorithm
                        alg_metrics = self._evaluate_single_run(
                            algorithm, inputs, true_complexity, f"{algorithm_name}_run{run_idx}"
                        )
                        algorithm_run_metrics.append(alg_metrics)
                        
                        # Test baseline
                        base_metrics = self._evaluate_single_run(
                            baseline, inputs, true_complexity, f"{baseline_name}_run{run_idx}"
                        )
                        baseline_run_metrics.append(base_metrics)
                        
                    # Store run results
                    condition = {
                        'complexity_pattern': pattern,
                        'sequence_length': seq_len,
                        'batch_size': batch_size
                    }
                    
                    results['algorithm_metrics'].append(algorithm_run_metrics)
                    results['baseline_metrics'].append(baseline_run_metrics)
                    results['experimental_conditions'].append(condition)
                    
        return results
        
    def _evaluate_single_run(
        self,
        algorithm: Any,
        inputs: np.ndarray,
        true_complexity: np.ndarray,
        run_name: str
    ) -> Dict[str, float]:
        """Evaluate algorithm on a single run."""
        
        start_time = time.time()
        
        # Route through algorithm
        if hasattr(algorithm, 'route'):
            expert_indices, expert_weights, routing_info = algorithm.route(
                inputs, return_routing_info=True
            )
        else:
            # For baseline algorithms
            expert_indices, expert_weights, routing_info = algorithm.route(inputs)
            
        routing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Compute metrics
        metrics = {
            'routing_latency_ms': routing_time,
            'average_experts_per_token': routing_info.get('average_experts_per_token', 0),
            'routing_entropy': routing_info.get('routing_entropy', 0),
            'memory_usage_mb': self._estimate_memory_usage(expert_indices, expert_weights),
        }
        
        # Compute additional metrics if available
        if 'flop_reduction' in routing_info:
            metrics['flop_reduction_percentage'] = routing_info['flop_reduction'] * 100
        else:
            # Estimate FLOP reduction based on expert usage
            static_experts = expert_indices.shape[-1]
            dynamic_experts = metrics['average_experts_per_token']
            metrics['flop_reduction_percentage'] = (1 - dynamic_experts/static_experts) * 100
            
        # Complexity correlation
        if true_complexity is not None:
            metrics['complexity_correlation'] = self._compute_complexity_correlation(
                expert_indices, expert_weights, true_complexity
            )
        else:
            metrics['complexity_correlation'] = 0.0
            
        # Expert utilization balance
        metrics['expert_utilization_balance'] = self._compute_expert_balance(expert_indices)
        
        # Routing consistency (if multiple samples)
        metrics['routing_consistency'] = self._compute_routing_consistency(expert_indices)
        
        return metrics
        
    def _statistical_analysis(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis on comparison results."""
        
        statistical_results = {}
        
        # Extract metrics for analysis
        algorithm_metrics = comparison_results['algorithm_metrics']
        baseline_metrics = comparison_results['baseline_metrics']
        
        # Flatten metrics across conditions and runs
        metric_names = algorithm_metrics[0][0].keys()
        
        for metric_name in metric_names:
            # Collect all values for this metric
            alg_values = []
            base_values = []
            
            for condition_idx in range(len(algorithm_metrics)):
                for run_metrics in algorithm_metrics[condition_idx]:
                    alg_values.append(run_metrics[metric_name])
                for run_metrics in baseline_metrics[condition_idx]:
                    base_values.append(run_metrics[metric_name])
                    
            alg_values = np.array(alg_values)
            base_values = np.array(base_values)
            
            # Statistical tests
            t_stat, p_value = self.statistical_analyzer.welch_t_test(alg_values, base_values)
            effect_size = self.statistical_analyzer.cohen_d(alg_values, base_values)
            
            # Confidence intervals
            alg_ci = self.statistical_analyzer.bootstrap_confidence_interval(
                alg_values, confidence=self.config.confidence_interval
            )
            base_ci = self.statistical_analyzer.bootstrap_confidence_interval(
                base_values, confidence=self.config.confidence_interval
            )
            
            # Statistical power
            power = self.statistical_analyzer.statistical_power_analysis(
                effect_size, len(alg_values), self.config.significance_level
            )
            
            statistical_results[metric_name] = {
                'algorithm_mean': float(np.mean(alg_values)),
                'baseline_mean': float(np.mean(base_values)),
                'algorithm_std': float(np.std(alg_values)),
                'baseline_std': float(np.std(base_values)),
                'algorithm_ci': alg_ci,
                'baseline_ci': base_ci,
                't_statistic': t_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'statistical_power': power,
                'significant': p_value < self.config.significance_level,
                'practical_significance': abs(effect_size) >= self.config.min_effect_size
            }
            
        return statistical_results
        
    def _compute_performance_metrics(
        self, 
        comparison_results: Dict[str, Any],
        statistical_results: Dict[str, Any]
    ) -> PerformanceMetrics:
        """Compute final performance metrics."""
        
        # Extract key metrics
        flop_reduction = statistical_results['flop_reduction_percentage']
        experts_per_token = statistical_results['average_experts_per_token']
        routing_latency = statistical_results['routing_latency_ms']
        memory_usage = statistical_results['memory_usage_mb']
        routing_entropy = statistical_results['routing_entropy']
        expert_balance = statistical_results['expert_utilization_balance']
        complexity_corr = statistical_results['complexity_correlation']
        routing_consistency = statistical_results['routing_consistency']
        
        # Overall significance
        significant_metrics = sum(1 for m in statistical_results.values() if m['significant'])
        overall_significance = significant_metrics > len(statistical_results) / 2
        
        # Combined p-value (Fisher's method)
        p_values = [m['p_value'] for m in statistical_results.values()]
        combined_p = -2 * sum(np.log(p) for p in p_values if p > 0)
        
        # Combined effect size (average)
        effect_sizes = [m['effect_size'] for m in statistical_results.values()]
        combined_effect_size = np.mean(effect_sizes)
        
        return PerformanceMetrics(
            average_experts_per_token=experts_per_token['algorithm_mean'],
            flop_reduction_percentage=flop_reduction['algorithm_mean'],
            routing_latency_ms=routing_latency['algorithm_mean'],
            memory_usage_mb=memory_usage['algorithm_mean'],
            routing_entropy=routing_entropy['algorithm_mean'],
            expert_utilization_balance=expert_balance['algorithm_mean'],
            complexity_correlation=complexity_corr['algorithm_mean'],
            routing_consistency=routing_consistency['algorithm_mean'],
            expert_specialization_score=0.0,  # Would need expert feedback
            confidence_interval=flop_reduction['algorithm_ci'],
            statistical_significance=overall_significance,
            p_value=np.exp(combined_p / (-2 * len(p_values))),  # Back-transform
            effect_size=combined_effect_size
        )
        
    def _estimate_memory_usage(self, expert_indices: np.ndarray, expert_weights: np.ndarray) -> float:
        """Estimate memory usage in MB."""
        # Simple estimation based on array sizes
        indices_memory = expert_indices.nbytes / (1024 * 1024)
        weights_memory = expert_weights.nbytes / (1024 * 1024)
        return indices_memory + weights_memory
        
    def _compute_complexity_correlation(
        self,
        expert_indices: np.ndarray,
        expert_weights: np.ndarray,
        true_complexity: np.ndarray
    ) -> float:
        """Compute correlation between routing decisions and true complexity."""
        # Compute effective number of experts used per token
        valid_mask = expert_indices >= 0
        experts_per_token = np.sum(valid_mask, axis=-1).astype(float)
        
        # Flatten arrays
        flat_experts = experts_per_token.flatten()
        flat_complexity = true_complexity.flatten()
        
        # Compute correlation
        correlation = np.corrcoef(flat_experts, flat_complexity)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
        
    def _compute_expert_balance(self, expert_indices: np.ndarray) -> float:
        """Compute expert utilization balance (1.0 = perfectly balanced)."""
        valid_indices = expert_indices[expert_indices >= 0]
        if len(valid_indices) == 0:
            return 0.0
            
        # Count expert usage
        max_expert = np.max(valid_indices) + 1
        expert_counts = np.bincount(valid_indices, minlength=max_expert)
        
        # Compute balance (inverse of coefficient of variation)
        if len(expert_counts) <= 1 or np.mean(expert_counts) == 0:
            return 0.0
            
        cv = np.std(expert_counts) / np.mean(expert_counts)
        balance = 1.0 / (1.0 + cv)  # Normalize to [0, 1]
        
        return float(balance)
        
    def _compute_routing_consistency(self, expert_indices: np.ndarray) -> float:
        """Compute routing consistency across similar inputs."""
        # Simplified consistency measure
        # In practice, this would compare routing for similar inputs
        batch_size, seq_len, k = expert_indices.shape
        
        if batch_size <= 1:
            return 1.0
            
        # Compute pairwise similarities in routing decisions
        similarities = []
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                # Jaccard similarity of expert sets
                set_i = set(expert_indices[i].flatten())
                set_j = set(expert_indices[j].flatten())
                
                # Remove invalid indices
                set_i.discard(-1)
                set_j.discard(-1)
                
                if len(set_i) == 0 and len(set_j) == 0:
                    similarity = 1.0
                else:
                    intersection = len(set_i & set_j)
                    union = len(set_i | set_j)
                    similarity = intersection / union if union > 0 else 0.0
                    
                similarities.append(similarity)
                
        return float(np.mean(similarities)) if similarities else 0.0
        
    def _save_validation_results(self, algorithm_name: str, results: Dict[str, PerformanceMetrics]):
        """Save validation results to file."""
        
        # Convert to serializable format
        serializable_results = {}
        for baseline_name, metrics in results.items():
            serializable_results[baseline_name] = metrics.to_dict()
            
        # Save JSON results
        results_file = self.output_dir / f"{algorithm_name}_validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Validation results saved to {results_file}")
        
        # Generate summary report
        self._generate_summary_report(algorithm_name, results)
        
    def _generate_summary_report(self, algorithm_name: str, results: Dict[str, PerformanceMetrics]):
        """Generate a human-readable summary report."""
        
        report_file = self.output_dir / f"{algorithm_name}_validation_summary.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Validation Report: {algorithm_name}\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            
            # Overall performance
            best_baseline = min(results.keys(), key=lambda k: results[k].p_value)
            best_metrics = results[best_baseline]
            
            f.write(f"**Best Performance Against**: {best_baseline}\n")
            f.write(f"**FLOP Reduction**: {best_metrics.flop_reduction_percentage:.1f}%\n")
            f.write(f"**Statistical Significance**: {best_metrics.statistical_significance}\n")
            f.write(f"**Effect Size**: {best_metrics.effect_size:.3f}\n\n")
            
            # Detailed results
            f.write("## Detailed Results\n\n")
            
            for baseline_name, metrics in results.items():
                f.write(f"### vs {baseline_name}\n\n")
                f.write(f"- **FLOP Reduction**: {metrics.flop_reduction_percentage:.1f}%\n")
                f.write(f"- **Avg Experts/Token**: {metrics.average_experts_per_token:.2f}\n")
                f.write(f"- **Routing Latency**: {metrics.routing_latency_ms:.2f}ms\n")
                f.write(f"- **Complexity Correlation**: {metrics.complexity_correlation:.3f}\n")
                f.write(f"- **Expert Balance**: {metrics.expert_utilization_balance:.3f}\n")
                f.write(f"- **Statistical Significance**: {metrics.statistical_significance}\n")
                f.write(f"- **P-value**: {metrics.p_value:.6f}\n")
                f.write(f"- **Effect Size**: {metrics.effect_size:.3f}\n\n")
                
        logger.info(f"Summary report saved to {report_file}")


# Export main classes
__all__ = [
    'ExperimentConfig',
    'PerformanceMetrics',
    'BaselineAlgorithm',
    'StaticTopKBaseline',
    'RandomRoutingBaseline',
    'ExperimentalDataGenerator',
    'StatisticalAnalyzer',
    'RoutingAlgorithmValidator'
]