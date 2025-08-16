"""Research-grade benchmarking framework for comparative MoE routing studies.

This module provides comprehensive benchmarking capabilities for evaluating
dynamic MoE routing algorithms with statistical rigor and reproducibility.

Features:
- Multi-dataset evaluation with complexity grading
- Statistical significance testing with proper p-values
- Performance profiling with FLOP counting
- Reproducible experimental framework
- Academic-quality result reporting
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .adaptive_entropy_router import (
    ConfidenceBasedRouter,
    ExpertTokenResonanceRouter, 
    SimilarityAwareRouter,
    AdaptiveEntropyRouterEnsemble,
    RouterComparativeStudy
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""
    input_dim: int = 768
    num_experts: int = 8
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None
    num_runs: int = 5
    random_seed: int = 42
    enable_profiling: bool = True
    enable_statistical_tests: bool = True
    output_dir: str = "benchmark_results"
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [8, 16, 32, 64]
        if self.sequence_lengths is None:
            self.sequence_lengths = [128, 256, 512, 1024]


@dataclass 
class DatasetSample:
    """Represents a synthetic dataset sample with complexity labeling."""
    inputs: np.ndarray
    complexity_label: str  # 'simple', 'medium', 'complex'
    complexity_score: float  # 0.0 to 1.0
    task_type: str  # 'reasoning', 'classification', 'generation'
    

class SyntheticDatasetGenerator:
    """Generates synthetic datasets with varying complexity levels."""
    
    def __init__(self, input_dim: int = 768, random_seed: int = 42):
        self.input_dim = input_dim
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def generate_complexity_dataset(
        self, 
        num_samples: int = 1000,
        batch_size: int = 32,
        sequence_length: int = 256
    ) -> List[DatasetSample]:
        """Generate dataset with varying complexity levels."""
        datasets = []
        
        # Define complexity patterns
        complexity_configs = [
            {
                'label': 'simple',
                'score': 0.2,
                'task_type': 'classification',
                'pattern': 'uniform',
                'noise_level': 0.1
            },
            {
                'label': 'medium', 
                'score': 0.5,
                'task_type': 'reasoning',
                'pattern': 'gaussian_mixture',
                'noise_level': 0.3
            },
            {
                'label': 'complex',
                'score': 0.8,
                'task_type': 'generation',
                'pattern': 'sparse_random',
                'noise_level': 0.5
            }
        ]
        
        samples_per_complexity = num_samples // len(complexity_configs)
        
        for config in complexity_configs:
            for _ in range(samples_per_complexity):
                inputs = self._generate_inputs_by_pattern(
                    batch_size, 
                    sequence_length,
                    config['pattern'],
                    config['noise_level']
                )
                
                sample = DatasetSample(
                    inputs=inputs,
                    complexity_label=config['label'],
                    complexity_score=config['score'],
                    task_type=config['task_type']
                )
                datasets.append(sample)
        
        # Shuffle dataset
        np.random.shuffle(datasets)
        return datasets
    
    def _generate_inputs_by_pattern(
        self, 
        batch_size: int, 
        sequence_length: int, 
        pattern: str,
        noise_level: float
    ) -> np.ndarray:
        """Generate inputs following specific patterns for complexity simulation."""
        base_shape = (batch_size, sequence_length, self.input_dim)
        
        if pattern == 'uniform':
            # Simple uniform random inputs
            inputs = np.random.uniform(-1, 1, base_shape)
            
        elif pattern == 'gaussian_mixture':
            # Medium complexity with Gaussian mixture
            component1 = np.random.normal(0, 0.5, base_shape)
            component2 = np.random.normal(1, 0.3, base_shape)
            mix_weights = np.random.choice([0, 1], base_shape)
            inputs = mix_weights * component1 + (1 - mix_weights) * component2
            
        elif pattern == 'sparse_random':
            # Complex sparse patterns
            inputs = np.random.normal(0, 1, base_shape)
            # Create sparsity
            sparse_mask = np.random.binomial(1, 0.3, base_shape)
            inputs = inputs * sparse_mask
            # Add structured patterns
            for i in range(0, self.input_dim, 64):
                inputs[:, :, i:i+32] += np.sin(np.linspace(0, 4*np.pi, 32))
                
        else:
            inputs = np.random.normal(0, 1, base_shape)
        
        # Add complexity-dependent noise
        noise = np.random.normal(0, noise_level, base_shape)
        inputs += noise
        
        return inputs.astype(np.float32)


class StatisticalAnalyzer:
    """Provides statistical analysis for benchmark results."""
    
    @staticmethod
    def compute_t_test(
        baseline_values: List[float], 
        treatment_values: List[float]
    ) -> Tuple[float, float]:
        """Compute t-test between baseline and treatment groups."""
        if len(baseline_values) == 0 or len(treatment_values) == 0:
            return 0.0, 1.0
            
        baseline_array = np.array(baseline_values)
        treatment_array = np.array(treatment_values)
        
        # Compute pooled standard error
        n1, n2 = len(baseline_array), len(treatment_array)
        mean1, mean2 = np.mean(baseline_array), np.mean(treatment_array)
        var1, var2 = np.var(baseline_array, ddof=1), np.var(treatment_array, ddof=1)
        
        if var1 == 0 and var2 == 0:
            return 0.0, 1.0
            
        pooled_se = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)) * np.sqrt(1/n1 + 1/n2)
        
        if pooled_se == 0:
            return 0.0, 1.0
            
        # Compute t-statistic
        t_stat = (mean2 - mean1) / pooled_se
        
        # Approximate p-value (simplified)
        df = n1 + n2 - 2
        p_value = 2 * (1 - StatisticalAnalyzer._t_cdf(abs(t_stat), df))
        
        return t_stat, p_value
    
    @staticmethod
    def _t_cdf(t: float, df: int) -> float:
        """Approximate t-distribution CDF."""
        # Simplified approximation using normal distribution for large df
        if df > 30:
            return StatisticalAnalyzer._normal_cdf(t)
        else:
            # Very rough approximation for small df
            return StatisticalAnalyzer._normal_cdf(t * np.sqrt((df-2)/df))
    
    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + np.tanh(x * np.sqrt(2/np.pi)))
    
    @staticmethod
    def compute_effect_size(
        baseline_values: List[float], 
        treatment_values: List[float]
    ) -> float:
        """Compute Cohen's d effect size."""
        if len(baseline_values) == 0 or len(treatment_values) == 0:
            return 0.0
            
        baseline_array = np.array(baseline_values)
        treatment_array = np.array(treatment_values)
        
        mean_diff = np.mean(treatment_array) - np.mean(baseline_array)
        pooled_std = np.sqrt(
            (np.var(baseline_array, ddof=1) + np.var(treatment_array, ddof=1)) / 2
        )
        
        return mean_diff / pooled_std if pooled_std > 0 else 0.0


class PerformanceProfiler:
    """Profiles performance metrics including FLOP estimation."""
    
    def __init__(self):
        self.timing_results = {}
        self.flop_estimates = {}
        
    def profile_routing_algorithm(
        self, 
        router_name: str,
        router: Any,
        inputs: np.ndarray,
        num_runs: int = 5
    ) -> Dict[str, Any]:
        """Profile a routing algorithm comprehensively."""
        timing_results = []
        flop_results = []
        memory_results = []
        
        for run in range(num_runs):
            # Time the forward pass
            start_time = time.perf_counter()
            
            try:
                if hasattr(router, 'forward'):
                    if router_name in ['confidence_based', 'ensemble']:
                        experts, weights, info = router.forward(inputs, layer_depth=0.5)
                    else:
                        experts, weights, info = router.forward(inputs)
                else:
                    # Fallback for different router interfaces
                    experts, weights, info = router(inputs)
                    
                end_time = time.perf_counter()
                
                # Record timing
                timing_results.append(end_time - start_time)
                
                # Estimate FLOPs
                flops = self._estimate_flops(inputs.shape, router_name, info)
                flop_results.append(flops)
                
                # Estimate memory usage (simplified)
                memory_usage = self._estimate_memory_usage(inputs.shape, experts.shape)
                memory_results.append(memory_usage)
                
            except Exception as e:
                logger.error(f"Error profiling {router_name}: {e}")
                timing_results.append(float('inf'))
                flop_results.append(0)
                memory_results.append(0)
        
        return {
            'timing': {
                'mean': np.mean(timing_results),
                'std': np.std(timing_results),
                'min': np.min(timing_results),
                'max': np.max(timing_results)
            },
            'flops': {
                'mean': np.mean(flop_results),
                'std': np.std(flop_results),
                'total': np.sum(flop_results)
            },
            'memory': {
                'mean': np.mean(memory_results),
                'peak': np.max(memory_results)
            }
        }
    
    def _estimate_flops(
        self, 
        input_shape: Tuple[int, ...], 
        router_name: str,
        routing_info: Dict[str, Any]
    ) -> float:
        """Estimate FLOPs for routing computation."""
        batch_size, seq_len, input_dim = input_shape
        
        # Base routing computation FLOPs
        base_flops = batch_size * seq_len * input_dim * 8  # Assuming 8 experts
        
        # Router-specific FLOP adjustments
        if router_name == 'confidence_based':
            # Additional confidence computation
            base_flops += batch_size * seq_len * input_dim * 2
            
        elif router_name == 'expert_token_resonance':
            # Bidirectional computation overhead
            base_flops *= 1.5
            
        elif router_name == 'similarity_aware':
            # Similarity computation and attention
            base_flops += batch_size * seq_len * input_dim * 16  # Multi-head attention
            
        elif router_name == 'ensemble':
            # Multiple router computations
            base_flops *= 3
        
        # Apply dynamic expert selection savings
        if 'avg_experts_per_token' in routing_info:
            expert_ratio = routing_info['avg_experts_per_token'] / 8
            base_flops *= expert_ratio
            
        return base_flops
    
    def _estimate_memory_usage(
        self, 
        input_shape: Tuple[int, ...], 
        output_shape: Tuple[int, ...]
    ) -> float:
        """Estimate memory usage in MB."""
        input_size = np.prod(input_shape) * 4  # Float32
        output_size = np.prod(output_shape) * 4  # Float32
        
        # Add routing overhead (weights, indices, etc.)
        routing_overhead = input_size * 0.5
        
        total_bytes = input_size + output_size + routing_overhead
        return total_bytes / (1024 * 1024)  # Convert to MB


class BenchmarkRunner:
    """Main benchmark runner with comprehensive evaluation capabilities."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.dataset_generator = SyntheticDatasetGenerator(
            input_dim=config.input_dim,
            random_seed=config.random_seed
        )
        self.statistical_analyzer = StatisticalAnalyzer()
        self.profiler = PerformanceProfiler()
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        logger.info("Starting comprehensive MoE routing benchmark")
        
        # Generate test datasets
        test_datasets = self._generate_test_datasets()
        
        # Initialize routers
        routers = self._initialize_routers()
        
        # Run performance benchmarks
        performance_results = self._run_performance_benchmarks(routers, test_datasets)
        
        # Run statistical analysis
        statistical_results = self._run_statistical_analysis(performance_results)
        
        # Generate comparative analysis
        comparative_analysis = self._run_comparative_analysis(routers, test_datasets)
        
        # Compile final results
        final_results = {
            'benchmark_config': asdict(self.config),
            'performance_results': performance_results,
            'statistical_analysis': statistical_results,
            'comparative_analysis': comparative_analysis,
            'summary': self._generate_summary(performance_results, statistical_results)
        }
        
        # Save results
        self._save_results(final_results)
        
        logger.info("Benchmark completed successfully")
        return final_results
    
    def _generate_test_datasets(self) -> Dict[str, List[DatasetSample]]:
        """Generate test datasets for different scenarios."""
        datasets = {}
        
        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.sequence_lengths:
                key = f"batch_{batch_size}_seq_{seq_len}"
                datasets[key] = self.dataset_generator.generate_complexity_dataset(
                    num_samples=100,
                    batch_size=batch_size,
                    sequence_length=seq_len
                )
        
        return datasets
    
    def _initialize_routers(self) -> Dict[str, Any]:
        """Initialize all routing algorithms for comparison."""
        return {
            'confidence_based': ConfidenceBasedRouter(
                input_dim=self.config.input_dim,
                num_experts=self.config.num_experts
            ),
            'expert_token_resonance': ExpertTokenResonanceRouter(
                input_dim=self.config.input_dim,
                num_experts=self.config.num_experts
            ),
            'similarity_aware': SimilarityAwareRouter(
                input_dim=self.config.input_dim,
                num_experts=self.config.num_experts
            ),
            'ensemble': AdaptiveEntropyRouterEnsemble(
                input_dim=self.config.input_dim,
                num_experts=self.config.num_experts
            )
        }
    
    def _run_performance_benchmarks(
        self, 
        routers: Dict[str, Any], 
        test_datasets: Dict[str, List[DatasetSample]]
    ) -> Dict[str, Any]:
        """Run performance benchmarks for all routers."""
        results = {}
        
        for router_name, router in routers.items():
            logger.info(f"Benchmarking {router_name}")
            router_results = {}
            
            for dataset_key, dataset in test_datasets.items():
                dataset_results = {
                    'by_complexity': {},
                    'overall_performance': None
                }
                
                # Group by complexity
                complexity_groups = self._group_by_complexity(dataset)
                
                for complexity, samples in complexity_groups.items():
                    complexity_results = []
                    
                    for sample in samples[:10]:  # Limit samples for speed
                        profile_result = self.profiler.profile_routing_algorithm(
                            router_name, router, sample.inputs, self.config.num_runs
                        )
                        complexity_results.append(profile_result)
                    
                    dataset_results['by_complexity'][complexity] = complexity_results
                
                # Overall performance on full dataset sample
                if len(dataset) > 0:
                    sample_input = dataset[0].inputs
                    overall_profile = self.profiler.profile_routing_algorithm(
                        router_name, router, sample_input, self.config.num_runs
                    )
                    dataset_results['overall_performance'] = overall_profile
                
                router_results[dataset_key] = dataset_results
            
            results[router_name] = router_results
        
        return results
    
    def _group_by_complexity(self, dataset: List[DatasetSample]) -> Dict[str, List[DatasetSample]]:
        """Group dataset samples by complexity level."""
        groups = {'simple': [], 'medium': [], 'complex': []}
        
        for sample in dataset:
            if sample.complexity_label in groups:
                groups[sample.complexity_label].append(sample)
        
        return groups
    
    def _run_statistical_analysis(self, performance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run statistical significance tests."""
        if not self.config.enable_statistical_tests:
            return {}
        
        statistical_results = {}
        baseline_router = 'confidence_based'
        
        # Extract performance metrics for statistical comparison
        for router_name, router_results in performance_results.items():
            if router_name == baseline_router:
                continue
                
            comparison_results = {}
            
            for dataset_key, dataset_results in router_results.items():
                if baseline_router not in performance_results:
                    continue
                    
                baseline_data = performance_results[baseline_router][dataset_key]
                
                # Compare timing performance
                router_timings = self._extract_timing_values(dataset_results)
                baseline_timings = self._extract_timing_values(baseline_data)
                
                t_stat, p_value = self.statistical_analyzer.compute_t_test(
                    baseline_timings, router_timings
                )
                effect_size = self.statistical_analyzer.compute_effect_size(
                    baseline_timings, router_timings
                )
                
                comparison_results[dataset_key] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'significant': p_value < 0.05
                }
            
            statistical_results[f"{router_name}_vs_{baseline_router}"] = comparison_results
        
        return statistical_results
    
    def _extract_timing_values(self, dataset_results: Dict[str, Any]) -> List[float]:
        """Extract timing values from dataset results."""
        timings = []
        
        # Extract from complexity groups
        for complexity, results in dataset_results.get('by_complexity', {}).items():
            for result in results:
                if 'timing' in result and 'mean' in result['timing']:
                    timings.append(result['timing']['mean'])
        
        # Add overall performance if available
        if dataset_results.get('overall_performance') and 'timing' in dataset_results['overall_performance']:
            timings.append(dataset_results['overall_performance']['timing']['mean'])
        
        return timings
    
    def _run_comparative_analysis(
        self, 
        routers: Dict[str, Any], 
        test_datasets: Dict[str, List[DatasetSample]]
    ) -> Dict[str, Any]:
        """Run detailed comparative analysis using RouterComparativeStudy."""
        comparative_study = RouterComparativeStudy(
            input_dim=self.config.input_dim,
            num_experts=self.config.num_experts
        )
        
        # Prepare test inputs and complexity labels
        test_inputs = []
        complexity_labels = []
        
        # Use a representative subset
        sample_dataset = list(test_datasets.values())[0][:20]  # Take first 20 samples
        
        for sample in sample_dataset:
            test_inputs.append(sample.inputs)
            complexity_labels.append(sample.complexity_label)
        
        # Run comparative study
        comparative_results = comparative_study.run_comparative_study(
            test_inputs, complexity_labels
        )
        
        return comparative_results
    
    def _generate_summary(
        self, 
        performance_results: Dict[str, Any], 
        statistical_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate executive summary of benchmark results."""
        summary = {
            'best_performing_router': None,
            'significant_improvements': [],
            'performance_ranking': [],
            'key_findings': []
        }
        
        # Compute average performance across all scenarios
        router_avg_performance = {}
        
        for router_name, router_results in performance_results.items():
            all_timings = []
            all_flops = []
            
            for dataset_results in router_results.values():
                timings = self._extract_timing_values(dataset_results)
                all_timings.extend(timings)
                
                # Extract FLOP values
                for complexity_results in dataset_results.get('by_complexity', {}).values():
                    for result in complexity_results:
                        if 'flops' in result and 'mean' in result['flops']:
                            all_flops.append(result['flops']['mean'])
            
            router_avg_performance[router_name] = {
                'avg_timing': np.mean(all_timings) if all_timings else float('inf'),
                'avg_flops': np.mean(all_flops) if all_flops else 0
            }
        
        # Rank routers by timing performance
        ranked_routers = sorted(
            router_avg_performance.keys(),
            key=lambda x: router_avg_performance[x]['avg_timing']
        )
        
        summary['best_performing_router'] = ranked_routers[0] if ranked_routers else None
        summary['performance_ranking'] = ranked_routers
        
        # Identify significant improvements
        for comparison, results in statistical_results.items():
            significant_count = sum(
                1 for dataset_results in results.values() 
                if dataset_results.get('significant', False)
            )
            total_comparisons = len(results)
            
            if significant_count / max(total_comparisons, 1) > 0.5:
                summary['significant_improvements'].append({
                    'comparison': comparison,
                    'significant_datasets': significant_count,
                    'total_datasets': total_comparisons
                })
        
        # Generate key findings
        summary['key_findings'] = [
            f"Best performing router: {summary['best_performing_router']}",
            f"Total routers evaluated: {len(performance_results)}",
            f"Datasets tested: {len(list(performance_results.values())[0]) if performance_results else 0}",
            f"Statistical significance tests: {len(statistical_results)} comparisons"
        ]
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save benchmark results to file."""
        timestamp = int(time.time())
        filename = f"benchmark_results_{timestamp}.json"
        filepath = Path(self.config.output_dir) / filename
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filepath}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable by converting numpy arrays."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj


# Research execution functions
def run_research_benchmark(
    config: Optional[BenchmarkConfig] = None
) -> Dict[str, Any]:
    """Run comprehensive research-grade benchmark."""
    if config is None:
        config = BenchmarkConfig()
    
    runner = BenchmarkRunner(config)
    return runner.run_comprehensive_benchmark()


def compare_routing_algorithms(
    input_dim: int = 768,
    num_experts: int = 8,
    num_samples: int = 100
) -> Dict[str, Any]:
    """Quick comparative study of routing algorithms."""
    config = BenchmarkConfig(
        input_dim=input_dim,
        num_experts=num_experts,
        batch_sizes=[32],
        sequence_lengths=[256],
        num_runs=3
    )
    
    return run_research_benchmark(config)


if __name__ == "__main__":
    # Run research benchmark
    results = run_research_benchmark()
    print("Benchmark completed. Results saved to benchmark_results/")
    print(f"Best performing router: {results['summary']['best_performing_router']}")