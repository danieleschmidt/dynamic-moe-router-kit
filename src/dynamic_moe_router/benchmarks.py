"""Comprehensive benchmarking suite for dynamic MoE routing."""

import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from .caching import create_cached_router
from .router import DynamicRouter
from .scaling import create_optimized_router


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    name: str
    num_runs: int = 100
    warmup_runs: int = 10
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None
    hidden_dims: List[int] = None
    num_experts_list: List[int] = None
    min_experts_list: List[int] = None
    max_experts_list: List[int] = None

    def __post_init__(self):
        # Set defaults
        self.batch_sizes = self.batch_sizes or [1, 4, 16, 32]
        self.sequence_lengths = self.sequence_lengths or [128, 512, 1024]
        self.hidden_dims = self.hidden_dims or [768, 1024]
        self.num_experts_list = self.num_experts_list or [4, 8, 16]
        self.min_experts_list = self.min_experts_list or [1, 2]
        self.max_experts_list = self.max_experts_list or [2, 4, 8]


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    config_name: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'config_name': self.config_name,
            'parameters': self.parameters,
            'metrics': self.metrics,
            'timestamp': self.timestamp
        }


class DynamicMoEBenchmark:
    """Comprehensive benchmarking suite for dynamic MoE routing."""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        self.results: List[BenchmarkResult] = []

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def run_latency_benchmark(self, config: BenchmarkConfig) -> List[BenchmarkResult]:
        """Benchmark routing latency across different configurations."""
        print(f"Running latency benchmark: {config.name}")
        results = []

        for batch_size in config.batch_sizes:
            for seq_len in config.sequence_lengths:
                for hidden_dim in config.hidden_dims:
                    for num_experts in config.num_experts_list:
                        for min_experts in config.min_experts_list:
                            for max_experts in config.max_experts_list:
                                if min_experts <= max_experts <= num_experts:
                                    result = self._benchmark_single_config(
                                        config, batch_size, seq_len, hidden_dim,
                                        num_experts, min_experts, max_experts
                                    )
                                    results.append(result)
                                    print(f"  Completed: {result.parameters}")

        self.results.extend(results)
        return results

    def _benchmark_single_config(
        self,
        config: BenchmarkConfig,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_experts: int,
        min_experts: int,
        max_experts: int
    ) -> BenchmarkResult:
        """Benchmark a single configuration."""

        # Create router
        router = DynamicRouter(
            input_dim=hidden_dim,
            num_experts=num_experts,
            min_experts=min_experts,
            max_experts=max_experts
        )

        # Generate test data
        test_input = np.random.randn(batch_size, seq_len, hidden_dim)

        # Warmup runs
        for _ in range(config.warmup_runs):
            router.route(test_input)

        # Benchmark runs
        latencies = []
        flop_reductions = []
        avg_experts = []

        for _ in range(config.num_runs):
            start_time = time.perf_counter()
            result = router.route(test_input)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            routing_info = result['routing_info']
            flop_reductions.append(routing_info['flop_reduction'])
            avg_experts.append(routing_info['avg_experts_per_token'])

        # Compute statistics
        metrics = {
            'latency_mean_ms': statistics.mean(latencies),
            'latency_median_ms': statistics.median(latencies),
            'latency_std_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'latency_p95_ms': np.percentile(latencies, 95),
            'latency_p99_ms': np.percentile(latencies, 99),
            'flop_reduction_mean': statistics.mean(flop_reductions),
            'avg_experts_mean': statistics.mean(avg_experts),
            'throughput_tokens_per_sec': (batch_size * seq_len * config.num_runs) / (sum(latencies) / 1000)
        }

        parameters = {
            'batch_size': batch_size,
            'sequence_length': seq_len,
            'hidden_dim': hidden_dim,
            'num_experts': num_experts,
            'min_experts': min_experts,
            'max_experts': max_experts
        }

        return BenchmarkResult(
            config_name=config.name,
            parameters=parameters,
            metrics=metrics,
            timestamp=time.time()
        )

    def run_throughput_benchmark(self, config: BenchmarkConfig,
                                concurrent_requests: List[int] = None) -> List[BenchmarkResult]:
        """Benchmark throughput under concurrent load."""
        print(f"Running throughput benchmark: {config.name}")

        concurrent_requests = concurrent_requests or [1, 4, 8, 16]
        results = []

        # Use a representative configuration
        batch_size, seq_len, hidden_dim = 16, 512, 768
        num_experts, min_experts, max_experts = 8, 1, 4

        router = DynamicRouter(
            input_dim=hidden_dim,
            num_experts=num_experts,
            min_experts=min_experts,
            max_experts=max_experts
        )

        for num_concurrent in concurrent_requests:
            result = self._benchmark_concurrent_throughput(
                config, router, batch_size, seq_len, hidden_dim, num_concurrent
            )
            results.append(result)
            print(f"  Completed concurrent={num_concurrent}: {result.metrics['throughput_requests_per_sec']:.2f} req/sec")

        self.results.extend(results)
        return results

    def _benchmark_concurrent_throughput(
        self,
        config: BenchmarkConfig,
        router: DynamicRouter,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_concurrent: int
    ) -> BenchmarkResult:
        """Benchmark throughput with concurrent requests."""

        def single_request():
            test_input = np.random.randn(batch_size, seq_len, hidden_dim)
            start_time = time.perf_counter()
            router.route(test_input)
            return time.perf_counter() - start_time

        # Run concurrent requests
        total_requests = config.num_runs
        latencies = []

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            start_time = time.perf_counter()
            futures = [executor.submit(single_request) for _ in range(total_requests)]

            for future in as_completed(futures):
                latency = future.result()
                latencies.append(latency)

            total_time = time.perf_counter() - start_time

        # Compute metrics
        metrics = {
            'concurrent_requests': num_concurrent,
            'total_requests': total_requests,
            'total_time_sec': total_time,
            'throughput_requests_per_sec': total_requests / total_time,
            'avg_latency_ms': np.mean(latencies) * 1000,
            'p95_latency_ms': np.percentile(latencies, 95) * 1000
        }

        parameters = {
            'batch_size': batch_size,
            'sequence_length': seq_len,
            'hidden_dim': hidden_dim,
            'concurrent_requests': num_concurrent
        }

        return BenchmarkResult(
            config_name=f"{config.name}_throughput",
            parameters=parameters,
            metrics=metrics,
            timestamp=time.time()
        )

    def run_scaling_benchmark(self, config: BenchmarkConfig) -> List[BenchmarkResult]:
        """Benchmark scaling performance with optimizations."""
        print(f"Running scaling benchmark: {config.name}")
        results = []

        # Test different optimization combinations
        optimization_configs = [
            {'name': 'baseline', 'cache': False, 'optimize': False},
            {'name': 'cached', 'cache': True, 'optimize': False},
            {'name': 'optimized', 'cache': False, 'optimize': True},
            {'name': 'full', 'cache': True, 'optimize': True}
        ]

        # Use representative configuration
        batch_size, seq_len, hidden_dim = 32, 512, 768
        num_experts, min_experts, max_experts = 8, 1, 4

        for opt_config in optimization_configs:
            result = self._benchmark_optimization_config(
                config, opt_config, batch_size, seq_len, hidden_dim,
                num_experts, min_experts, max_experts
            )
            results.append(result)
            print(f"  {opt_config['name']}: {result.metrics['latency_mean_ms']:.2f}ms")

        self.results.extend(results)
        return results

    def _benchmark_optimization_config(
        self,
        config: BenchmarkConfig,
        opt_config: Dict[str, Any],
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_experts: int,
        min_experts: int,
        max_experts: int
    ) -> BenchmarkResult:
        """Benchmark a specific optimization configuration."""

        # Create base router
        base_router = DynamicRouter(
            input_dim=hidden_dim,
            num_experts=num_experts,
            min_experts=min_experts,
            max_experts=max_experts
        )

        # Apply optimizations
        router = base_router
        if opt_config['cache']:
            router = create_cached_router(router, cache_size=100, adaptive=True)
        if opt_config['optimize']:
            router = create_optimized_router(router, enable_autoscaling=False)

        # Generate test data
        test_input = np.random.randn(batch_size, seq_len, hidden_dim)

        # Warmup
        for _ in range(config.warmup_runs):
            router.route(test_input)

        # Benchmark
        latencies = []
        for _ in range(config.num_runs):
            start_time = time.perf_counter()
            result = router.route(test_input)
            latency = (time.perf_counter() - start_time) * 1000
            latencies.append(latency)

        # Compute metrics
        metrics = {
            'optimization_type': opt_config['name'],
            'latency_mean_ms': statistics.mean(latencies),
            'latency_p95_ms': np.percentile(latencies, 95),
            'throughput_tokens_per_sec': (batch_size * seq_len * config.num_runs) / (sum(latencies) / 1000)
        }

        # Add cache stats if applicable
        if opt_config['cache'] and hasattr(router, 'get_cache_stats'):
            cache_stats = router.get_cache_stats()
            metrics['cache_hit_rate'] = cache_stats.get('hit_rate', 0)

        parameters = {
            'batch_size': batch_size,
            'sequence_length': seq_len,
            'hidden_dim': hidden_dim,
            'optimization_config': opt_config
        }

        return BenchmarkResult(
            config_name=f"{config.name}_scaling",
            parameters=parameters,
            metrics=metrics,
            timestamp=time.time()
        )

    def run_memory_benchmark(self, config: BenchmarkConfig) -> List[BenchmarkResult]:
        """Benchmark memory usage across configurations."""
        print(f"Running memory benchmark: {config.name}")
        results = []

        # Test memory usage with different input sizes
        test_configs = [
            (8, 128, 768),   # Small
            (16, 512, 768),  # Medium
            (32, 1024, 1024) # Large
        ]

        for batch_size, seq_len, hidden_dim in test_configs:
            result = self._benchmark_memory_usage(
                config, batch_size, seq_len, hidden_dim
            )
            results.append(result)

        self.results.extend(results)
        return results

    def _benchmark_memory_usage(
        self,
        config: BenchmarkConfig,
        batch_size: int,
        seq_len: int,
        hidden_dim: int
    ) -> BenchmarkResult:
        """Benchmark memory usage for a specific configuration."""

        router = DynamicRouter(
            input_dim=hidden_dim,
            num_experts=8,
            min_experts=1,
            max_experts=4
        )

        test_input = np.random.randn(batch_size, seq_len, hidden_dim)

        # Measure memory before routing
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Run routing
        for _ in range(config.num_runs):
            router.route(test_input)

        memory_after = process.memory_info().rss / 1024 / 1024  # MB

        metrics = {
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_increase_mb': memory_after - memory_before,
            'memory_per_token_kb': (memory_after - memory_before) * 1024 / (batch_size * seq_len)
        }

        parameters = {
            'batch_size': batch_size,
            'sequence_length': seq_len,
            'hidden_dim': hidden_dim,
            'total_tokens': batch_size * seq_len
        }

        return BenchmarkResult(
            config_name=f"{config.name}_memory",
            parameters=parameters,
            metrics=metrics,
            timestamp=time.time()
        )

    def save_results(self, filename: str = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"benchmark_results_{timestamp}.json"

        filepath = os.path.join(self.output_dir, filename)

        results_data = {
            'metadata': {
                'total_results': len(self.results),
                'generated_at': time.time(),
                'python_version': '3.12+',
                'numpy_version': np.__version__
            },
            'results': [result.to_dict() for result in self.results]
        }

        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"Results saved to: {filepath}")

    def generate_report(self) -> str:
        """Generate a summary report of benchmark results."""
        if not self.results:
            return "No benchmark results available."

        report_lines = ["# Dynamic MoE Router Benchmark Report\n"]

        # Group results by benchmark type
        by_type = {}
        for result in self.results:
            benchmark_type = result.config_name.split('_')[0]
            if benchmark_type not in by_type:
                by_type[benchmark_type] = []
            by_type[benchmark_type].append(result)

        # Generate summary for each type
        for benchmark_type, results in by_type.items():
            report_lines.append(f"## {benchmark_type.title()} Benchmark")
            report_lines.append(f"Total runs: {len(results)}\n")

            if 'latency' in benchmark_type.lower():
                latencies = [r.metrics.get('latency_mean_ms', 0) for r in results]
                report_lines.append(f"Average latency: {statistics.mean(latencies):.2f}ms")
                report_lines.append(f"Best latency: {min(latencies):.2f}ms")
                report_lines.append(f"Worst latency: {max(latencies):.2f}ms\n")

            if 'throughput' in benchmark_type.lower():
                throughputs = [r.metrics.get('throughput_requests_per_sec', 0) for r in results]
                report_lines.append(f"Average throughput: {statistics.mean(throughputs):.2f} req/sec")
                report_lines.append(f"Peak throughput: {max(throughputs):.2f} req/sec\n")

        return '\n'.join(report_lines)


def run_comprehensive_benchmark() -> DynamicMoEBenchmark:
    """Run a comprehensive benchmark suite."""

    benchmark = DynamicMoEBenchmark()

    # Define benchmark configurations
    configs = [
        BenchmarkConfig(
            name="latency_comprehensive",
            num_runs=50,
            warmup_runs=5,
            batch_sizes=[1, 8, 16, 32],
            sequence_lengths=[128, 512],
            hidden_dims=[768],
            num_experts_list=[4, 8],
            min_experts_list=[1, 2],
            max_experts_list=[2, 4]
        ),
        BenchmarkConfig(
            name="throughput_stress",
            num_runs=100,
            warmup_runs=10
        ),
        BenchmarkConfig(
            name="scaling_comparison",
            num_runs=30,
            warmup_runs=5
        )
    ]

    # Run all benchmarks
    for config in configs:
        benchmark.run_latency_benchmark(config)
        benchmark.run_throughput_benchmark(config)
        benchmark.run_scaling_benchmark(config)

    # Save results and generate report
    benchmark.save_results()
    report = benchmark.generate_report()

    # Save report
    with open(os.path.join(benchmark.output_dir, "benchmark_report.md"), 'w') as f:
        f.write(report)

    print(f"\nBenchmark completed! Results saved in: {benchmark.output_dir}")
    print(f"Total benchmark runs: {len(benchmark.results)}")

    return benchmark


if __name__ == "__main__":
    # Run benchmarks if script is executed directly
    run_comprehensive_benchmark()
