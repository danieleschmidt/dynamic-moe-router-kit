#!/usr/bin/env python3
"""Benchmark CLI for dynamic MoE routing performance comparison."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..benchmarks import BenchmarkRunner
from ..profiler import FLOPProfiler, ComparisonProfiler


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def run_benchmark_suite(
    model_name: str,
    tasks: List[str],
    compare_static: bool = True,
    output_dir: Optional[Path] = None,
    batch_size: int = 32,
    sequence_length: int = 128,
    input_dim: int = 768,
    num_experts: int = 8,
    verbose: bool = False
) -> Dict:
    """Run comprehensive benchmark suite."""
    
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for benchmarking")
    
    results = {
        "model": model_name,
        "config": {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "input_dim": input_dim,
            "num_experts": num_experts,
            "tasks": tasks
        },
        "benchmarks": {}
    }
    
    # Initialize profiler
    profiler = ComparisonProfiler()
    
    # Generate synthetic data
    torch.manual_seed(42)
    test_input = torch.randn(batch_size, sequence_length, input_dim)
    
    if verbose:
        print(f"Running benchmarks for {model_name}")
        print(f"Input shape: {test_input.shape}")
    
    # Run dynamic routing benchmark
    from ..torch import TorchDynamicRouter, TorchMoELayer
    
    dynamic_router = TorchDynamicRouter(
        input_dim=input_dim,
        num_experts=num_experts,
        min_experts=1,
        max_experts=num_experts // 2
    )
    
    dynamic_moe = TorchMoELayer(
        router=dynamic_router,
        expert_fn=lambda: torch.nn.Linear(input_dim, input_dim),
        num_experts=num_experts
    )
    
    # Warmup
    for _ in range(3):
        _ = dynamic_moe(test_input)
    
    # Dynamic routing benchmark
    start_time = time.time()
    dynamic_output, dynamic_info = dynamic_moe(test_input, return_router_logits=True)
    dynamic_time = time.time() - start_time
    
    results["benchmarks"]["dynamic"] = {
        "inference_time": dynamic_time,
        "avg_experts_per_token": float(dynamic_info.get("avg_experts_per_token", 0)),
        "flop_reduction": float(dynamic_info.get("flop_reduction", 0)),
        "output_shape": list(dynamic_output.shape)
    }
    
    if compare_static:
        # Static routing comparison
        static_router = TorchDynamicRouter(
            input_dim=input_dim,
            num_experts=num_experts,
            min_experts=num_experts,  # Force all experts
            max_experts=num_experts
        )
        
        static_moe = TorchMoELayer(
            router=static_router,
            expert_fn=lambda: torch.nn.Linear(input_dim, input_dim),
            num_experts=num_experts
        )
        
        # Warmup
        for _ in range(3):
            _ = static_moe(test_input)
        
        start_time = time.time()
        static_output, static_info = static_moe(test_input, return_router_logits=True)
        static_time = time.time() - start_time
        
        results["benchmarks"]["static"] = {
            "inference_time": static_time,
            "avg_experts_per_token": float(static_info.get("avg_experts_per_token", num_experts)),
            "flop_reduction": 0.0,
            "output_shape": list(static_output.shape)
        }
        
        # Comparison metrics
        speedup = static_time / dynamic_time if dynamic_time > 0 else 0
        results["comparison"] = {
            "speedup": speedup,
            "time_reduction": (static_time - dynamic_time) / static_time if static_time > 0 else 0,
            "accuracy_maintained": float(torch.allclose(dynamic_output, static_output, rtol=1e-3))
        }
    
    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / f"benchmark_{model_name}_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_file}")
    
    return results


def print_benchmark_summary(results: Dict) -> None:
    """Print formatted benchmark summary."""
    print("\n" + "="*60)
    print(f"BENCHMARK RESULTS: {results['model']}")
    print("="*60)
    
    config = results['config']
    print(f"Configuration:")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Sequence length: {config['sequence_length']}")
    print(f"  - Input dimension: {config['input_dim']}")
    print(f"  - Number of experts: {config['num_experts']}")
    
    benchmarks = results['benchmarks']
    
    if 'dynamic' in benchmarks:
        dyn = benchmarks['dynamic']
        print(f"\nDynamic Routing:")
        print(f"  - Inference time: {dyn['inference_time']:.4f}s")
        print(f"  - Avg experts/token: {dyn['avg_experts_per_token']:.2f}")
        print(f"  - FLOP reduction: {dyn['flop_reduction']:.1%}")
    
    if 'static' in benchmarks:
        stat = benchmarks['static']
        print(f"\nStatic Routing:")
        print(f"  - Inference time: {stat['inference_time']:.4f}s")
        print(f"  - Avg experts/token: {stat['avg_experts_per_token']:.2f}")
    
    if 'comparison' in results:
        comp = results['comparison']
        print(f"\nComparison:")
        print(f"  - Speedup: {comp['speedup']:.2f}x")
        print(f"  - Time reduction: {comp['time_reduction']:.1%}")
        print(f"  - Accuracy maintained: {comp['accuracy_maintained']}")


def main():
    """Main CLI entry point for benchmarking."""
    parser = argparse.ArgumentParser(
        description="Benchmark dynamic MoE routing performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        default="test-model",
        help="Model name for benchmarking"
    )
    
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["inference"],
        help="Tasks to benchmark"
    )
    
    parser.add_argument(
        "--compare-static",
        action="store_true",
        default=True,
        help="Compare against static routing"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for benchmarking"
    )
    
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=128,
        help="Sequence length for inputs"
    )
    
    parser.add_argument(
        "--input-dim",
        type=int,
        default=768,
        help="Input dimension"
    )
    
    parser.add_argument(
        "--num-experts",
        type=int,
        default=8,
        help="Number of experts"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    try:
        results = run_benchmark_suite(
            model_name=args.model,
            tasks=args.tasks,
            compare_static=args.compare_static,
            output_dir=args.output,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            input_dim=args.input_dim,
            num_experts=args.num_experts,
            verbose=args.verbose
        )
        
        print_benchmark_summary(results)
        
    except Exception as e:
        logging.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()