#!/usr/bin/env python3
"""Performance profiling utilities for dynamic-moe-router-kit."""

import argparse
import cProfile
import pstats
import time
from pathlib import Path
from typing import Any, Dict, Optional

import psutil


class PerformanceProfiler:
    """Comprehensive performance profiler for MoE routing operations."""
    
    def __init__(self, output_dir: Path = Path("./profiling_results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.profiler = cProfile.Profile()
        self.start_time = None
        self.process = psutil.Process()
        
    def start_profiling(self):
        """Start performance profiling."""
        self.start_time = time.time()
        self.profiler.enable()
        
    def stop_profiling(self, name: str = "profile"):
        """Stop profiling and save results."""
        self.profiler.disable()
        
        # Save cProfile results
        profile_file = self.output_dir / f"{name}.prof"
        self.profiler.dump_stats(str(profile_file))
        
        # Generate text report
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumulative')
        
        report_file = self.output_dir / f"{name}_report.txt"
        with open(report_file, 'w') as f:
            stats.print_stats(30, file=f)
            
        # Memory usage report
        memory_info = self.process.memory_info()
        memory_report = {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'cpu_percent': self.process.cpu_percent(),
            'duration_seconds': time.time() - self.start_time
        }
        
        memory_file = self.output_dir / f"{name}_memory.txt"
        with open(memory_file, 'w') as f:
            for key, value in memory_report.items():
                f.write(f"{key}: {value}\n")
                
        return profile_file, report_file, memory_file


def profile_routing_operation(
    model_name: str,
    batch_size: int,
    sequence_length: int,
    num_experts: int,
    backend: str = "torch"
):
    """Profile a specific routing operation."""
    profiler = PerformanceProfiler()
    
    # Import appropriate backend
    if backend == "torch":
        import torch
        inputs = torch.randn(batch_size, sequence_length, 768)
    elif backend == "jax":
        import jax.numpy as jnp
        inputs = jnp.ones((batch_size, sequence_length, 768))
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    profiler.start_profiling()
    
    # Simulate routing operations
    for _ in range(100):  # Multiple iterations for better profiling
        # This would be replaced with actual routing code
        if backend == "torch":
            complexity = torch.rand(batch_size, sequence_length)
            k = torch.randint(1, num_experts//2, (batch_size, sequence_length))
        elif backend == "jax":
            import jax.random as random
            key = random.PRNGKey(42)
            complexity = random.uniform(key, (batch_size, sequence_length))
    
    results = profiler.stop_profiling(f"routing_{model_name}_{backend}")
    
    print(f"Profiling completed for {model_name} ({backend})")
    print(f"Results saved to: {results[0].parent}")
    
    return results


def benchmark_complexity_estimators():
    """Benchmark different complexity estimation methods."""
    estimators = [
        "gradient_norm",
        "attention_entropy", 
        "perplexity_proxy",
        "custom"
    ]
    
    results = {}
    
    for estimator in estimators:
        profiler = PerformanceProfiler()
        profiler.start_profiling()
        
        # Simulate complexity estimation
        import torch
        inputs = torch.randn(32, 128, 768)
        
        for _ in range(50):
            if estimator == "gradient_norm":
                # Simulate gradient norm computation
                grad_norm = torch.norm(inputs, dim=-1)
                complexity = torch.sigmoid(grad_norm - grad_norm.mean())
            elif estimator == "attention_entropy":
                # Simulate attention entropy
                attention = torch.softmax(inputs @ inputs.transpose(-2, -1), dim=-1)
                entropy = -torch.sum(attention * torch.log(attention + 1e-8), dim=-1)
                complexity = torch.sigmoid(entropy - entropy.mean())
        
        profile_results = profiler.stop_profiling(f"complexity_{estimator}")
        results[estimator] = profile_results
        
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performance profiling for dynamic MoE router")
    parser.add_argument("--operation", choices=["routing", "complexity", "full"], 
                       default="routing", help="Operation to profile")
    parser.add_argument("--model", default="test-model", help="Model name")
    parser.add_argument("--backend", choices=["torch", "jax", "tf"], 
                       default="torch", help="Backend to use")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--num-experts", type=int, default=8, help="Number of experts")
    
    args = parser.parse_args()
    
    if args.operation == "routing":
        profile_routing_operation(
            args.model, args.batch_size, args.seq_len, 
            args.num_experts, args.backend
        )
    elif args.operation == "complexity":
        benchmark_complexity_estimators()
    elif args.operation == "full":
        print("Running full profiling suite...")
        profile_routing_operation(
            args.model, args.batch_size, args.seq_len,
            args.num_experts, args.backend
        )
        benchmark_complexity_estimators()
        
    print("Profiling complete!")