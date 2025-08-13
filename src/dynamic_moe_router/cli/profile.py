#!/usr/bin/env python3
"""Profiling CLI for dynamic MoE routing analysis."""

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

from ..profiler import FLOPProfiler, ComparisonProfiler


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def load_input_data(input_file: Path, batch_size: int, input_dim: int) -> torch.Tensor:
    """Load input data from file or generate synthetic data."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for profiling")
    
    if input_file and input_file.exists():
        # Try to load from file
        try:
            if input_file.suffix == '.json':
                with open(input_file, 'r') as f:
                    data = json.load(f)
                return torch.tensor(data, dtype=torch.float32)
            elif input_file.suffix == '.npy':
                data = np.load(input_file)
                return torch.from_numpy(data).float()
            else:
                logging.warning(f"Unsupported file format: {input_file.suffix}")
        except Exception as e:
            logging.warning(f"Failed to load input file: {e}")
    
    # Generate synthetic data
    logging.info("Generating synthetic input data")
    torch.manual_seed(42)
    return torch.randn(batch_size, 128, input_dim)


def profile_model(
    model_path: Optional[Path] = None,
    input_file: Optional[Path] = None,
    batch_size: int = 32,
    input_dim: int = 768,
    num_experts: int = 8,
    verbose: bool = False
) -> Dict:
    """Profile model performance and resource usage."""
    
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for profiling")
    
    # Load input data
    input_data = load_input_data(input_file, batch_size, input_dim)
    
    if verbose:
        print(f"Input data shape: {input_data.shape}")
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Initialize profiler
    profiler = FLOPProfiler()
    
    # Create model
    from ..torch import TorchDynamicRouter, TorchMoELayer
    
    router = TorchDynamicRouter(
        input_dim=input_dim,
        num_experts=num_experts,
        min_experts=1,
        max_experts=num_experts // 2,
        complexity_estimator="gradient_norm"
    )
    
    moe_layer = TorchMoELayer(
        router=router,
        expert_fn=lambda: torch.nn.Linear(input_dim, input_dim),
        num_experts=num_experts
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    moe_layer = moe_layer.to(device)
    input_data = input_data.to(device)
    
    # Warmup
    for _ in range(3):
        _ = moe_layer(input_data)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Profile execution
    start_time = time.time()
    
    with profiler:
        output, routing_info = moe_layer(input_data, return_router_logits=True)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    execution_time = time.time() - start_time
    
    # Collect profiling results
    profile_summary = profiler.summary()
    
    results = {
        "model_config": {
            "input_dim": input_dim,
            "num_experts": num_experts,
            "batch_size": batch_size,
            "sequence_length": input_data.shape[1],
            "device": str(device)
        },
        "performance": {
            "execution_time": execution_time,
            "throughput_tokens_per_sec": (batch_size * input_data.shape[1]) / execution_time,
        },
        "routing": {
            "avg_experts_per_token": float(routing_info.get("avg_experts_per_token", 0)),
            "flop_reduction": float(routing_info.get("flop_reduction", 0)),
            "routing_efficiency": float(routing_info.get("routing_efficiency", 0))
        },
        "resource_usage": {
            "flop_profile": profile_summary,
            "output_shape": list(output.shape)
        }
    }
    
    # GPU memory usage
    if device.type == 'cuda':
        results["resource_usage"]["gpu_memory"] = {
            "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024
        }
    
    return results


def print_profile_summary(results: Dict) -> None:
    """Print formatted profiling summary."""
    print("\n" + "="*60)
    print("PROFILING RESULTS")
    print("="*60)
    
    config = results['model_config']
    print(f"Model Configuration:")
    print(f"  - Input dimension: {config['input_dim']}")
    print(f"  - Number of experts: {config['num_experts']}")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Sequence length: {config['sequence_length']}")
    print(f"  - Device: {config['device']}")
    
    perf = results['performance']
    print(f"\nPerformance:")
    print(f"  - Execution time: {perf['execution_time']:.4f}s")
    print(f"  - Throughput: {perf['throughput_tokens_per_sec']:.1f} tokens/sec")
    
    routing = results['routing']
    print(f"\nRouting Efficiency:")
    print(f"  - Avg experts/token: {routing['avg_experts_per_token']:.2f}")
    print(f"  - FLOP reduction: {routing['flop_reduction']:.1%}")
    print(f"  - Routing efficiency: {routing['routing_efficiency']:.3f}")
    
    if 'gpu_memory' in results['resource_usage']:
        gpu = results['resource_usage']['gpu_memory']
        print(f"\nGPU Memory Usage:")
        print(f"  - Allocated: {gpu['allocated_mb']:.1f} MB")
        print(f"  - Reserved: {gpu['reserved_mb']:.1f} MB")
        print(f"  - Peak allocated: {gpu['max_allocated_mb']:.1f} MB")


def main():
    """Main CLI entry point for profiling."""
    parser = argparse.ArgumentParser(
        description="Profile dynamic MoE routing performance and resource usage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to pre-trained model (optional)"
    )
    
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Input data file (.json or .npy)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for profiling"
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
        "--output",
        type=Path,
        help="Output file for profiling results"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    try:
        results = profile_model(
            model_path=args.model_path,
            input_file=args.input_file,
            batch_size=args.batch_size,
            input_dim=args.input_dim,
            num_experts=args.num_experts,
            verbose=args.verbose
        )
        
        print_profile_summary(results)
        
        # Save results if output specified
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
        
    except Exception as e:
        logging.error(f"Profiling failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()