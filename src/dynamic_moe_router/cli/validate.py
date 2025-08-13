#!/usr/bin/env python3
"""Validation CLI for dynamic MoE routing correctness and compliance."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..validation import (
    validate_router_config,
    validate_tensor_shape,
    validate_expert_weights,
    validate_complexity_scores,
    check_memory_usage
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def validate_routing_correctness(
    input_dim: int = 768,
    num_experts: int = 8,
    batch_size: int = 32,
    sequence_length: int = 128,
    verbose: bool = False
) -> Dict:
    """Validate routing algorithm correctness."""
    
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for validation")
    
    results = {
        "test_config": {
            "input_dim": input_dim,
            "num_experts": num_experts,
            "batch_size": batch_size,
            "sequence_length": sequence_length
        },
        "validation_results": {},
        "errors": []
    }
    
    try:
        from ..torch import TorchDynamicRouter, TorchMoELayer
        
        # Test 1: Router configuration validation
        if verbose:
            print("Testing router configuration validation...")
        
        try:
            validate_router_config({
                "input_dim": input_dim,
                "num_experts": num_experts,
                "min_experts": 1,
                "max_experts": num_experts // 2
            })
            results["validation_results"]["config_validation"] = "PASS"
        except Exception as e:
            results["validation_results"]["config_validation"] = "FAIL"
            results["errors"].append(f"Config validation: {e}")
        
        # Test 2: Router initialization
        if verbose:
            print("Testing router initialization...")
        
        try:
            router = TorchDynamicRouter(
                input_dim=input_dim,
                num_experts=num_experts,
                min_experts=1,
                max_experts=num_experts // 2
            )
            results["validation_results"]["router_init"] = "PASS"
        except Exception as e:
            results["validation_results"]["router_init"] = "FAIL"
            results["errors"].append(f"Router initialization: {e}")
            return results
        
        # Test 3: Input shape validation
        if verbose:
            print("Testing input shape validation...")
        
        torch.manual_seed(42)
        test_input = torch.randn(batch_size, sequence_length, input_dim)
        
        try:
            validate_tensor_shape(test_input, expected_shape=(batch_size, sequence_length, input_dim))
            results["validation_results"]["input_shape"] = "PASS"
        except Exception as e:
            results["validation_results"]["input_shape"] = "FAIL"
            results["errors"].append(f"Input shape validation: {e}")
        
        # Test 4: MoE layer functionality
        if verbose:
            print("Testing MoE layer functionality...")
        
        try:
            moe_layer = TorchMoELayer(
                router=router,
                expert_fn=lambda: torch.nn.Linear(input_dim, input_dim),
                num_experts=num_experts
            )
            
            output, routing_info = moe_layer(test_input, return_router_logits=True)
            
            # Validate output shape
            expected_output_shape = (batch_size, sequence_length, input_dim)
            if output.shape == expected_output_shape:
                results["validation_results"]["moe_output_shape"] = "PASS"
            else:
                results["validation_results"]["moe_output_shape"] = "FAIL"
                results["errors"].append(f"Output shape mismatch: got {output.shape}, expected {expected_output_shape}")
            
            # Validate routing info
            if isinstance(routing_info, dict) and "avg_experts_per_token" in routing_info:
                avg_experts = routing_info["avg_experts_per_token"]
                if 1 <= avg_experts <= num_experts // 2:
                    results["validation_results"]["routing_bounds"] = "PASS"
                else:
                    results["validation_results"]["routing_bounds"] = "FAIL"
                    results["errors"].append(f"Expert count out of bounds: {avg_experts}")
            else:
                results["validation_results"]["routing_info"] = "FAIL"
                results["errors"].append("Missing or invalid routing info")
            
        except Exception as e:
            results["validation_results"]["moe_functionality"] = "FAIL"
            results["errors"].append(f"MoE layer functionality: {e}")
        
        # Test 5: Complexity estimation validation
        if verbose:
            print("Testing complexity estimation...")
        
        try:
            complexity_scores = router.complexity_estimator.estimate(test_input)
            validate_complexity_scores(complexity_scores)
            
            # Check score range [0, 1]
            if torch.all(complexity_scores >= 0) and torch.all(complexity_scores <= 1):
                results["validation_results"]["complexity_range"] = "PASS"
            else:
                results["validation_results"]["complexity_range"] = "FAIL"
                results["errors"].append("Complexity scores out of [0, 1] range")
            
        except Exception as e:
            results["validation_results"]["complexity_estimation"] = "FAIL"
            results["errors"].append(f"Complexity estimation: {e}")
        
        # Test 6: Memory usage validation
        if verbose:
            print("Testing memory usage...")
        
        try:
            check_memory_usage(test_input, max_memory_gb=1.0)
            results["validation_results"]["memory_usage"] = "PASS"
        except Exception as e:
            results["validation_results"]["memory_usage"] = "FAIL"
            results["errors"].append(f"Memory usage: {e}")
        
        # Test 7: Gradient flow (if supported)
        if verbose:
            print("Testing gradient flow...")
        
        try:
            test_input.requires_grad_(True)
            output, _ = moe_layer(test_input, return_router_logits=True)
            loss = output.sum()
            loss.backward()
            
            if test_input.grad is not None and not torch.all(test_input.grad == 0):
                results["validation_results"]["gradient_flow"] = "PASS"
            else:
                results["validation_results"]["gradient_flow"] = "FAIL"
                results["errors"].append("No gradient flow detected")
                
        except Exception as e:
            results["validation_results"]["gradient_flow"] = "FAIL"
            results["errors"].append(f"Gradient flow: {e}")
    
    except Exception as e:
        results["validation_results"]["general"] = "FAIL"
        results["errors"].append(f"General validation error: {e}")
    
    return results


def validate_performance_bounds(
    input_dim: int = 768,
    num_experts: int = 8,
    batch_size: int = 32,
    sequence_length: int = 128,
    verbose: bool = False
) -> Dict:
    """Validate performance is within expected bounds."""
    
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for performance validation")
    
    results = {
        "performance_bounds": {},
        "errors": []
    }
    
    try:
        from ..torch import TorchDynamicRouter, TorchMoELayer
        
        router = TorchDynamicRouter(
            input_dim=input_dim,
            num_experts=num_experts,
            min_experts=1,
            max_experts=num_experts // 2
        )
        
        moe_layer = TorchMoELayer(
            router=router,
            expert_fn=lambda: torch.nn.Linear(input_dim, input_dim),
            num_experts=num_experts
        )
        
        torch.manual_seed(42)
        test_input = torch.randn(batch_size, sequence_length, input_dim)
        
        # Warmup
        for _ in range(3):
            _ = moe_layer(test_input)
        
        # Measure inference time
        start_time = time.time()
        output, routing_info = moe_layer(test_input, return_router_logits=True)
        inference_time = time.time() - start_time
        
        # Performance bounds validation
        tokens_per_second = (batch_size * sequence_length) / inference_time
        
        # Expected minimum performance (adjust based on hardware)
        min_tokens_per_sec = 1000  # Conservative baseline
        if tokens_per_second >= min_tokens_per_sec:
            results["performance_bounds"]["throughput"] = "PASS"
        else:
            results["performance_bounds"]["throughput"] = "FAIL"
            results["errors"].append(f"Low throughput: {tokens_per_second:.1f} tokens/sec < {min_tokens_per_sec}")
        
        # FLOP reduction validation
        flop_reduction = routing_info.get("flop_reduction", 0)
        if flop_reduction > 0:
            results["performance_bounds"]["flop_reduction"] = "PASS"
        else:
            results["performance_bounds"]["flop_reduction"] = "FAIL"
            results["errors"].append(f"No FLOP reduction: {flop_reduction}")
        
        # Expert utilization validation
        avg_experts = routing_info.get("avg_experts_per_token", num_experts)
        if avg_experts < num_experts:
            results["performance_bounds"]["expert_efficiency"] = "PASS"
        else:
            results["performance_bounds"]["expert_efficiency"] = "FAIL"
            results["errors"].append(f"No expert savings: {avg_experts} >= {num_experts}")
        
    except Exception as e:
        results["performance_bounds"]["general"] = "FAIL"
        results["errors"].append(f"Performance validation error: {e}")
    
    return results


def print_validation_summary(results: Dict) -> None:
    """Print formatted validation summary."""
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    if "test_config" in results:
        config = results["test_config"]
        print(f"Test Configuration:")
        print(f"  - Input dimension: {config['input_dim']}")
        print(f"  - Number of experts: {config['num_experts']}")
        print(f"  - Batch size: {config['batch_size']}")
        print(f"  - Sequence length: {config['sequence_length']}")
    
    if "validation_results" in results:
        print(f"\nValidation Tests:")
        for test_name, status in results["validation_results"].items():
            status_symbol = "✓" if status == "PASS" else "✗"
            print(f"  {status_symbol} {test_name}: {status}")
    
    if "performance_bounds" in results:
        print(f"\nPerformance Bounds:")
        for test_name, status in results["performance_bounds"].items():
            status_symbol = "✓" if status == "PASS" else "✗"
            print(f"  {status_symbol} {test_name}: {status}")
    
    if results.get("errors"):
        print(f"\nErrors:")
        for error in results["errors"]:
            print(f"  - {error}")
    
    # Overall status
    all_tests = list(results.get("validation_results", {}).values()) + list(results.get("performance_bounds", {}).values())
    if all_tests:
        passed = sum(1 for status in all_tests if status == "PASS")
        total = len(all_tests)
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL VALIDATIONS PASSED!")
        else:
            print("⚠️  Some validations failed. Review errors above.")


def main():
    """Main CLI entry point for validation."""
    parser = argparse.ArgumentParser(
        description="Validate dynamic MoE routing correctness and performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for validation"
    )
    
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=128,
        help="Sequence length for inputs"
    )
    
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Include performance bounds validation"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for validation results"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    try:
        # Run correctness validation
        correctness_results = validate_routing_correctness(
            input_dim=args.input_dim,
            num_experts=args.num_experts,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            verbose=args.verbose
        )
        
        # Combine results
        results = correctness_results
        
        # Run performance validation if requested
        if args.performance:
            performance_results = validate_performance_bounds(
                input_dim=args.input_dim,
                num_experts=args.num_experts,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                verbose=args.verbose
            )
            results.update(performance_results)
        
        print_validation_summary(results)
        
        # Save results if output specified
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
        
        # Exit with error code if validations failed
        all_tests = list(results.get("validation_results", {}).values()) + list(results.get("performance_bounds", {}).values())
        if all_tests and not all(status == "PASS" for status in all_tests):
            sys.exit(1)
        
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()