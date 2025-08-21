#!/usr/bin/env python3
"""Simple CLI demo for dynamic MoE routing."""

import argparse
import time
import numpy as np
from typing import Dict, Any

def main():
    """Main CLI demo function."""
    parser = argparse.ArgumentParser(
        description="Dynamic MoE Router Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m dynamic_moe_router.cli.demo --experts 8 --batch-size 32
  python -m dynamic_moe_router.cli.demo --input-dim 1024 --experts 16
        """
    )
    
    parser.add_argument(
        "--input-dim", type=int, default=768,
        help="Input dimension (default: 768)"
    )
    parser.add_argument(
        "--experts", type=int, default=8,
        help="Number of experts (default: 8)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=128,
        help="Sequence length (default: 128)"
    )
    parser.add_argument(
        "--min-experts", type=int, default=1,
        help="Minimum experts per token (default: 1)"
    )
    parser.add_argument(
        "--max-experts", type=int, default=4,
        help="Maximum experts per token (default: 4)"
    )
    parser.add_argument(
        "--complexity-estimator", default="gradient_norm",
        choices=["gradient_norm", "attention_entropy", "perplexity_proxy", "threshold"],
        help="Complexity estimator to use (default: gradient_norm)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Dynamic MoE Router Demo")
    print("=" * 50)
    print(f"📊 Configuration:")
    print(f"   Input dimension: {args.input_dim}")
    print(f"   Number of experts: {args.experts}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Sequence length: {args.seq_len}")
    print(f"   Min experts per token: {args.min_experts}")
    print(f"   Max experts per token: {args.max_experts}")
    print(f"   Complexity estimator: {args.complexity_estimator}")
    print()
    
    # Import and initialize router
    try:
        from dynamic_moe_router import DynamicRouter, ProductionMoERouter
        from dynamic_moe_router.production_fixed import ProductionConfig
        
        print("✅ Successfully imported dynamic_moe_router")
        
        # Create production router
        config = ProductionConfig(
            input_dim=args.input_dim,
            num_experts=args.experts,
            min_experts=args.min_experts,
            max_experts=args.max_experts,
            complexity_estimator=args.complexity_estimator
        )
        
        router = ProductionMoERouter(config)
        print("✅ Created production router")
        
        # Generate test data
        print("📋 Generating test data...")
        test_input = np.random.randn(args.batch_size, args.seq_len, args.input_dim).astype(np.float32)
        
        # Perform routing
        print("⚡ Performing dynamic routing...")
        start_time = time.time()
        
        result = router.route(test_input)
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000
        
        # Display results
        print()
        print("📈 Routing Results:")
        print("=" * 50)
        
        routing_info = result.get('routing_info', {})
        production_info = result.get('production_info', {})
        
        print(f"✨ Average experts per token: {routing_info.get('avg_experts_per_token', 'N/A'):.2f}")
        print(f"⚡ FLOP reduction: {routing_info.get('flop_reduction', 0)*100:.1f}%")
        print(f"⏱️  Processing time: {processing_time:.2f}ms")
        print(f"🏥 Router health: {production_info.get('router_health', 'unknown')}")
        
        if 'expert_utilization' in routing_info:
            expert_util = routing_info['expert_utilization']
            print(f"👥 Expert utilization:")
            for i, util in enumerate(expert_util[:8]):  # Show first 8 experts
                print(f"   Expert {i}: {util:.3f}")
                if i >= 7:
                    break
        
        # Health check
        print()
        print("🏥 Health Check:")
        print("=" * 30)
        health = router.health_check()
        print(f"Status: {health['status']}")
        print(f"Uptime: {health['uptime_seconds']:.1f}s")
        print(f"Total requests: {health['total_requests']}")
        
        for component, status in health.get('components', {}).items():
            print(f"{component}: {status}")
        
        print()
        print("✅ Demo completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure dynamic_moe_router is properly installed.")
        return 1
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())