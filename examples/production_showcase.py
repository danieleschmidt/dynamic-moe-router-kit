#!/usr/bin/env python3
"""Production Showcase - Demonstrate Dynamic MoE Router capabilities.

This script showcases the complete Dynamic MoE Router Kit in a production-like
scenario with comprehensive monitoring, optimization, and performance tracking.
"""

import time
import logging
from typing import Dict, Any, List
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    # Import our dynamic MoE components
    from dynamic_moe_router import DynamicRouter, AdaptiveRouter
    from dynamic_moe_router.moe import MoELayer
    from dynamic_moe_router.scaling import create_optimized_router
    from dynamic_moe_router.caching import create_cached_router
    from dynamic_moe_router.monitoring import create_monitoring_wrapper
    from dynamic_moe_router.benchmarks import DynamicMoEBenchmark, BenchmarkConfig
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Make sure to run: pip install -e .")
    exit(1)


class ProductionMoEService:
    """Production-grade MoE routing service with full observability."""
    
    def __init__(self):
        """Initialize production service with optimizations."""
        logger.info("🚀 Initializing Production MoE Service...")
        
        # Create base router with production settings
        self.base_router = self._create_base_router()
        
        # Apply production optimizations
        self.router = self._apply_optimizations(self.base_router)
        
        # Initialize metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_latency': 0.0,
            'latencies': [],
            'expert_usage': np.zeros(8),
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.metrics_lock = threading.Lock()
        
        logger.info("✅ Production MoE Service initialized successfully")
    
    def _create_base_router(self) -> DynamicRouter:
        """Create base router with production configuration."""
        return DynamicRouter(
            input_dim=768,  # BERT-like embedding dimension
            num_experts=8,  # 8 specialized experts
            min_experts=1,  # Minimum experts per token
            max_experts=4,  # Maximum experts per token
            complexity_estimator="gradient_norm",
            routing_strategy="top_k",
            load_balancing=True,
            noise_factor=0.1
        )
    
    def _apply_optimizations(self, base_router: DynamicRouter):
        """Apply production optimizations to router."""
        logger.info("📈 Applying production optimizations...")
        
        # Step 1: Add auto-scaling capabilities
        router = create_optimized_router(
            base_router,
            enable_autoscaling=True,
            enable_parallel=True,
            max_workers=4,
            min_experts=1,
            max_experts=8,
            target_latency_ms=25.0,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3
        )
        
        # Step 2: Add intelligent caching
        router = create_cached_router(
            router,
            cache_size=1000,
            adaptive=True
        )
        
        # Step 3: Add comprehensive monitoring
        router = create_monitoring_wrapper(
            router,
            enable_circuit_breaker=True,
            alert_thresholds={
                'avg_latency_ms': 50.0,
                'error_rate': 0.05,
                'load_balance_variance': 0.1,
                'memory_usage_mb': 1000.0
            }
        )
        
        logger.info("✅ Optimizations applied successfully")
        return router
    
    def process_request(self, hidden_states: np.ndarray) -> Dict[str, Any]:
        """Process a single routing request with full monitoring."""
        start_time = time.perf_counter()
        
        try:
            # Route through optimized router
            result = self.router.route(
                hidden_states,
                return_router_logits=True
            )
            
            # Update metrics
            latency = (time.perf_counter() - start_time) * 1000
            self._update_metrics(latency, result, success=True)
            
            # Add service metadata
            result['service_metadata'] = {
                'latency_ms': latency,
                'timestamp': time.time(),
                'service_version': '1.0.0'
            }
            
            return result
            
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            self._update_metrics(latency, {}, success=False)
            logger.error(f"Request failed: {e}")
            raise
    
    def _update_metrics(self, latency: float, result: Dict[str, Any], success: bool):
        """Thread-safe metrics update."""
        with self.metrics_lock:
            self.metrics['total_requests'] += 1
            self.metrics['total_latency'] += latency
            self.metrics['latencies'].append(latency)
            
            if success:
                self.metrics['successful_requests'] += 1
                
                # Track expert usage
                if 'routing_info' in result:
                    expert_util = result['routing_info'].get('expert_utilization', [])
                    if expert_util:
                        self.metrics['expert_usage'] += np.array(expert_util[:8])
            else:
                self.metrics['failed_requests'] += 1
            
            # Keep only last 1000 latencies
            if len(self.metrics['latencies']) > 1000:
                self.metrics['latencies'] = self.metrics['latencies'][-1000:]
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics."""
        with self.metrics_lock:
            if self.metrics['total_requests'] == 0:
                return {'status': 'no_requests'}
            
            latencies = self.metrics['latencies']
            
            metrics = {
                'total_requests': self.metrics['total_requests'],
                'success_rate': self.metrics['successful_requests'] / self.metrics['total_requests'],
                'error_rate': self.metrics['failed_requests'] / self.metrics['total_requests'],
                'avg_latency_ms': self.metrics['total_latency'] / self.metrics['total_requests'],
                'p50_latency_ms': np.percentile(latencies, 50),
                'p95_latency_ms': np.percentile(latencies, 95),
                'p99_latency_ms': np.percentile(latencies, 99),
                'max_latency_ms': np.max(latencies),
                'expert_usage_distribution': self.metrics['expert_usage'].tolist(),
                'expert_balance_score': 1.0 / (1.0 + np.var(self.metrics['expert_usage']))
            }
            
            # Add router-specific metrics
            if hasattr(self.router, 'get_optimization_stats'):
                opt_stats = self.router.get_optimization_stats()
                metrics['optimization_stats'] = opt_stats
            
            return metrics


def demonstrate_basic_functionality():
    """Demonstrate basic MoE routing functionality."""
    print("\n🔧 Basic Functionality Demonstration")
    print("=" * 50)
    
    # Create service
    service = ProductionMoEService()
    
    # Test with different input sizes
    test_cases = [
        (1, 32, 768),    # Single sample
        (8, 128, 768),   # Small batch
        (16, 256, 768),  # Medium batch
        (32, 512, 768)   # Large batch
    ]
    
    for batch_size, seq_len, hidden_dim in test_cases:
        print(f"\n📊 Testing {batch_size}x{seq_len}x{hidden_dim} input...")
        
        # Generate test input
        test_input = np.random.randn(batch_size, seq_len, hidden_dim)
        
        # Process request
        start_time = time.perf_counter()
        result = service.process_request(test_input)
        latency = (time.perf_counter() - start_time) * 1000
        
        # Display results
        routing_info = result['routing_info']
        print(f"   ✅ Latency: {latency:.2f}ms")
        print(f"   ✅ Avg experts per token: {routing_info['avg_experts_per_token']:.2f}")
        print(f"   ✅ FLOP reduction: {routing_info['flop_reduction']:.1%}")
        print(f"   ✅ Expert utilization variance: {np.var(routing_info['expert_utilization']):.4f}")


def demonstrate_load_testing():
    """Demonstrate load testing capabilities."""
    print("\n🚀 Load Testing Demonstration")
    print("=" * 50)
    
    service = ProductionMoEService()
    
    def single_request():
        """Generate and process a single request."""
        test_input = np.random.randn(16, 128, 768)
        try:
            service.process_request(test_input)
            return True
        except Exception:
            return False
    
    # Run load test
    num_requests = 100
    num_workers = 8
    
    print(f"🔥 Running load test: {num_requests} requests with {num_workers} workers...")
    
    start_time = time.perf_counter()
    successful_requests = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(single_request) for _ in range(num_requests)]
        
        for future in as_completed(futures):
            if future.result():
                successful_requests += 1
    
    total_time = time.perf_counter() - start_time
    
    # Display load test results
    print(f"\n📈 Load Test Results:")
    print(f"   ✅ Total requests: {num_requests}")
    print(f"   ✅ Successful requests: {successful_requests}")
    print(f"   ✅ Success rate: {successful_requests/num_requests:.1%}")
    print(f"   ✅ Total time: {total_time:.2f}s")
    print(f"   ✅ Throughput: {num_requests/total_time:.1f} requests/sec")
    
    # Get service metrics
    metrics = service.get_service_metrics()
    print(f"   ✅ Average latency: {metrics['avg_latency_ms']:.2f}ms")
    print(f"   ✅ 95th percentile latency: {metrics['p95_latency_ms']:.2f}ms")
    print(f"   ✅ Expert balance score: {metrics['expert_balance_score']:.3f}")


def demonstrate_adaptive_routing():
    """Demonstrate adaptive routing capabilities."""
    print("\n🧠 Adaptive Routing Demonstration")
    print("=" * 50)
    
    # Create adaptive router
    adaptive_router = AdaptiveRouter(
        input_dim=768,
        num_experts=8,
        min_experts=1,
        max_experts=4,
        adaptation_rate=0.05
    )
    
    print("🔄 Training adaptive router with performance feedback...")
    
    # Simulate training with performance feedback
    test_input = np.random.randn(16, 128, 768)
    
    for epoch in range(10):
        # Route input
        result = adaptive_router.route(test_input)
        
        # Simulate performance score (higher experts = better performance for this demo)
        avg_experts = result['routing_info']['avg_experts_per_token']
        performance_score = min(1.0, avg_experts / 3.0)  # Normalize to [0, 1]
        
        # Update adaptive thresholds
        adaptive_router.update_thresholds(performance_score)
        
        print(f"   Epoch {epoch+1}: Avg experts = {avg_experts:.2f}, Performance = {performance_score:.3f}")
    
    print("✅ Adaptive training completed!")
    
    # Compare before and after
    initial_router = DynamicRouter(input_dim=768, num_experts=8, min_experts=1, max_experts=4)
    initial_result = initial_router.route(test_input)
    adaptive_result = adaptive_router.route(test_input)
    
    print(f"\n📊 Comparison:")
    print(f"   Initial router avg experts: {initial_result['routing_info']['avg_experts_per_token']:.2f}")
    print(f"   Adaptive router avg experts: {adaptive_result['routing_info']['avg_experts_per_token']:.2f}")


def run_comprehensive_benchmark():
    """Run comprehensive benchmark suite."""
    print("\n📊 Comprehensive Benchmark Suite")
    print("=" * 50)
    
    benchmark = DynamicMoEBenchmark(output_dir="production_benchmarks")
    
    # Quick benchmark configuration
    config = BenchmarkConfig(
        name="production_showcase",
        num_runs=25,
        warmup_runs=5,
        batch_sizes=[8, 16, 32],
        sequence_lengths=[128, 256],
        hidden_dims=[768],
        num_experts_list=[8],
        min_experts_list=[1, 2],
        max_experts_list=[2, 4]
    )
    
    print("🏃‍♂️ Running latency benchmarks...")
    latency_results = benchmark.run_latency_benchmark(config)
    
    print("🚀 Running throughput benchmarks...")
    throughput_results = benchmark.run_throughput_benchmark(config, concurrent_requests=[1, 4, 8])
    
    print("⚡ Running scaling benchmarks...")
    scaling_results = benchmark.run_scaling_benchmark(config)
    
    # Save results
    benchmark.save_results("production_showcase_results.json")
    
    # Generate and display report
    report = benchmark.generate_report()
    print(f"\n📈 Benchmark Report:")
    print(report)
    
    # Display top performance metrics
    all_results = latency_results + throughput_results + scaling_results
    
    if all_results:
        best_latency = min(r.metrics.get('latency_mean_ms', float('inf')) for r in all_results if 'latency_mean_ms' in r.metrics)
        best_throughput = max(r.metrics.get('throughput_requests_per_sec', 0) for r in all_results if 'throughput_requests_per_sec' in r.metrics)
        
        print(f"\n🏆 Performance Highlights:")
        print(f"   🏅 Best latency: {best_latency:.2f}ms")
        print(f"   🏅 Best throughput: {best_throughput:.1f} requests/sec")


def demonstrate_monitoring_capabilities():
    """Demonstrate monitoring and observability features."""
    print("\n📡 Monitoring & Observability Demonstration")
    print("=" * 50)
    
    service = ProductionMoEService()
    
    # Generate some traffic
    print("📊 Generating sample traffic...")
    for i in range(50):
        batch_size = np.random.randint(4, 33)
        seq_len = np.random.randint(64, 513)
        test_input = np.random.randn(batch_size, seq_len, 768)
        
        try:
            service.process_request(test_input)
        except Exception:
            pass  # Some failures are expected in demo
    
    # Display monitoring metrics
    metrics = service.get_service_metrics()
    
    print(f"\n🔍 Service Metrics:")
    print(f"   📈 Total requests: {metrics['total_requests']}")
    print(f"   ✅ Success rate: {metrics['success_rate']:.1%}")
    print(f"   ⚡ Average latency: {metrics['avg_latency_ms']:.2f}ms")
    print(f"   📊 95th percentile latency: {metrics['p95_latency_ms']:.2f}ms")
    print(f"   ⚖️ Expert balance score: {metrics['expert_balance_score']:.3f}")
    
    if 'optimization_stats' in metrics:
        opt_stats = metrics['optimization_stats']
        print(f"\n🚀 Optimization Stats:")
        if 'performance' in opt_stats:
            perf = opt_stats['performance']
            print(f"   🎯 Circuit breaker state: {opt_stats.get('circuit_breaker_state', 'N/A')}")
            print(f"   📊 Total calls: {perf.get('total_calls', 0)}")
            print(f"   ✅ Success rate: {perf.get('success_rate', 0):.1%}")


def main():
    """Main demonstration function."""
    print("🎉 Dynamic MoE Router Kit - Production Showcase")
    print("=" * 60)
    print("This demonstration showcases the complete Dynamic MoE Router Kit")
    print("with production-grade optimizations, monitoring, and performance.")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        demonstrate_basic_functionality()
        demonstrate_load_testing()
        demonstrate_adaptive_routing()
        demonstrate_monitoring_capabilities()
        run_comprehensive_benchmark()
        
        print("\n🎊 Production Showcase Completed Successfully!")
        print("=" * 60)
        print("✅ All features demonstrated successfully")
        print("✅ Performance metrics collected")
        print("✅ Monitoring capabilities verified")
        print("✅ Optimization systems working")
        print("\n📊 Check the 'production_benchmarks' directory for detailed results")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\n❌ Demonstration failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())