#!/usr/bin/env python3
"""Demo of the real-time metrics dashboard."""

import time
import threading
import numpy as np
from dynamic_moe_router.production_fixed import ProductionConfig, ProductionRouter
from dynamic_moe_router.metrics_dashboard import create_dashboard


def simulate_workload(router, dashboard, duration_seconds=30):
    """Simulate a realistic workload for demonstration."""
    start_time = time.time()
    request_count = 0
    
    while time.time() - start_time < duration_seconds:
        try:
            # Generate random input
            batch_size = np.random.randint(8, 32)
            seq_len = np.random.randint(64, 256)
            input_data = np.random.randn(batch_size, seq_len, 768).astype(np.float32)
            
            # Simulate processing
            process_start = time.time()
            result = router.route(input_data)
            processing_time = (time.time() - process_start) * 1000
            
            # Record metrics
            dashboard.record_request(result, processing_time, error=False)
            
            request_count += 1
            
            # Variable delay to simulate realistic load
            delay = np.random.exponential(0.1)  # Average 100ms between requests
            time.sleep(min(delay, 1.0))  # Cap at 1 second
            
        except Exception as e:
            print(f"Error in workload simulation: {e}")
            # Record error
            dashboard.record_request({}, 0, error=True)
    
    print(f"\n✅ Workload simulation completed. Processed {request_count} requests.")


def main():
    """Main demo function."""
    print("🔥 Real-time Dashboard Demo")
    print("=" * 50)
    
    # Create router
    config = ProductionConfig(
        input_dim=768,
        num_experts=8,
        min_experts=1,
        max_experts=4,
        complexity_estimator="gradient_norm"
    )
    
    router = ProductionRouter(config)
    print("✅ Created production router")
    
    # Create dashboard
    dashboard = create_dashboard(router)
    print("✅ Created real-time dashboard")
    
    # Start dashboard
    dashboard.start()
    print("✅ Started dashboard")
    
    try:
        print("\n🚀 Starting workload simulation...")
        print("The dashboard will update in real-time showing:")
        print("  - Request throughput and processing times")
        print("  - Expert utilization patterns")
        print("  - FLOP reduction efficiency")
        print("  - Error rates and health status")
        print("\nPress Ctrl+C to stop the demo\n")
        
        time.sleep(2)  # Give dashboard time to start
        
        # Start workload simulation in background
        workload_thread = threading.Thread(
            target=simulate_workload,
            args=(router, dashboard, 60),  # Run for 60 seconds
            daemon=True
        )
        workload_thread.start()
        
        # Keep dashboard running
        workload_thread.join()
        
        # Show final summary
        print("\n📊 Final Metrics Summary:")
        print("=" * 40)
        summary = dashboard.get_metrics_summary()
        
        if summary:
            print(f"Total Requests: {summary['total_requests']}")
            print(f"Average Throughput: {summary['throughput_rps']:.2f} req/sec")
            print(f"Average Processing Time: {summary['avg_processing_time_ms']:.2f}ms")
            print(f"Error Rate: {summary['error_rate']:.3f}")
            print(f"Average Experts per Token: {summary['avg_experts_per_token']:.2f}")
            print(f"FLOP Reduction: {summary['avg_flop_reduction']*100:.1f}%")
            
            # Export metrics
            dashboard.export_metrics("demo_metrics.json")
            print("📄 Metrics exported to demo_metrics.json")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    finally:
        dashboard.stop()
        print("✅ Dashboard stopped")


if __name__ == "__main__":
    main()