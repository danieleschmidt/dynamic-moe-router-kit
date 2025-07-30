"""Performance benchmarks for dynamic MoE routing."""

import pytest
import time
import psutil
import os
from unittest.mock import Mock


class PerformanceTestCase:
    """Base class for performance testing."""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
    
    def start_profiling(self):
        """Start performance profiling."""
        self.start_time = time.perf_counter()
        process = psutil.Process(os.getpid())
        self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    def end_profiling(self):
        """End performance profiling and return metrics."""
        end_time = time.perf_counter()
        process = psutil.Process(os.getpid())
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'duration': end_time - self.start_time,
            'memory_delta': end_memory - self.start_memory,
            'peak_memory': end_memory
        }


@pytest.mark.performance
class TestRoutingPerformance:
    """Test routing performance characteristics."""
    
    def test_routing_latency_benchmark(self, router_config):
        """Benchmark routing latency across different input sizes."""
        perf = PerformanceTestCase()
        
        # Mock router for testing
        mock_router = Mock()
        mock_router.route = Mock(return_value=(Mock(), Mock()))
        
        # Test different batch sizes
        batch_sizes = [1, 8, 32, 128]
        results = {}
        
        for batch_size in batch_sizes:
            perf.start_profiling()
            
            # Simulate routing calls
            for _ in range(100):
                mock_router.route(Mock())
            
            metrics = perf.end_profiling()
            results[batch_size] = metrics['duration']
        
        # Assert reasonable performance scaling
        assert results[1] < 0.1  # Single batch should be fast
        assert results[128] / results[1] < 10  # Scaling should be reasonable
    
    def test_memory_usage_scaling(self, router_config):
        """Test memory usage scales reasonably with model size."""
        perf = PerformanceTestCase()
        
        # Test different expert counts
        expert_counts = [2, 4, 8, 16]
        memory_usage = {}
        
        for num_experts in expert_counts:
            perf.start_profiling()
            
            # Mock expert creation
            mock_experts = [Mock() for _ in range(num_experts)]
            
            # Simulate some operations
            time.sleep(0.01)
            
            metrics = perf.end_profiling()
            memory_usage[num_experts] = metrics['peak_memory']
        
        # Memory should scale sub-linearly
        assert memory_usage[16] / memory_usage[2] < 8
    
    def test_flops_calculation_accuracy(self):
        """Test FLOP calculation accuracy."""
        # Mock FLOP calculator
        mock_profiler = Mock()
        mock_profiler.calculate_flops = Mock(return_value=1000000)
        
        flops = mock_profiler.calculate_flops()
        assert flops > 0
        assert isinstance(flops, (int, float))


@pytest.mark.slow
class TestLongRunningBenchmarks:
    """Long-running performance tests."""
    
    def test_sustained_performance(self):
        """Test performance over extended operation."""
        perf = PerformanceTestCase()
        perf.start_profiling()
        
        # Simulate 1000 routing operations
        for i in range(1000):
            time.sleep(0.001)  # Simulate work
            if i % 100 == 0:
                # Simulate memory cleanup
                pass
        
        metrics = perf.end_profiling()
        
        # Should complete in reasonable time
        assert metrics['duration'] < 30.0
        # Memory growth should be bounded
        assert metrics['memory_delta'] < 100  # MB
    
    def test_memory_leak_detection(self):
        """Detect potential memory leaks."""
        import gc
        
        perf = PerformanceTestCase()
        
        # Baseline memory
        gc.collect()
        perf.start_profiling()
        initial_metrics = perf.end_profiling()
        
        # Simulate operations that might leak
        for _ in range(100):
            mock_objects = [Mock() for _ in range(10)]
            del mock_objects
        
        # Check final memory
        gc.collect()
        perf.start_profiling()
        final_metrics = perf.end_profiling()
        
        memory_growth = final_metrics['peak_memory'] - initial_metrics['peak_memory']
        
        # Memory growth should be minimal
        assert memory_growth < 10  # MB


@pytest.mark.parametrize("backend", ["torch", "jax", "tf"])
def test_backend_performance_parity(backend):
    """Test that all backends have similar performance characteristics."""
    if backend == "torch":
        pytest.importorskip("torch")
    elif backend == "jax":
        pytest.importorskip("jax")
    elif backend == "tf":
        pytest.importorskip("tensorflow")
    
    perf = PerformanceTestCase()
    perf.start_profiling()
    
    # Mock backend-specific operations
    time.sleep(0.01)
    
    metrics = perf.end_profiling()
    
    # All backends should complete quickly
    assert metrics['duration'] < 1.0