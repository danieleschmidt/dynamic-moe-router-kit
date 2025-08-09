"""FLOP profiling and performance analysis for dynamic MoE routing."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class FLOPCount:
    """Container for FLOP counting information."""
    total_flops: int = 0
    expert_flops: Dict[int, int] = field(default_factory=dict)
    routing_flops: int = 0
    complexity_estimation_flops: int = 0

    def add_expert_flops(self, expert_id: int, flops: int):
        """Add FLOPs for a specific expert."""
        if expert_id not in self.expert_flops:
            self.expert_flops[expert_id] = 0
        self.expert_flops[expert_id] += flops
        self.total_flops += flops

    def add_routing_flops(self, flops: int):
        """Add routing overhead FLOPs."""
        self.routing_flops += flops
        self.total_flops += flops

    def add_complexity_flops(self, flops: int):
        """Add complexity estimation FLOPs."""
        self.complexity_estimation_flops += flops
        self.total_flops += flops


@dataclass
class ProfilerStats:
    """Statistics from profiling session."""
    wall_time: float = 0.0
    flop_count: FLOPCount = field(default_factory=FLOPCount)
    memory_usage: Dict[str, int] = field(default_factory=dict)
    expert_utilization: Dict[int, int] = field(default_factory=dict)
    routing_decisions: List[Dict[str, Any]] = field(default_factory=list)

    def add_routing_decision(self, decision: Dict[str, Any]):
        """Record a routing decision."""
        self.routing_decisions.append(decision)

        # Update expert utilization
        if 'expert_indices' in decision:
            indices = decision['expert_indices']
            if hasattr(indices, 'flatten'):
                # Handle numpy/torch arrays
                flat_indices = indices.flatten()
                for idx in flat_indices:
                    if idx >= 0:  # Valid expert index
                        self.expert_utilization[int(idx)] = self.expert_utilization.get(int(idx), 0) + 1


class FLOPProfiler:
    """Profile computational cost of dynamic MoE routing.
    
    This profiler tracks FLOPs, memory usage, and timing information
    for MoE layers with dynamic routing.
    """

    def __init__(self, detailed_tracking: bool = True):
        self.detailed_tracking = detailed_tracking
        self.stats = ProfilerStats()
        self.is_active = False
        self._start_time = None

        # FLOP estimation constants (rough estimates for common operations)
        self.FLOP_ESTIMATES = {
            'linear': lambda input_size, output_size: 2 * input_size * output_size,
            'softmax': lambda size: 3 * size,  # exp + sum + div
            'layer_norm': lambda size: 5 * size,  # mean + var + norm + scale + shift
            'attention': lambda seq_len, hidden_dim: 4 * seq_len * hidden_dim * seq_len,
            'complexity_estimation': lambda batch_size, seq_len, hidden_dim: batch_size * seq_len * hidden_dim * 10,
            'router_logits': lambda batch_size, seq_len, hidden_dim, num_experts: 2 * batch_size * seq_len * hidden_dim * num_experts,
        }

    def start(self):
        """Start profiling session."""
        self.is_active = True
        self._start_time = time.time()
        self.stats = ProfilerStats()

    def stop(self):
        """Stop profiling session."""
        if self.is_active and self._start_time:
            self.stats.wall_time = time.time() - self._start_time
        self.is_active = False

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def profile_routing_decision(
        self,
        hidden_states_shape: tuple,
        num_experts: int,
        routing_result: Dict[str, Any]
    ):
        """Profile a single routing decision.
        
        Args:
            hidden_states_shape: Shape of input tensor (batch, seq_len, hidden_dim)
            num_experts: Total number of experts
            routing_result: Result from router.route() call
        """
        if not self.is_active:
            return

        batch_size, seq_len, hidden_dim = hidden_states_shape

        # Estimate complexity estimation FLOPs
        complexity_flops = self.FLOP_ESTIMATES['complexity_estimation'](
            batch_size, seq_len, hidden_dim
        )
        self.stats.flop_count.add_complexity_flops(complexity_flops)

        # Estimate routing FLOPs
        routing_flops = self.FLOP_ESTIMATES['router_logits'](
            batch_size, seq_len, hidden_dim, num_experts
        )

        # Add softmax for expert selection
        routing_flops += self.FLOP_ESTIMATES['softmax'](batch_size * seq_len * num_experts)
        self.stats.flop_count.add_routing_flops(routing_flops)

        # Track routing decision
        self.stats.add_routing_decision(routing_result)

    def profile_expert_computation(
        self,
        expert_id: int,
        input_shape: tuple,
        output_shape: tuple,
        operation_type: str = 'linear'
    ):
        """Profile computation for a specific expert.
        
        Args:
            expert_id: ID of the expert
            input_shape: Shape of expert input
            output_shape: Shape of expert output
            operation_type: Type of operation ('linear', 'attention', etc.)
        """
        if not self.is_active:
            return

        # Estimate FLOPs based on operation type
        if operation_type == 'linear':
            input_size = np.prod(input_shape)
            output_size = np.prod(output_shape)
            flops = self.FLOP_ESTIMATES['linear'](input_size, output_size)
        elif operation_type == 'mlp':
            # Standard MoE expert: 2-layer MLP with expansion factor 4
            hidden_dim = input_shape[-1]
            intermediate_dim = hidden_dim * 4
            # First layer + activation + second layer
            flops = (self.FLOP_ESTIMATES['linear'](hidden_dim, intermediate_dim) +
                    intermediate_dim +  # ReLU activation
                    self.FLOP_ESTIMATES['linear'](intermediate_dim, hidden_dim))
        else:
            # Fallback estimation
            flops = np.prod(input_shape) * 100  # Rough estimate

        self.stats.flop_count.add_expert_flops(expert_id, flops)

    def estimate_static_moe_flops(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_experts: int,
        expert_type: str = 'mlp'
    ) -> int:
        """Estimate FLOPs for equivalent static MoE.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            hidden_dim: Hidden dimension
            num_experts: Number of experts
            expert_type: Type of expert network
            
        Returns:
            Estimated FLOPs for static MoE
        """
        total_tokens = batch_size * seq_len

        # Routing overhead (same for static and dynamic)
        routing_flops = self.FLOP_ESTIMATES['router_logits'](
            batch_size, seq_len, hidden_dim, num_experts
        )
        routing_flops += self.FLOP_ESTIMATES['softmax'](total_tokens * num_experts)

        # Expert computation (all experts for all tokens in static MoE)
        if expert_type == 'mlp':
            intermediate_dim = hidden_dim * 4
            expert_flops_per_token = (
                self.FLOP_ESTIMATES['linear'](hidden_dim, intermediate_dim) +
                intermediate_dim +  # ReLU
                self.FLOP_ESTIMATES['linear'](intermediate_dim, hidden_dim)
            )
        else:
            expert_flops_per_token = hidden_dim * 100  # Fallback

        total_expert_flops = total_tokens * num_experts * expert_flops_per_token

        return routing_flops + total_expert_flops

    def compute_efficiency_metrics(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_experts: int
    ) -> Dict[str, Any]:
        """Compute efficiency metrics compared to static MoE.
        
        Returns:
            Dictionary with efficiency metrics
        """
        if not self.stats.routing_decisions:
            return {'error': 'No routing decisions recorded'}

        # Actual FLOPs from dynamic routing
        dynamic_flops = self.stats.flop_count.total_flops

        # Estimated static MoE FLOPs
        static_flops = self.estimate_static_moe_flops(
            batch_size, seq_len, hidden_dim, num_experts
        )

        # FLOP reduction
        flop_reduction = (static_flops - dynamic_flops) / static_flops if static_flops > 0 else 0.0

        # Expert utilization statistics
        total_expert_calls = sum(self.stats.expert_utilization.values())
        max_possible_calls = len(self.stats.routing_decisions) * num_experts
        utilization_efficiency = 1.0 - (total_expert_calls / max_possible_calls) if max_possible_calls > 0 else 0.0

        # Load balance analysis
        if self.stats.expert_utilization:
            utilization_values = list(self.stats.expert_utilization.values())
            load_balance_variance = np.var(utilization_values)
            load_balance_score = 1.0 / (1.0 + load_balance_variance)  # Higher is better
        else:
            load_balance_variance = 0.0
            load_balance_score = 1.0

        return {
            'flop_reduction_ratio': flop_reduction,
            'flop_reduction_percent': flop_reduction * 100,
            'dynamic_flops': dynamic_flops,
            'static_flops_estimate': static_flops,
            'utilization_efficiency': utilization_efficiency,
            'total_expert_calls': total_expert_calls,
            'max_possible_calls': max_possible_calls,
            'load_balance_variance': load_balance_variance,
            'load_balance_score': load_balance_score,
            'wall_time_seconds': self.stats.wall_time,
            'flops_per_second': dynamic_flops / self.stats.wall_time if self.stats.wall_time > 0 else 0
        }

    def summary(self) -> str:
        """Generate a human-readable summary of profiling results."""
        if not self.stats.routing_decisions:
            return "No profiling data available. Use profiler as context manager or call start()/stop()."

        # Get basic stats
        total_decisions = len(self.stats.routing_decisions)
        total_flops = self.stats.flop_count.total_flops
        wall_time = self.stats.wall_time

        # Expert utilization
        expert_calls = self.stats.expert_utilization
        most_used = max(expert_calls.items(), key=lambda x: x[1]) if expert_calls else (0, 0)
        least_used = min(expert_calls.items(), key=lambda x: x[1]) if expert_calls else (0, 0)

        summary = f"""
Dynamic MoE Profiling Summary
============================

Timing:
  Wall time: {wall_time:.3f} seconds
  Decisions: {total_decisions}
  Avg time per decision: {wall_time/total_decisions*1000:.2f} ms

FLOP Analysis:
  Total FLOPs: {total_flops:,}
  Routing FLOPs: {self.stats.flop_count.routing_flops:,}
  Expert FLOPs: {sum(self.stats.flop_count.expert_flops.values()):,}
  Complexity estimation FLOPs: {self.stats.flop_count.complexity_estimation_flops:,}
  
Expert Utilization:
  Unique experts used: {len(expert_calls)}
  Most used expert: #{most_used[0]} ({most_used[1]} calls)
  Least used expert: #{least_used[0]} ({least_used[1]} calls)
  Total expert calls: {sum(expert_calls.values())}
"""

        return summary

    def detailed_breakdown(self) -> Dict[str, Any]:
        """Get detailed breakdown of all profiling data."""
        return {
            'wall_time': self.stats.wall_time,
            'flop_count': {
                'total': self.stats.flop_count.total_flops,
                'routing': self.stats.flop_count.routing_flops,
                'complexity_estimation': self.stats.flop_count.complexity_estimation_flops,
                'expert_breakdown': dict(self.stats.flop_count.expert_flops)
            },
            'expert_utilization': dict(self.stats.expert_utilization),
            'routing_decisions_count': len(self.stats.routing_decisions),
            'memory_usage': dict(self.stats.memory_usage)
        }


class ComparisonProfiler:
    """Compare dynamic vs static MoE performance."""

    def __init__(self):
        self.dynamic_profiler = FLOPProfiler()
        self.static_baseline = None

    def set_static_baseline(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_experts: int,
        wall_time: float
    ):
        """Set baseline performance for static MoE."""
        static_flops = self.dynamic_profiler.estimate_static_moe_flops(
            batch_size, seq_len, hidden_dim, num_experts
        )

        self.static_baseline = {
            'flops': static_flops,
            'wall_time': wall_time,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'hidden_dim': hidden_dim,
            'num_experts': num_experts
        }

    def compare_performance(self) -> Dict[str, Any]:
        """Compare dynamic vs static performance."""
        if not self.static_baseline:
            return {'error': 'Static baseline not set'}

        # Get dynamic performance
        dynamic_flops = self.dynamic_profiler.stats.flop_count.total_flops
        dynamic_time = self.dynamic_profiler.stats.wall_time

        # Compare
        flop_speedup = self.static_baseline['flops'] / dynamic_flops if dynamic_flops > 0 else 0
        time_speedup = self.static_baseline['wall_time'] / dynamic_time if dynamic_time > 0 else 0

        return {
            'flop_speedup': flop_speedup,
            'time_speedup': time_speedup,
            'flop_reduction_percent': (1 - 1/flop_speedup) * 100 if flop_speedup > 0 else 0,
            'time_reduction_percent': (1 - 1/time_speedup) * 100 if time_speedup > 0 else 0,
            'dynamic_flops': dynamic_flops,
            'static_flops': self.static_baseline['flops'],
            'dynamic_time': dynamic_time,
            'static_time': self.static_baseline['wall_time']
        }
