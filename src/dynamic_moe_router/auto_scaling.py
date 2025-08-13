"""Auto-scaling and adaptive resource management for MoE routing."""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from .health_monitoring import HealthMonitor, PerformanceSnapshot

logger = logging.getLogger(__name__)


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    metric_name: str
    scale_up_threshold: float
    scale_down_threshold: float
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.8
    min_instances: int = 1
    max_instances: int = 10
    cooldown_seconds: int = 300
    evaluation_periods: int = 3


@dataclass
class ResourceAllocation:
    """Resource allocation configuration."""
    cpu_cores: int
    memory_mb: int
    gpu_memory_mb: int = 0
    instance_count: int = 1


class AutoScaler:
    """Intelligent auto-scaling for MoE routing systems."""
    
    def __init__(
        self,
        health_monitor: HealthMonitor,
        initial_allocation: ResourceAllocation,
        scaling_policies: Optional[List[ScalingPolicy]] = None,
        enable_predictive_scaling: bool = True,
        enable_load_balancing: bool = True
    ):
        self.health_monitor = health_monitor
        self.current_allocation = initial_allocation
        self.enable_predictive_scaling = enable_predictive_scaling
        self.enable_load_balancing = enable_load_balancing
        
        # Default scaling policies if none provided
        if scaling_policies is None:
            scaling_policies = self._create_default_policies()
        
        self.scaling_policies = {policy.metric_name: policy for policy in scaling_policies}
        
        # Scaling state
        self.last_scaling_action = {}
        self.scaling_history = []
        self.prediction_buffer = []
        
        # Auto-scaling thread
        self.scaling_active = False
        self.scaling_thread = None
        
        # Load balancing
        self.instance_loads = {}
        self.load_balancer_weights = {}
        
        # Callbacks
        self.scaling_callbacks = []
        
        logger.info(f"Auto-scaler initialized with {len(scaling_policies)} policies")
    
    def _create_default_policies(self) -> List[ScalingPolicy]:
        """Create default auto-scaling policies."""
        return [
            ScalingPolicy(
                metric_name="throughput_tokens_per_sec",
                scale_up_threshold=80,
                scale_down_threshold=20,
                scale_up_factor=1.5,
                scale_down_factor=0.8,
                min_instances=1,
                max_instances=8,
                cooldown_seconds=300
            ),
            ScalingPolicy(
                metric_name="error_rate",
                scale_up_threshold=0.05,  # 5% error rate
                scale_down_threshold=0.01,  # 1% error rate
                scale_up_factor=1.3,
                scale_down_factor=0.9,
                min_instances=1,
                max_instances=5,
                cooldown_seconds=180
            ),
            ScalingPolicy(
                metric_name="memory_usage_mb",
                scale_up_threshold=800,  # 800MB memory usage
                scale_down_threshold=400,  # 400MB memory usage
                scale_up_factor=1.4,
                scale_down_factor=0.7,
                min_instances=1,
                max_instances=6,
                cooldown_seconds=240
            ),
            ScalingPolicy(
                metric_name="response_time_ms",
                scale_up_threshold=1000,  # 1 second response time
                scale_down_threshold=200,  # 200ms response time
                scale_up_factor=1.6,
                scale_down_factor=0.8,
                min_instances=1,
                max_instances=10,
                cooldown_seconds=200
            )
        ]
    
    def start_auto_scaling(self, check_interval_seconds: int = 60):
        """Start auto-scaling monitoring and adjustment."""
        if self.scaling_active:
            logger.warning("Auto-scaling already active")
            return
        
        self.scaling_active = True
        
        def scaling_loop():
            while self.scaling_active:
                try:
                    self._evaluate_scaling_decisions()
                    time.sleep(check_interval_seconds)
                except Exception as e:
                    logger.error(f"Auto-scaling evaluation failed: {e}")
                    time.sleep(check_interval_seconds)
        
        self.scaling_thread = threading.Thread(target=scaling_loop, daemon=True)
        self.scaling_thread.start()
        
        logger.info(f"Auto-scaling started with {check_interval_seconds}s interval")
    
    def stop_auto_scaling(self):
        """Stop auto-scaling monitoring."""
        self.scaling_active = False
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5.0)
        logger.info("Auto-scaling stopped")
    
    def _evaluate_scaling_decisions(self):
        """Evaluate and execute scaling decisions based on metrics."""
        
        # Get current health status
        health_status = self.health_monitor.get_health_status()
        
        if not health_status.get("latest_metrics"):
            logger.debug("No metrics available for scaling evaluation")
            return
        
        scaling_decisions = []
        
        # Evaluate each scaling policy
        for metric_name, policy in self.scaling_policies.items():
            if metric_name not in health_status["latest_metrics"]:
                continue
            
            metric_data = health_status["latest_metrics"][metric_name]
            current_value = metric_data["value"]
            
            # Check cooldown period
            last_action_time = self.last_scaling_action.get(metric_name, 0)
            if time.time() - last_action_time < policy.cooldown_seconds:
                continue
            
            # Determine scaling action
            scaling_action = self._determine_scaling_action(current_value, policy)
            
            if scaling_action:
                scaling_decisions.append((metric_name, policy, scaling_action, current_value))
        
        # Execute scaling decisions
        if scaling_decisions:
            self._execute_scaling_decisions(scaling_decisions)
        
        # Predictive scaling
        if self.enable_predictive_scaling:
            self._evaluate_predictive_scaling()
    
    def _determine_scaling_action(self, current_value: float, policy: ScalingPolicy) -> Optional[str]:
        """Determine if scaling action is needed based on policy."""
        
        # For error metrics (higher is worse)
        if policy.metric_name in ["error_rate", "response_time_ms", "memory_usage_mb"]:
            if current_value >= policy.scale_up_threshold:
                return "scale_up"
            elif current_value <= policy.scale_down_threshold and self.current_allocation.instance_count > policy.min_instances:
                return "scale_down"
        
        # For performance metrics (lower is worse)
        elif policy.metric_name in ["throughput_tokens_per_sec"]:
            if current_value <= policy.scale_up_threshold:
                return "scale_up"
            elif current_value >= policy.scale_down_threshold and self.current_allocation.instance_count > policy.min_instances:
                return "scale_down"
        
        return None
    
    def _execute_scaling_decisions(self, decisions: List[Tuple]):
        """Execute scaling decisions with conflict resolution."""
        
        # Resolve conflicting decisions (prioritize scale up)
        scale_up_count = sum(1 for _, _, action, _ in decisions if action == "scale_up")
        scale_down_count = sum(1 for _, _, action, _ in decisions if action == "scale_down")
        
        if scale_up_count > 0:
            final_action = "scale_up"
            primary_decision = next(d for d in decisions if d[2] == "scale_up")
        elif scale_down_count > 0:
            final_action = "scale_down"
            primary_decision = next(d for d in decisions if d[2] == "scale_down")
        else:
            return
        
        metric_name, policy, action, current_value = primary_decision
        
        # Calculate new instance count
        if action == "scale_up":
            new_instances = min(
                int(self.current_allocation.instance_count * policy.scale_up_factor),
                policy.max_instances
            )
        else:  # scale_down
            new_instances = max(
                int(self.current_allocation.instance_count * policy.scale_down_factor),
                policy.min_instances
            )
        
        if new_instances == self.current_allocation.instance_count:
            return  # No change needed
        
        # Execute scaling
        old_instances = self.current_allocation.instance_count
        self.current_allocation.instance_count = new_instances
        
        # Update scaling history
        scaling_event = {
            "timestamp": time.time(),
            "trigger_metric": metric_name,
            "trigger_value": current_value,
            "action": action,
            "old_instances": old_instances,
            "new_instances": new_instances,
            "reason": f"{metric_name} = {current_value} triggered {action}"
        }
        
        self.scaling_history.append(scaling_event)
        self.last_scaling_action[metric_name] = time.time()
        
        # Trigger callbacks
        for callback in self.scaling_callbacks:
            try:
                callback(scaling_event)
            except Exception as e:
                logger.error(f"Scaling callback failed: {e}")
        
        logger.info(f"Auto-scaling: {action} from {old_instances} to {new_instances} instances "
                   f"(triggered by {metric_name} = {current_value})")
    
    def _evaluate_predictive_scaling(self):
        """Evaluate predictive scaling based on trend analysis."""
        
        if not self.enable_predictive_scaling:
            return
        
        # Get performance trends
        trends = self.health_monitor.get_performance_trends(hours=1)
        
        if not trends.get("trends"):
            return
        
        # Analyze trends for predictive scaling
        for metric_name, trend_data in trends["trends"].items():
            if metric_name not in self.scaling_policies:
                continue
            
            policy = self.scaling_policies[metric_name]
            direction = trend_data["direction"]
            current_value = trend_data["current_value"]
            
            # Predict future value (simple linear extrapolation)
            slope = trend_data["slope"]
            predicted_value = current_value + (slope * 3600)  # 1 hour prediction
            
            # Check if predicted value would trigger scaling
            if metric_name in ["error_rate", "response_time_ms", "memory_usage_mb"]:
                if direction == "increasing" and predicted_value >= policy.scale_up_threshold:
                    self._trigger_predictive_scaling("scale_up", metric_name, predicted_value)
            elif metric_name == "throughput_tokens_per_sec":
                if direction == "decreasing" and predicted_value <= policy.scale_up_threshold:
                    self._trigger_predictive_scaling("scale_up", metric_name, predicted_value)
    
    def _trigger_predictive_scaling(self, action: str, metric_name: str, predicted_value: float):
        """Trigger predictive scaling action."""
        
        # Conservative predictive scaling (smaller adjustments)
        if action == "scale_up" and self.current_allocation.instance_count < self.scaling_policies[metric_name].max_instances:
            new_instances = self.current_allocation.instance_count + 1
            
            old_instances = self.current_allocation.instance_count
            self.current_allocation.instance_count = new_instances
            
            scaling_event = {
                "timestamp": time.time(),
                "trigger_metric": metric_name,
                "trigger_value": predicted_value,
                "action": "predictive_scale_up",
                "old_instances": old_instances,
                "new_instances": new_instances,
                "reason": f"Predictive scaling: {metric_name} predicted to reach {predicted_value}"
            }
            
            self.scaling_history.append(scaling_event)
            
            logger.info(f"Predictive scaling: added 1 instance (predicted {metric_name} = {predicted_value})")
    
    def add_scaling_callback(self, callback: Callable[[Dict], None]):
        """Add callback for scaling events."""
        self.scaling_callbacks.append(callback)
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current auto-scaling status and statistics."""
        
        return {
            "auto_scaling_active": self.scaling_active,
            "current_allocation": {
                "instance_count": self.current_allocation.instance_count,
                "cpu_cores": self.current_allocation.cpu_cores,
                "memory_mb": self.current_allocation.memory_mb,
                "gpu_memory_mb": self.current_allocation.gpu_memory_mb
            },
            "scaling_policies": {
                name: {
                    "scale_up_threshold": policy.scale_up_threshold,
                    "scale_down_threshold": policy.scale_down_threshold,
                    "min_instances": policy.min_instances,
                    "max_instances": policy.max_instances,
                    "cooldown_seconds": policy.cooldown_seconds
                }
                for name, policy in self.scaling_policies.items()
            },
            "recent_scaling_events": self.scaling_history[-10:],  # Last 10 events
            "total_scaling_events": len(self.scaling_history),
            "predictive_scaling_enabled": self.enable_predictive_scaling
        }
    
    def update_scaling_policy(self, metric_name: str, policy: ScalingPolicy):
        """Update a scaling policy."""
        self.scaling_policies[metric_name] = policy
        logger.info(f"Updated scaling policy for {metric_name}")
    
    def reset_scaling_history(self):
        """Reset scaling history and state."""
        self.scaling_history = []
        self.last_scaling_action = {}
        self.prediction_buffer = []
        logger.info("Scaling history reset")


class LoadBalancer:
    """Intelligent load balancing for multiple router instances."""
    
    def __init__(
        self,
        balancing_strategy: str = "least_connections",
        health_check_interval: int = 30,
        enable_circuit_breaker: bool = True
    ):
        self.balancing_strategy = balancing_strategy
        self.health_check_interval = health_check_interval
        self.enable_circuit_breaker = enable_circuit_breaker
        
        # Instance tracking
        self.instances = {}
        self.instance_health = {}
        self.instance_connections = {}
        self.instance_response_times = {}
        
        # Circuit breaker state
        self.circuit_breaker_state = {}
        
        # Load balancing weights
        self.weights = {}
        
        logger.info(f"Load balancer initialized with {balancing_strategy} strategy")
    
    def register_instance(self, instance_id: str, router_instance: Any, weight: float = 1.0):
        """Register a router instance for load balancing."""
        self.instances[instance_id] = router_instance
        self.instance_health[instance_id] = True
        self.instance_connections[instance_id] = 0
        self.instance_response_times[instance_id] = []
        self.weights[instance_id] = weight
        
        if self.enable_circuit_breaker:
            self.circuit_breaker_state[instance_id] = {
                "failures": 0,
                "last_failure": 0,
                "state": "closed"  # closed, open, half_open
            }
        
        logger.info(f"Registered instance {instance_id} with weight {weight}")
    
    def remove_instance(self, instance_id: str):
        """Remove a router instance from load balancing."""
        if instance_id in self.instances:
            del self.instances[instance_id]
            del self.instance_health[instance_id]
            del self.instance_connections[instance_id]
            del self.instance_response_times[instance_id]
            del self.weights[instance_id]
            
            if instance_id in self.circuit_breaker_state:
                del self.circuit_breaker_state[instance_id]
            
            logger.info(f"Removed instance {instance_id}")
    
    def route_request(self, *args, **kwargs):
        """Route request to optimal instance based on load balancing strategy."""
        
        # Get available instances
        available_instances = self._get_available_instances()
        
        if not available_instances:
            raise RuntimeError("No healthy instances available")
        
        # Select instance based on strategy
        selected_instance_id = self._select_instance(available_instances)
        selected_instance = self.instances[selected_instance_id]
        
        # Track connection
        self.instance_connections[selected_instance_id] += 1
        start_time = time.time()
        
        try:
            result = selected_instance.route(*args, **kwargs)
            
            # Record successful response time
            response_time = time.time() - start_time
            self._record_response_time(selected_instance_id, response_time)
            
            # Reset circuit breaker on success
            if self.enable_circuit_breaker:
                self.circuit_breaker_state[selected_instance_id]["failures"] = 0
            
            return result
            
        except Exception as e:
            # Record failure
            self._record_failure(selected_instance_id)
            raise
        
        finally:
            self.instance_connections[selected_instance_id] -= 1
    
    def _get_available_instances(self) -> List[str]:
        """Get list of healthy and available instances."""
        available = []
        
        for instance_id in self.instances.keys():
            # Check health
            if not self.instance_health[instance_id]:
                continue
            
            # Check circuit breaker
            if self.enable_circuit_breaker:
                cb_state = self.circuit_breaker_state[instance_id]
                if cb_state["state"] == "open":
                    # Check if should transition to half-open
                    if time.time() - cb_state["last_failure"] > 60:  # 1 minute timeout
                        cb_state["state"] = "half_open"
                    else:
                        continue
            
            available.append(instance_id)
        
        return available
    
    def _select_instance(self, available_instances: List[str]) -> str:
        """Select instance based on load balancing strategy."""
        
        if self.balancing_strategy == "round_robin":
            return self._round_robin_selection(available_instances)
        elif self.balancing_strategy == "least_connections":
            return self._least_connections_selection(available_instances)
        elif self.balancing_strategy == "fastest_response":
            return self._fastest_response_selection(available_instances)
        elif self.balancing_strategy == "weighted":
            return self._weighted_selection(available_instances)
        else:
            # Default to round robin
            return self._round_robin_selection(available_instances)
    
    def _round_robin_selection(self, available_instances: List[str]) -> str:
        """Simple round-robin selection."""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        selected = available_instances[self._round_robin_index % len(available_instances)]
        self._round_robin_index += 1
        return selected
    
    def _least_connections_selection(self, available_instances: List[str]) -> str:
        """Select instance with least active connections."""
        return min(available_instances, key=lambda x: self.instance_connections[x])
    
    def _fastest_response_selection(self, available_instances: List[str]) -> str:
        """Select instance with fastest average response time."""
        def avg_response_time(instance_id):
            times = self.instance_response_times[instance_id]
            return np.mean(times) if times else 0
        
        return min(available_instances, key=avg_response_time)
    
    def _weighted_selection(self, available_instances: List[str]) -> str:
        """Weighted random selection based on instance weights."""
        weights = [self.weights[instance_id] for instance_id in available_instances]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return available_instances[0]
        
        # Weighted random selection
        r = np.random.random() * total_weight
        cumulative = 0
        
        for i, instance_id in enumerate(available_instances):
            cumulative += weights[i]
            if r <= cumulative:
                return instance_id
        
        return available_instances[-1]
    
    def _record_response_time(self, instance_id: str, response_time: float):
        """Record response time for instance."""
        times = self.instance_response_times[instance_id]
        times.append(response_time)
        
        # Keep only recent response times
        if len(times) > 100:
            self.instance_response_times[instance_id] = times[-100:]
    
    def _record_failure(self, instance_id: str):
        """Record failure for circuit breaker."""
        if not self.enable_circuit_breaker:
            return
        
        cb_state = self.circuit_breaker_state[instance_id]
        cb_state["failures"] += 1
        cb_state["last_failure"] = time.time()
        
        # Open circuit breaker if too many failures
        if cb_state["failures"] >= 5:  # Threshold
            cb_state["state"] = "open"
            logger.warning(f"Circuit breaker opened for instance {instance_id}")
    
    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        stats = {
            "strategy": self.balancing_strategy,
            "total_instances": len(self.instances),
            "healthy_instances": sum(self.instance_health.values()),
            "instances": {}
        }
        
        for instance_id in self.instances.keys():
            times = self.instance_response_times[instance_id]
            avg_response = np.mean(times) if times else 0
            
            stats["instances"][instance_id] = {
                "healthy": self.instance_health[instance_id],
                "active_connections": self.instance_connections[instance_id],
                "avg_response_time_ms": avg_response * 1000,
                "weight": self.weights[instance_id]
            }
            
            if self.enable_circuit_breaker:
                cb_state = self.circuit_breaker_state[instance_id]
                stats["instances"][instance_id]["circuit_breaker"] = {
                    "state": cb_state["state"],
                    "failures": cb_state["failures"]
                }
        
        return stats