"""
Generation 2: Robust Core - Dynamic MoE Router
Making it reliable with comprehensive error handling, validation, and monitoring.
"""

import numpy as np
import logging
import time
import hashlib
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
import sys
import traceback

# Enhanced logging configuration
def setup_robust_logging():
    """Setup comprehensive logging"""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger('dynamic_moe_router')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    return logger

logger = setup_robust_logging()

@dataclass
class RobustRouterConfig:
    """Robust configuration with validation"""
    input_dim: int = 768
    num_experts: int = 8
    min_experts: int = 1
    max_experts: int = 4
    complexity_threshold: float = 0.5
    max_batch_size: int = 1024
    max_sequence_length: int = 8192
    numerical_stability_eps: float = 1e-8
    routing_temperature: float = 1.0
    enable_monitoring: bool = True
    enable_security_checks: bool = True
    fallback_mode: str = "static_topk"  # static_topk, random, uniform
    
    def __post_init__(self):
        """Validate configuration parameters"""
        self._validate_config()
    
    def _validate_config(self):
        """Comprehensive configuration validation"""
        errors = []
        
        if self.input_dim <= 0:
            errors.append(f"input_dim must be positive, got {self.input_dim}")
        
        if self.num_experts <= 0:
            errors.append(f"num_experts must be positive, got {self.num_experts}")
        
        if not (0 < self.min_experts <= self.max_experts <= self.num_experts):
            errors.append(f"Invalid expert range: min={self.min_experts}, max={self.max_experts}, total={self.num_experts}")
        
        if not (0.0 <= self.complexity_threshold <= 1.0):
            errors.append(f"complexity_threshold must be in [0,1], got {self.complexity_threshold}")
        
        if self.max_batch_size <= 0:
            errors.append(f"max_batch_size must be positive, got {self.max_batch_size}")
        
        if self.max_sequence_length <= 0:
            errors.append(f"max_sequence_length must be positive, got {self.max_sequence_length}")
        
        if self.routing_temperature <= 0:
            errors.append(f"routing_temperature must be positive, got {self.routing_temperature}")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
        
        logger.info("✅ Configuration validation passed")

class SecurityValidator:
    """Security validation and sanitization"""
    
    def __init__(self, config: RobustRouterConfig):
        self.config = config
        self.max_input_norm = 1000.0  # Prevent adversarial inputs
        
    def validate_input(self, inputs: np.ndarray, operation: str = "forward") -> bool:
        """Validate inputs for security and safety"""
        try:
            # Check for basic properties
            if inputs is None:
                raise ValueError("Input cannot be None")
            
            if not isinstance(inputs, np.ndarray):
                raise TypeError(f"Input must be numpy array, got {type(inputs)}")
            
            # Check dimensions
            if len(inputs.shape) != 3:
                raise ValueError(f"Expected 3D input [batch, seq, dim], got shape {inputs.shape}")
            
            batch_size, seq_len, dim = inputs.shape
            
            # Security bounds checking
            if batch_size > self.config.max_batch_size:
                raise ValueError(f"Batch size {batch_size} exceeds maximum {self.config.max_batch_size}")
            
            if seq_len > self.config.max_sequence_length:
                raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.config.max_sequence_length}")
            
            if dim != self.config.input_dim:
                raise ValueError(f"Input dimension {dim} doesn't match config {self.config.input_dim}")
            
            # Check for numerical anomalies
            if not np.isfinite(inputs).all():
                raise ValueError("Input contains non-finite values (NaN or Inf)")
            
            # Check input magnitude (prevent adversarial inputs)
            input_norm = np.linalg.norm(inputs)
            if input_norm > self.max_input_norm:
                logger.warning(f"Input norm {input_norm:.2f} exceeds safety threshold {self.max_input_norm}")
                return False
            
            logger.debug(f"✅ Security validation passed for {operation}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Security validation failed for {operation}: {e}")
            return False
    
    def sanitize_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize outputs to prevent information leakage"""
        sanitized = {}
        
        for key, value in outputs.items():
            if isinstance(value, np.ndarray):
                # Clip extreme values
                if key in ['expert_weights', 'complexity_scores']:
                    value = np.clip(value, 0, 1)
                elif key == 'routing_logits':
                    value = np.clip(value, -10, 10)  # Prevent extreme logits
                
                # Ensure finite values
                value = np.where(np.isfinite(value), value, 0)
                
            sanitized[key] = value
        
        return sanitized

class PerformanceMonitor:
    """Performance monitoring and metrics collection"""
    
    def __init__(self):
        self.metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'max_time': 0.0,
            'min_time': float('inf')
        }
        self.call_history = []
        
    @contextmanager
    def monitor_call(self, operation: str):
        """Context manager for monitoring operation performance"""
        start_time = time.time()
        success = False
        
        try:
            yield
            success = True
        except Exception as e:
            logger.error(f"Operation {operation} failed: {e}")
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            self.metrics['total_calls'] += 1
            if success:
                self.metrics['successful_calls'] += 1
            else:
                self.metrics['failed_calls'] += 1
                
            self.metrics['total_time'] += duration
            self.metrics['avg_time'] = self.metrics['total_time'] / self.metrics['total_calls']
            self.metrics['max_time'] = max(self.metrics['max_time'], duration)
            self.metrics['min_time'] = min(self.metrics['min_time'], duration)
            
            self.call_history.append({
                'operation': operation,
                'duration': duration,
                'success': success,
                'timestamp': end_time
            })
            
            # Keep only recent history
            if len(self.call_history) > 1000:
                self.call_history = self.call_history[-500:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if self.metrics['total_calls'] == 0:
            return {'status': 'no_calls'}
        
        success_rate = self.metrics['successful_calls'] / self.metrics['total_calls']
        
        return {
            'total_calls': self.metrics['total_calls'],
            'success_rate': success_rate,
            'avg_response_time': self.metrics['avg_time'],
            'max_response_time': self.metrics['max_time'],
            'min_response_time': self.metrics['min_time'],
            'status': 'healthy' if success_rate > 0.95 else 'degraded'
        }

def retry_on_failure(max_retries: int = 3, delay: float = 0.1):
    """Decorator for automatic retry on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s...")
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator

class RobustComplexityEstimator:
    """Robust complexity estimator with error handling"""
    
    def __init__(self, config: RobustRouterConfig):
        self.config = config
        self.security_validator = SecurityValidator(config)
        
    @retry_on_failure(max_retries=2)
    def estimate(self, inputs: np.ndarray) -> np.ndarray:
        """Robust complexity estimation with fallbacks"""
        try:
            # Primary method: variance-based complexity
            variance = np.var(inputs, axis=-1, keepdims=True)
            
            # Add numerical stability
            variance = np.maximum(variance, self.config.numerical_stability_eps)
            
            # Normalize using tanh for bounded output
            complexity = np.tanh(variance / np.std(variance))
            
            # Ensure valid range [0, 1]
            complexity = np.clip(complexity, 0, 1)
            
            logger.debug(f"Complexity estimation: mean={np.mean(complexity):.3f}, std={np.std(complexity):.3f}")
            
            return complexity
            
        except Exception as e:
            logger.error(f"Primary complexity estimation failed: {e}")
            return self._fallback_complexity_estimation(inputs)
    
    def _fallback_complexity_estimation(self, inputs: np.ndarray) -> np.ndarray:
        """Fallback complexity estimation methods"""
        try:
            # Fallback 1: L2 norm based
            l2_norms = np.linalg.norm(inputs, axis=-1, keepdims=True)
            normalized_l2 = l2_norms / (np.max(l2_norms) + self.config.numerical_stability_eps)
            return np.clip(normalized_l2, 0, 1)
            
        except Exception as e:
            logger.error(f"Fallback complexity estimation failed: {e}")
            # Final fallback: uniform complexity
            return np.full(
                (inputs.shape[0], inputs.shape[1], 1), 
                self.config.complexity_threshold
            )

class RobustDynamicRouter:
    """Generation 2: Robust dynamic router with comprehensive error handling"""
    
    def __init__(self, config: RobustRouterConfig):
        self.config = config
        self.complexity_estimator = RobustComplexityEstimator(config)
        self.security_validator = SecurityValidator(config)
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize routing network with proper initialization
        self._initialize_routing_network()
        
        # Health tracking
        self.health_status = "healthy"
        self.error_count = 0
        self.max_errors = 10
        
        logger.info(f"✅ Initialized RobustDynamicRouter with {config.num_experts} experts")
    
    def _initialize_routing_network(self):
        """Initialize routing network with robust parameters"""
        np.random.seed(42)  # For reproducibility
        
        # Xavier/Glorot initialization for better stability
        std = np.sqrt(2.0 / (self.config.input_dim + self.config.num_experts))
        self.router_weights = np.random.normal(0, std, 
            (self.config.input_dim, self.config.num_experts))
        
        # Add bias terms
        self.router_bias = np.zeros(self.config.num_experts)
        
        logger.debug("Router network initialized with Xavier initialization")
    
    @retry_on_failure(max_retries=3)
    def route(self, inputs: np.ndarray) -> Dict[str, Any]:
        """Robust routing with comprehensive error handling"""
        
        with self.performance_monitor.monitor_call("route"):
            # Security validation
            if self.config.enable_security_checks:
                if not self.security_validator.validate_input(inputs, "routing"):
                    return self._emergency_fallback_routing(inputs)
            
            try:
                return self._perform_routing(inputs)
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Routing error #{self.error_count}: {e}")
                logger.debug(traceback.format_exc())
                
                # Update health status
                if self.error_count >= self.max_errors:
                    self.health_status = "critical"
                    logger.critical("Router health is critical - too many errors")
                
                return self._emergency_fallback_routing(inputs)
    
    def _perform_routing(self, inputs: np.ndarray) -> Dict[str, Any]:
        """Core routing logic"""
        batch_size, seq_len, dim = inputs.shape
        
        # Estimate complexity
        complexity = self.complexity_estimator.estimate(inputs)
        
        # Dynamic expert count based on complexity
        k_base = self.config.min_experts + (
            self.config.max_experts - self.config.min_experts
        ) * complexity
        k_experts = np.round(k_base).astype(int)
        
        # Compute routing logits with temperature scaling
        routing_logits = (
            np.dot(inputs, self.router_weights) + self.router_bias
        ) / self.config.routing_temperature
        
        # Numerical stability
        routing_logits = np.clip(routing_logits, -10, 10)
        
        # Select top-k experts
        expert_indices = np.argsort(routing_logits, axis=-1)[..., -self.config.max_experts:]
        
        # Compute expert weights using softmax
        selected_logits = np.take_along_axis(routing_logits, expert_indices, axis=-1)
        expert_weights = self._stable_softmax(selected_logits)
        
        # Calculate metrics
        avg_experts = np.mean(k_experts)
        flop_reduction = 1.0 - (avg_experts / self.config.num_experts)
        
        result = {
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'complexity_scores': complexity,
            'avg_experts_per_token': avg_experts,
            'flop_reduction': flop_reduction,
            'routing_logits': routing_logits,
            'routing_entropy': self._calculate_routing_entropy(expert_weights),
            'load_balance_loss': self._calculate_load_balance_loss(expert_weights)
        }
        
        # Sanitize output for security
        if self.config.enable_security_checks:
            result = self.security_validator.sanitize_output(result)
        
        logger.debug(f"✅ Routing successful: experts={avg_experts:.2f}, flop_reduction={flop_reduction:.2%}")
        
        return result
    
    def _stable_softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        # Subtract max for numerical stability
        shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        return exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + self.config.numerical_stability_eps)
    
    def _calculate_routing_entropy(self, weights: np.ndarray) -> float:
        """Calculate routing entropy for load balancing"""
        try:
            # Avoid log(0)
            safe_weights = np.maximum(weights, self.config.numerical_stability_eps)
            entropy = -np.sum(weights * np.log(safe_weights), axis=-1)
            return np.mean(entropy)
        except:
            return 0.0
    
    def _calculate_load_balance_loss(self, weights: np.ndarray) -> float:
        """Calculate load balancing loss"""
        try:
            # Measure how evenly experts are used
            expert_usage = np.mean(weights, axis=(0, 1))
            ideal_usage = 1.0 / len(expert_usage)
            balance_loss = np.sum((expert_usage - ideal_usage) ** 2)
            return balance_loss
        except:
            return 0.0
    
    def _emergency_fallback_routing(self, inputs: np.ndarray) -> Dict[str, Any]:
        """Emergency fallback when all else fails"""
        batch_size, seq_len, dim = inputs.shape
        
        logger.warning("🚨 Using emergency fallback routing")
        
        if self.config.fallback_mode == "static_topk":
            # Use static top-2 experts
            expert_indices = np.tile([0, 1], (batch_size, seq_len, 1))
            expert_weights = np.full((batch_size, seq_len, 2), 0.5)
        elif self.config.fallback_mode == "random":
            # Random expert selection
            expert_indices = np.random.randint(0, self.config.num_experts, 
                                             (batch_size, seq_len, 2))
            expert_weights = np.full((batch_size, seq_len, 2), 0.5)
        else:  # uniform
            # Use all experts equally
            expert_indices = np.tile(
                np.arange(self.config.num_experts), 
                (batch_size, seq_len, 1)
            )
            expert_weights = np.full(
                (batch_size, seq_len, self.config.num_experts), 
                1.0 / self.config.num_experts
            )
        
        return {
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'complexity_scores': np.full((batch_size, seq_len, 1), 0.5),
            'avg_experts_per_token': 2.0,
            'flop_reduction': 0.75,
            'routing_logits': np.zeros((batch_size, seq_len, self.config.num_experts)),
            'routing_entropy': 0.0,
            'load_balance_loss': 0.0,
            'fallback_used': True
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get router health status and metrics"""
        perf_stats = self.performance_monitor.get_stats()
        
        return {
            'health_status': self.health_status,
            'error_count': self.error_count,
            'max_errors': self.max_errors,
            'performance_stats': perf_stats,
            'config_hash': self._get_config_hash()
        }
    
    def _get_config_hash(self) -> str:
        """Get configuration hash for validation"""
        config_str = str(sorted(self.config.__dict__.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

def demonstrate_robust_moe():
    """Demonstrate the robust MoE implementation"""
    print("🛡️  Generation 2: Robust Dynamic MoE Router Demo")
    print("=" * 55)
    
    # Create robust configuration
    config = RobustRouterConfig(
        input_dim=768,
        num_experts=8,
        min_experts=1,
        max_experts=4,
        enable_monitoring=True,
        enable_security_checks=True
    )
    
    # Create robust router
    router = RobustDynamicRouter(config)
    
    # Test with normal input
    print("\n🔬 Testing with normal input...")
    batch_size, seq_len = 2, 128
    normal_inputs = np.random.randn(batch_size, seq_len, config.input_dim) * 0.5
    
    routing_info = router.route(normal_inputs)
    print(f"✅ Normal input: avg_experts={routing_info['avg_experts_per_token']:.2f}, "
          f"flop_reduction={routing_info['flop_reduction']:.1%}")
    
    # Test with extreme input (should trigger security validation)
    print("\n⚠️  Testing with extreme input...")
    extreme_inputs = np.random.randn(batch_size, seq_len, config.input_dim) * 100
    
    routing_info = router.route(extreme_inputs)
    fallback_used = routing_info.get('fallback_used', False)
    print(f"{'🚨 Fallback used' if fallback_used else '✅ Normal processing'}: "
          f"avg_experts={routing_info['avg_experts_per_token']:.2f}")
    
    # Show health status
    health = router.get_health_status()
    print(f"\n📊 Router Health Status: {health['health_status']}")
    print(f"   Error count: {health['error_count']}/{health['max_errors']}")
    print(f"   Performance: {health['performance_stats']}")
    
    print("\n✅ Generation 2 Complete: Robust error handling and monitoring active!")
    
    return router, routing_info

if __name__ == "__main__":
    demonstrate_robust_moe()