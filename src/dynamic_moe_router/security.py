"""Security utilities for dynamic MoE routing."""

import logging
import time
import threading
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import hashlib

import numpy as np

from .exceptions import ValidationError, BackendError

logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    """Represents a security-related event."""
    timestamp: float
    event_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    source_hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SecurityMonitor:
    """Real-time security monitoring for MoE operations."""
    
    def __init__(self, max_events: int = 1000):
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.threat_scores: Dict[str, float] = defaultdict(float)
        self.blocked_hashes: Set[str] = set()
        self.rate_limits: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def log_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        source_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a security event."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            description=description,
            source_hash=source_hash,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.events.append(event)
            
            # Update threat scores
            if source_hash:
                severity_weights = {'low': 0.1, 'medium': 0.5, 'high': 1.0, 'critical': 2.0}
                self.threat_scores[source_hash] += severity_weights.get(severity, 0.5)
                
                # Auto-block high-threat sources
                if self.threat_scores[source_hash] > 5.0:
                    self.blocked_hashes.add(source_hash)
                    logger.warning(f"Auto-blocked source {source_hash[:16]} due to high threat score")
    
    def check_rate_limit(self, source_id: str, limit: int = 100, window: int = 60) -> bool:
        """Check if source exceeds rate limit."""
        current_time = time.time()
        
        with self._lock:
            # Clean old entries
            self.rate_limits[source_id] = [
                t for t in self.rate_limits[source_id] 
                if current_time - t < window
            ]
            
            # Check limit
            if len(self.rate_limits[source_id]) >= limit:
                self.log_event(
                    'rate_limit_exceeded',
                    'medium',
                    f'Source {source_id} exceeded rate limit: {len(self.rate_limits[source_id])}/{limit}',
                    source_hash=source_id
                )
                return False
            
            # Record request
            self.rate_limits[source_id].append(current_time)
            return True
    
    def is_blocked(self, source_hash: str) -> bool:
        """Check if source is blocked."""
        return source_hash in self.blocked_hashes
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security monitoring summary."""
        with self._lock:
            recent_events = [e for e in self.events if time.time() - e.timestamp < 300]  # 5 min
            
            severity_counts = defaultdict(int)
            for event in recent_events:
                severity_counts[event.severity] += 1
            
            return {
                'total_events': len(self.events),
                'recent_events': len(recent_events),
                'severity_counts': dict(severity_counts),
                'threat_sources': len(self.threat_scores),
                'blocked_sources': len(self.blocked_hashes),
                'top_threats': sorted(
                    self.threat_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }


class InputSanitizer:
    """Advanced input sanitization for MoE routing."""
    
    def __init__(self, security_monitor: Optional[SecurityMonitor] = None):
        self.security_monitor = security_monitor or SecurityMonitor()
    
    def sanitize_tensor(self, tensor: Any, name: str = "tensor") -> Tuple[Any, str]:
        """Comprehensively sanitize input tensor.
        
        Args:
            tensor: Input tensor to sanitize
            name: Tensor name for logging
            
        Returns:
            Tuple of (sanitized_tensor, integrity_hash)
            
        Raises:
            ValidationError: If tensor fails security checks
        """
        # Generate hash for tracking
        tensor_hash = self._compute_tensor_hash(tensor)
        
        # Check if source is blocked
        if self.security_monitor.is_blocked(tensor_hash):
            raise ValidationError(f"Blocked tensor source: {tensor_hash[:16]}")
        
        # Check rate limits
        if not self.security_monitor.check_rate_limit(tensor_hash):
            raise ValidationError(f"Rate limit exceeded for tensor source")
        
        # Comprehensive validation
        try:
            self._validate_tensor_security(tensor, name, tensor_hash)
            sanitized_tensor = self._apply_sanitization(tensor, name)
            
            self.security_monitor.log_event(
                'tensor_sanitized',
                'low',
                f'Successfully sanitized {name}',
                source_hash=tensor_hash,
                metadata={'shape': getattr(tensor, 'shape', None)}
            )
            
            return sanitized_tensor, tensor_hash
            
        except Exception as e:
            self.security_monitor.log_event(
                'sanitization_failed',
                'high',
                f'Failed to sanitize {name}: {str(e)}',
                source_hash=tensor_hash
            )
            raise
    
    def _compute_tensor_hash(self, tensor: Any) -> str:
        """Compute secure hash of tensor."""
        try:
            if hasattr(tensor, 'tobytes'):
                data = tensor.tobytes()
            else:
                data = str(tensor).encode('utf-8')
            return hashlib.sha256(data).hexdigest()
        except Exception:
            # Fallback for problematic tensors
            return hashlib.sha256(str(type(tensor)).encode()).hexdigest()
    
    def _validate_tensor_security(self, tensor: Any, name: str, tensor_hash: str) -> None:
        """Validate tensor security properties."""
        # Shape validation
        if hasattr(tensor, 'shape'):
            shape = tensor.shape
            if len(shape) > 10:  # Prevent dimension explosion attacks
                raise ValidationError(f"{name} has too many dimensions: {len(shape)}")
            
            if any(dim <= 0 for dim in shape):
                raise ValidationError(f"{name} has invalid dimensions: {shape}")
        
        # Data type validation
        if hasattr(tensor, 'dtype'):
            dtype_str = str(tensor.dtype).lower()
            dangerous_dtypes = ['object', 'unicode', 'string']
            if any(dangerous in dtype_str for dangerous in dangerous_dtypes):
                self.security_monitor.log_event(
                    'dangerous_dtype',
                    'high',
                    f'{name} has potentially dangerous dtype: {tensor.dtype}',
                    source_hash=tensor_hash
                )
        
        # Value range validation
        if hasattr(tensor, 'min') and hasattr(tensor, 'max'):
            min_val, max_val = float(tensor.min()), float(tensor.max())
            
            # Check for adversarial patterns
            if abs(max_val - min_val) < 1e-10 and abs(min_val) > 1e6:
                self.security_monitor.log_event(
                    'suspicious_values',
                    'medium',
                    f'{name} has suspicious value patterns',
                    source_hash=tensor_hash
                )
    
    def _apply_sanitization(self, tensor: Any, name: str) -> Any:
        """Apply sanitization transformations to tensor."""
        if not hasattr(tensor, 'clip'):
            return tensor
        
        # Clip extreme values
        sanitized = np.clip(tensor, -1e6, 1e6)
        
        # Replace NaN/Inf with zeros
        if hasattr(sanitized, 'isnan'):
            sanitized[np.isnan(sanitized)] = 0.0
        if hasattr(sanitized, 'isinf'):
            sanitized[np.isinf(sanitized)] = 0.0
        
        return sanitized


class RouterSecurityPolicy:
    """Security policy enforcement for router operations."""
    
    def __init__(self):
        self.allowed_estimators = {
            'gradient_norm', 'attention_entropy', 'perplexity_proxy', 'threshold'
        }
        self.max_experts = 128
        self.max_routing_ratio = 0.9  # Max 90% of experts can be used
        self.require_load_balancing = True
    
    def validate_router_config(self, config: Dict[str, Any]) -> None:
        """Validate router configuration against security policy."""
        # Validate estimator
        estimator = config.get('complexity_estimator', 'unknown')
        if isinstance(estimator, str) and estimator not in self.allowed_estimators:
            raise ValidationError(f"Estimator '{estimator}' not in allowed list")
        
        # Validate expert counts
        num_experts = config.get('num_experts', 0)
        if num_experts > self.max_experts:
            raise ValidationError(f"num_experts {num_experts} exceeds limit {self.max_experts}")
        
        min_experts = config.get('min_experts', 1)
        max_experts = config.get('max_experts', num_experts)
        
        if max_experts / num_experts > self.max_routing_ratio:
            raise ValidationError(
                f"max_experts ratio {max_experts/num_experts:.2f} exceeds policy limit {self.max_routing_ratio}"
            )
        
        # Enforce load balancing for security
        if self.require_load_balancing and not config.get('load_balancing', True):
            logger.warning("Load balancing disabled - this may create security vulnerabilities")


# Global security monitor instance
_global_security_monitor = SecurityMonitor()


def get_security_monitor() -> SecurityMonitor:
    """Get global security monitor instance."""
    return _global_security_monitor


def create_secure_sanitizer() -> InputSanitizer:
    """Create input sanitizer with global security monitor."""
    return InputSanitizer(_global_security_monitor)


def reset_security_state() -> None:
    """Reset global security state (for testing)."""
    global _global_security_monitor
    _global_security_monitor = SecurityMonitor()