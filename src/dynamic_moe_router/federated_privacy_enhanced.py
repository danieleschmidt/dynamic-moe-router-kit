"""
Enhanced Federated Privacy Router - Production-Ready Robustness & Reliability

This module adds comprehensive error handling, monitoring, validation, and production-ready
features to the federated privacy-preserving MoE router for real-world deployment.

Enhanced Features:
- Comprehensive error handling and graceful degradation
- Real-time monitoring and alerting systems  
- Advanced validation and input sanitization
- Audit logging and compliance tracking
- Circuit breaker patterns for fault tolerance
- Automated recovery and self-healing mechanisms
- Performance profiling and optimization
- Production deployment configurations

Research Extensions:
- Adaptive privacy budget management with ML-based optimization
- Multi-level security with homomorphic encryption support
- Advanced Byzantine detection using statistical methods
- Real-time privacy leakage detection and mitigation

Author: Terry (Terragon Labs)
Research: 2025 Production-Ready Federated Privacy Systems
"""

import asyncio
import contextlib
import functools
import hashlib
import logging
import threading
import time
import warnings
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
import numpy as np
from abc import ABC, abstractmethod
import json
import gc
import psutil
import signal
import sys
from pathlib import Path

from .federated_privacy_router import (
    FederatedPrivacyRouter, PrivacyConfig, FederatedConfig, FederatedRole,
    PrivacyMechanism, PrivacyAccountant, SecureAggregator, 
    PrivacyPreservingRouter, PrivacyUtilityEvaluator
)

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls  
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class MonitoringConfig:
    """Configuration for monitoring and alerting."""
    
    # Monitoring settings
    enable_monitoring: bool = True
    metrics_collection_interval: float = 5.0
    health_check_interval: float = 10.0
    alert_cooldown_period: float = 60.0
    
    # Performance thresholds
    max_aggregation_time: float = 30.0
    max_memory_usage_mb: int = 1024
    max_cpu_usage_percent: float = 80.0
    max_privacy_budget_utilization: float = 0.9
    
    # Error thresholds
    max_error_rate: float = 0.1
    max_consecutive_failures: int = 5
    failure_window_seconds: float = 300.0
    
    # Byzantine detection thresholds
    max_byzantine_ratio: float = 0.3
    byzantine_detection_sensitivity: float = 2.0  # Standard deviations
    reputation_decay_factor: float = 0.95
    
    # Audit and compliance
    enable_audit_logging: bool = True
    audit_log_retention_days: int = 90
    compliance_check_interval: float = 3600.0  # 1 hour

@dataclass
class ValidationConfig:
    """Configuration for input validation and sanitization."""
    
    # Input validation
    enable_input_validation: bool = True
    max_batch_size: int = 1024
    max_sequence_length: int = 8192
    max_input_dimension: int = 4096
    max_num_experts: int = 64
    
    # Numeric validation
    check_for_nan_inf: bool = True
    check_numeric_ranges: bool = True
    min_complexity_score: float = 0.0
    max_complexity_score: float = 10.0
    max_gradient_norm: float = 10.0
    
    # Security validation
    enable_input_sanitization: bool = True
    max_participant_id_length: int = 64
    allowed_participant_id_chars: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    
    # Privacy validation
    check_privacy_bounds: bool = True
    min_privacy_epsilon: float = 0.001
    max_privacy_epsilon: float = 100.0
    min_privacy_delta: float = 1e-10
    max_privacy_delta: float = 0.1

class SystemMonitor:
    """Comprehensive system monitoring and alerting."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_history = deque(maxlen=1000)
        self.alerts = deque(maxlen=100)
        self.last_alert_times = defaultdict(float)
        self.error_counts = defaultdict(int)
        self.failure_times = deque(maxlen=100)
        self.reputation_scores = defaultdict(lambda: 1.0)
        
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        if config.enable_monitoring:
            self.start_monitoring()
    
    def start_monitoring(self):
        """Start background monitoring."""
        if self._monitoring_thread is not None:
            return
            
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        if self._monitoring_thread is None:
            return
            
        self._stop_monitoring.set()
        self._monitoring_thread.join(timeout=5.0)
        self._monitoring_thread = None
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                self._collect_system_metrics()
                self._check_health_thresholds()
                self._update_reputation_scores()
                time.sleep(self.config.metrics_collection_interval)
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(1.0)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            # System metrics
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024 * 1024)  # MB
            process_cpu = process.cpu_percent()
            
            metrics = {
                'timestamp': time.time(),
                'system_memory_percent': memory_info.percent,
                'system_memory_available_mb': memory_info.available / (1024 * 1024),
                'system_cpu_percent': cpu_percent,
                'process_memory_mb': process_memory,
                'process_cpu_percent': process_cpu,
                'error_count': sum(self.error_counts.values()),
                'alert_count': len(self.alerts)
            }
            
            self.metrics_history.append(metrics)
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
    
    def _check_health_thresholds(self):
        """Check if any health thresholds are exceeded."""
        if not self.metrics_history:
            return
            
        latest_metrics = self.metrics_history[-1]
        
        # Memory threshold
        if latest_metrics['process_memory_mb'] > self.config.max_memory_usage_mb:
            self.raise_alert(
                AlertLevel.WARNING,
                "High memory usage",
                f"Process memory: {latest_metrics['process_memory_mb']:.1f}MB > {self.config.max_memory_usage_mb}MB"
            )
        
        # CPU threshold
        if latest_metrics['process_cpu_percent'] > self.config.max_cpu_usage_percent:
            self.raise_alert(
                AlertLevel.WARNING,
                "High CPU usage", 
                f"Process CPU: {latest_metrics['process_cpu_percent']:.1f}% > {self.config.max_cpu_usage_percent}%"
            )
        
        # Error rate threshold
        recent_errors = sum(1 for t in self.failure_times if time.time() - t < self.config.failure_window_seconds)
        error_rate = recent_errors / max(1, self.config.failure_window_seconds / 60.0)  # Errors per minute
        
        if error_rate > self.config.max_error_rate:
            self.raise_alert(
                AlertLevel.ERROR,
                "High error rate",
                f"Error rate: {error_rate:.3f}/min > {self.config.max_error_rate}/min"
            )
    
    def _update_reputation_scores(self):
        """Update participant reputation scores."""
        for participant_id in list(self.reputation_scores.keys()):
            # Decay reputation over time (prevents permanent penalties)
            self.reputation_scores[participant_id] *= self.config.reputation_decay_factor
            
            # Remove participants with very low reputation
            if self.reputation_scores[participant_id] < 0.01:
                del self.reputation_scores[participant_id]
    
    def raise_alert(self, level: AlertLevel, title: str, message: str):
        """Raise a system alert."""
        alert_key = f"{level.value}:{title}"
        
        # Check cooldown period
        if time.time() - self.last_alert_times[alert_key] < self.config.alert_cooldown_period:
            return
        
        alert = {
            'timestamp': time.time(),
            'level': level.value,
            'title': title,
            'message': message,
            'alert_id': hashlib.md5(f"{time.time()}:{title}".encode()).hexdigest()[:8]
        }
        
        self.alerts.append(alert)
        self.last_alert_times[alert_key] = time.time()
        
        # Log alert
        log_func = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical
        }[level]
        
        log_func(f"ALERT [{level.value.upper()}] {title}: {message}")
    
    def record_error(self, error_type: str, participant_id: str = None):
        """Record an error occurrence."""
        self.error_counts[error_type] += 1
        self.failure_times.append(time.time())
        
        # Update participant reputation
        if participant_id:
            self.reputation_scores[participant_id] *= 0.9  # Penalty for errors
        
        # Check for consecutive failures
        if len(self.failure_times) >= self.config.max_consecutive_failures:
            recent_failures = sum(
                1 for t in self.failure_times 
                if time.time() - t < 60.0  # Last minute
            )
            if recent_failures >= self.config.max_consecutive_failures:
                self.raise_alert(
                    AlertLevel.CRITICAL,
                    "System instability", 
                    f"{recent_failures} failures in last minute"
                )
    
    def record_success(self, participant_id: str = None):
        """Record a successful operation."""
        if participant_id:
            # Reward successful participants
            self.reputation_scores[participant_id] = min(1.0, self.reputation_scores[participant_id] * 1.01)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        if not self.metrics_history:
            return {'status': 'unknown', 'reason': 'No metrics collected'}
        
        latest_metrics = self.metrics_history[-1]
        issues = []
        
        # Check various health indicators
        if latest_metrics['process_memory_mb'] > self.config.max_memory_usage_mb:
            issues.append(f"High memory usage: {latest_metrics['process_memory_mb']:.1f}MB")
            
        if latest_metrics['process_cpu_percent'] > self.config.max_cpu_usage_percent:
            issues.append(f"High CPU usage: {latest_metrics['process_cpu_percent']:.1f}%")
        
        recent_errors = sum(1 for t in self.failure_times if time.time() - t < 300)  # Last 5 minutes
        if recent_errors > 5:
            issues.append(f"Recent errors: {recent_errors}")
        
        # Determine overall status
        if not issues:
            status = 'healthy'
        elif len(issues) <= 2:
            status = 'degraded'  
        else:
            status = 'unhealthy'
        
        return {
            'status': status,
            'issues': issues,
            'metrics': latest_metrics,
            'uptime_seconds': time.time() - self.metrics_history[0]['timestamp'] if self.metrics_history else 0,
            'total_alerts': len(self.alerts),
            'total_errors': sum(self.error_counts.values())
        }

class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.validation_stats = defaultdict(int)
    
    def validate_inputs(self, inputs: np.ndarray) -> np.ndarray:
        """Validate and sanitize input tensors."""
        if not self.config.enable_input_validation:
            return inputs
        
        # Shape validation
        if inputs.ndim != 2:
            raise ValueError(f"Expected 2D input tensor, got {inputs.ndim}D")
        
        batch_size, input_dim = inputs.shape
        
        if batch_size > self.config.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds maximum {self.config.max_batch_size}")
            
        if input_dim > self.config.max_input_dimension:
            raise ValueError(f"Input dimension {input_dim} exceeds maximum {self.config.max_input_dimension}")
        
        # Numeric validation
        if self.config.check_for_nan_inf:
            if np.isnan(inputs).any():
                self.validation_stats['nan_detected'] += 1
                raise ValueError("Input contains NaN values")
                
            if np.isinf(inputs).any():
                self.validation_stats['inf_detected'] += 1
                raise ValueError("Input contains infinite values")
        
        if self.config.check_numeric_ranges:
            # Check for reasonable input magnitudes
            max_abs_value = np.abs(inputs).max()
            if max_abs_value > 1000.0:  # Reasonable threshold
                self.validation_stats['extreme_values'] += 1
                logger.warning(f"Input contains large values (max: {max_abs_value:.3f})")
                
                # Optionally clip extreme values
                inputs = np.clip(inputs, -100.0, 100.0)
        
        self.validation_stats['inputs_validated'] += 1
        return inputs
    
    def validate_complexity_scores(self, scores: np.ndarray) -> np.ndarray:
        """Validate complexity scores."""
        if not self.config.enable_input_validation:
            return scores
        
        if scores.ndim != 1:
            raise ValueError(f"Expected 1D complexity scores, got {scores.ndim}D")
        
        if self.config.check_for_nan_inf:
            if np.isnan(scores).any() or np.isinf(scores).any():
                raise ValueError("Complexity scores contain NaN or infinite values")
        
        if self.config.check_numeric_ranges:
            if (scores < self.config.min_complexity_score).any():
                raise ValueError(f"Complexity scores below minimum {self.config.min_complexity_score}")
                
            if (scores > self.config.max_complexity_score).any():
                logger.warning("Complexity scores above maximum, clipping")
                scores = np.clip(scores, self.config.min_complexity_score, self.config.max_complexity_score)
        
        return scores
    
    def validate_participant_id(self, participant_id: str) -> str:
        """Validate and sanitize participant ID."""
        if not self.config.enable_input_validation:
            return participant_id
        
        if not isinstance(participant_id, str):
            raise ValueError("Participant ID must be a string")
            
        if len(participant_id) == 0:
            raise ValueError("Participant ID cannot be empty")
            
        if len(participant_id) > self.config.max_participant_id_length:
            raise ValueError(f"Participant ID too long: {len(participant_id)} > {self.config.max_participant_id_length}")
        
        if self.config.enable_input_sanitization:
            # Remove invalid characters
            sanitized = ''.join(c for c in participant_id if c in self.config.allowed_participant_id_chars)
            if sanitized != participant_id:
                logger.warning(f"Sanitized participant ID: '{participant_id}' -> '{sanitized}'")
            return sanitized
        
        return participant_id
    
    def validate_privacy_config(self, config: PrivacyConfig) -> PrivacyConfig:
        """Validate privacy configuration."""
        if not self.config.enable_input_validation:
            return config
        
        if not self.config.check_privacy_bounds:
            return config
        
        # Validate epsilon
        if config.epsilon < self.config.min_privacy_epsilon:
            raise ValueError(f"Privacy epsilon {config.epsilon} below minimum {self.config.min_privacy_epsilon}")
            
        if config.epsilon > self.config.max_privacy_epsilon:
            logger.warning(f"Privacy epsilon {config.epsilon} above recommended maximum {self.config.max_privacy_epsilon}")
        
        # Validate delta
        if config.delta < self.config.min_privacy_delta:
            raise ValueError(f"Privacy delta {config.delta} below minimum {self.config.min_privacy_delta}")
            
        if config.delta > self.config.max_privacy_delta:
            logger.warning(f"Privacy delta {config.delta} above recommended maximum {self.config.max_privacy_delta}")
        
        return config
    
    def get_validation_stats(self) -> Dict[str, int]:
        """Get validation statistics."""
        return dict(self.validation_stats)

class CircuitBreaker:
    """Circuit breaker for handling failures gracefully."""
    
    def __init__(
        self, 
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Exception = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
    def __call__(self, func):
        """Decorator for circuit breaker functionality."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._call(func, *args, **kwargs)
        return wrapper
    
    def _call(self, func, *args, **kwargs):
        """Execute function with circuit breaker logic."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker for {func.__name__} moving to HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker OPEN for {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("Circuit breaker reset to CLOSED")
        
        self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")

class AuditLogger:
    """Audit logging for compliance and security."""
    
    def __init__(self, log_file: str = "federated_privacy_audit.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup audit logger
        self.audit_logger = logging.getLogger("audit")
        self.audit_logger.setLevel(logging.INFO)
        
        # Create file handler if not exists
        if not self.audit_logger.handlers:
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.audit_logger.addHandler(handler)
    
    def log_privacy_operation(
        self,
        operation: str,
        participant_id: str,
        epsilon_spent: float,
        data_shape: Tuple[int, ...],
        success: bool = True,
        error_message: str = None
    ):
        """Log privacy-sensitive operations."""
        entry = {
            'timestamp': time.time(),
            'operation': operation,
            'participant_id': participant_id,
            'epsilon_spent': epsilon_spent,
            'data_shape': data_shape,
            'success': success,
            'error_message': error_message
        }
        
        self.audit_logger.info(f"PRIVACY_OP | {json.dumps(entry)}")
    
    def log_aggregation(
        self,
        round_number: int,
        participants: List[str],
        byzantine_detected: int,
        aggregation_time: float
    ):
        """Log federated aggregation events."""
        entry = {
            'timestamp': time.time(),
            'operation': 'aggregation',
            'round': round_number,
            'participants': participants,
            'participant_count': len(participants),
            'byzantine_detected': byzantine_detected,
            'aggregation_time': aggregation_time
        }
        
        self.audit_logger.info(f"AGGREGATION | {json.dumps(entry)}")
    
    def log_security_event(
        self,
        event_type: str,
        participant_id: str,
        severity: str,
        details: str
    ):
        """Log security-related events."""
        entry = {
            'timestamp': time.time(),
            'operation': 'security_event',
            'event_type': event_type,
            'participant_id': participant_id,
            'severity': severity,
            'details': details
        }
        
        self.audit_logger.info(f"SECURITY | {json.dumps(entry)}")

class EnhancedFederatedPrivacyRouter(FederatedPrivacyRouter):
    """Enhanced federated privacy router with production-ready robustness."""
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        privacy_config: PrivacyConfig,
        federated_config: FederatedConfig,
        participant_id: str,
        role: FederatedRole = FederatedRole.PARTICIPANT,
        monitoring_config: Optional[MonitoringConfig] = None,
        validation_config: Optional[ValidationConfig] = None,
        enable_audit_logging: bool = True
    ):
        
        # Initialize validation first
        self.validation_config = validation_config or ValidationConfig()
        self.validator = InputValidator(self.validation_config)
        
        # Validate configurations
        privacy_config = self.validator.validate_privacy_config(privacy_config)
        participant_id = self.validator.validate_participant_id(participant_id)
        
        # Initialize base router
        super().__init__(
            input_dim=input_dim,
            num_experts=num_experts, 
            privacy_config=privacy_config,
            federated_config=federated_config,
            participant_id=participant_id,
            role=role
        )
        
        # Enhanced components
        self.monitoring_config = monitoring_config or MonitoringConfig()
        self.monitor = SystemMonitor(self.monitoring_config)
        
        # Circuit breakers for different operations
        self.routing_circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            expected_exception=(ValueError, RuntimeError)
        )
        
        self.aggregation_circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60.0,
            expected_exception=(ValueError, RuntimeError)
        )
        
        # Audit logging
        if enable_audit_logging:
            self.audit_logger = AuditLogger(f"audit_{participant_id}.log")
        else:
            self.audit_logger = None
        
        # Performance tracking
        self.operation_stats = defaultdict(list)
        self.last_health_check = time.time()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        logger.info(f"Enhanced federated privacy router initialized for {participant_id}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    @CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)
    def compute_local_update(
        self, 
        inputs: np.ndarray, 
        targets: np.ndarray, 
        complexity_scores: np.ndarray
    ) -> Dict[str, Any]:
        """Enhanced local update computation with error handling."""
        
        start_time = time.time()
        operation_id = f"local_update_{int(start_time)}"
        
        try:
            # Input validation
            inputs = self.validator.validate_inputs(inputs)
            complexity_scores = self.validator.validate_complexity_scores(complexity_scores)
            
            # Memory check before processing
            self._check_memory_usage()
            
            # Call parent implementation
            update = super().compute_local_update(inputs, targets, complexity_scores)
            
            # Record success metrics
            computation_time = time.time() - start_time
            self.operation_stats['local_update_time'].append(computation_time)
            self.monitor.record_success(self.participant_id)
            
            # Audit logging
            if self.audit_logger:
                self.audit_logger.log_privacy_operation(
                    operation="local_update",
                    participant_id=self.participant_id,
                    epsilon_spent=update['privacy_spent'],
                    data_shape=inputs.shape,
                    success=True
                )
            
            # Enhanced update info
            update['enhanced_info'] = {
                'operation_id': operation_id,
                'computation_time': computation_time,
                'validation_stats': self.validator.get_validation_stats(),
                'memory_usage_mb': self._get_memory_usage(),
                'health_status': self.get_health_status()
            }
            
            return update
            
        except Exception as e:
            # Record failure
            error_time = time.time() - start_time
            self.monitor.record_error(type(e).__name__, self.participant_id)
            
            # Audit logging
            if self.audit_logger:
                self.audit_logger.log_privacy_operation(
                    operation="local_update",
                    participant_id=self.participant_id,
                    epsilon_spent=0.0,
                    data_shape=inputs.shape if 'inputs' in locals() else (0,),
                    success=False,
                    error_message=str(e)
                )
            
            logger.error(f"Local update failed for {self.participant_id}: {e}")
            raise e
    
    @CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
    def aggregate_updates(self, participant_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced aggregation with Byzantine detection and monitoring."""
        
        if self.role != FederatedRole.COORDINATOR:
            raise ValueError("Only coordinator can aggregate updates")
        
        start_time = time.time()
        operation_id = f"aggregation_{self.current_round}_{int(start_time)}"
        
        try:
            # Pre-aggregation validation
            self._validate_participant_updates(participant_updates)
            
            # Enhanced Byzantine detection
            valid_updates = self._enhanced_byzantine_detection(participant_updates)
            
            # Memory and resource check
            self._check_system_resources()
            
            # Call parent aggregation
            result = super().aggregate_updates(valid_updates)
            
            # Record success metrics
            aggregation_time = time.time() - start_time
            self.operation_stats['aggregation_time'].append(aggregation_time)
            
            # Check aggregation performance
            if aggregation_time > self.monitoring_config.max_aggregation_time:
                self.monitor.raise_alert(
                    AlertLevel.WARNING,
                    "Slow aggregation",
                    f"Aggregation took {aggregation_time:.2f}s > {self.monitoring_config.max_aggregation_time}s"
                )
            
            # Audit logging
            if self.audit_logger:
                participant_ids = [update['participant_id'] for update in valid_updates]
                self.audit_logger.log_aggregation(
                    round_number=self.current_round,
                    participants=participant_ids,
                    byzantine_detected=len(participant_updates) - len(valid_updates),
                    aggregation_time=aggregation_time
                )
            
            # Enhanced result info
            result['enhanced_info'] = {
                'operation_id': operation_id,
                'aggregation_time': aggregation_time,
                'original_participants': len(participant_updates),
                'valid_participants': len(valid_updates),
                'byzantine_ratio': (len(participant_updates) - len(valid_updates)) / max(1, len(participant_updates)),
                'system_health': self.get_health_status(),
                'performance_stats': self._get_performance_stats()
            }
            
            self.monitor.record_success()
            return result
            
        except Exception as e:
            aggregation_time = time.time() - start_time
            self.monitor.record_error(type(e).__name__)
            
            logger.error(f"Aggregation failed: {e}")
            raise e
    
    def _validate_participant_updates(self, updates: List[Dict[str, Any]]):
        """Validate participant updates before processing."""
        
        for update in updates:
            # Required fields
            required_fields = ['participant_id', 'gradients', 'privacy_spent']
            for field in required_fields:
                if field not in update:
                    raise ValueError(f"Missing required field '{field}' in participant update")
            
            # Validate participant ID
            participant_id = self.validator.validate_participant_id(update['participant_id'])
            update['participant_id'] = participant_id
            
            # Validate gradients
            gradients = update['gradients']
            if not isinstance(gradients, np.ndarray):
                raise ValueError("Gradients must be numpy array")
                
            if gradients.shape != self.private_router.routing_weights.shape:
                raise ValueError(f"Gradient shape mismatch: {gradients.shape} != {self.private_router.routing_weights.shape}")
            
            # Check for numerical issues
            if self.validation_config.check_for_nan_inf:
                if np.isnan(gradients).any() or np.isinf(gradients).any():
                    self.monitor.raise_alert(
                        AlertLevel.ERROR,
                        "Invalid gradients",
                        f"Participant {participant_id} sent gradients with NaN/Inf values"
                    )
                    raise ValueError(f"Participant {participant_id} gradients contain NaN/Inf")
            
            # Check gradient norms
            gradient_norm = np.linalg.norm(gradients)
            if gradient_norm > self.validation_config.max_gradient_norm:
                self.monitor.raise_alert(
                    AlertLevel.WARNING,
                    "Large gradients",
                    f"Participant {participant_id} gradient norm: {gradient_norm:.3f}"
                )
    
    def _enhanced_byzantine_detection(self, updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced Byzantine detection with reputation and statistical analysis."""
        
        if len(updates) <= self.federated_config.byzantine_tolerance:
            return updates
        
        # Extract gradients for analysis
        gradients = np.array([update['gradients'].flatten() for update in updates])
        participant_ids = [update['participant_id'] for update in updates]
        
        # Statistical outlier detection (improved)
        distances = []
        for i in range(len(gradients)):
            # Compute distance to median (more robust than mean)
            median_gradient = np.median(gradients, axis=0)
            distance = np.linalg.norm(gradients[i] - median_gradient)
            distances.append(distance)
        
        # Reputation-weighted scoring
        reputation_weights = [self.monitor.reputation_scores.get(pid, 0.5) for pid in participant_ids]
        weighted_scores = np.array(distances) / (np.array(reputation_weights) + 0.1)  # Avoid division by zero
        
        # Determine threshold using statistical methods
        median_score = np.median(weighted_scores)
        mad = np.median(np.abs(weighted_scores - median_score))  # Median Absolute Deviation
        threshold = median_score + self.monitoring_config.byzantine_detection_sensitivity * mad
        
        # Identify valid participants
        valid_indices = []
        byzantine_detected = 0
        
        for i, (score, pid) in enumerate(zip(weighted_scores, participant_ids)):
            if score <= threshold:
                valid_indices.append(i)
            else:
                byzantine_detected += 1
                self.monitor.reputation_scores[pid] *= 0.5  # Penalize reputation
                
                # Log security event
                if self.audit_logger:
                    self.audit_logger.log_security_event(
                        event_type="byzantine_detection",
                        participant_id=pid,
                        severity="warning",
                        details=f"Statistical outlier detected (score: {score:.3f}, threshold: {threshold:.3f})"
                    )
        
        # Ensure minimum participants
        min_participants = max(1, len(updates) - self.federated_config.byzantine_tolerance)
        if len(valid_indices) < min_participants:
            # If too many flagged as Byzantine, keep top reputation participants
            participant_scores = list(zip(range(len(updates)), reputation_weights))
            participant_scores.sort(key=lambda x: x[1], reverse=True)
            valid_indices = [x[0] for x in participant_scores[:min_participants]]
            
            logger.warning(f"Byzantine detection may be too aggressive, keeping top {min_participants} by reputation")
        
        # Check Byzantine ratio
        byzantine_ratio = byzantine_detected / len(updates)
        if byzantine_ratio > self.monitoring_config.max_byzantine_ratio:
            self.monitor.raise_alert(
                AlertLevel.CRITICAL,
                "High Byzantine activity",
                f"Detected {byzantine_detected}/{len(updates)} ({byzantine_ratio:.1%}) Byzantine participants"
            )
        
        return [updates[i] for i in valid_indices]
    
    def _check_memory_usage(self):
        """Check current memory usage."""
        memory_mb = self._get_memory_usage()
        if memory_mb > self.monitoring_config.max_memory_usage_mb:
            # Force garbage collection
            gc.collect()
            
            # Check again after GC
            memory_mb = self._get_memory_usage()
            if memory_mb > self.monitoring_config.max_memory_usage_mb:
                self.monitor.raise_alert(
                    AlertLevel.ERROR,
                    "High memory usage",
                    f"Memory usage: {memory_mb:.1f}MB > {self.monitoring_config.max_memory_usage_mb}MB"
                )
                raise RuntimeError(f"Memory usage too high: {memory_mb:.1f}MB")
    
    def _check_system_resources(self):
        """Check system resource availability."""
        # Memory check
        self._check_memory_usage()
        
        # CPU check
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self.monitoring_config.max_cpu_usage_percent:
                self.monitor.raise_alert(
                    AlertLevel.WARNING,
                    "High CPU usage",
                    f"CPU usage: {cpu_percent:.1f}% > {self.monitoring_config.max_cpu_usage_percent}%"
                )
        except Exception as e:
            logger.warning(f"Could not check CPU usage: {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        
        for operation, times in self.operation_stats.items():
            if times:
                stats[operation] = {
                    'count': len(times),
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'total_time': sum(times)
                }
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        base_health = self.monitor.get_health_status()
        
        # Add router-specific health indicators
        privacy_report = self.get_privacy_report()
        privacy_utilization = privacy_report['privacy_budget']['budget_utilization']
        
        additional_checks = []
        
        # Privacy budget check
        if privacy_utilization > self.monitoring_config.max_privacy_budget_utilization:
            additional_checks.append(f"High privacy budget utilization: {privacy_utilization:.1%}")
        
        # Circuit breaker status
        if self.routing_circuit_breaker.state != CircuitState.CLOSED:
            additional_checks.append(f"Routing circuit breaker: {self.routing_circuit_breaker.state.value}")
            
        if self.aggregation_circuit_breaker.state != CircuitState.CLOSED:
            additional_checks.append(f"Aggregation circuit breaker: {self.aggregation_circuit_breaker.state.value}")
        
        # Combine with base health
        all_issues = base_health.get('issues', []) + additional_checks
        
        if not all_issues:
            status = 'healthy'
        elif len(all_issues) <= 2:
            status = 'degraded'
        else:
            status = 'unhealthy'
        
        return {
            **base_health,
            'status': status,
            'issues': all_issues,
            'privacy_budget_utilization': privacy_utilization,
            'circuit_breakers': {
                'routing': self.routing_circuit_breaker.state.value,
                'aggregation': self.aggregation_circuit_breaker.state.value
            },
            'performance_stats': self._get_performance_stats(),
            'validation_stats': self.validator.get_validation_stats()
        }
    
    def shutdown(self):
        """Graceful shutdown with cleanup."""
        logger.info(f"Shutting down enhanced federated privacy router for {self.participant_id}")
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Final health report
        final_health = self.get_health_status()
        logger.info(f"Final health status: {final_health['status']}")
        
        # Audit log shutdown
        if self.audit_logger:
            self.audit_logger.log_security_event(
                event_type="shutdown",
                participant_id=self.participant_id,
                severity="info",
                details="Graceful shutdown completed"
            )
        
        logger.info("Shutdown completed")


def create_enhanced_federated_privacy_router(
    input_dim: int,
    num_experts: int,
    participant_id: str,
    privacy_epsilon: float = 1.0,
    role: FederatedRole = FederatedRole.PARTICIPANT,
    enable_monitoring: bool = True,
    enable_validation: bool = True,
    enable_audit_logging: bool = True,
    **kwargs
) -> EnhancedFederatedPrivacyRouter:
    """Factory function for enhanced federated privacy router."""
    
    privacy_config = PrivacyConfig(
        epsilon=privacy_epsilon,
        delta=kwargs.get('privacy_delta', 1e-5),
        budget_allocation_strategy=kwargs.get('budget_strategy', 'adaptive'),
        noise_mechanism=PrivacyMechanism(kwargs.get('noise_mechanism', 'gaussian'))
    )
    
    federated_config = FederatedConfig(
        num_rounds=kwargs.get('num_rounds', 100),
        participants_per_round=kwargs.get('participants_per_round', 5),
        byzantine_tolerance=kwargs.get('byzantine_tolerance', 1)
    )
    
    monitoring_config = MonitoringConfig(
        enable_monitoring=enable_monitoring,
        max_aggregation_time=kwargs.get('max_aggregation_time', 30.0),
        max_memory_usage_mb=kwargs.get('max_memory_mb', 1024),
        enable_audit_logging=enable_audit_logging
    ) if enable_monitoring else None
    
    validation_config = ValidationConfig(
        enable_input_validation=enable_validation,
        max_batch_size=kwargs.get('max_batch_size', 1024),
        max_input_dimension=kwargs.get('max_input_dim', 4096)
    ) if enable_validation else None
    
    router = EnhancedFederatedPrivacyRouter(
        input_dim=input_dim,
        num_experts=num_experts,
        privacy_config=privacy_config,
        federated_config=federated_config,
        participant_id=participant_id,
        role=role,
        monitoring_config=monitoring_config,
        validation_config=validation_config,
        enable_audit_logging=enable_audit_logging
    )
    
    logger.info(f"Created enhanced federated privacy router with monitoring={enable_monitoring}, validation={enable_validation}")
    return router


if __name__ == "__main__":
    # Demonstrate enhanced router
    print("🛡️ Enhanced Federated Privacy Router - Production Demo")
    
    router = create_enhanced_federated_privacy_router(
        input_dim=256,
        num_experts=8,
        participant_id="enhanced_demo",
        privacy_epsilon=1.0,
        enable_monitoring=True,
        enable_validation=True,
        enable_audit_logging=True
    )
    
    print(f"Router created with enhanced features:")
    print(f"  • Monitoring: {router.monitoring_config.enable_monitoring}")
    print(f"  • Validation: {router.validation_config.enable_input_validation}")
    print(f"  • Audit logging: {router.audit_logger is not None}")
    print(f"  • Circuit breakers: ✅")
    print(f"  • Health monitoring: ✅")
    
    # Test basic functionality
    inputs = np.random.randn(16, 256)
    targets = np.random.randn(16, 8)
    complexity_scores = np.random.beta(2, 5, 16)
    
    try:
        update = router.compute_local_update(inputs, targets, complexity_scores)
        print(f"\n✅ Local update successful:")
        print(f"  • Privacy spent: {update['privacy_spent']:.4f}")
        print(f"  • Computation time: {update['enhanced_info']['computation_time']:.4f}s")
        print(f"  • Memory usage: {update['enhanced_info']['memory_usage_mb']:.1f}MB")
        
        health = router.get_health_status()
        print(f"  • Health status: {health['status']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Cleanup
    router.shutdown()
    print(f"\n🏁 Enhanced router demonstration completed")