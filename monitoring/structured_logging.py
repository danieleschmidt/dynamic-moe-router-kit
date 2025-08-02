"""Structured logging configuration for dynamic MoE router."""

import json
import logging
import logging.config
import time
import traceback
from typing import Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
import sys
import os


class LogLevel(Enum):
    """Standardized log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogEvent(Enum):
    """Standardized log event types."""
    # Router events
    ROUTING_START = "routing.start"
    ROUTING_COMPLETE = "routing.complete"
    ROUTING_ERROR = "routing.error"
    COMPLEXITY_ESTIMATION = "complexity.estimation"
    EXPERT_SELECTION = "expert.selection"
    EXPERT_COMPUTATION = "expert.computation"
    
    # Performance events
    PERFORMANCE_MEASUREMENT = "performance.measurement"
    MEMORY_WARNING = "memory.warning"
    FLOP_COMPUTATION = "flop.computation"
    
    # Model events
    MODEL_LOAD = "model.load"
    MODEL_INFERENCE = "model.inference"
    MODEL_ERROR = "model.error"
    
    # System events
    SYSTEM_START = "system.start"
    SYSTEM_SHUTDOWN = "system.shutdown"
    HEALTH_CHECK = "health.check"
    
    # Security events
    SECURITY_WARNING = "security.warning"
    AUTHENTICATION = "auth.attempt"


class StructuredLogger:
    """Structured logger with consistent formatting."""
    
    def __init__(
        self,
        name: str,
        service_name: str = "dynamic-moe-router",
        service_version: str = "0.1.0",
        environment: str = "development",
        extra_fields: Optional[Dict[str, Any]] = None
    ):
        self.logger = logging.getLogger(name)
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.extra_fields = extra_fields or {}
        
        # Add correlation ID tracking
        self._correlation_id: Optional[str] = None
    
    def _create_log_record(
        self,
        level: LogLevel,
        event: LogEvent,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ) -> Dict[str, Any]:
        """Create a structured log record."""
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level.value,
            "service": {
                "name": self.service_name,
                "version": self.service_version,
                "environment": self.environment
            },
            "event": {
                "type": event.value,
                "message": message
            },
            "process": {
                "pid": os.getpid(),
                "thread": threading.current_thread().name if 'threading' in sys.modules else "main"
            }
        }
        
        # Add correlation ID if available
        if self._correlation_id:
            record["correlation_id"] = self._correlation_id
        
        # Add extra fields
        if extra:
            record["data"] = extra
        
        # Add exception information
        if exception:
            record["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc()
            }
        
        # Add global extra fields
        if self.extra_fields:
            record["metadata"] = self.extra_fields
        
        return record
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for request tracking."""
        self._correlation_id = correlation_id
    
    def clear_correlation_id(self):
        """Clear correlation ID."""
        self._correlation_id = None
    
    def debug(
        self,
        event: LogEvent,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ):
        """Log debug message."""
        if self.logger.isEnabledFor(logging.DEBUG):
            record = self._create_log_record(LogLevel.DEBUG, event, message, extra, exception)
            self.logger.debug(json.dumps(record))
    
    def info(
        self,
        event: LogEvent,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ):
        """Log info message."""
        record = self._create_log_record(LogLevel.INFO, event, message, extra, exception)
        self.logger.info(json.dumps(record))
    
    def warning(
        self,
        event: LogEvent,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ):
        """Log warning message."""
        record = self._create_log_record(LogLevel.WARNING, event, message, extra, exception)
        self.logger.warning(json.dumps(record))
    
    def error(
        self,
        event: LogEvent,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ):
        """Log error message."""
        record = self._create_log_record(LogLevel.ERROR, event, message, extra, exception)
        self.logger.error(json.dumps(record))
    
    def critical(
        self,
        event: LogEvent,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ):
        """Log critical message."""
        record = self._create_log_record(LogLevel.CRITICAL, event, message, extra, exception)
        self.logger.critical(json.dumps(record))


class PerformanceLogger:
    """Specialized logger for performance metrics."""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
    
    def log_routing_performance(
        self,
        duration_seconds: float,
        num_tokens: int,
        num_experts_used: float,
        flop_reduction: float,
        model_name: str,
        backend: str
    ):
        """Log routing performance metrics."""
        self.logger.info(
            LogEvent.PERFORMANCE_MEASUREMENT,
            "Routing performance measurement",
            extra={
                "performance": {
                    "duration_seconds": duration_seconds,
                    "tokens_per_second": num_tokens / duration_seconds if duration_seconds > 0 else 0,
                    "num_tokens": num_tokens,
                    "avg_experts_per_token": num_experts_used,
                    "flop_reduction_percent": flop_reduction * 100,
                },
                "model": {
                    "name": model_name,
                    "backend": backend
                }
            }
        )
    
    def log_memory_usage(
        self,
        current_memory_mb: float,
        peak_memory_mb: float,
        memory_delta_mb: float,
        component: str
    ):
        """Log memory usage metrics."""
        self.logger.info(
            LogEvent.PERFORMANCE_MEASUREMENT,
            "Memory usage measurement",
            extra={
                "memory": {
                    "current_mb": current_memory_mb,
                    "peak_mb": peak_memory_mb,
                    "delta_mb": memory_delta_mb,
                    "component": component
                }
            }
        )
    
    def log_flop_computation(
        self,
        flops_computed: int,
        flops_saved: int,
        static_flops: int,
        model_name: str
    ):
        """Log FLOP computation metrics."""
        flop_reduction = flops_saved / static_flops if static_flops > 0 else 0
        
        self.logger.info(
            LogEvent.FLOP_COMPUTATION,
            "FLOP computation measurement",
            extra={
                "flops": {
                    "computed": flops_computed,
                    "saved": flops_saved,
                    "static_baseline": static_flops,
                    "reduction_percent": flop_reduction * 100,
                    "efficiency_ratio": flops_computed / static_flops if static_flops > 0 else 0
                },
                "model": {
                    "name": model_name
                }
            }
        )


class RouterLogger:
    """Specialized logger for router operations."""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
    
    def log_complexity_estimation(
        self,
        estimator_type: str,
        input_shape: tuple,
        complexity_stats: Dict[str, float],
        duration_seconds: float
    ):
        """Log complexity estimation operation."""
        self.logger.debug(
            LogEvent.COMPLEXITY_ESTIMATION,
            "Complexity estimation completed",
            extra={
                "complexity": {
                    "estimator_type": estimator_type,
                    "input_shape": list(input_shape),
                    "statistics": complexity_stats,
                    "duration_seconds": duration_seconds
                }
            }
        )
    
    def log_expert_selection(
        self,
        num_experts: int,
        avg_experts_selected: float,
        load_balance_score: float,
        expert_utilization: Dict[int, float],
        duration_seconds: float
    ):
        """Log expert selection operation."""
        self.logger.debug(
            LogEvent.EXPERT_SELECTION,
            "Expert selection completed",
            extra={
                "routing": {
                    "total_experts": num_experts,
                    "avg_experts_selected": avg_experts_selected,
                    "load_balance_score": load_balance_score,
                    "expert_utilization": expert_utilization,
                    "duration_seconds": duration_seconds
                }
            }
        )
    
    def log_routing_error(
        self,
        error_type: str,
        error_message: str,
        input_shape: Optional[tuple] = None,
        model_name: Optional[str] = None,
        exception: Optional[Exception] = None
    ):
        """Log routing error."""
        extra_data = {
            "error": {
                "type": error_type,
                "message": error_message
            }
        }
        
        if input_shape:
            extra_data["input"] = {"shape": list(input_shape)}
        
        if model_name:
            extra_data["model"] = {"name": model_name}
        
        self.logger.error(
            LogEvent.ROUTING_ERROR,
            f"Routing error: {error_message}",
            extra=extra_data,
            exception=exception
        )


class SecurityLogger:
    """Specialized logger for security events."""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
    
    def log_security_warning(
        self,
        warning_type: str,
        message: str,
        severity: str = "medium",
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        """Log security warning."""
        extra_data = {
            "security": {
                "warning_type": warning_type,
                "severity": severity,
                "source_ip": source_ip,
                "user_id": user_id
            }
        }
        
        if additional_context:
            extra_data["security"]["context"] = additional_context
        
        self.logger.warning(
            LogEvent.SECURITY_WARNING,
            message,
            extra=extra_data
        )


def configure_logging(
    log_level: str = "INFO",
    log_format: str = "json",  # json or text
    log_file: Optional[str] = None,
    service_name: str = "dynamic-moe-router",
    service_version: str = "0.1.0",
    environment: str = "development"
) -> Dict[str, Any]:
    """Configure application logging."""
    
    # Base configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "format": "%(message)s"
            },
            "text": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": log_format,
                "stream": sys.stdout
            }
        },
        "loggers": {
            "dynamic_moe_router": {
                "level": log_level,
                "handlers": ["console"],
                "propagate": False
            },
            "": {  # Root logger
                "level": log_level,
                "handlers": ["console"]
            }
        }
    }
    
    # Add file handler if specified
    if log_file:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": log_format,
            "filename": log_file,
            "maxBytes": 100 * 1024 * 1024,  # 100MB
            "backupCount": 5
        }
        
        # Add file handler to loggers
        config["loggers"]["dynamic_moe_router"]["handlers"].append("file")
        config["loggers"][""]["handlers"].append("file")
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    return config


def get_logger(
    name: str,
    service_name: str = "dynamic-moe-router",
    service_version: str = "0.1.0",
    environment: str = None,
    extra_fields: Optional[Dict[str, Any]] = None
) -> StructuredLogger:
    """Get a configured structured logger."""
    
    # Get environment from environment variable if not specified
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
    
    return StructuredLogger(
        name=name,
        service_name=service_name,
        service_version=service_version,
        environment=environment,
        extra_fields=extra_fields
    )


# Example usage and testing
if __name__ == "__main__":
    import threading
    import uuid
    
    # Configure logging
    configure_logging(
        log_level="DEBUG",
        log_format="json",
        service_name="dynamic-moe-router-test"
    )
    
    # Create logger
    logger = get_logger(
        "test_module",
        extra_fields={"test_run_id": str(uuid.uuid4())}
    )
    
    # Create specialized loggers
    perf_logger = PerformanceLogger(logger)
    router_logger = RouterLogger(logger)
    security_logger = SecurityLogger(logger)
    
    # Set correlation ID for request tracking
    correlation_id = str(uuid.uuid4())
    logger.set_correlation_id(correlation_id)
    
    # Example logs
    logger.info(
        LogEvent.SYSTEM_START,
        "Dynamic MoE Router starting up",
        extra={"startup_time": time.time()}
    )
    
    # Performance logging
    perf_logger.log_routing_performance(
        duration_seconds=0.15,
        num_tokens=128,
        num_experts_used=3.2,
        flop_reduction=0.35,
        model_name="test-model",
        backend="torch"
    )
    
    # Router logging
    router_logger.log_complexity_estimation(
        estimator_type="gradient_norm",
        input_shape=(4, 32, 768),
        complexity_stats={"mean": 0.45, "std": 0.15, "min": 0.1, "max": 0.8},
        duration_seconds=0.02
    )
    
    router_logger.log_expert_selection(
        num_experts=8,
        avg_experts_selected=3.2,
        load_balance_score=0.85,
        expert_utilization={0: 0.15, 1: 0.12, 2: 0.18, 3: 0.13, 4: 0.14, 5: 0.11, 6: 0.09, 7: 0.08},
        duration_seconds=0.01
    )
    
    # Security logging
    security_logger.log_security_warning(
        warning_type="unusual_activity",
        message="High number of routing requests from single source",
        severity="medium",
        source_ip="192.168.1.100",
        additional_context={"request_count": 1000, "time_window": "5min"}
    )
    
    # Error logging
    try:
        raise ValueError("Test error for logging")
    except Exception as e:
        logger.error(
            LogEvent.ROUTING_ERROR,
            "Example error occurred",
            extra={"error_context": "test_scenario"},
            exception=e
        )
    
    logger.info(
        LogEvent.SYSTEM_SHUTDOWN,
        "Logging example completed",
        extra={"completion_time": time.time()}
    )
    
    print("\nStructured logging example completed")
    print("All logs are in JSON format for easy parsing and analysis")