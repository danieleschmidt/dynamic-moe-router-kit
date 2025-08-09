"""Logging configuration for dynamic MoE router kit."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    include_timestamp: bool = True,
    include_module: bool = True
) -> None:
    """Setup logging configuration for the dynamic MoE router.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string (optional)
        include_timestamp: Whether to include timestamp in logs
        include_module: Whether to include module name in logs
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Default format string
    if format_string is None:
        parts = []
        if include_timestamp:
            parts.append("%(asctime)s")
        if include_module:
            parts.append("%(name)s")
        parts.extend(["%(levelname)s", "%(message)s"])
        format_string = " - ".join(parts)

    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Set specific loggers to appropriate levels
    logging.getLogger("dynamic_moe_router").setLevel(numeric_level)

    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger
    """
    return logging.getLogger(f"dynamic_moe_router.{name}")


def set_debug_mode(enabled: bool = True) -> None:
    """Enable or disable debug mode logging.
    
    Args:
        enabled: Whether to enable debug mode
    """
    level = logging.DEBUG if enabled else logging.INFO
    logging.getLogger("dynamic_moe_router").setLevel(level)


class PerformanceLogger:
    """Logger for performance monitoring."""

    def __init__(self, name: str):
        self.logger = get_logger(f"performance.{name}")
        self._timers = {}

    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        import time
        self._timers[operation] = time.time()
        self.logger.debug(f"Started {operation}")

    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        import time
        if operation not in self._timers:
            self.logger.warning(f"Timer for '{operation}' was not started")
            return 0.0

        duration = time.time() - self._timers[operation]
        self.logger.info(f"Completed {operation} in {duration:.3f}s")
        del self._timers[operation]
        return duration

    def log_memory_usage(self, operation: str, tensor_size_mb: float) -> None:
        """Log memory usage for an operation."""
        self.logger.info(f"{operation}: {tensor_size_mb:.1f} MB")

    def log_flop_count(self, operation: str, flops: int) -> None:
        """Log FLOP count for an operation."""
        if flops > 1e9:
            self.logger.info(f"{operation}: {flops/1e9:.2f} GFLOPs")
        elif flops > 1e6:
            self.logger.info(f"{operation}: {flops/1e6:.2f} MFLOPs")
        else:
            self.logger.info(f"{operation}: {flops:,} FLOPs")


class RouterLogger:
    """Specialized logger for router operations."""

    def __init__(self, router_id: str):
        self.logger = get_logger(f"router.{router_id}")
        self.performance = PerformanceLogger(f"router.{router_id}")

    def log_initialization(self, config: dict) -> None:
        """Log router initialization."""
        self.logger.info(f"Initializing router with config: {config}")

    def log_routing_decision(self, batch_size: int, seq_len: int, avg_experts: float, flop_reduction: float) -> None:
        """Log routing decision statistics."""
        self.logger.debug(
            f"Routed batch [{batch_size}, {seq_len}]: "
            f"avg_experts={avg_experts:.2f}, flop_reduction={flop_reduction:.1%}"
        )

    def log_load_balancing(self, expert_utilization: list, variance: float) -> None:
        """Log load balancing statistics."""
        self.logger.debug(
            f"Load balancing: utilization_variance={variance:.4f}, "
            f"max_util={max(expert_utilization):.3f}, min_util={min(expert_utilization):.3f}"
        )

    def log_error(self, operation: str, error: Exception) -> None:
        """Log routing errors."""
        self.logger.error(f"Error in {operation}: {type(error).__name__}: {error}")

    def log_warning(self, message: str) -> None:
        """Log routing warnings."""
        self.logger.warning(message)
