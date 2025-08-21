"""Structured logging for dynamic MoE routing."""

import json
import logging
import time
import threading
from typing import Any, Dict, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import sys
import os

# Custom log levels
PERFORMANCE_LEVEL = 25  # Between INFO (20) and WARNING (30)
ROUTING_LEVEL = 22     # Between INFO (20) and PERFORMANCE (25)

# Add custom levels
logging.addLevelName(PERFORMANCE_LEVEL, "PERFORMANCE")
logging.addLevelName(ROUTING_LEVEL, "ROUTING")


@dataclass
class LogEvent:
    """Structured log event."""
    timestamp: float
    level: str
    logger_name: str
    message: str
    request_id: Optional[str] = None
    processing_time_ms: Optional[float] = None
    expert_count: Optional[int] = None
    flop_reduction: Optional[float] = None
    error_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': record.created,
            'datetime': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add custom fields if present
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'processing_time_ms'):
            log_entry['processing_time_ms'] = record.processing_time_ms
        if hasattr(record, 'expert_count'):
            log_entry['expert_count'] = record.expert_count
        if hasattr(record, 'flop_reduction'):
            log_entry['flop_reduction'] = record.flop_reduction
        if hasattr(record, 'error_type'):
            log_entry['error_type'] = record.error_type
        if hasattr(record, 'metadata'):
            log_entry['metadata'] = record.metadata
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)


class PerformanceLogger:
    """Logger specifically for performance metrics."""
    
    def __init__(self, name: str = "dynamic_moe_router.performance"):
        self.logger = logging.getLogger(name)
        self._lock = threading.Lock()
        
    def log_request(self, 
                   request_id: str,
                   processing_time_ms: float,
                   expert_count: int,
                   flop_reduction: float,
                   success: bool = True,
                   error_type: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None):
        """Log a routing request with performance metrics."""
        with self._lock:
            extra = {
                'request_id': request_id,
                'processing_time_ms': processing_time_ms,
                'expert_count': expert_count,
                'flop_reduction': flop_reduction,
                'metadata': metadata or {}
            }
            
            if success:
                self.logger.log(
                    PERFORMANCE_LEVEL,
                    f"Request {request_id} completed successfully in {processing_time_ms:.2f}ms",
                    extra=extra
                )
            else:
                extra['error_type'] = error_type
                self.logger.error(
                    f"Request {request_id} failed after {processing_time_ms:.2f}ms: {error_type}",
                    extra=extra
                )
    
    def log_routing_decision(self,
                           request_id: str,
                           complexity_scores: List[float],
                           expert_selection: List[int],
                           routing_strategy: str,
                           metadata: Optional[Dict[str, Any]] = None):
        """Log detailed routing decisions."""
        extra = {
            'request_id': request_id,
            'metadata': {
                'complexity_scores_avg': sum(complexity_scores) / len(complexity_scores),
                'complexity_scores_std': float(
                    (sum((x - sum(complexity_scores) / len(complexity_scores))**2 for x in complexity_scores) / len(complexity_scores))**0.5
                ),
                'unique_experts_used': len(set(expert_selection)),
                'total_experts_selected': len(expert_selection),
                'routing_strategy': routing_strategy,
                **(metadata or {})
            }
        }
        
        self.logger.log(
            ROUTING_LEVEL,
            f"Routing decision for {request_id}: {len(set(expert_selection))} unique experts selected",
            extra=extra
        )


class LogAggregator:
    """Aggregates and analyzes log events."""
    
    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self.events: List[LogEvent] = []
        self._lock = threading.Lock()
    
    def add_event(self, event: LogEvent):
        """Add a log event to the aggregator."""
        with self._lock:
            self.events.append(event)
            if len(self.events) > self.max_events:
                self.events.pop(0)  # Remove oldest event
    
    def get_performance_summary(self, last_n_minutes: int = 5) -> Dict[str, Any]:
        """Get performance summary for the last N minutes."""
        cutoff_time = time.time() - (last_n_minutes * 60)
        
        with self._lock:
            recent_events = [
                event for event in self.events 
                if event.timestamp >= cutoff_time and event.level in ['PERFORMANCE', 'ERROR']
            ]
        
        if not recent_events:
            return {
                'total_requests': 0,
                'error_rate': 0.0,
                'avg_processing_time_ms': 0.0,
                'avg_experts_per_request': 0.0,
                'avg_flop_reduction': 0.0
            }
        
        total_requests = len(recent_events)
        error_count = len([e for e in recent_events if e.level == 'ERROR'])
        
        # Calculate averages for successful requests
        successful_events = [e for e in recent_events if e.level == 'PERFORMANCE']
        
        avg_processing_time = 0.0
        avg_experts = 0.0
        avg_flop_reduction = 0.0
        
        if successful_events:
            avg_processing_time = sum(
                e.processing_time_ms for e in successful_events if e.processing_time_ms
            ) / len(successful_events)
            
            expert_counts = [e.expert_count for e in successful_events if e.expert_count]
            if expert_counts:
                avg_experts = sum(expert_counts) / len(expert_counts)
            
            flop_reductions = [e.flop_reduction for e in successful_events if e.flop_reduction]
            if flop_reductions:
                avg_flop_reduction = sum(flop_reductions) / len(flop_reductions)
        
        return {
            'total_requests': total_requests,
            'successful_requests': len(successful_events),
            'error_count': error_count,
            'error_rate': error_count / total_requests,
            'avg_processing_time_ms': avg_processing_time,
            'avg_experts_per_request': avg_experts,
            'avg_flop_reduction': avg_flop_reduction,
            'time_window_minutes': last_n_minutes
        }
    
    def get_error_breakdown(self, last_n_minutes: int = 30) -> Dict[str, int]:
        """Get breakdown of errors by type."""
        cutoff_time = time.time() - (last_n_minutes * 60)
        
        with self._lock:
            error_events = [
                event for event in self.events 
                if event.timestamp >= cutoff_time and event.level == 'ERROR' and event.error_type
            ]
        
        error_counts = {}
        for event in error_events:
            error_type = event.error_type
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return error_counts


def setup_structured_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_performance_logging: bool = True
) -> Tuple[logging.Logger, PerformanceLogger, LogAggregator]:
    """Setup structured logging for MoE routing."""
    
    # Create root logger
    root_logger = logging.getLogger("dynamic_moe_router")
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = StructuredFormatter()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Create performance logger
    perf_logger = PerformanceLogger() if enable_performance_logging else None
    
    # Create log aggregator
    aggregator = LogAggregator()
    
    # Custom handler to feed aggregator
    class AggregatorHandler(logging.Handler):
        def emit(self, record):
            try:
                event = LogEvent(
                    timestamp=record.created,
                    level=record.levelname,
                    logger_name=record.name,
                    message=record.getMessage(),
                    request_id=getattr(record, 'request_id', None),
                    processing_time_ms=getattr(record, 'processing_time_ms', None),
                    expert_count=getattr(record, 'expert_count', None),
                    flop_reduction=getattr(record, 'flop_reduction', None),
                    error_type=getattr(record, 'error_type', None),
                    metadata=getattr(record, 'metadata', None)
                )
                aggregator.add_event(event)
            except Exception:
                pass  # Don't let logging errors break the application
    
    aggregator_handler = AggregatorHandler()
    root_logger.addHandler(aggregator_handler)
    
    return root_logger, perf_logger, aggregator


def create_request_logger(base_logger: logging.Logger, request_id: str) -> logging.LoggerAdapter:
    """Create a logger adapter that automatically includes request ID."""
    
    class RequestAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            kwargs['extra'] = kwargs.get('extra', {})
            kwargs['extra']['request_id'] = self.extra['request_id']
            return msg, kwargs
    
    return RequestAdapter(base_logger, {'request_id': request_id})


# Global logging setup - can be imported and used directly
DEFAULT_LOGGER, DEFAULT_PERF_LOGGER, DEFAULT_AGGREGATOR = setup_structured_logging(
    log_level=os.getenv('MOE_LOG_LEVEL', 'INFO'),
    log_file=os.getenv('MOE_LOG_FILE'),
    enable_console=os.getenv('MOE_LOG_CONSOLE', 'true').lower() == 'true',
    enable_performance_logging=os.getenv('MOE_LOG_PERFORMANCE', 'true').lower() == 'true'
)