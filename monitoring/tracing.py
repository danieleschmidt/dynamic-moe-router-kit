"""OpenTelemetry tracing for dynamic MoE router components."""

import time
import functools
import logging
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.auto_instrumentation import sitecustomize
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.b3 import B3MultiFormat
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.semconv.resource import ResourceAttributes
    
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

logger = logging.getLogger(__name__)


class TracingConfig:
    """Configuration for distributed tracing."""
    
    def __init__(
        self,
        service_name: str = "dynamic-moe-router",
        service_version: str = "0.1.0",
        environment: str = "development",
        exporter_type: str = "console",  # console, jaeger, otlp
        jaeger_endpoint: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        sample_rate: float = 1.0
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.exporter_type = exporter_type
        self.jaeger_endpoint = jaeger_endpoint or "http://localhost:14268/api/traces"
        self.otlp_endpoint = otlp_endpoint or "http://localhost:4317"
        self.sample_rate = sample_rate


class TracingManager:
    """Manager for OpenTelemetry distributed tracing."""
    
    def __init__(self, config: TracingConfig):
        self.config = config
        self.tracer: Optional[trace.Tracer] = None
        self.initialized = False
        
        if OTEL_AVAILABLE:
            self._setup_tracing()
        else:
            logger.warning("OpenTelemetry not available, tracing disabled")
    
    def _setup_tracing(self):
        """Initialize OpenTelemetry tracing."""
        try:
            # Create resource
            resource = Resource.create({
                ResourceAttributes.SERVICE_NAME: self.config.service_name,
                ResourceAttributes.SERVICE_VERSION: self.config.service_version,
                ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.config.environment,
            })
            
            # Create tracer provider
            provider = TracerProvider(resource=resource)
            
            # Configure exporter
            if self.config.exporter_type == "jaeger":
                exporter = JaegerExporter(
                    agent_host_name="localhost",
                    agent_port=6831,
                    collector_endpoint=self.config.jaeger_endpoint,
                )
            elif self.config.exporter_type == "otlp":
                exporter = OTLPSpanExporter(endpoint=self.config.otlp_endpoint)
            else:
                # Console exporter for development
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter
                exporter = ConsoleSpanExporter()
            
            # Add span processor
            provider.add_span_processor(BatchSpanProcessor(exporter))
            
            # Set global tracer provider
            trace.set_tracer_provider(provider)
            
            # Set up propagation
            set_global_textmap(B3MultiFormat())
            
            # Create tracer
            self.tracer = trace.get_tracer(
                "dynamic_moe_router",
                version=self.config.service_version
            )
            
            self.initialized = True
            logger.info(f"Tracing initialized with {self.config.exporter_type} exporter")
            
        except Exception as e:
            logger.error(f"Failed to initialize tracing: {e}")
            self.tracer = None
    
    def get_tracer(self) -> Optional[trace.Tracer]:
        """Get the configured tracer."""
        return self.tracer if self.initialized else None


# Global tracing manager instance
_tracing_manager: Optional[TracingManager] = None


def initialize_tracing(config: TracingConfig) -> TracingManager:
    """Initialize global tracing manager."""
    global _tracing_manager
    _tracing_manager = TracingManager(config)
    return _tracing_manager


def get_tracer() -> Optional[trace.Tracer]:
    """Get the global tracer instance."""
    if _tracing_manager:
        return _tracing_manager.get_tracer()
    return None


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    parent_span: Optional[trace.Span] = None
):
    """Context manager for creating traced spans."""
    tracer = get_tracer()
    
    if not tracer:
        # If tracing not available, just yield without creating span
        yield None
        return
    
    # Create span
    with tracer.start_as_current_span(name, parent=parent_span) as span:
        # Add attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        
        try:
            yield span
        except Exception as e:
            # Record exception in span
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise


def trace_function(
    span_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    record_args: bool = False,
    record_result: bool = False
):
    """Decorator to trace function calls."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            if not tracer:
                # If tracing not available, just call function
                return func(*args, **kwargs)
            
            # Determine span name
            name = span_name or f"{func.__module__}.{func.__name__}"
            
            # Build attributes
            span_attributes = attributes.copy() if attributes else {}
            span_attributes["function.name"] = func.__name__
            span_attributes["function.module"] = func.__module__
            
            if record_args:
                span_attributes["function.args"] = str(args)
                span_attributes["function.kwargs"] = str(kwargs)
            
            with trace_span(name, span_attributes) as span:
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    if span:
                        duration = time.time() - start_time
                        span.set_attribute("function.duration_seconds", duration)
                        
                        if record_result and result is not None:
                            span.set_attribute("function.result", str(result)[:1000])  # Truncate
                    
                    return result
                    
                except Exception as e:
                    if span:
                        span.record_exception(e)
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
        
        return wrapper
    return decorator


class RouterTracing:
    """Specialized tracing for dynamic MoE router operations."""
    
    @staticmethod
    @trace_function(
        span_name="router.complexity_estimation",
        attributes={"component": "complexity_estimator"}
    )
    def trace_complexity_estimation(
        estimator_func: Callable,
        inputs: Any,
        **kwargs
    ):
        """Trace complexity estimation operations."""
        with trace_span("complexity_estimation.compute") as span:
            if span:
                # Add input metadata
                if hasattr(inputs, 'shape'):
                    span.set_attribute("input.shape", str(inputs.shape))
                if hasattr(inputs, 'dtype'):
                    span.set_attribute("input.dtype", str(inputs.dtype))
            
            # Execute complexity estimation
            complexity_scores = estimator_func(inputs, **kwargs)
            
            if span and hasattr(complexity_scores, 'shape'):
                span.set_attribute("output.shape", str(complexity_scores.shape))
                # Add statistics
                try:
                    import numpy as np
                    scores_array = np.asarray(complexity_scores)
                    span.set_attribute("complexity.mean", float(np.mean(scores_array)))
                    span.set_attribute("complexity.std", float(np.std(scores_array)))
                    span.set_attribute("complexity.min", float(np.min(scores_array)))
                    span.set_attribute("complexity.max", float(np.max(scores_array)))
                except Exception:
                    pass  # Skip statistics if conversion fails
            
            return complexity_scores
    
    @staticmethod
    @trace_function(
        span_name="router.expert_selection", 
        attributes={"component": "expert_router"}
    )
    def trace_expert_selection(
        router_func: Callable,
        complexity_scores: Any,
        num_experts: int,
        **kwargs
    ):
        """Trace expert selection operations."""
        with trace_span("expert_selection.routing") as span:
            if span:
                span.set_attribute("router.num_experts", num_experts)
                
                if hasattr(complexity_scores, 'shape'):
                    span.set_attribute("complexity.shape", str(complexity_scores.shape))
            
            # Execute expert selection
            expert_indices, expert_weights = router_func(
                complexity_scores, num_experts, **kwargs
            )
            
            if span:
                # Add routing statistics
                try:
                    import numpy as np
                    
                    # Expert usage statistics
                    if hasattr(expert_indices, 'shape'):
                        span.set_attribute("routing.output_shape", str(expert_indices.shape))
                        
                        indices_array = np.asarray(expert_indices)
                        avg_experts = np.mean(np.sum(indices_array >= 0, axis=-1))
                        span.set_attribute("routing.avg_experts_per_token", float(avg_experts))
                        
                        # Expert utilization
                        unique_experts, counts = np.unique(indices_array[indices_array >= 0], return_counts=True)
                        utilization = np.zeros(num_experts)
                        utilization[unique_experts] = counts
                        utilization = utilization / np.sum(utilization) if np.sum(utilization) > 0 else utilization
                        
                        # Load balance metrics
                        cv = np.std(utilization) / np.mean(utilization) if np.mean(utilization) > 0 else 0
                        span.set_attribute("routing.load_balance_cv", float(cv))
                        span.set_attribute("routing.active_experts", len(unique_experts))
                        
                except Exception:
                    pass  # Skip statistics if computation fails
            
            return expert_indices, expert_weights
    
    @staticmethod
    @trace_function(
        span_name="router.expert_computation",
        attributes={"component": "expert_network"}
    )
    def trace_expert_computation(
        expert_func: Callable,
        inputs: Any,
        expert_indices: Any,
        expert_weights: Any,
        **kwargs
    ):
        """Trace expert computation operations."""
        with trace_span("expert_computation.forward") as span:
            if span:
                if hasattr(inputs, 'shape'):
                    span.set_attribute("input.shape", str(inputs.shape))
                if hasattr(expert_indices, 'shape'):
                    span.set_attribute("expert_indices.shape", str(expert_indices.shape))
            
            # Execute expert computation
            outputs = expert_func(inputs, expert_indices, expert_weights, **kwargs)
            
            if span:
                if hasattr(outputs, 'shape'):
                    span.set_attribute("output.shape", str(outputs.shape))
            
            return outputs


class ModelTracing:
    """Tracing utilities for model operations."""
    
    @staticmethod
    def trace_model_forward(model_name: str):
        """Decorator for tracing model forward passes."""
        def decorator(func):
            @trace_function(
                span_name=f"model.{model_name}.forward",
                attributes={"model.name": model_name, "operation": "forward"}
            )
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def trace_model_backward(model_name: str):
        """Decorator for tracing model backward passes."""
        def decorator(func):
            @trace_function(
                span_name=f"model.{model_name}.backward",
                attributes={"model.name": model_name, "operation": "backward"}
            )
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator


# Convenience functions for common tracing patterns
def trace_inference(model_name: str = "dynamic-moe-router"):
    """Decorator for tracing inference operations."""
    return trace_function(
        span_name=f"inference.{model_name}",
        attributes={"operation": "inference", "model.name": model_name}
    )


def trace_training_step(model_name: str = "dynamic-moe-router"):
    """Decorator for tracing training steps."""
    return trace_function(
        span_name=f"training.{model_name}.step",
        attributes={"operation": "training_step", "model.name": model_name}
    )


# Example usage and testing
if __name__ == "__main__":
    # Initialize tracing
    config = TracingConfig(
        service_name="dynamic-moe-router-test",
        exporter_type="console"
    )
    
    manager = initialize_tracing(config)
    
    # Example traced function
    @trace_function("test.computation", {"test": "true"})
    def example_computation(x: int, y: int) -> int:
        time.sleep(0.1)  # Simulate work
        return x + y
    
    # Example context manager usage
    with trace_span("test.main", {"example": "true"}) as span:
        if span:
            span.add_event("Starting computation")
        
        result = example_computation(5, 3)
        
        if span:
            span.add_event("Computation completed", {"result": result})
    
    print(f"Computation result: {result}")
    
    # Example router tracing
    def mock_complexity_estimator(inputs):
        time.sleep(0.05)
        import numpy as np
        return np.random.rand(*inputs.shape[:2])
    
    def mock_expert_selector(complexity_scores, num_experts):
        time.sleep(0.02)
        import numpy as np
        batch_size, seq_len = complexity_scores.shape
        k = 4  # Mock top-k
        indices = np.random.randint(0, num_experts, (batch_size, seq_len, k))
        weights = np.random.rand(batch_size, seq_len, k)
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        return indices, weights
    
    # Trace router operations
    import numpy as np
    mock_input = np.random.randn(4, 32, 768)
    
    complexity = RouterTracing.trace_complexity_estimation(
        mock_complexity_estimator, mock_input
    )
    
    expert_indices, expert_weights = RouterTracing.trace_expert_selection(
        mock_expert_selector, complexity, num_experts=8
    )
    
    print("Tracing example completed")