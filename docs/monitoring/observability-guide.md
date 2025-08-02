# Observability Guide

## Overview

This guide covers the comprehensive observability stack for dynamic-moe-router-kit, including metrics, logging, tracing, and health monitoring.

## Observability Stack

### Components

1. **Metrics**: Prometheus metrics for quantitative monitoring
2. **Logging**: Structured JSON logging for event tracking
3. **Tracing**: OpenTelemetry distributed tracing
4. **Health Checks**: System health monitoring and alerting
5. **Profiling**: Performance profiling and optimization

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │───▶│   Prometheus    │───▶│    Grafana      │
│                 │    │     Metrics     │    │   Dashboards    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                                              ▲
         ▼                                              │
┌─────────────────┐    ┌─────────────────┐             │
│  Structured     │───▶│   Elasticsearch │─────────────┘
│    Logging      │    │     /Loki       │
└─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐
│  OpenTelemetry  │───▶│     Jaeger      │
│    Tracing      │    │   /Zipkin       │
└─────────────────┘    └─────────────────┘
```

## Metrics Collection

### Prometheus Metrics

Key metrics automatically collected:

#### Router Performance
```python
# Request metrics
dynamic_moe_routing_requests_total{model_name, backend, status}
dynamic_moe_routing_duration_seconds{model_name, backend}

# Expert usage
dynamic_moe_experts_used_per_token{model_name}
dynamic_moe_load_balance_score{model_name, layer_id}

# Computational efficiency
dynamic_moe_flops_computed_total{model_name, layer_type}
dynamic_moe_flops_saved_total{model_name}
```

#### System Health
```python
# Memory and resources
dynamic_moe_memory_usage_bytes{model_name, component}

# Error tracking
dynamic_moe_routing_errors_total{model_name, error_type}
dynamic_moe_expert_failures_total{model_name, expert_id, failure_type}
```

### Usage Example

```python
from monitoring.prometheus_metrics import MetricsCollector

# Initialize metrics collection
collector = MetricsCollector("my-model")

# Record routing metrics
collector.record_routing_request("torch", "success")
collector.record_routing_duration(0.15, "torch")
collector.record_experts_used(3.2)

# Record performance metrics
collector.record_flops(1000000, 350000, "moe")  # computed, saved
collector.update_memory_usage(1024*1024*512, "model")  # 512MB

# Record errors
collector.record_error("complexity_estimation_failed")
```

### Custom Metrics

Add custom metrics for specific use cases:

```python
from prometheus_client import Counter, Histogram, Gauge

# Custom business metric
CUSTOM_METRIC = Counter(
    'dynamic_moe_custom_events_total',
    'Custom events for specific tracking',
    ['event_type', 'model_name']
)

# Usage
CUSTOM_METRIC.labels(
    event_type="user_interaction",
    model_name="production-model"
).inc()
```

## Structured Logging

### Log Configuration

Configure structured JSON logging:

```python
from monitoring.structured_logging import configure_logging, get_logger

# Configure logging
configure_logging(
    log_level="INFO",
    log_format="json",
    log_file="/var/log/dynamic-moe-router.log",
    service_name="dynamic-moe-router",
    environment="production"
)

# Get logger
logger = get_logger("my_module")
```

### Logging Best Practices

#### Standard Event Types

```python
from monitoring.structured_logging import LogEvent

# Router events
logger.info(LogEvent.ROUTING_START, "Starting routing operation")
logger.info(LogEvent.ROUTING_COMPLETE, "Routing completed successfully")
logger.error(LogEvent.ROUTING_ERROR, "Routing failed", exception=e)

# Performance events
logger.info(LogEvent.PERFORMANCE_MEASUREMENT, "Performance measured")
logger.warning(LogEvent.MEMORY_WARNING, "High memory usage detected")

# Model events
logger.info(LogEvent.MODEL_LOAD, "Model loaded successfully")
logger.error(LogEvent.MODEL_ERROR, "Model inference failed")
```

#### Structured Data

Include relevant context in log entries:

```python
logger.info(
    LogEvent.ROUTING_COMPLETE,
    "Dynamic routing completed",
    extra={
        "performance": {
            "duration_seconds": 0.15,
            "tokens_processed": 128,
            "experts_used": 3.2,
            "flop_reduction": 0.35
        },
        "model": {
            "name": "mixtral-8x7b",
            "backend": "torch",
            "batch_size": 4
        }
    }
)
```

#### Correlation IDs

Track requests across components:

```python
import uuid

# Set correlation ID at request start
correlation_id = str(uuid.uuid4())
logger.set_correlation_id(correlation_id)

# All subsequent logs will include this ID
logger.info(LogEvent.ROUTING_START, "Processing request")
# ... other operations
logger.info(LogEvent.ROUTING_COMPLETE, "Request completed")

# Clear correlation ID
logger.clear_correlation_id()
```

## Distributed Tracing

### OpenTelemetry Setup

Initialize tracing:

```python
from monitoring.tracing import TracingConfig, initialize_tracing

# Configure tracing
config = TracingConfig(
    service_name="dynamic-moe-router",
    service_version="0.1.0",
    environment="production",
    exporter_type="jaeger",  # or "otlp", "console"
    jaeger_endpoint="http://jaeger:14268/api/traces"
)

# Initialize tracing
manager = initialize_tracing(config)
```

### Tracing Router Operations

```python
from monitoring.tracing import RouterTracing

# Trace complexity estimation
complexity_scores = RouterTracing.trace_complexity_estimation(
    estimator_function,
    input_tensors
)

# Trace expert selection
expert_indices, weights = RouterTracing.trace_expert_selection(
    router_function,
    complexity_scores,
    num_experts=8
)

# Trace expert computation
outputs = RouterTracing.trace_expert_computation(
    expert_function,
    inputs,
    expert_indices,
    weights
)
```

### Custom Spans

Create custom spans for specific operations:

```python
from monitoring.tracing import trace_span, trace_function

# Context manager
with trace_span("custom_operation", {"operation_type": "preprocessing"}) as span:
    result = preprocess_data(inputs)
    if span:
        span.set_attribute("processed_items", len(result))

# Decorator
@trace_function("model_inference", {"model_type": "dynamic_moe"})
def run_inference(model, inputs):
    return model(inputs)
```

## Health Monitoring

### Health Checks

Set up comprehensive health monitoring:

```python
from monitoring.health_checks import (
    HealthMonitor, MemoryHealthCheck, DiskSpaceHealthCheck,
    ModelHealthCheck, ExternalServiceHealthCheck
)

# Create health monitor
monitor = HealthMonitor()

# Add health checks
monitor.add_check(MemoryHealthCheck(warning_threshold=0.8))
monitor.add_check(DiskSpaceHealthCheck(path="/data"))
monitor.add_check(ModelHealthCheck("my-model"))
monitor.add_check(ExternalServiceHealthCheck("model-service", "http://model-api:8080/health"))

# Run health checks
health_status = await monitor.get_overall_health()
print(health_status)
```

### Health Check Results

Health check responses include:

```json
{
  "status": "healthy",
  "timestamp": 1640995200.0,
  "summary": {
    "total_checks": 4,
    "healthy": 3,
    "degraded": 1,
    "unhealthy": 0
  },
  "checks": {
    "memory": {
      "status": "healthy",
      "message": "Memory usage normal: 65.2%",
      "duration_ms": 12.5,
      "details": {
        "usage_percent": 0.652,
        "available_bytes": 2147483648
      }
    }
  }
}
```

### Custom Health Checks

Create domain-specific health checks:

```python
from monitoring.health_checks import BaseHealthCheck, HealthStatus

class ExpertBalanceHealthCheck(BaseHealthCheck):
    def __init__(self, model_name: str, balance_threshold: float = 0.5):
        super().__init__(f"expert_balance_{model_name}")
        self.model_name = model_name
        self.balance_threshold = balance_threshold
    
    async def _perform_check(self) -> Dict[str, Any]:
        # Check expert load balance
        balance_score = get_current_balance_score(self.model_name)
        
        if balance_score < self.balance_threshold:
            return {
                'status': HealthStatus.DEGRADED,
                'message': f'Expert load imbalance: {balance_score:.2f}',
                'details': {'balance_score': balance_score}
            }
        
        return {
            'status': HealthStatus.HEALTHY,
            'message': f'Expert balance healthy: {balance_score:.2f}',
            'details': {'balance_score': balance_score}
        }
```

## Dashboard Configuration

### Grafana Dashboards

Example Grafana dashboard configuration:

```json
{
  "dashboard": {
    "title": "Dynamic MoE Router - Overview",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(dynamic_moe_routing_requests_total[5m])",
            "legendFormat": "{{model_name}} - {{status}}"
          }
        ]
      },
      {
        "title": "Routing Duration",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(dynamic_moe_routing_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Expert Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "dynamic_moe_experts_used_per_token",
            "legendFormat": "{{model_name}}"
          }
        ]
      },
      {
        "title": "FLOP Efficiency",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(dynamic_moe_flops_saved_total[5m]) / rate(dynamic_moe_flops_computed_total[5m]) * 100",
            "legendFormat": "FLOP Reduction %"
          }
        ]
      }
    ]
  }
}
```

### Key Performance Indicators (KPIs)

Monitor these critical metrics:

1. **Routing Performance**
   - Request latency (p50, p95, p99)
   - Throughput (requests/second)
   - Error rate

2. **Computational Efficiency**
   - FLOP reduction percentage
   - Average experts per token
   - Memory usage

3. **Model Quality**
   - Expert load balance
   - Complexity distribution
   - Model accuracy (if available)

4. **System Health**
   - Memory usage
   - CPU utilization
   - Disk space
   - Error rates

## Alerting

### Prometheus Alerts

Configure alerts for critical conditions:

```yaml
# alerts.yml
groups:
  - name: dynamic-moe-router
    rules:
      - alert: HighErrorRate
        expr: rate(dynamic_moe_routing_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in dynamic MoE router"
          description: "Error rate is {{ $value }} errors/second"
      
      - alert: HighMemoryUsage
        expr: dynamic_moe_memory_usage_bytes / (1024*1024*1024) > 8
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}GB"
      
      - alert: ExpertImbalance
        expr: dynamic_moe_load_balance_score < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Expert load imbalance detected"
          description: "Load balance score is {{ $value }}"
```

### Log-based Alerts

Set up alerts on log patterns:

```yaml
# For ELK stack
{
  "trigger": {
    "schedule": {
      "interval": "1m"
    }
  },
  "input": {
    "search": {
      "request": {
        "indices": ["dynamic-moe-router-*"],
        "body": {
          "query": {
            "bool": {
              "must": [
                {"match": {"event.type": "routing.error"}},
                {"range": {"timestamp": {"gte": "now-5m"}}}
              ]
            }
          }
        }
      }
    }
  },
  "condition": {
    "compare": {
      "ctx.payload.hits.total": {
        "gt": 10
      }
    }
  },
  "actions": {
    "send_alert": {
      "webhook": {
        "url": "https://hooks.slack.com/...",
        "body": "High number of routing errors detected"
      }
    }
  }
}
```

## Performance Profiling

### Built-in Profiler

Use the integrated performance profiler:

```python
from monitoring.prometheus_metrics import PerformanceProfiler, MetricsCollector

collector = MetricsCollector("my-model")
profiler = PerformanceProfiler(collector)

# Profile a code section
profiler.start_profile("inference")
result = model.forward(inputs)
profile_data = profiler.end_profile("inference")

print(f"Duration: {profile_data['duration']:.3f}s")
print(f"Memory delta: {profile_data['memory_delta']/1024/1024:.1f}MB")
```

### External Profiling

Integration with external profilers:

```python
# py-spy integration
import subprocess

def profile_with_pyspy(duration=30, output_file="profile.svg"):
    """Profile application with py-spy."""
    cmd = [
        "py-spy", "record",
        "-d", str(duration),
        "-o", output_file,
        "-p", str(os.getpid())
    ]
    subprocess.run(cmd)

# Memory profiling with memory_profiler
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Function to profile
    pass
```

## Deployment Integration

### Docker Configuration

Enable observability in Docker:

```dockerfile
# Dockerfile
EXPOSE 8000  # Metrics endpoint
EXPOSE 8080  # Health check endpoint

# Environment variables
ENV ENABLE_METRICS=true
ENV METRICS_PORT=8000
ENV LOG_LEVEL=INFO
ENV LOG_FORMAT=json
```

### Kubernetes Integration

Deploy with observability:

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dynamic-moe-router
  labels:
    app: dynamic-moe-router
spec:
  template:
    metadata:
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: dynamic-moe-router
        ports:
        - containerPort: 8000
          name: metrics
        - containerPort: 8080
          name: health
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
```

## Troubleshooting

### Common Issues

1. **Missing Metrics**
   - Check metrics server is running: `curl http://localhost:8000/metrics`
   - Verify Prometheus configuration
   - Check firewall settings

2. **Log Parsing Errors**
   - Validate JSON format
   - Check log aggregation pipeline
   - Verify log rotation settings

3. **Tracing Data Missing**
   - Confirm OpenTelemetry configuration
   - Check exporter endpoints
   - Verify sampling rates

4. **High Cardinality Metrics**
   - Review label usage
   - Implement metric aggregation
   - Use recording rules

### Debug Mode

Enable debug mode for detailed observability:

```python
# Enable debug logging
configure_logging(log_level="DEBUG")

# Enable detailed metrics
collector = MetricsCollector("model", enable_detailed_metrics=True)

# Enable console tracing
config = TracingConfig(exporter_type="console")
```

## Best Practices

1. **Metric Design**
   - Use consistent naming conventions
   - Avoid high cardinality labels
   - Include units in metric names

2. **Logging Strategy**
   - Log at appropriate levels
   - Include context and correlation IDs
   - Use structured data

3. **Tracing Efficiency**
   - Sample appropriately in production
   - Focus on critical paths
   - Include relevant attributes

4. **Performance Impact**
   - Monitor observability overhead
   - Use async operations where possible
   - Implement circuit breakers

For more information, see the [Monitoring Configuration](../deployment/MONITORING.md) guide.