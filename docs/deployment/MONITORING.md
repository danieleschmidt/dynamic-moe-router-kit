# Monitoring and Observability Guide

## Overview
This document provides comprehensive monitoring and observability setup for dynamic-moe-router-kit.

## Metrics Collection

### Application Metrics
The following metrics are automatically collected:

#### Router Performance
- `routing_latency_seconds`: Time spent in routing decisions
- `expert_selection_count`: Number of experts selected per token
- `complexity_estimation_time`: Time for complexity estimation
- `flop_reduction_ratio`: Computational savings ratio

#### Expert Utilization
- `expert_utilization_ratio`: Per-expert usage percentage
- `expert_load_distribution`: Load balance across experts
- `active_experts_per_batch`: Average experts used per batch

#### Model Performance
- `inference_latency_seconds`: End-to-end inference time
- `throughput_tokens_per_second`: Processing throughput
- `memory_usage_mb`: Memory consumption
- `gpu_utilization_percent`: GPU usage (if applicable)

### System Metrics
- CPU and memory usage
- GPU metrics (CUDA/ROCm)
- Disk I/O and network traffic
- Container resource usage

## Alerting Rules

### Performance Alerts
```yaml
# High latency alert
- alert: HighInferenceLatency
  expr: inference_latency_seconds > 1.0
  for: 2m
  
# Low throughput alert  
- alert: LowThroughput
  expr: throughput_tokens_per_second < 10
  for: 5m
```

### Resource Alerts
```yaml
# High memory usage
- alert: HighMemoryUsage
  expr: memory_usage_mb > 1000
  for: 5m
  
# GPU utilization
- alert: LowGPUUtilization
  expr: gpu_utilization_percent < 50
  for: 10m
```

## Dashboard Setup

### Grafana Dashboards
1. **MoE Router Performance**: Routing metrics and expert utilization
2. **System Resources**: CPU, memory, GPU monitoring
3. **Model Training**: Training progress and validation metrics

### Key Visualizations
- Expert utilization heatmap
- Latency distribution histograms
- FLOP reduction trends
- Resource usage timeseries

## Log Aggregation

### Structured Logging
```python
import structlog

logger = structlog.get_logger("dynamic_moe_router")

logger.info(
    "routing_decision",
    num_experts=k,
    complexity_score=complexity,
    flop_reduction=reduction_ratio
)
```

### Log Levels
- **DEBUG**: Detailed routing decisions
- **INFO**: High-level operations
- **WARNING**: Performance degradation
- **ERROR**: Routing failures
- **CRITICAL**: System failures

## Tracing Integration

### OpenTelemetry Setup
```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Instrument routing operations
with tracer.start_as_current_span("expert_routing"):
    experts = router.select_experts(inputs)
```

### Trace Attributes
- `model.name`: Model identifier
- `routing.algorithm`: Routing algorithm used
- `experts.selected`: Number of experts selected
- `batch.size`: Input batch size

## Deployment Integration

### Docker Compose
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - PROMETHEUS_ENDPOINT=http://prometheus:9090
      
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/config/prometheus.yml:/etc/prometheus/prometheus.yml
      
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/config/grafana.yml:/etc/grafana/provisioning/datasources/datasources.yml
```

### Kubernetes Deployment
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: monitoring-config
data:
  prometheus.yml: |
    # Prometheus configuration
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'dynamic-moe-router'
      kubernetes_sd_configs:
      - role: pod
```

## Custom Metrics

### Adding New Metrics
```python
from prometheus_client import Counter, Histogram, Gauge

# Define custom metrics
routing_decisions = Counter(
    'routing_decisions_total',
    'Total routing decisions made',
    ['model_name', 'complexity_level']
)

expert_load = Gauge(
    'expert_load_current',
    'Current load per expert',
    ['expert_id']
)

complexity_distribution = Histogram(
    'complexity_scores',
    'Distribution of complexity scores',
    buckets=[0.1, 0.2, 0.5, 0.8, 1.0]
)
```

### Metric Collection
```python
# In routing code
routing_decisions.labels(
    model_name="mixtral-8x7b",
    complexity_level="high"
).inc()

expert_load.labels(expert_id=expert_idx).set(load_value)
complexity_distribution.observe(complexity_score)
```

## Performance Profiling

### PyTorch Profiler Integration
```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    outputs = model(inputs)

# Export trace
prof.export_chrome_trace("trace.json")
```

### JAX Profiler
```python
import jax
from jax import profiler

# Start profiling
profiler.start_trace("/tmp/jax-trace")

# Your JAX code here
outputs = jax_model(inputs)

# Stop profiling
profiler.stop_trace()
```

## Troubleshooting

### Common Issues
1. **High memory usage**: Check for memory leaks in expert caching
2. **Low expert utilization**: Review complexity estimation logic
3. **High latency**: Profile routing decision overhead
4. **Load imbalance**: Adjust expert selection algorithm

### Debug Commands
```bash
# Check metrics endpoint
curl http://localhost:8080/metrics

# View logs
docker logs dynamic-moe-router

# Profile memory usage
python -m memory_profiler script.py
```

## Integration Examples

### WandB Integration
```python
import wandb

wandb.init(project="dynamic-moe-router")

# Log metrics
wandb.log({
    "expert_utilization": utilization_ratio,
    "flop_reduction": reduction_ratio,
    "inference_latency": latency
})
```

### TensorBoard Integration
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/dynamic_moe_router')

# Log scalars
writer.add_scalar('Performance/Latency', latency, step)
writer.add_scalar('Efficiency/FLOP_Reduction', reduction, step)

# Log histograms
writer.add_histogram('Routing/Complexity_Scores', complexity_scores, step)
```

This monitoring setup provides comprehensive observability for dynamic MoE routing performance and system health.