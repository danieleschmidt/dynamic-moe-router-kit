# Dynamic MoE Router - Production Deployment Guide

## Overview

This guide covers the deployment and operation of the Dynamic MoE Router in production environments. The router has been enhanced with comprehensive security, monitoring, caching, and resilience features for enterprise-grade deployments.

## Architecture

### Core Components

1. **Dynamic Router**: Adaptive expert selection with complexity estimation
2. **Security Layer**: Input validation, sanitization, and threat detection
3. **Monitoring System**: Real-time performance tracking and alerting
4. **Caching Layer**: Intelligent caching with prefetching
5. **Resilience Patterns**: Circuit breakers, retries, and fallback strategies

### Performance Features

- **Adaptive Caching**: LRU cache with predictive prefetching
- **Parallel Processing**: Multi-threaded expert selection for large batches
- **Load Balancing**: Automatic expert utilization balancing
- **Memory Optimization**: Efficient tensor handling and cleanup

## Deployment Options

### 1. Docker Deployment

```bash
# Build production image
docker build -f deploy/docker/Dockerfile.production -t dynamic-moe-router:latest .

# Run with production config
docker run -d \
  --name moe-router \
  -p 8000:8000 \
  -v $(pwd)/config:/app/config \
  -e MOE_ENVIRONMENT=production \
  -e MOE_LOG_LEVEL=INFO \
  dynamic-moe-router:latest
```

### 2. Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f deploy/kubernetes/deployment.yaml

# Check deployment status
kubectl get pods -l app=dynamic-moe-router
kubectl logs -l app=dynamic-moe-router

# Port forward for testing
kubectl port-forward svc/dynamic-moe-router-service 8080:80
```

### 3. Direct Python Deployment

```python
from dynamic_moe_router.production_fixed import RouterFactory

# Create production router
router = RouterFactory.create_optimized_for_inference()

# Health check
health = router.health_check()
print(f"Router status: {health['status']}")

# Process requests
import numpy as np
test_input = np.random.randn(4, 128, 768).astype(np.float32)
result = router.route(test_input)
```

## Configuration

### Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MOE_INPUT_DIM` | Input tensor dimension | 768 | 1024 |
| `MOE_NUM_EXPERTS` | Total number of experts | 8 | 16 |
| `MOE_MIN_EXPERTS` | Minimum experts per token | 1 | 2 |
| `MOE_MAX_EXPERTS` | Maximum experts per token | 4 | 8 |
| `MOE_ENABLE_CACHING` | Enable result caching | true | false |
| `MOE_ENABLE_MONITORING` | Enable performance monitoring | true | false |
| `MOE_SECURITY_LEVEL` | Security validation level | standard | strict |
| `MOE_LOG_LEVEL` | Logging verbosity | INFO | DEBUG |

### Configuration Files

Create `config/production.json`:

```json
{
  "router_config": {
    "input_dim": 768,
    "num_experts": 8,
    "min_experts": 1,
    "max_experts": 3,
    "complexity_estimator": "gradient_norm",
    "routing_strategy": "top_k",
    "load_balancing": true,
    "noise_factor": 0.1
  },
  "performance": {
    "enable_caching": true,
    "enable_monitoring": true,
    "cache_size": 10000,
    "parallel_threshold": 1000
  },
  "security": {
    "enable_validation": true,
    "max_tensor_size": 1000000000,
    "max_batch_size": 1024
  }
}
```

## Monitoring and Observability

### Metrics

The router exposes comprehensive metrics:

- **Performance**: Request rate, latency percentiles, throughput
- **Expert Usage**: Utilization distribution, load balance score
- **Resource Usage**: Memory, CPU, cache hit rates
- **Errors**: Error rate, failure types, circuit breaker state

### Health Checks

Health check endpoint provides:
- Router component status
- Configuration validation
- Memory usage
- Recent performance metrics

```python
health_status = router.health_check()
print(f"Status: {health_status['status']}")
print(f"Uptime: {health_status['uptime_seconds']}s")
```

### Alerting

Configure alerts for:
- High error rate (> 5%)
- Excessive latency (> 500ms p95)
- Poor load balancing (score < 0.5)
- Memory usage spikes
- Circuit breaker activation

### Grafana Dashboard

Use the provided Grafana dashboard (`monitoring/grafana-dashboard.json`) for visualization:

1. Import the dashboard JSON
2. Configure Prometheus data source
3. Set up alert rules based on thresholds

## Performance Optimization

### Batch Size Optimization

Optimal batch sizes vary by hardware:
- **CPU-only**: 1-8 samples
- **GPU inference**: 16-64 samples  
- **GPU training**: 32-128 samples

### Caching Strategy

- **Inference workloads**: Enable caching with large cache size (10K+ entries)
- **Training workloads**: Disable caching (unique inputs)
- **Mixed workloads**: Enable with moderate cache size (5K entries)

### Expert Configuration

Balance complexity vs. efficiency:
- **High accuracy**: More experts (12-16), higher max experts per token (4-6)
- **Low latency**: Fewer experts (6-8), lower max experts per token (2-3)
- **Balanced**: 8 experts, 1-3 experts per token (recommended)

## Security Considerations

### Input Validation

- Automatic tensor shape and dtype validation
- Size limits to prevent memory exhaustion
- NaN/Inf value detection and sanitization

### Rate Limiting

- Built-in rate limiting per client/session
- Automatic blocking of suspicious traffic patterns
- Configurable thresholds per security level

### Security Levels

1. **Minimal**: Basic validation only
2. **Standard**: Comprehensive validation + rate limiting
3. **Strict**: All security features + enhanced monitoring

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Reduce cache size
   - Lower max batch size
   - Check for memory leaks in custom estimators

2. **Poor Performance**  
   - Enable parallel processing for large batches
   - Tune expert count configuration
   - Verify hardware utilization

3. **Load Imbalance**
   - Enable load balancing
   - Adjust noise factor (0.1-0.2)
   - Check expert capacity constraints

4. **Cache Misses**
   - Increase cache size
   - Verify input consistency
   - Check cache TTL settings

### Debugging

Enable debug logging:
```bash
export MOE_LOG_LEVEL=DEBUG
```

Access detailed metrics:
```python
stats = router.get_metrics()
print(f"Cache hit rate: {stats.get('cache_hit_ratio', 'N/A')}")
print(f"Expert usage: {stats['router_stats']}")
```

## Scaling Guidelines

### Horizontal Scaling

- Deploy multiple router instances behind a load balancer
- Use sticky sessions if caching is enabled
- Monitor inter-instance load distribution

### Vertical Scaling

- Scale CPU cores for parallel processing
- Increase memory for larger caches
- Use GPU acceleration for expert computations

### Auto-scaling Triggers

- CPU usage > 70%
- Memory usage > 80%  
- Request queue length > 100
- Average response time > 200ms

## Maintenance

### Updates

1. Test new versions in staging environment
2. Use blue-green deployment for zero-downtime updates
3. Verify metrics and health checks post-deployment

### Monitoring

- Set up automated health checks (every 30s)
- Configure log aggregation and analysis
- Implement performance regression detection

### Backup and Recovery

- Export router configuration and trained thresholds
- Back up performance baselines and thresholds
- Document rollback procedures

## Support

For production support:
- Check logs for detailed error messages
- Use health check endpoint for diagnostics  
- Monitor Grafana dashboard for performance insights
- Contact support with metrics and configuration details

---

This production guide ensures reliable, scalable deployment of the Dynamic MoE Router in enterprise environments.