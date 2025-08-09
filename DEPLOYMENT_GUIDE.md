# 🚀 Dynamic MoE Router Kit - Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Dynamic MoE Router Kit in production environments with enterprise-grade scalability, monitoring, and reliability.

## 📋 Pre-Deployment Checklist

### System Requirements
- ✅ Python 3.8+ with virtual environment support
- ✅ NumPy 1.21.0+ 
- ✅ At least 4GB RAM (8GB+ recommended for large models)
- ✅ Multi-core CPU (GPU optional but recommended for PyTorch backend)
- ✅ Network connectivity for monitoring/logging services

### Security Requirements
- ✅ Secrets management system (HashiCorp Vault, AWS Secrets Manager, etc.)
- ✅ SSL/TLS certificates for API endpoints
- ✅ Network segmentation and firewall rules
- ✅ Access logging and audit trails

### Monitoring Infrastructure
- ✅ Prometheus/Grafana for metrics collection
- ✅ Centralized logging (ELK stack, Splunk, etc.)
- ✅ Alerting system (PagerDuty, Slack, etc.)
- ✅ Health check endpoints

## 🏗️ Infrastructure Setup

### 1. Container Deployment (Recommended)

```bash
# Build production Docker image
docker build -f Dockerfile.production -t dynamic-moe-router:latest .

# Run with monitoring enabled
docker run -d \
  --name moe-router-prod \
  -p 8080:8080 \
  -e ENVIRONMENT=production \
  -e LOG_LEVEL=INFO \
  -e METRICS_ENABLED=true \
  -v /host/config:/app/config:ro \
  -v /host/logs:/app/logs \
  --restart unless-stopped \
  dynamic-moe-router:latest
```

### 2. Kubernetes Deployment

```yaml
# moe-router-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dynamic-moe-router
  namespace: ml-services
spec:
  replicas: 3
  selector:
    matchLabels:
      app: dynamic-moe-router
  template:
    metadata:
      labels:
        app: dynamic-moe-router
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: moe-router
        image: dynamic-moe-router:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: REPLICAS
          value: "3"
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: moe-router-config
```

### 3. Load Balancer Configuration

```nginx
# nginx.conf
upstream moe_router_backend {
    least_conn;
    server moe-router-1:8080 max_fails=3 fail_timeout=30s;
    server moe-router-2:8080 max_fails=3 fail_timeout=30s;
    server moe-router-3:8080 max_fails=3 fail_timeout=30s;
}

server {
    listen 443 ssl http2;
    server_name moe-router.your-domain.com;
    
    ssl_certificate /etc/ssl/certs/moe-router.crt;
    ssl_certificate_key /etc/ssl/private/moe-router.key;
    
    location / {
        proxy_pass http://moe_router_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffer settings for large requests
        proxy_buffering on;
        proxy_buffer_size 16k;
        proxy_buffers 8 16k;
        proxy_busy_buffers_size 32k;
    }
    
    location /health {
        access_log off;
        proxy_pass http://moe_router_backend;
    }
}
```

## 📊 Production Configuration

### 1. Optimized Router Configuration

```python
# production_config.py
from dynamic_moe_router import DynamicRouter
from dynamic_moe_router.scaling import create_optimized_router
from dynamic_moe_router.caching import create_cached_router
from dynamic_moe_router.monitoring import create_monitoring_wrapper

# Create base router with production settings
base_router = DynamicRouter(
    input_dim=1024,
    num_experts=16,
    min_experts=2,
    max_experts=8,
    complexity_estimator="gradient_norm",
    routing_strategy="top_k",
    load_balancing=True,
    noise_factor=0.1
)

# Add production optimizations
router = create_optimized_router(
    base_router,
    enable_autoscaling=True,
    enable_parallel=True,
    max_workers=8,
    min_experts=2,
    max_experts=16,
    target_latency_ms=50.0,
    scale_up_threshold=0.8,
    scale_down_threshold=0.3
)

# Add caching for better performance
router = create_cached_router(
    router,
    cache_size=5000,
    adaptive=True
)

# Add comprehensive monitoring
router = create_monitoring_wrapper(
    router,
    enable_circuit_breaker=True,
    alert_thresholds={
        'avg_latency_ms': 100.0,
        'error_rate': 0.05,
        'load_balance_variance': 0.1,
        'memory_usage_mb': 2000.0
    }
)
```

### 2. Environment Configuration

```bash
# .env.production
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Performance settings
CACHE_SIZE=5000
MAX_WORKERS=8
AUTOSCALING_ENABLED=true
CIRCUIT_BREAKER_ENABLED=true

# Monitoring settings
METRICS_ENABLED=true
METRICS_PORT=9090
HEALTH_CHECK_PORT=8080
TRACING_ENABLED=true
TRACING_ENDPOINT=http://jaeger:14268/api/traces

# Security settings
SSL_ENABLED=true
SSL_CERT_PATH=/etc/ssl/certs/server.crt
SSL_KEY_PATH=/etc/ssl/private/server.key
API_KEY_REQUIRED=true

# Database settings (for metrics storage)
METRICS_DB_HOST=postgresql://metrics-db:5432
METRICS_DB_NAME=moe_metrics
METRICS_DB_USER=moe_user

# Alert settings
ALERT_WEBHOOK=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
PAGERDUTY_KEY=your-pagerduty-integration-key
```

## 📈 Monitoring and Observability

### 1. Prometheus Metrics

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "moe_router_rules.yml"

scrape_configs:
  - job_name: 'dynamic-moe-router'
    static_configs:
      - targets: ['moe-router:8080']
    metrics_path: '/metrics'
    scrape_interval: 10s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 2. Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Dynamic MoE Router Production Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(moe_router_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Latency",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(moe_router_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(moe_router_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(moe_router_errors_total[5m]) / rate(moe_router_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ],
        "thresholds": {
          "steps": [
            {"color": "green", "value": 0},
            {"color": "yellow", "value": 0.01},
            {"color": "red", "value": 0.05}
          ]
        }
      },
      {
        "title": "Expert Utilization",
        "type": "heatmap",
        "targets": [
          {
            "expr": "moe_router_expert_utilization",
            "legendFormat": "Expert {{expert_id}}"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "moe_router_cache_hits / (moe_router_cache_hits + moe_router_cache_misses)",
            "legendFormat": "Cache Hit Rate"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "process_resident_memory_bytes{job=\"dynamic-moe-router\"}",
            "legendFormat": "Memory Usage"
          }
        ]
      }
    ]
  }
}
```

### 3. Alert Rules

```yaml
# moe_router_rules.yml
groups:
  - name: moe_router_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(moe_router_errors_total[5m]) / rate(moe_router_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate in MoE Router"
          description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(moe_router_request_duration_seconds_bucket[5m])) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency in MoE Router"
          description: "95th percentile latency is {{ $value }}s"

      - alert: LowCacheHitRate
        expr: moe_router_cache_hits / (moe_router_cache_hits + moe_router_cache_misses) < 0.7
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit rate"
          description: "Cache hit rate is {{ $value | humanizePercentage }}"

      - alert: ExpertImbalance
        expr: stddev(moe_router_expert_utilization) > 0.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Expert load imbalance detected"
          description: "Expert utilization variance is {{ $value }}"

      - alert: ServiceDown
        expr: up{job="dynamic-moe-router"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "MoE Router service is down"
          description: "Service has been down for more than 1 minute"
```

## 🔒 Security Configuration

### 1. API Authentication

```python
# security.py
import jwt
from functools import wraps
from flask import request, jsonify

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or not validate_api_key(api_key):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

def validate_api_key(api_key):
    # Validate against secure key store
    return api_key in get_valid_api_keys()

@require_api_key
def route_endpoint():
    # Your routing logic here
    pass
```

### 2. Rate Limiting

```python
# rate_limiting.py
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["1000 per hour", "100 per minute"]
)

@app.route('/route')
@limiter.limit("10 per minute")
def route_with_limits():
    # Your routing logic
    pass
```

## 🔧 Performance Optimization

### 1. Connection Pooling

```python
# connection_pool.py
import threading
from concurrent.futures import ThreadPoolExecutor

class RouterPool:
    def __init__(self, router_factory, pool_size=8):
        self.pool = [router_factory() for _ in range(pool_size)]
        self.lock = threading.Lock()
        self.index = 0
    
    def get_router(self):
        with self.lock:
            router = self.pool[self.index]
            self.index = (self.index + 1) % len(self.pool)
            return router

router_pool = RouterPool(create_production_router, pool_size=8)
```

### 2. Async Processing

```python
# async_processing.py
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

class AsyncRouterService:
    def __init__(self, router_pool):
        self.router_pool = router_pool
        self.executor = ThreadPoolExecutor(max_workers=16)
    
    async def route_async(self, hidden_states):
        loop = asyncio.get_event_loop()
        router = self.router_pool.get_router()
        
        # Run in thread pool to avoid blocking
        result = await loop.run_in_executor(
            self.executor,
            router.route,
            hidden_states
        )
        
        return result
```

## 📝 Operational Procedures

### 1. Health Checks

```python
# health_checks.py
from flask import Flask, jsonify
import time

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Basic health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'version': get_version()
    })

@app.route('/ready')
def readiness_check():
    """Readiness check for Kubernetes"""
    try:
        # Verify router functionality
        test_router_functionality()
        
        # Check dependencies
        check_external_dependencies()
        
        return jsonify({'status': 'ready'})
    except Exception as e:
        return jsonify({'status': 'not ready', 'error': str(e)}), 503

def test_router_functionality():
    """Test basic router functionality"""
    import numpy as np
    router = get_health_check_router()
    test_input = np.random.randn(1, 10, 768)
    router.route(test_input)
```

### 2. Graceful Shutdown

```python
# graceful_shutdown.py
import signal
import sys
import logging

def signal_handler(sig, frame):
    logging.info('Received shutdown signal, initiating graceful shutdown...')
    
    # Stop accepting new requests
    app.shutdown()
    
    # Complete existing requests (with timeout)
    router_pool.shutdown(wait=True, timeout=30)
    
    # Close monitoring connections
    monitoring.cleanup()
    
    logging.info('Shutdown completed successfully')
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
```

### 3. Deployment Automation

```bash
#!/bin/bash
# deploy.sh - Production deployment script

set -e

VERSION=${1:-latest}
ENVIRONMENT=${2:-production}

echo "Deploying Dynamic MoE Router $VERSION to $ENVIRONMENT"

# Pre-deployment checks
echo "Running pre-deployment checks..."
kubectl get nodes
kubectl get pods -n ml-services

# Update deployment
echo "Updating deployment..."
kubectl set image deployment/dynamic-moe-router \
  moe-router=dynamic-moe-router:$VERSION \
  -n ml-services

# Wait for rollout to complete
echo "Waiting for rollout to complete..."
kubectl rollout status deployment/dynamic-moe-router -n ml-services --timeout=300s

# Post-deployment verification
echo "Running post-deployment checks..."
kubectl get pods -n ml-services -l app=dynamic-moe-router
curl -f http://moe-router.ml-services.svc.cluster.local:8080/health

# Run smoke tests
echo "Running smoke tests..."
python3 tests/smoke_tests.py --environment $ENVIRONMENT

echo "Deployment completed successfully!"
```

## 🆘 Troubleshooting

### Common Issues and Solutions

1. **High Memory Usage**
   ```bash
   # Check memory usage
   kubectl top pods -n ml-services
   
   # Adjust memory limits
   kubectl patch deployment dynamic-moe-router -p '{"spec":{"template":{"spec":{"containers":[{"name":"moe-router","resources":{"limits":{"memory":"8Gi"}}}]}}}}'
   ```

2. **Load Balancing Issues**
   ```python
   # Check expert utilization
   router_stats = router.get_expert_usage_stats()
   print(f"Load balance score: {router_stats['load_balance_score']}")
   
   # Adjust load balancing parameters
   router.load_balancing_threshold = 0.05
   ```

3. **Performance Degradation**
   ```bash
   # Check metrics
   curl http://moe-router:8080/metrics | grep latency
   
   # Scale up instances
   kubectl scale deployment dynamic-moe-router --replicas=5
   ```

## 📊 Production Benchmarks

Expected performance metrics in production:

| Metric | Target | Alert Threshold |
|--------|---------|----------------|
| 95th percentile latency | < 50ms | > 100ms |
| Error rate | < 0.1% | > 5% |
| Cache hit rate | > 80% | < 70% |
| Memory usage | < 4GB per instance | > 6GB |
| Expert load balance score | > 0.8 | < 0.6 |
| Throughput | > 1000 req/sec | < 500 req/sec |

## 📞 Support and Maintenance

### Monitoring Checklist
- [ ] All metrics are being collected
- [ ] Alerts are configured and tested
- [ ] Log aggregation is working
- [ ] Dashboards are accessible
- [ ] Health checks are passing

### Regular Maintenance Tasks
- [ ] Review and rotate API keys
- [ ] Update SSL certificates
- [ ] Clear old logs and metrics
- [ ] Test backup and recovery procedures
- [ ] Review and update monitoring thresholds
- [ ] Performance testing and optimization

## 🔄 Rollback Procedures

```bash
#!/bin/bash
# rollback.sh - Emergency rollback script

PREVIOUS_VERSION=${1:-$(kubectl rollout history deployment/dynamic-moe-router -n ml-services | tail -n 2 | head -n 1 | awk '{print $1}')}

echo "Rolling back to version $PREVIOUS_VERSION"

kubectl rollout undo deployment/dynamic-moe-router \
  --to-revision=$PREVIOUS_VERSION \
  -n ml-services

kubectl rollout status deployment/dynamic-moe-router -n ml-services

echo "Rollback completed. Running health checks..."
curl -f http://moe-router.ml-services.svc.cluster.local:8080/health
```

---

This deployment guide provides enterprise-ready configurations for running Dynamic MoE Router Kit in production with comprehensive monitoring, security, and operational procedures.