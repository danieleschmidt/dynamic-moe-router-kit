# Deployment Guide

## Overview

This guide covers various deployment strategies for dynamic-moe-router-kit, from local development to production environments.

## Quick Start

### Local Installation

```bash
# Basic installation
pip install dynamic-moe-router-kit

# With specific backend
pip install dynamic-moe-router-kit[torch]  # PyTorch
pip install dynamic-moe-router-kit[jax]    # JAX/Flax
pip install dynamic-moe-router-kit[tf]     # TensorFlow

# Development installation
git clone https://github.com/yourusername/dynamic-moe-router-kit.git
cd dynamic-moe-router-kit
pip install -e ".[dev]"
```

### Docker Quick Start

```bash
# Pull and run basic image
docker run --rm dynamic-moe-router-kit:latest

# With GPU support
docker run --gpus all --rm dynamic-moe-router-kit:gpu

# Development with mounted code
docker-compose up dev
```

## Docker Deployment

### Available Images

We provide several Docker images for different use cases:

#### Development Image (`Dockerfile`)
- **Purpose**: Development and testing
- **Size**: ~2GB
- **Includes**: All dependencies, development tools, tests
- **Use case**: Local development, CI/CD

```bash
docker build -t dynamic-moe-router:dev .
docker run -it --rm -v $(pwd):/app dynamic-moe-router:dev bash
```

#### Production Image (`Dockerfile.production`)
- **Purpose**: Production deployments
- **Size**: ~500MB (optimized)
- **Includes**: Minimal runtime dependencies only
- **Use case**: Production serving, edge deployment

```bash
docker build -f Dockerfile.production -t dynamic-moe-router:prod .
docker run --rm dynamic-moe-router:prod
```

#### GPU Image (`Dockerfile.gpu`)
- **Purpose**: GPU-accelerated workloads
- **Size**: ~5GB (includes CUDA)
- **Includes**: CUDA runtime, GPU-optimized libraries
- **Use case**: High-performance inference, training

```bash
docker build -f Dockerfile.gpu -t dynamic-moe-router:gpu .
docker run --gpus all --rm dynamic-moe-router:gpu
```

### Docker Compose Services

Use the provided `docker-compose.yml` for common workflows:

```bash
# Development environment
docker-compose up dev

# Run tests
docker-compose run test

# Backend-specific testing
docker-compose run test-torch
docker-compose run test-jax
docker-compose run test-tf

# Build and serve documentation
docker-compose up docs

# Run benchmarks
docker-compose run benchmark
```

## Cloud Deployment

### AWS Deployment

#### ECS Fargate

```yaml
# ecs-task-definition.json
{
  "family": "dynamic-moe-router",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "dynamic-moe-router",
      "image": "your-account.dkr.ecr.region.amazonaws.com/dynamic-moe-router:prod",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/dynamic-moe-router",
          "awslogs-region": "us-west-2"
        }
      }
    }
  ]
}
```

#### EKS Kubernetes

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dynamic-moe-router
spec:
  replicas: 3
  selector:
    matchLabels:
      app: dynamic-moe-router
  template:
    metadata:
      labels:
        app: dynamic-moe-router
    spec:
      containers:
      - name: dynamic-moe-router
        image: dynamic-moe-router:prod
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        env:
        - name: LOG_LEVEL
          value: "INFO"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: dynamic-moe-router-service
spec:
  selector:
    app: dynamic-moe-router
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

### Google Cloud Platform

#### Cloud Run

```bash
# Deploy to Cloud Run
gcloud run deploy dynamic-moe-router \
  --image gcr.io/your-project/dynamic-moe-router:prod \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1000m
```

#### GKE

```yaml
# gke-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dynamic-moe-router
spec:
  replicas: 2
  selector:
    matchLabels:
      app: dynamic-moe-router
  template:
    metadata:
      labels:
        app: dynamic-moe-router
    spec:
      containers:
      - name: dynamic-moe-router
        image: gcr.io/your-project/dynamic-moe-router:gpu
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
```

### Azure Deployment

#### Container Instances

```bash
# Deploy to Azure Container Instances
az container create \
  --resource-group myResourceGroup \
  --name dynamic-moe-router \
  --image your-registry.azurecr.io/dynamic-moe-router:prod \
  --cpu 1 \
  --memory 2 \
  --ports 8000
```

## Environment Configuration

### Environment Variables

Key environment variables for deployment:

```bash
# Python settings
PYTHONPATH=/app/src
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1

# ML Framework settings
TORCH_HOME=/tmp/torch
HF_HOME=/tmp/hf_cache
CUDA_VISIBLE_DEVICES=0

# Application settings
LOG_LEVEL=INFO
LOG_FORMAT=json
ENABLE_METRICS=true

# Performance settings
ENABLE_MIXED_PRECISION=true
PROFILE_MEMORY=false
```

### Configuration Files

#### Production Configuration (`config/production.yaml`)

```yaml
# Model settings
model:
  default_backend: "torch"
  cache_models: true
  max_batch_size: 32

# Router settings  
router:
  complexity_estimator: "gradient_norm"
  load_balance_factor: 0.01
  min_experts: 1
  max_experts: 4

# Performance settings
performance:
  mixed_precision: true
  gradient_checkpointing: false
  compile_model: true

# Monitoring
monitoring:
  enable_metrics: true
  metrics_port: 9090
  health_check_interval: 30

# Logging
logging:
  level: "INFO"
  format: "json"
  output: "stdout"
```

## Performance Optimization

### CPU Optimization

```bash
# Set CPU affinity
taskset -c 0-7 python your_script.py

# Enable threading optimizations
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### GPU Optimization

```bash
# CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# JAX GPU settings
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# TensorFlow GPU settings
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

### Memory Optimization

```python
# Enable gradient checkpointing
model = DynamicMoEModel(
    gradient_checkpointing=True,
    mixed_precision=True
)

# Use memory mapping for large models
torch.hub.set_dir('/tmp/torch_cache')
```

## Monitoring and Observability

### Health Checks

The Docker images include built-in health checks:

```bash
# Check container health
docker ps  # Shows health status

# Manual health check
curl http://localhost:8000/health
```

### Metrics Collection

Enable Prometheus metrics:

```python
from dynamic_moe_router.monitoring import PrometheusMetrics

metrics = PrometheusMetrics(port=9090)
metrics.start()
```

### Logging

Configure structured logging:

```python
import logging
import json

# JSON logging for production
class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module
        })

logging.basicConfig(
    format='%(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
```

## Security Considerations

### Container Security

1. **Non-root user**: All images run as non-root user
2. **Minimal dependencies**: Production images include only required packages
3. **Security scanning**: Regular vulnerability scans

```bash
# Scan for vulnerabilities
docker scan dynamic-moe-router:prod

# Run with security options
docker run --rm --security-opt no-new-privileges \
  --cap-drop ALL dynamic-moe-router:prod
```

### Network Security

```bash
# Restrict network access
docker run --rm --network none dynamic-moe-router:prod

# Use custom network
docker network create --driver bridge isolated
docker run --rm --network isolated dynamic-moe-router:prod
```

### Secrets Management

```bash
# Use Docker secrets
echo "api_key_value" | docker secret create api_key -
docker service create --secret api_key dynamic-moe-router:prod

# Use environment files
docker run --rm --env-file .env.production dynamic-moe-router:prod
```

## Scaling and Load Balancing

### Horizontal Scaling

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: dynamic-moe-router-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: dynamic-moe-router
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Load Balancing

```nginx
# Nginx load balancer configuration
upstream dynamic_moe_router {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    location / {
        proxy_pass http://dynamic_moe_router;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Reduce batch size
   export MAX_BATCH_SIZE=16
   
   # Enable gradient checkpointing
   export ENABLE_GRADIENT_CHECKPOINTING=true
   ```

2. **CUDA Out of Memory**
   ```bash
   # Clear CUDA cache
   python -c "import torch; torch.cuda.empty_cache()"
   
   # Reduce CUDA memory allocation
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
   ```

3. **Model Loading Issues**
   ```bash
   # Check model cache
   ls -la ~/.cache/huggingface/
   
   # Clear cache if corrupted
   rm -rf ~/.cache/huggingface/transformers/
   ```

### Debug Mode

Enable debug logging:

```bash
docker run --rm -e LOG_LEVEL=DEBUG dynamic-moe-router:prod
```

### Performance Profiling

```bash
# Profile with py-spy
py-spy top --pid $(pgrep python)

# Memory profiling
docker stats dynamic-moe-router
```

## Best Practices

1. **Resource Planning**: Size containers based on model requirements
2. **Health Checks**: Always implement proper health checks
3. **Graceful Shutdown**: Handle SIGTERM for clean shutdowns
4. **Monitoring**: Track key metrics (latency, throughput, errors)
5. **Security**: Regular security updates and scans
6. **Backup**: Backup model weights and configurations
7. **Testing**: Test deployments in staging environment first

## Support

For deployment issues:

- 📖 [Documentation](https://dynamic-moe-router.readthedocs.io)
- 💬 [GitHub Discussions](https://github.com/yourusername/dynamic-moe-router-kit/discussions)
- 🐛 [Issue Tracker](https://github.com/yourusername/dynamic-moe-router-kit/issues)
- 📧 [Support Email](mailto:support@example.com)