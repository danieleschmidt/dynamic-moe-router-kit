# Security Testing Guide

## Overview

This guide outlines security testing practices for the dynamic-moe-router-kit project, covering static analysis, dependency scanning, and ML-specific security considerations.

## Automated Security Testing

### Static Code Analysis

```bash
# Run bandit security linter
bandit -r src/ -f json -o security-report.json

# Run with confidence levels
bandit -r src/ -ll -i  # Low confidence, medium severity

# Exclude specific tests
bandit -r src/ -s B101,B601
```

### Dependency Vulnerability Scanner

```bash
# Check for known vulnerabilities
safety check --json --output safety-report.json

# Use policy file
safety check --policy-file .safety-policy.json

# Check specific requirements file
safety check -r requirements.txt
```

### Secret Detection

```bash
# Run detect-secrets
detect-secrets scan --all-files

# Create baseline
detect-secrets scan --all-files --baseline .secrets.baseline

# Audit findings
detect-secrets audit .secrets.baseline
```

## Manual Security Testing

### Input Validation Testing

```python
import pytest
from dynamic_moe_router import DynamicRouter

class TestInputValidation:
    
    def test_malformed_input_handling(self):
        """Test handling of malformed inputs."""
        router = DynamicRouter(input_dim=768, num_experts=8)
        
        # Test various malformed inputs
        malformed_inputs = [
            None,
            [],
            "invalid_string",
            {"wrong": "format"},
            # Add tensor-specific malformed inputs
        ]
        
        for bad_input in malformed_inputs:
            with pytest.raises((ValueError, TypeError)):
                router.forward(bad_input)
    
    def test_adversarial_input_detection(self):
        """Test detection of adversarial inputs."""
        router = DynamicRouter(input_dim=768, num_experts=8)
        
        # Test extreme values
        import numpy as np
        extreme_input = np.full((1, 10, 768), 1e6)
        
        # Should not crash or cause overflow
        try:
            output = router.forward(extreme_input)
            assert output is not None
        except (ValueError, OverflowError) as e:
            # Acceptable to reject extreme inputs
            pass
```

### Resource Exhaustion Testing

```python
def test_memory_exhaustion_protection():
    """Test protection against memory exhaustion attacks."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    router = DynamicRouter(input_dim=768, num_experts=8)
    
    # Try to create large inputs that might exhaust memory
    try:
        large_batch = np.random.randn(1000, 1000, 768)
        router.forward(large_batch)
    except MemoryError:
        # Expected behavior for very large inputs
        pass
    
    final_memory = process.memory_info().rss
    memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB
    
    # Memory growth should be reasonable
    assert memory_growth < 1000  # Less than 1GB growth
```

## AI/ML Security Testing

### Model Integrity Testing

```python
def test_model_weight_integrity():
    """Test that model weights haven't been tampered with."""
    router = DynamicRouter(input_dim=768, num_experts=8)
    
    # Get initial weight checksums
    import hashlib
    
    initial_checksums = {}
    for name, param in router.named_parameters():
        param_bytes = param.detach().numpy().tobytes()
        initial_checksums[name] = hashlib.sha256(param_bytes).hexdigest()
    
    # Perform some operations
    test_input = np.random.randn(1, 10, 768)
    router.forward(test_input)
    
    # Verify weights haven't changed unexpectedly
    for name, param in router.named_parameters():
        param_bytes = param.detach().numpy().tobytes()
        current_checksum = hashlib.sha256(param_bytes).hexdigest()
        assert current_checksum == initial_checksums[name], f"Weight {name} changed unexpectedly"
```

### Privacy Testing

```python
def test_data_leakage_prevention():
    """Test that training data cannot be extracted from model."""
    router = DynamicRouter(input_dim=768, num_experts=8)
    
    # Test that model doesn't memorize specific inputs
    sensitive_input = "This is sensitive information that should not be memorized"
    
    # Convert to tensor representation
    # ... (implementation depends on tokenization)
    
    # Multiple forward passes shouldn't reveal the input
    for _ in range(100):
        output = router.forward(test_tensor)
        # Verify output doesn't contain input patterns
        assert "sensitive information" not in str(output)
```

## Container Security Testing

### Dockerfile Security

```bash
# Scan Dockerfile for security issues
docker run --rm -v "$PWD":/project clair-scanner:latest

# Use Trivy for comprehensive scanning
trivy image dynamic-moe-router:latest

# Check for secrets in layers
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  wagoodman/dive:latest dynamic-moe-router:latest
```

### Runtime Security

```yaml
# Example security context for Kubernetes
apiVersion: v1
kind: Pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
  containers:
  - name: dynamic-moe-router
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
```

## CI/CD Security Pipeline

### GitHub Actions Security

```yaml
# Example security workflow
name: Security Scan

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run Bandit
      run: |
        pip install bandit
        bandit -r src/ -f json -o bandit-report.json
    
    - name: Run Safety
      run: |
        pip install safety
        safety check --json --output safety-report.json
    
    - name: Upload Security Reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json
```

## Compliance Testing

### SLSA Provenance

```bash
# Generate SLSA provenance
slsa-generator generate \
  --artifact-path dist/ \
  --source-uri github.com/yourusername/dynamic-moe-router-kit \
  --source-ref $GITHUB_SHA
```

### SBOM Generation

```bash
# Generate Software Bill of Materials
syft packages dir:. -o spdx-json > sbom.spdx.json

# Scan SBOM for vulnerabilities
grype sbom:sbom.spdx.json
```

## Security Metrics

### Test Coverage

```python
# Security test markers
@pytest.mark.security
def test_security_feature():
    pass

# Run only security tests
pytest -m security --cov=dynamic_moe_router --cov-report=html
```

### Vulnerability Tracking

```bash
# Track vulnerability remediation
echo "High: 0, Medium: 2, Low: 5" > vulnerability-summary.txt

# Trend analysis
python scripts/vulnerability_trend.py
```

## Incident Response Testing

### Tabletop Exercises

1. **Data Breach Simulation**: Test response to potential data exposure
2. **Supply Chain Attack**: Test response to compromised dependency
3. **Model Poisoning**: Test detection and response to adversarial training

### Recovery Testing

```bash
# Test backup and recovery procedures
docker run --rm -v backup-volume:/backup dynamic-moe-router:latest \
  python scripts/verify_backup.py

# Test rollback capabilities
git tag security-rollback-point
# ... make changes ...
git reset --hard security-rollback-point
```

## Security Test Automation

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml security additions
repos:
- repo: https://github.com/PyCQA/bandit
  rev: 1.7.5
  hooks:
  - id: bandit
    args: ['-c', '.bandit']

- repo: https://github.com/Yelp/detect-secrets
  rev: v1.4.0
  hooks:
  - id: detect-secrets
    args: ['--baseline', '.secrets.baseline']
```

### Continuous Monitoring

```python
# Example monitoring script
import requests
import json

def check_vulnerability_feeds():
    """Check for new vulnerabilities affecting our dependencies."""
    # Query vulnerability databases
    nvd_api = "https://services.nvd.nist.gov/rest/json/cves/1.0"
    
    # Check our dependencies against known CVEs
    # ... implementation ...
    
    return vulnerability_report

if __name__ == "__main__":
    report = check_vulnerability_feeds()
    with open("vulnerability_report.json", "w") as f:
        json.dump(report, f)
```

This comprehensive security testing guide ensures thorough security validation across all aspects of the dynamic-moe-router-kit project.