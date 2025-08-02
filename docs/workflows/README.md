# GitHub Workflows Documentation

## Overview

This directory contains documentation and templates for GitHub Actions workflows. Due to GitHub App permission limitations, these are provided as templates that must be manually created by repository maintainers.

## Required Workflows

The following workflows are essential for maintaining code quality, security, and automation:

### 1. Continuous Integration (`ci.yml`)
- **Purpose**: Validate pull requests and commits
- **Triggers**: Push to main, pull requests
- **Responsibilities**:
  - Run tests across multiple Python versions
  - Execute linting and type checking
  - Perform security scans
  - Build and validate packages

### 2. Continuous Deployment (`cd.yml`)
- **Purpose**: Automated releases and deployments
- **Triggers**: Push to main (after CI passes), release tags
- **Responsibilities**:
  - Build and publish to PyPI
  - Create GitHub releases
  - Deploy documentation
  - Update Docker images

### 3. Security Scanning (`security-scan.yml`)
- **Purpose**: Comprehensive security analysis
- **Triggers**: Schedule (daily), pull requests
- **Responsibilities**:
  - Dependency vulnerability scanning
  - Code security analysis
  - Container image scanning
  - License compliance checking

### 4. Dependency Updates (`dependency-update.yml`)
- **Purpose**: Automated dependency management
- **Triggers**: Schedule (weekly)
- **Responsibilities**:
  - Update dependencies via Dependabot/Renovate
  - Run compatibility tests
  - Create pull requests for updates

## Workflow Templates

All workflow templates are provided in the `examples/` directory:

- [`ci.yml`](examples/ci.yml) - Continuous Integration
- [`cd.yml`](examples/cd.yml) - Continuous Deployment
- [`security-scan.yml`](examples/security-scan.yml) - Security Scanning
- [`dependency-update.yml`](examples/dependency-update.yml) - Dependency Updates
- [`docker-build.yml`](examples/docker-build.yml) - Docker Image Building
- [`docs-deploy.yml`](examples/docs-deploy.yml) - Documentation Deployment

## Setup Instructions

### 1. Manual Workflow Creation

Repository maintainers must manually create these workflows:

```bash
# Create .github/workflows directory
mkdir -p .github/workflows

# Copy templates to workflow directory
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
```

### 2. Required Secrets

Configure the following secrets in your repository settings:

#### PyPI Publishing
- `PYPI_API_TOKEN` - PyPI API token for package publishing
- `TEST_PYPI_API_TOKEN` - Test PyPI token for staging

#### Container Registry
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password/token
- `GHCR_TOKEN` - GitHub Container Registry token

#### Security & Monitoring
- `CODECOV_TOKEN` - Codecov token for coverage reporting
- `SONAR_TOKEN` - SonarCloud token for code analysis

#### Documentation
- `DOCS_DEPLOY_KEY` - SSH key for documentation deployment

### 3. Repository Settings

Configure the following repository settings:

#### Branch Protection Rules
```yaml
Branch: main
Settings:
  - Require pull request reviews before merging
  - Require status checks to pass before merging
    - Required checks: CI / test (3.8, 3.9, 3.10, 3.11)
    - Required checks: CI / lint
    - Required checks: CI / security-scan
  - Require branches to be up to date before merging
  - Restrict pushes that create files larger than 100MB
```

#### Environment Protection Rules
```yaml
Environment: production
Settings:
  - Required reviewers: [maintainer-team]
  - Wait timer: 0 minutes
  - Deployment branches: main only
```

### 4. Dependabot Configuration

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    reviewers:
      - "maintainer-team"
    labels:
      - "dependencies"
      - "python"
  
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "dependencies"
      - "docker"

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "dependencies"
      - "github-actions"
```

## Workflow Features

### Multi-Backend Testing

All workflows support testing across multiple ML backends:

```yaml
strategy:
  matrix:
    python-version: ['3.8', '3.9', '3.10', '3.11']
    backend: ['torch', 'jax', 'tf']
    os: [ubuntu-latest, windows-latest, macos-latest]
    exclude:
      # Exclude specific combinations if needed
      - python-version: '3.8'
        backend: 'jax'
        os: 'windows-latest'
```

### Caching Strategy

Optimized caching for faster builds:

```yaml
- name: Cache Python dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
    restore-keys: |
      ${{ runner.os }}-pip-

- name: Cache pre-commit
  uses: actions/cache@v3
  with:
    path: ~/.cache/pre-commit
    key: ${{ runner.os }}-pre-commit-${{ hashFiles('.pre-commit-config.yaml') }}
```

### Security Features

Built-in security scanning:

```yaml
- name: Run Bandit security scan
  run: bandit -r src/ -f json -o bandit-report.json

- name: Run Safety check
  run: safety check --json --output safety-report.json

- name: Upload security reports
  uses: actions/upload-artifact@v3
  with:
    name: security-reports
    path: |
      bandit-report.json
      safety-report.json
```

## Monitoring & Observability

### Workflow Metrics

Track workflow performance:

- **Success Rate**: Percentage of successful workflow runs
- **Duration**: Average workflow execution time
- **Failure Analysis**: Common failure patterns and causes
- **Resource Usage**: Compute and storage consumption

### Notifications

Configure notifications for workflow events:

```yaml
- name: Notify on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    channel: '#ci-alerts'
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## Troubleshooting

### Common Issues

1. **Permission Errors**
   - Verify GitHub App permissions
   - Check repository settings
   - Confirm secret configurations

2. **Test Failures**
   - Review test logs in workflow runs
   - Check for environment-specific issues
   - Validate dependencies and versions

3. **Deployment Failures**
   - Verify deployment credentials
   - Check target environment availability
   - Review deployment logs

### Debug Mode

Enable debug logging:

```yaml
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

## Best Practices

### 1. Workflow Design
- Keep workflows focused and single-purpose
- Use reusable workflows for common patterns
- Implement proper error handling and notifications
- Cache dependencies for faster execution

### 2. Security
- Never commit secrets to repository
- Use environment protection for sensitive operations
- Implement proper access controls
- Regular security audits of workflow permissions

### 3. Performance
- Optimize workflow execution time
- Use matrix builds for parallel testing
- Implement intelligent caching strategies
- Monitor resource usage and costs

### 4. Maintenance
- Regular updates to action versions
- Monitor deprecated actions and features
- Review and update workflow logic
- Document workflow changes and rationale

## Workflow Templates Guide

Each template includes:

- **Comprehensive comments** explaining each step
- **Conditional logic** for different scenarios  
- **Error handling** and notification strategies
- **Security best practices** and scanning
- **Performance optimizations** and caching
- **Multi-platform support** where applicable

## Migration Guide

When updating existing workflows:

1. **Backup current workflows**
2. **Review template changes**
3. **Update secrets and settings**
4. **Test in staging environment**
5. **Deploy incrementally**
6. **Monitor for issues**

## Support

For workflow-related issues:

- 📖 [GitHub Actions Documentation](https://docs.github.com/en/actions)
- 💬 [Repository Discussions](../../../discussions)
- 🐛 [Issue Tracker](../../../issues)
- 📧 [Maintainer Contact](mailto:maintainers@example.com)

---

**⚠️ IMPORTANT**: Repository maintainers must manually create workflow files from these templates due to GitHub App permission limitations. This documentation provides all necessary templates and configuration guidance.