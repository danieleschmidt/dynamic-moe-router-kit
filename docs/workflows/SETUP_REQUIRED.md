# Manual Setup Required

⚠️ **IMPORTANT**: Due to GitHub App permission limitations, the following components must be manually set up by repository maintainers.

## Required Actions

### 1. Create GitHub Workflows

Copy the workflow templates to the `.github/workflows/` directory:

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy workflow templates
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/

# Commit the workflows
git add .github/workflows/
git commit -m "Add GitHub Actions workflows"
git push
```

### 2. Configure Repository Secrets

Add the following secrets in **Repository Settings > Secrets and variables > Actions**:

#### Required Secrets
- `PYPI_API_TOKEN` - PyPI API token for package publishing
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password or access token

#### Optional Secrets (for enhanced features)
- `TEST_PYPI_API_TOKEN` - Test PyPI token for staging releases
- `SLACK_WEBHOOK` - Slack webhook URL for notifications
- `CODECOV_TOKEN` - Codecov token for coverage reporting
- `SNYK_TOKEN` - Snyk token for security scanning
- `FOSSA_API_KEY` - FOSSA API key for license compliance

### 3. Configure Branch Protection

Set up branch protection rules for the `main` branch:

1. Go to **Settings > Branches**
2. Click **Add rule** 
3. Configure the following settings:

```yaml
Branch name pattern: main
Settings:
  ✅ Require a pull request before merging
  ✅ Require approvals (minimum: 1)
  ✅ Dismiss stale PR approvals when new commits are pushed
  ✅ Require review from code owners
  ✅ Require status checks to pass before merging
  ✅ Require branches to be up to date before merging
  ✅ Require conversation resolution before merging
  
Required status checks:
  - CI / lint
  - CI / security
  - CI / test (3.8, torch)
  - CI / test (3.9, torch)
  - CI / test (3.10, torch)
  - CI / test (3.11, torch)
  - CI / build

  ✅ Restrict pushes that create files larger than 100MB
  ✅ Do not allow bypassing the above settings
```

### 4. Configure Dependabot

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "08:00"
    open-pull-requests-limit: 5
    reviewers:
      - "maintainer-username"
    labels:
      - "dependencies"
      - "python"
    commit-message:
      prefix: "deps"
      include: "scope"

  # Docker dependencies  
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "dependencies"
      - "docker"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "dependencies"
      - "github-actions"
```

### 5. Set Up Environment Protection (Optional)

For production deployments, configure environment protection:

1. Go to **Settings > Environments**
2. Create a `production` environment
3. Configure protection rules:
   - Required reviewers: Add maintainer team
   - Wait timer: 0 minutes
   - Deployment branches: Limit to `main` branch only

### 6. Configure Repository Settings

Update repository settings:

1. **General Settings**:
   - Enable issues
   - Enable discussions (recommended)
   - Enable wiki (optional)
   - Enable sponsorships (optional)

2. **Security & Analysis**:
   - Enable Dependency graph
   - Enable Dependabot alerts
   - Enable Dependabot security updates
   - Enable Secret scanning
   - Enable Code scanning (will be populated by workflows)

3. **Pages** (for documentation):
   - Source: Deploy from a branch
   - Branch: `gh-pages`
   - Folder: `/ (root)`

### 7. Team and Collaborator Access

Configure team access if using GitHub organization:

1. Go to **Settings > Manage access**
2. Add teams/collaborators with appropriate permissions:
   - **Maintainers**: Admin access
   - **Contributors**: Write access
   - **Reviewers**: Triage access

### 8. Configure Issue and PR Templates

The repository already includes templates in `.github/`, but verify they're working:

- Issue templates for bug reports and feature requests
- Pull request template with checklist
- Code of conduct and contributing guidelines

## Verification Steps

After setup, verify everything is working:

1. **Test CI**: Create a test PR and ensure all checks pass
2. **Test Security**: Verify security scans are running and uploading results
3. **Test Dependencies**: Ensure Dependabot is creating PRs for updates
4. **Test Release**: Create a test release to verify CD pipeline

## Troubleshooting

### Common Issues

1. **Workflow Permissions**: If workflows fail with permission errors:
   - Check repository settings > Actions > General
   - Ensure "Read and write permissions" is enabled for GITHUB_TOKEN

2. **Secret Access**: If secrets are not accessible:
   - Verify secrets are added at the repository level
   - Check secret names match exactly (case-sensitive)

3. **Branch Protection**: If PRs can be merged without checks:
   - Verify required status checks are correctly named
   - Ensure branch protection is enabled for the correct branch

4. **Dependabot Issues**: If Dependabot PRs aren't created:
   - Check `.github/dependabot.yml` syntax
   - Verify Dependabot is enabled in security settings

### Getting Help

If you encounter issues during setup:

1. Check the [GitHub Actions documentation](https://docs.github.com/en/actions)
2. Review workflow run logs for error details
3. Create an issue in the repository with setup problems
4. Contact the maintainer team for assistance

## Security Considerations

- Never commit secrets or tokens to the repository
- Use environment protection for production deployments
- Regularly review and rotate access tokens
- Monitor security scan results and act on vulnerabilities
- Keep workflow dependencies up to date

## Maintenance

Regular maintenance tasks:

1. **Monthly**: Review and update GitHub Actions versions
2. **Quarterly**: Review branch protection rules and team access
3. **As needed**: Update secrets when they expire or are compromised
4. **After incidents**: Review and improve security configurations

---

**✅ Setup Complete**: Once all steps are completed, the repository will have a fully functional CI/CD pipeline with comprehensive security scanning and automated dependency management.