# CI/CD Pipeline Documentation

## Overview

This directory contains the GitHub Actions CI/CD pipeline for the Federated Fraud Detection system. The pipeline ensures code quality, security, and correctness through automated testing and validation.

## Pipeline Structure

### Workflow: `ci-cd.yml`

The main CI/CD workflow runs on:
- Pull requests to `main` and `develop` branches
- Pushes to `main` and `develop` branches
- Manual workflow dispatch

## Jobs

### 1. Code Quality Checks (`code-quality`)

Enforces code quality standards using:

- **Black**: Code formatting verification
  - Line length: 120 characters
  - Ignores: E203, W503
  
- **Flake8**: Linting and style checking
  - Catches syntax errors, undefined names, unused imports
  
- **MyPy**: Static type checking
  - Ensures type safety across the codebase
  
- **Bandit**: Security vulnerability scanning
  - Identifies common security issues in Python code
  - Generates JSON report for review

**Artifacts**: `bandit-security-report`

### 2. Dependency Security (`dependency-security`)

Scans dependencies for known vulnerabilities:

- **Safety**: Checks `requirements.txt` against vulnerability database
- Generates JSON report of findings
- Fails on critical vulnerabilities

**Artifacts**: `safety-dependency-report`

### 3. Unit Tests (`unit-tests`)

Runs comprehensive unit test suite:

- Executes all non-slow tests
- Generates coverage reports (XML, HTML, terminal)
- Uploads coverage to Codecov
- Minimum coverage threshold enforced

**Artifacts**: 
- `coverage-reports` (XML and HTML)
- Coverage uploaded to Codecov

### 4. Property-Based Tests (`property-tests`)

Runs property-based tests using Hypothesis:

- Tests universal correctness properties
- Uses `ci` Hypothesis profile (200 examples)
- Timeout: 30 minutes
- Validates all 26 correctness properties from design

**Artifacts**: `.hypothesis/` (test database)

### 5. Integration Tests (`integration-tests`)

Validates component integration:

- Tests federated learning workflow
- Validates MLOps integration
- Tests container orchestration
- Verifies end-to-end scenarios

### 6. Container Security (`container-security`)

Scans Docker images for vulnerabilities:

- Builds Bank Client and Aggregation Server images
- **Trivy**: Vulnerability scanner for containers
- Generates SARIF reports
- Uploads findings to GitHub Security tab

**Images Scanned**:
- `federated-fraud/bank-client:test`
- `federated-fraud/aggregation-server:test`

**Artifacts**: SARIF reports uploaded to GitHub Security

### 7. Performance Benchmarks (`performance-tests`)

Measures system performance with synthetic datasets:

- Data preprocessing throughput
- Model training speed
- Inference latency
- Weight aggregation performance

**Script**: `tests/performance_benchmarks.py`

**Artifacts**: `benchmark_results.json`

**Metrics Tracked**:
- Samples per second
- Training time per epoch
- Inference latency (ms/sample)
- Memory usage

### 8. Privacy Verification (`privacy-verification`)

Verifies differential privacy guarantees:

- Privacy budget tracking accuracy
- Noise addition to gradients
- Privacy accounting across FL rounds
- Support for epsilon values [0.5, 1, 2, 4, 8]

**Script**: `tests/verify_dp_guarantees.py`

**Artifacts**: `dp_verification_results.json`

**Validates**: Requirements 4.1, 4.3, 4.4, 4.6, 4.7

### 9. End-to-End Workflow (`e2e-workflow`)

Validates complete federated learning workflow:

- Data preprocessing and partitioning
- Model initialization
- Privacy engine integration
- Federated training simulation
- Model evaluation

**Script**: `tests/e2e_workflow_test.py`

**Artifacts**: `e2e_test_results.json`

**Timeout**: 45 minutes

### 10. Build Summary (`build-summary`)

Aggregates results from all jobs:

- Displays status of each job
- Runs even if previous jobs fail
- Provides quick overview of pipeline status

### 11. Deploy to Staging (`deploy-staging`)

Automated deployment to staging environment:

- **Trigger**: Push to `develop` branch
- **Dependencies**: Requires code-quality, unit-tests, property-tests, integration-tests to pass
- Builds and tags Docker images with `staging` tag
- Placeholder for actual deployment logic

**Images Built**:
- `federated-fraud/bank-client:staging`
- `federated-fraud/aggregation-server:staging`
- `federated-fraud/mlflow:staging`

## Environment Variables

- `PYTHON_VERSION`: Python version for all jobs (default: 3.10)
- `HYPOTHESIS_PROFILE`: Hypothesis testing profile (default: ci)

## Artifacts

All jobs upload artifacts that can be downloaded from the Actions tab:

| Artifact | Job | Description |
|----------|-----|-------------|
| `bandit-security-report` | code-quality | Security scan results |
| `safety-dependency-report` | dependency-security | Dependency vulnerabilities |
| `coverage-reports` | unit-tests | Code coverage (XML/HTML) |
| `.hypothesis/` | property-tests | Hypothesis test database |
| `benchmark_results.json` | performance-tests | Performance metrics |
| `dp_verification_results.json` | privacy-verification | DP guarantee verification |
| `e2e_test_results.json` | e2e-workflow | E2E workflow results |

## Running Locally

### Code Quality Checks

```bash
# Black formatting
black --check --diff src/ tests/

# Flake8 linting
flake8 src/ tests/ --max-line-length=120 --extend-ignore=E203,W503

# MyPy type checking
mypy src/ --ignore-missing-imports --no-strict-optional

# Bandit security scan
bandit -r src/ -ll
```

### Dependency Security

```bash
# Safety check
pip install safety
safety check
```

### Unit Tests

```bash
# Run with coverage
pytest tests/ -v --cov=src --cov-report=html -m "not slow"
```

### Property-Based Tests

```bash
# Run property tests
pytest tests/ -v -m property --hypothesis-profile=ci
```

### Performance Benchmarks

```bash
python tests/performance_benchmarks.py
```

### Privacy Verification

```bash
python tests/verify_dp_guarantees.py
```

### End-to-End Workflow

```bash
python tests/e2e_workflow_test.py
```

## Quality Gates

The pipeline enforces the following quality gates:

1. **Code Quality**: All linting and type checking must pass
2. **Security**: No critical vulnerabilities in code or dependencies
3. **Test Coverage**: Minimum coverage threshold must be met
4. **Property Tests**: All 26 correctness properties must hold
5. **Integration**: All integration tests must pass
6. **Container Security**: No critical vulnerabilities in Docker images
7. **Performance**: Benchmarks must complete within acceptable time
8. **Privacy**: DP guarantees must be verified
9. **E2E**: Complete workflow must execute successfully

## Troubleshooting

### Job Failures

1. **Code Quality**: Check formatting with `black --diff` and fix issues
2. **Security Scans**: Review Bandit/Safety reports and update dependencies
3. **Unit Tests**: Check test logs for failures and fix code
4. **Property Tests**: Review Hypothesis counterexamples and fix logic
5. **Container Security**: Update base images or fix vulnerabilities
6. **Performance**: Check for performance regressions
7. **Privacy**: Verify DP implementation correctness
8. **E2E**: Check workflow logs for integration issues

### Skipping Jobs

To skip CI for a commit (use sparingly):

```bash
git commit -m "docs: update README [skip ci]"
```

## Maintenance

### Updating Dependencies

1. Update `requirements.txt`
2. Run `safety check` locally
3. Update workflow if new tools are added

### Adding New Tests

1. Add test files to `tests/` directory
2. Mark with appropriate pytest markers
3. Update this README with new test information

### Modifying Workflow

1. Edit `.github/workflows/ci-cd.yml`
2. Test locally with `act` (GitHub Actions local runner)
3. Create PR and verify workflow runs correctly

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Pytest Documentation](https://docs.pytest.org/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Trivy Documentation](https://aquasecurity.github.io/trivy/)
- [Codecov Documentation](https://docs.codecov.com/)
