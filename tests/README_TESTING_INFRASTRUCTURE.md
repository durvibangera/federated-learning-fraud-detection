# Testing Infrastructure Documentation

## Overview

This document describes the comprehensive testing infrastructure for the Federated Fraud Detection system, implementing Task 19 from the implementation plan.

## Testing Approach

The system uses a **dual testing approach** as specified in the design document:

1. **Property-Based Testing**: Uses Hypothesis library to verify universal properties across all valid inputs (minimum 100 iterations per test)
2. **Unit Testing**: Tests specific examples, edge cases, and integration scenarios

## Test Organization

### Core Testing Files

#### `conftest.py`
- Pytest configuration and shared fixtures
- Hypothesis profile configuration (default: 100 examples, CI: 200 examples)
- Common fixtures for device, random seeds, sample configurations
- Test markers: `@pytest.mark.property`, `@pytest.mark.unit`, `@pytest.mark.integration`

#### `hypothesis_strategies.py`
- Custom Hypothesis strategies for generating test data
- **IEEE-CIS Dataset Strategies**: Generate realistic transaction records with proper structure
- **Model Weight Strategies**: Generate PyTorch model weights for federated learning
- **Configuration Strategies**: Generate valid privacy, model, and FL configurations
- **Federated Learning Strategies**: Generate FL round metrics and client updates

### Test Suites

#### 1. Property-Based Tests (Task 19.1)

**`test_property_17_monitoring.py`** (Example)
- Property 17: Real-time Monitoring Integration
- Tests Prometheus metrics exposure, alerting, and Grafana integration
- Validates: Requirements 7.3, 7.4, 7.6
- Uses Hypothesis strategies for randomized testing

**Future Property Tests** (Optional, marked with * in tasks.md):
- Property 1-4: Data preprocessing properties
- Property 5-6: Model architecture properties
- Property 7-8: Federated learning protocol properties
- Property 9-11: Privacy properties
- Property 12-13: Explainability properties
- Property 14-15: Container orchestration properties
- Property 16-19: Monitoring and evaluation properties
- Property 20-21: Serialization properties
- Property 22-26: System resilience properties

#### 2. Unit Tests (Task 19.2)

**`test_preprocessing_edge_cases.py`**
- IEEE-CIS preprocessing edge cases
- Empty datasets, all missing values, invalid ProductCD
- Temporal split edge cases, encoding unseen categories
- Validates: Requirements 1.1-1.7

**`test_container_integration.py`**
- Container orchestration integration tests
- Docker Compose structure validation
- Network isolation, volume mounts, health checks
- Dockerfile structure and best practices
- Validates: Requirements 6.1-6.7

**`test_mlops_integration.py`**
- MLOps pipeline integration tests
- MLflow experiment tracking
- Prometheus metrics export
- Grafana dashboard configuration
- End-to-end monitoring pipeline
- Validates: Requirements 7.1-7.7

**`test_error_recovery.py`**
- Error conditions and recovery scenarios
- Client disconnection, network failures
- Model aggregation failures, memory constraints
- Corrupted data handling, privacy budget exhaustion
- Checkpointing and graceful degradation
- Validates: Requirements 11.1-11.7

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Suite
```bash
# Property-based tests
pytest tests/test_property_17_monitoring.py -v

# Unit tests
pytest tests/test_preprocessing_edge_cases.py -v
pytest tests/test_container_integration.py -v
pytest tests/test_mlops_integration.py -v
pytest tests/test_error_recovery.py -v
```

### Run Tests by Marker
```bash
# Property-based tests only
pytest tests/ -m property -v

# Unit tests only
pytest tests/ -m unit -v

# Integration tests only
pytest tests/ -m integration -v

# Slow tests (skip for quick validation)
pytest tests/ -m "not slow" -v
```

### Run with Hypothesis Statistics
```bash
pytest tests/test_property_17_monitoring.py -v --hypothesis-show-statistics
```

### Run with Different Hypothesis Profiles
```bash
# Development (50 examples, verbose)
pytest tests/ -v --hypothesis-profile=dev

# CI (200 examples)
pytest tests/ -v --hypothesis-profile=ci

# Debug (10 examples, very verbose)
pytest tests/ -v --hypothesis-profile=debug
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term
```

## Hypothesis Strategies Reference

### IEEE-CIS Dataset Strategies

```python
from tests.hypothesis_strategies import (
    ieee_cis_transaction_record,  # Single transaction
    ieee_cis_dataframe,            # Full DataFrame
    ieee_cis_product_cd,           # ProductCD values
    ieee_cis_card_features,        # Card features
)

@given(transaction=ieee_cis_transaction_record())
def test_transaction_processing(transaction):
    # Test with random transaction
    pass

@given(df=ieee_cis_dataframe(min_rows=10, max_rows=100))
def test_dataframe_processing(df):
    # Test with random DataFrame
    pass
```

### Model Weight Strategies

```python
from tests.hypothesis_strategies import (
    model_weights_dict,           # Single model weights
    federated_model_weights,      # Multiple client weights
    model_weight_tensor,          # Single tensor
)

@given(weights=federated_model_weights(num_clients=3))
def test_weight_aggregation(weights):
    # Test with random client weights
    pass
```

### Configuration Strategies

```python
from tests.hypothesis_strategies import (
    privacy_config,               # Privacy configuration
    model_config,                 # Model configuration
    fl_config,                    # FL configuration
    data_split_ratios,            # Train/val/test ratios
)

@given(config=privacy_config())
def test_privacy_engine(config):
    # Test with random privacy config
    pass
```

### Federated Learning Strategies

```python
from tests.hypothesis_strategies import (
    fl_round_metrics,             # FL round metrics
    client_update,                # Single client update
    fl_round_updates,             # Multiple client updates
)

@given(updates=fl_round_updates(num_clients=3))
def test_aggregation(updates):
    # Test with random client updates
    pass
```

## Test Fixtures Reference

### Common Fixtures (from conftest.py)

```python
def test_with_device(device):
    """Use PyTorch device fixture."""
    model = model.to(device)

def test_with_random_seed(set_random_seeds):
    """Use random seed fixture for reproducibility."""
    # All random operations will be deterministic

def test_with_sample_config(sample_model_config, sample_privacy_config, sample_fl_config):
    """Use sample configuration fixtures."""
    # Pre-configured test configurations
```

## Writing New Tests

### Property-Based Test Template

```python
from hypothesis import given, settings
from tests.hypothesis_strategies import ieee_cis_dataframe

@given(df=ieee_cis_dataframe(min_rows=10, max_rows=100))
@settings(max_examples=100, deadline=None)
def test_my_property(df):
    """
    Property: [Description of property being tested]
    
    For any [input description], the system should [expected behavior].
    
    Validates: Requirements X.Y, X.Z
    """
    # Arrange
    preprocessor = DataPreprocessor()
    
    # Act
    result = preprocessor.process(df)
    
    # Assert
    assert len(result) > 0
    assert result['column'].dtype == expected_type
```

### Unit Test Template

```python
import pytest

class TestMyFeature:
    """Unit tests for [feature name]."""
    
    def test_normal_case(self):
        """Test normal operation."""
        # Arrange
        input_data = create_test_data()
        
        # Act
        result = process(input_data)
        
        # Assert
        assert result is not None
    
    def test_edge_case(self):
        """Test edge case: [description]."""
        # Test edge case
        pass
    
    def test_error_condition(self):
        """Test error handling: [description]."""
        with pytest.raises(ValueError):
            process(invalid_data)
```

## Test Coverage Goals

### Current Coverage (Task 19.1 & 19.2)

- ✅ Hypothesis infrastructure and strategies
- ✅ IEEE-CIS preprocessing edge cases
- ✅ Container orchestration integration
- ✅ MLOps pipeline integration
- ✅ Error recovery scenarios
- ✅ Property 17: Real-time Monitoring

### Future Coverage (Optional Property Tests)

- ⏳ Properties 1-4: Data processing
- ⏳ Properties 5-6: Model architecture
- ⏳ Properties 7-8: Federated learning
- ⏳ Properties 9-11: Privacy
- ⏳ Properties 12-13: Explainability
- ⏳ Properties 14-15: Container lifecycle
- ⏳ Properties 16, 18-19: Evaluation
- ⏳ Properties 20-21: Serialization
- ⏳ Properties 22-26: System resilience

## Continuous Integration

### GitHub Actions Workflow (Future: Task 20.1)

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ -v --hypothesis-profile=ci
      - name: Generate coverage
        run: pytest tests/ --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Best Practices

### 1. Property-Based Testing
- Use Hypothesis for testing universal properties
- Minimum 100 iterations per property test (configurable via profiles)
- Generate realistic test data using custom strategies
- Test invariants that should hold for all valid inputs

### 2. Unit Testing
- Test specific examples and edge cases
- Test error conditions and recovery
- Use descriptive test names
- Follow Arrange-Act-Assert pattern

### 3. Integration Testing
- Test component interactions
- Test end-to-end workflows
- Use mocking for external dependencies
- Test both success and failure paths

### 4. Test Organization
- Group related tests in classes
- Use fixtures for common setup
- Mark tests appropriately (@pytest.mark.*)
- Keep tests independent and isolated

### 5. Test Maintenance
- Update tests when requirements change
- Keep test data generators in sync with schemas
- Document complex test scenarios
- Review test coverage regularly

## Troubleshooting

### Hypothesis Generates Invalid Data
- Review and refine strategy constraints
- Add `assume()` statements to filter invalid cases
- Use `@example()` decorator to add specific test cases

### Tests Are Too Slow
- Use `--hypothesis-profile=dev` for faster iteration
- Mark slow tests with `@pytest.mark.slow`
- Run subset of tests during development

### Flaky Tests
- Ensure proper random seed management
- Check for race conditions in async code
- Use `set_random_seeds` fixture for reproducibility

### Import Errors
- Ensure `src` is in Python path (handled by conftest.py)
- Check that all dependencies are installed
- Verify module structure matches imports

## References

- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Pytest Documentation](https://docs.pytest.org/)
- Design Document: `.kiro/specs/federated-fraud-detection/design.md`
- Requirements: `.kiro/specs/federated-fraud-detection/requirements.md`
- Tasks: `.kiro/specs/federated-fraud-detection/tasks.md`
