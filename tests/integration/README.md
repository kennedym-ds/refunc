# Integration Tests

This directory contains comprehensive integration tests for the refunc package that verify cross-module functionality and end-to-end workflows.

## Test Structure

```
tests/integration/
├── __init__.py                      # Integration test package
├── test_cross_module.py            # Cross-module interaction tests
├── test_complete_workflows.py      # End-to-end workflow tests
├── test_performance_regression.py  # Performance benchmark tests
├── test_real_world_scenarios.py    # Real-world usage scenarios
└── README.md                       # This file
```

## Test Categories

### 🔗 Cross-Module Integration (`test_cross_module.py`)
Tests interactions between different refunc modules:
- Config + Logging integration
- Decorators + Exception handling
- Utils + Logging integration
- Data Science + Utils integration
- Math/Stats + Logging integration
- ML workflow integration
- Full module integration workflows

**23 test scenarios** covering all major module interaction patterns.

### 🔄 Complete Workflows (`test_complete_workflows.py`)
Tests end-to-end workflows from start to finish:
- Basic ML pipeline (data → model → evaluation)
- Data processing pipeline with monitoring
- Config-driven adaptive workflows
- Performance-monitored workflows
- Error recovery workflows
- Complete ML experiment workflows

**6 test scenarios** covering realistic end-to-end usage patterns.

### 📊 Performance Regression (`test_performance_regression.py`)
Establishes baselines and detects performance regressions:
- Decorator performance overhead testing
- FileHandler performance benchmarks
- Logging performance impact analysis
- Memory usage pattern validation
- Cache performance scaling
- Regression benchmark tests
- Scalability testing
- Resource utilization monitoring

**12 test scenarios** with performance baselines and regression detection.

### 🌍 Real-World Scenarios (`test_real_world_scenarios.py`)
Simulates common production usage patterns:
- Data science EDA workflows
- Model training pipelines with monitoring
- Production batch processing with error handling
- Configuration-driven adaptive pipelines
- MLOps model deployment simulation
- Multi-source data integration
- Network failure recovery scenarios

**7 test scenarios** covering production-ready workflows.

## Test Markers

Integration tests use pytest markers for categorization:

- `@pytest.mark.integration` - All integration tests
- `@pytest.mark.slow` - Long-running tests (>10 seconds)
- `@pytest.mark.benchmark` - Performance benchmark tests

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r requirements/test.txt
```

### Basic Usage

```bash
# Run all integration tests
pytest tests/integration/

# Run only fast integration tests
pytest tests/integration/ -m "integration and not slow"

# Run only cross-module tests
pytest tests/integration/test_cross_module.py

# Run with coverage
pytest tests/integration/ --cov=refunc --cov-report=html

# Run performance benchmarks
pytest tests/integration/ -m benchmark --benchmark-only
```

### Advanced Usage

```bash
# Run tests in parallel (if pytest-xdist installed)
pytest tests/integration/ -n auto

# Run with detailed output
pytest tests/integration/ -v -s

# Run specific test class
pytest tests/integration/test_cross_module.py::TestConfigLoggingIntegration

# Run with performance monitoring
pytest tests/integration/ --durations=10
```

## Test Configuration

Integration tests are configured via:

1. **`conftest.py`** - Shared fixtures and configuration
2. **`pyproject.toml`** - Pytest settings and markers
3. **Environment variables** - Runtime configuration

### Key Fixtures

- `temp_dir` - Temporary directory for test files
- `sample_dataframe` - Sample pandas DataFrame
- `integration_config` - Integration test configuration
- `large_sample_dataframe` - Large dataset for performance tests
- `performance_baseline` - Performance baseline expectations
- `mock_ml_model` - Mock ML model for testing

## Test Design Principles

### 1. Graceful Degradation
Tests use `pytest.skip()` when dependencies are unavailable:
```python
try:
    from refunc.config import ConfigManager
    # Test implementation
except ImportError as e:
    pytest.skip(f"Module import failed: {e}")
```

### 2. Realistic Scenarios
Tests simulate real-world usage patterns:
- Actual file I/O operations
- Realistic data sizes
- Production-like error conditions
- Performance monitoring

### 3. Comprehensive Coverage
Tests cover:
- Happy path scenarios
- Error conditions and recovery
- Performance edge cases
- Cross-module interactions
- Configuration variations

### 4. Performance Awareness
Tests establish and validate performance baselines:
- Execution time limits
- Memory usage patterns
- Throughput requirements
- Scalability characteristics

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements/base.txt
   pip install -r requirements/test.txt
   ```

2. **Slow Tests**: Use markers to skip slow tests during development
   ```bash
   pytest tests/integration/ -m "not slow"
   ```

3. **Memory Issues**: Some tests create large datasets; ensure adequate RAM

4. **Network Timeouts**: Some tests simulate network failures; increase timeouts if needed

### Performance Test Failures

Performance tests may fail on slower systems. Adjust baselines in `conftest.py`:
```python
@pytest.fixture
def performance_baseline():
    return {
        'file_operations': {
            'save_1000_rows_max_seconds': 2.0,  # Increased from 1.0
            'load_1000_rows_max_seconds': 1.0   # Increased from 0.5
        }
    }
```

## Contributing

When adding new integration tests:

1. **Follow naming conventions**: `test_*.py` files, `test_*` methods
2. **Use appropriate markers**: `@pytest.mark.integration`, `@pytest.mark.slow`, etc.
3. **Handle import errors gracefully**: Use `pytest.skip()` for missing dependencies
4. **Document test purpose**: Clear docstrings explaining what's being tested
5. **Validate realistic scenarios**: Test actual usage patterns, not just API calls

## Test Statistics

- **Total test files**: 4
- **Total test methods**: 35
- **Integration scenarios**: 23 marked with `@pytest.mark.integration`
- **Slow tests**: 6 marked with `@pytest.mark.slow`
- **Benchmark tests**: 1 marked with `@pytest.mark.benchmark`
- **Lines of code**: ~2,900 lines

## Dependencies

Integration tests require:
- **Core**: pytest, pytest-mock
- **Optional**: pytest-benchmark, pytest-cov, pytest-xdist
- **Refunc dependencies**: pandas, numpy, psutil, pyyaml
- **Mock-friendly**: Tests gracefully handle missing dependencies