# Performance Benchmarking Guide

This guide covers the performance benchmarking and regression detection system in refunc.

## Overview

The refunc benchmarking suite provides comprehensive performance testing to ensure consistent performance characteristics across versions. It includes:

- **Benchmark tests** for critical functions and decorators
- **Performance regression detection** with configurable thresholds  
- **Memory profiling** benchmarks
- **CI/CD integration** for automated performance monitoring

## Quick Start

### Running Benchmarks

```bash
# Run all benchmarks
python scripts/run_benchmarks.py

# Run with custom threshold (default 20%)
python scripts/run_benchmarks.py --threshold 15

# Run specific benchmark category
pytest benchmarks/benchmark_decorators.py --benchmark-only

# Compare against baseline
python scripts/run_benchmarks.py --no-compare
```

### Saving Baselines

```bash
# Save current results as baseline
python scripts/run_benchmarks.py --save-baseline

# This creates benchmarks/.benchmarks/baseline.json
```

## Benchmark Structure

```
benchmarks/
├── __init__.py                    # Package initialization
├── benchmark_decorators.py       # Performance decorator benchmarks
├── benchmark_data_science.py     # Data processing benchmarks  
├── benchmark_file_handling.py    # File I/O benchmarks
├── benchmark_utils.py            # Core utility benchmarks
├── regression_detection.py       # Regression detection utilities
└── .benchmarks/                  # Results and baselines
    ├── results.json              # Latest benchmark results
    ├── baseline.json             # Performance baseline
    └── history/                  # Historical results
```

## Regression Detection

### Automatic Detection

The system automatically detects performance regressions:

```python
from benchmarks.regression_detection import PerformanceRegression

# Initialize with 25% threshold
detector = PerformanceRegression(threshold_percent=25)

# Check for regressions
alerts = detector.check_all_regressions(current_results)

for alert in alerts:
    print(f"Regression in {alert.benchmark_name}: {alert.regression_percent:.1f}% slower")
```

### Manual Baseline Management

```python
from benchmarks.regression_detection import BenchmarkResult
from datetime import datetime

# Create baseline result
baseline = BenchmarkResult(
    name="my_function",
    mean_time=0.001,  # 1ms
    min_time=0.0008,
    max_time=0.0012,
    stddev=0.0001,
    iterations=1000,
    timestamp=datetime.now().isoformat()
)

# Set as baseline
detector.set_baseline("my_function", baseline)
```

## Integration with Existing Performance Monitoring

The benchmarking system works seamlessly with refunc's existing decorators:

```python
from refunc.decorators import time_it, get_timing_stats
from benchmarks.regression_detection import has_regressed

@time_it(collect_stats=True)
def monitored_function():
    # Your function implementation
    pass

# Establish baseline
for _ in range(10):
    monitored_function()

# Get baseline performance
stats = get_timing_stats("monitored_function")
baseline_ms = stats.mean_time * 1000

# Later, check for regression
if has_regressed("monitored_function", baseline_ms, tolerance=0.25):
    print("Performance regression detected!")
```

## CI/CD Integration

### GitHub Actions

The benchmarking system includes GitHub Actions workflows:

- **`.github/workflows/performance.yml`**: Runs benchmarks on PRs and pushes
- **Regression detection**: Fails CI if performance degrades beyond threshold
- **Benchmark reports**: Comments benchmark results on PRs

### Custom CI Integration

```bash
# In your CI pipeline
python scripts/run_benchmarks.py --threshold 20

# Exit code 1 if regressions detected
if [ $? -ne 0 ]; then
    echo "Performance regression detected, failing build"
    exit 1
fi
```

## Configuration

### Pytest Configuration

Benchmarks are configured in `pytest.ini`:

```ini
[tool:pytest]
benchmark-columns = min,max,mean,stddev,outliers,ops,rounds
benchmark-group-by = group
benchmark-sort = mean
benchmark-compare-fail = mean:5%
benchmark-autosave = true
```

### Benchmark Categories

Benchmarks are organized by category:

- **Decorators**: Performance monitoring overhead
- **File Handling**: I/O operation performance  
- **Data Science**: Data processing performance
- **Utils**: Core utility performance

## Best Practices

### Writing Benchmarks

```python
import pytest

class TestMyFeaturePerformance:
    def test_my_function_performance(self, benchmark):
        """Benchmark my function with realistic workload."""
        
        # Setup
        test_data = create_realistic_test_data()
        
        # Benchmark the function
        result = benchmark(my_function, test_data)
        
        # Verify correctness
        assert result is not None
        assert validate_result(result)
    
    def test_scalability(self, benchmark):
        """Test performance with varying data sizes."""
        
        def run_with_size(size):
            data = generate_data(size)
            return my_function(data)
        
        # Benchmark with different sizes
        small_result = benchmark.pedantic(
            lambda: run_with_size(100),
            rounds=10,
            iterations=5
        )
        
        assert validate_result(small_result)
```

### Regression Thresholds

- **5-10%**: Acceptable variation for most functions
- **15-20%**: Reasonable threshold for complex operations
- **25%+**: Use only for highly variable workloads

### Memory Benchmarks

```python
def test_memory_efficiency(self, benchmark):
    """Test memory usage patterns."""
    
    @memory_profile()
    def memory_intensive_function():
        # Function that allocates memory
        return process_large_dataset()
    
    result = benchmark(memory_intensive_function)
    
    # Verify no memory leaks
    assert result is not None
```

## Troubleshooting

### Common Issues

1. **High variability**: Use `benchmark.pedantic()` with more rounds
2. **CI timeouts**: Reduce benchmark complexity or increase timeout
3. **False regressions**: Adjust threshold or use relative baselines

### Debug Mode

```bash
# Run benchmarks with detailed output
pytest benchmarks/ --benchmark-only -v --benchmark-verbose

# Analyze specific benchmark
pytest benchmarks/benchmark_decorators.py::TestTimingDecorators::test_time_it_decorator_overhead --benchmark-only -v
```

## Related Documentation

- [Performance Monitoring Guide](performance.md) - Using decorators for monitoring
- [Decorators API Reference](../api/decorators.md) - Decorator documentation
- [Contributing Guidelines](../developer/contributing.md) - Adding new benchmarks