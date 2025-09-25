# ⚡ Decorators API Reference

> **Performance monitoring decorators for timing, memory profiling, system monitoring, validation, and comprehensive performance tracking.**

## Overview

The decorators module provides a comprehensive suite of performance monitoring decorators designed to help you profile and optimize ML workflows. From simple timing measurements to advanced system monitoring and validation, these decorators integrate seamlessly into your code.

### Key Features

- **⏱️ Timing Decorators**: Precise execution time measurement with multiple timing modes
- **🧠 Memory Profiling**: Memory usage tracking and leak detection
- **🖥️ System Monitoring**: CPU, GPU, and system resource monitoring
- **✅ Validation**: Input/output validation with custom validators
- **📊 Combined Monitoring**: All-in-one performance tracking
- **📈 Statistics**: Comprehensive statistics and performance analytics

## Quick Start

```python
from refunc.decorators import time_it, memory_profile, validate_inputs

# Simple timing
@time_it
def train_model(data):
    # Training logic
    return model

# Memory profiling
@memory_profile(track_peak=True)
def process_large_dataset(data):
    # Data processing
    return processed_data

# Input validation
@validate_inputs(types={'data': np.ndarray, 'epochs': int})
def fit_model(data, epochs=100):
    # Model fitting
    return model

# Combined monitoring
from refunc.decorators import performance_monitor

@performance_monitor(timing=True, memory=True, system=True)
def comprehensive_training():
    # Complex training pipeline
    return results
```

---

## Timing Decorators

### time_it

Measure execution time with multiple timing modes and statistics collection.

```python
@time_it(
    mode: str = "wall_clock",
    logger: Optional[logging.Logger] = None,
    collect_stats: bool = True,
    print_result: bool = False,
    precision: int = 6
)
```

**Parameters:**

- `mode`: Timing mode (`"wall_clock"`, `"process_time"`, `"thread_time"`, `"perf_counter"`)
- `logger`: Logger for timing results
- `collect_stats`: Whether to collect statistics across calls
- `print_result`: Print timing result to console
- `precision`: Decimal precision for time display

**Example:**

```python
from refunc.decorators import time_it, get_timing_stats

@time_it(mode="wall_clock", collect_stats=True)
def expensive_computation(n):
    return sum(i**2 for i in range(n))

# Call function multiple times
for i in range(10):
    expensive_computation(100000)

# Get statistics
stats = get_timing_stats("expensive_computation")
print(f"Mean time: {stats.mean_time:.6f}s")
print(f"Min time: {stats.min_time:.6f}s")
print(f"Max time: {stats.max_time:.6f}s")
```

### time_it_async

Async version for asynchronous functions.

```python
@time_it_async(mode="wall_clock", collect_stats=True)
async def async_data_processing(data):
    await asyncio.sleep(0.1)  # Simulate async work
    return processed_data

# Usage
result = await async_data_processing(data)
```

### timer

Context manager for timing code blocks.

```python
from refunc.decorators import timer

with timer("data_loading") as t:
    data = load_large_dataset()
    
print(f"Data loading took: {t.elapsed:.3f}s")

# With custom timing mode
with timer("processing", mode="process_time") as t:
    processed = process_data(data)
```

### TimingProfiler

Advanced timing profiler with detailed analysis.

```python
from refunc.decorators import TimingProfiler

profiler = TimingProfiler()

@profiler.profile("model_training")
def train_model():
    # Training logic
    pass

@profiler.profile("data_preprocessing")
def preprocess_data():
    # Preprocessing logic
    pass

# Train multiple times
for i in range(5):
    preprocess_data()
    train_model()

# Get comprehensive report
report = profiler.get_report()
print(report)
```

### Timing Results

```python
@dataclass
class TimingResult:
    function_name: str
    execution_time: float
    timestamp: float
    args_hash: Optional[str] = None
    timing_mode: str = "wall_clock"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TimingStats:
    function_name: str
    call_count: int
    total_time: float
    mean_time: float
    min_time: float
    max_time: float
    std_dev: float
    median_time: float
    p95_time: float
    p99_time: float
```

---

## Memory Decorators

### memory_profile

Profile memory usage during function execution.

```python
@memory_profile(
    track_peak: bool = True,
    track_allocations: bool = False,
    interval: float = 0.1,
    logger: Optional[logging.Logger] = None,
    units: str = "MB"
)
```

**Parameters:**

- `track_peak`: Track peak memory usage
- `track_allocations`: Track individual allocations (requires tracemalloc)
- `interval`: Monitoring interval in seconds
- `logger`: Logger for memory results
- `units`: Memory units (`"B"`, `"KB"`, `"MB"`, `"GB"`)

**Example:**

```python
from refunc.decorators import memory_profile, get_peak_memory_usage

@memory_profile(track_peak=True, units="MB")
def memory_intensive_function():
    # Create large data structures
    large_list = [i for i in range(1000000)]
    large_dict = {i: i**2 for i in range(100000)}
    return large_list, large_dict

result = memory_intensive_function()
peak = get_peak_memory_usage("memory_intensive_function")
print(f"Peak memory usage: {peak:.2f} MB")
```

### memory_monitor

Continuous memory monitoring during execution.

```python
@memory_monitor(
    interval=0.1,
    threshold_mb=1000,
    alert_callback=None
)
def long_running_process():
    # Long running computation
    pass

# Custom alert callback
def memory_alert(usage_mb, threshold_mb):
    print(f"WARNING: Memory usage {usage_mb:.1f}MB exceeds threshold {threshold_mb}MB")

@memory_monitor(interval=0.5, threshold_mb=500, alert_callback=memory_alert)
def monitored_process():
    # Process with memory monitoring
    pass
```

### MemoryLeakDetector

Advanced memory leak detection.

```python
from refunc.decorators import MemoryLeakDetector

detector = MemoryLeakDetector()

@detector.monitor("data_processing")
def process_batch(batch_data):
    # Process data batch
    return processed_batch

# Process multiple batches
for batch in data_batches:
    process_batch(batch)

# Check for memory leaks
leaks = detector.detect_leaks()
if leaks:
    print("Memory leaks detected:")
    for leak in leaks:
        print(f"  {leak.function_name}: {leak.trend:.2f} MB/call")
```

### Memory Results

```python
@dataclass
class MemoryResult:
    function_name: str
    memory_before: float
    memory_after: float
    memory_peak: float
    memory_delta: float
    units: str = "MB"
    allocations: List[Dict] = field(default_factory=list)
    
@dataclass
class MemorySnapshot:
    timestamp: float
    rss: float
    vms: float
    percent: float
    available: float
    used: float
```

---

## System Monitoring Decorators

### system_monitor

Monitor system resources during function execution.

```python
@system_monitor(
    monitor_cpu: bool = True,
    monitor_memory: bool = True,
    monitor_disk: bool = False,
    monitor_network: bool = False,
    monitor_gpu: bool = False,
    interval: float = 1.0
)
```

**Example:**

```python
from refunc.decorators import system_monitor, get_system_info

@system_monitor(monitor_cpu=True, monitor_memory=True, monitor_gpu=True)
def gpu_intensive_training():
    # GPU-intensive training
    pass

# Get current system info
info = get_system_info()
print(f"CPU cores: {info.cpu_count}")
print(f"Memory: {info.memory_total:.1f} GB")

# GPU monitoring
@system_monitor(monitor_gpu=True, interval=0.5)
def gpu_computation():
    # GPU computation
    pass
```

### MonitoringResult

```python
@dataclass
class MonitoringResult:
    function_name: str
    duration: float
    cpu_percent: float
    memory_percent: float
    disk_io: Dict[str, int]
    network_io: Dict[str, int]
    gpu_utilization: List[float]
    gpu_memory: List[float]
    system_load: float
```

---

## Validation Decorators

### validate_inputs

Validate function inputs with flexible validators.

```python
@validate_inputs(
    types: Optional[Dict[str, type]] = None,
    ranges: Optional[Dict[str, tuple]] = None,
    shapes: Optional[Dict[str, tuple]] = None,
    custom: Optional[Dict[str, Callable]] = None,
    strict: bool = True
)
```

**Example:**

```python
from refunc.decorators import validate_inputs
import numpy as np

@validate_inputs(
    types={'data': np.ndarray, 'epochs': int, 'lr': float},
    ranges={'epochs': (1, 1000), 'lr': (0.0001, 1.0)},
    shapes={'data': (None, 784)}  # None means any size
)
def train_neural_network(data, epochs, lr=0.001):
    # Training logic
    pass

# This will raise ValidationError
try:
    train_neural_network("invalid_data", epochs=-5, lr=2.0)
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### validate_outputs

Validate function outputs.

```python
@validate_outputs(
    types={'return': np.ndarray},
    shapes={'return': (None, 10)},  # Output should have 10 features
    custom={'return': lambda x: x.min() >= 0}  # All values non-negative
)
def predict_probabilities(model, data):
    predictions = model.predict_proba(data)
    return predictions
```

### Custom Validators

```python
from refunc.decorators import CustomValidator

class DataFrameValidator(CustomValidator):
    def __init__(self, required_columns=None, min_rows=None):
        self.required_columns = required_columns or []
        self.min_rows = min_rows
    
    def validate(self, value, param_name):
        if not hasattr(value, 'columns'):
            raise ValidationError(f"{param_name} must be a DataFrame")
        
        for col in self.required_columns:
            if col not in value.columns:
                raise ValidationError(f"Missing column: {col}")
        
        if self.min_rows and len(value) < self.min_rows:
            raise ValidationError(f"Insufficient rows: {len(value)} < {self.min_rows}")

# Usage
@validate_inputs(custom={
    'df': DataFrameValidator(required_columns=['target'], min_rows=100)
})
def train_on_dataframe(df):
    # Training logic
    pass
```

### Validation Results

```python
@dataclass
class ValidationResult:
    is_valid: bool
    parameter: str
    expected: Any
    actual: Any
    error_message: str
    validator_type: str
```

---

## Combined Monitoring

### performance_monitor

All-in-one performance monitoring decorator.

```python
@performance_monitor(
    timing: bool = True,
    memory: bool = True,
    system: bool = False,
    validation: bool = False,
    config: Optional[MonitoringConfig] = None
)
```

**Example:**

```python
from refunc.decorators import performance_monitor, MonitoringConfig

# Custom monitoring configuration
config = MonitoringConfig(
    timing_mode="process_time",
    memory_units="GB",
    memory_interval=0.5,
    system_interval=1.0,
    collect_stats=True
)

@performance_monitor(
    timing=True,
    memory=True,
    system=True,
    config=config
)
def comprehensive_ml_pipeline(data):
    # Data preprocessing
    processed = preprocess_data(data)
    
    # Model training
    model = train_model(processed)
    
    # Evaluation
    results = evaluate_model(model, processed)
    
    return model, results

# Results contain all monitoring data
model, results = comprehensive_ml_pipeline(training_data)
```

### Monitoring Presets

```python
from refunc.decorators import quick_monitor, full_monitor, debug_monitor

# Quick monitoring (timing + basic memory)
@quick_monitor
def fast_function():
    pass

# Full monitoring (everything enabled)
@full_monitor
def comprehensive_function():
    pass

# Debug monitoring (detailed profiling)
@debug_monitor
def debug_function():
    pass
```

### CombinedResult

```python
@dataclass
class CombinedResult:
    timing: Optional[TimingResult] = None
    memory: Optional[MemoryResult] = None
    system: Optional[MonitoringResult] = None
    validation: Optional[ValidationResult] = None
    
    def summary(self) -> str:
        """Get formatted summary of all results."""
        lines = []
        
        if self.timing:
            lines.append(f"Execution time: {self.timing.execution_time:.6f}s")
        
        if self.memory:
            lines.append(f"Memory delta: {self.memory.memory_delta:.2f} {self.memory.units}")
        
        if self.system:
            lines.append(f"CPU usage: {self.system.cpu_percent:.1f}%")
        
        return "\n".join(lines)
```

---

## Advanced Usage

### Statistical Analysis

```python
from refunc.decorators import get_timing_stats, clear_timing_stats

# Run function multiple times
@time_it(collect_stats=True)
def algorithm_benchmark(n):
    return sorted([random.random() for _ in range(n)])

# Benchmark with different input sizes
for size in [1000, 5000, 10000, 50000]:
    for _ in range(10):
        algorithm_benchmark(size)

# Analyze performance characteristics
stats = get_timing_stats("algorithm_benchmark")
print(f"Function called {stats.call_count} times")
print(f"Average time: {stats.mean_time:.6f}s")
print(f"95th percentile: {stats.p95_time:.6f}s")
print(f"Standard deviation: {stats.std_dev:.6f}s")

# Clear statistics for fresh measurements
clear_timing_stats("algorithm_benchmark")
```

### Custom Monitoring Configuration

```python
from refunc.decorators import MonitoringConfig

# Create custom monitoring setup
config = MonitoringConfig(
    # Timing settings
    timing_mode="perf_counter",
    timing_precision=9,
    
    # Memory settings
    memory_units="MB",
    memory_interval=0.1,
    memory_track_peak=True,
    
    # System settings
    system_interval=2.0,
    system_monitor_gpu=True,
    
    # Statistics
    collect_stats=True,
    stats_window_size=100,  # Keep last 100 measurements
    
    # Output settings
    log_results=True,
    print_summary=True
)

@performance_monitor(
    timing=True,
    memory=True,
    system=True,
    config=config
)
def custom_monitored_function():
    # Function with custom monitoring
    pass
```

### Conditional Monitoring

```python
import os
from functools import wraps

def conditional_monitor(condition):
    """Apply monitoring only when condition is True."""
    def decorator(func):
        if condition:
            return performance_monitor(timing=True, memory=True)(func)
        else:
            return func
    return decorator

# Enable monitoring only in development
@conditional_monitor(os.getenv('ENVIRONMENT') == 'development')
def development_function():
    pass

# Enable monitoring for debugging
DEBUG = True

@conditional_monitor(DEBUG)
def debug_function():
    pass
```

### Performance Regression Detection

```python
from refunc.decorators import TimingProfiler

class PerformanceRegression:
    def __init__(self, threshold_percent=20):
        self.threshold = threshold_percent / 100
        self.baseline = {}
    
    def set_baseline(self, func_name, baseline_time):
        self.baseline[func_name] = baseline_time
    
    def check_regression(self, func_name, current_time):
        if func_name not in self.baseline:
            return False
        
        baseline = self.baseline[func_name]
        regression = (current_time - baseline) / baseline
        
        return regression > self.threshold

# Usage
regression_detector = PerformanceRegression(threshold_percent=25)

@time_it(collect_stats=True)
def monitored_algorithm(data):
    # Algorithm implementation
    return processed_data

# Set baseline performance
baseline_time = 0.1  # seconds
regression_detector.set_baseline("monitored_algorithm", baseline_time)

# Check for regressions during testing
stats = get_timing_stats("monitored_algorithm")
if stats and regression_detector.check_regression("monitored_algorithm", stats.mean_time):
    print("WARNING: Performance regression detected!")
```

---

## Error Handling

### Decorator Error Handling

```python
from refunc.decorators import time_it, ValidationError

@time_it(handle_errors=True)  # Continue timing even if function errors
def risky_function():
    if random.random() < 0.5:
        raise ValueError("Random error")
    return "success"

# Validation with error handling
@validate_inputs(
    types={'x': int}, 
    strict=False  # Don't raise on validation failure
)
def flexible_function(x):
    # Function handles both valid and invalid inputs
    if isinstance(x, int):
        return x * 2
    else:
        return str(x) * 2
```

### Graceful Degradation

```python
def safe_monitor(func):
    """Apply monitoring with graceful degradation."""
    try:
        return performance_monitor(timing=True, memory=True)(func)
    except ImportError:
        # Fall back to basic timing if advanced monitoring unavailable
        try:
            return time_it()(func)
        except ImportError:
            # No monitoring available, return original function
            return func

@safe_monitor
def production_function():
    # Function works with or without monitoring
    pass
```

---

## Best Practices

### 1. Choose Appropriate Monitoring Level

```python
# Development: Comprehensive monitoring
@performance_monitor(timing=True, memory=True, system=True)
def development_function():
    pass

# Production: Minimal monitoring
@time_it(collect_stats=True, print_result=False)
def production_function():
    pass

# Debugging: Maximum detail
@debug_monitor
def problematic_function():
    pass
```

### 2. Use Context Managers for Code Blocks

```python
# Monitor specific code sections
with timer("data_loading"):
    data = load_dataset()

with timer("preprocessing"):
    data = preprocess(data)

with timer("training"):
    model = train_model(data)
```

### 3. Combine with Logging

```python
import logging
from refunc.decorators import time_it, memory_profile

logger = logging.getLogger(__name__)

@time_it(logger=logger)
@memory_profile(logger=logger)
def logged_function():
    # Function with automatic logging
    pass
```

### 4. Statistical Analysis for Optimization

```python
# Collect statistics for performance analysis
@time_it(collect_stats=True)
def optimization_target():
    # Function to optimize
    pass

# Run multiple times to collect data
for _ in range(100):
    optimization_target()

# Analyze performance characteristics
stats = get_timing_stats("optimization_target")
if stats.std_dev > stats.mean_time * 0.1:
    print("High variability detected - investigate further")
```

---

## Examples

### Complete ML Pipeline Monitoring

```python
from refunc.decorators import (
    performance_monitor, validate_inputs, timer,
    get_timing_stats, MonitoringConfig
)
import numpy as np

# Configure comprehensive monitoring
config = MonitoringConfig(
    timing_mode="wall_clock",
    memory_units="MB",
    collect_stats=True,
    log_results=True
)

class MLPipeline:
    @validate_inputs(types={'data': np.ndarray})
    @performance_monitor(timing=True, memory=True, config=config)
    def preprocess(self, data):
        """Data preprocessing with validation and monitoring."""
        # Normalization
        normalized = (data - data.mean()) / data.std()
        
        # Feature engineering
        features = self._extract_features(normalized)
        
        return features
    
    @performance_monitor(timing=True, memory=True, system=True, config=config)
    def train(self, features, labels):
        """Model training with comprehensive monitoring."""
        model = self._initialize_model()
        
        for epoch in range(100):
            with timer(f"epoch_{epoch}"):
                model = self._train_epoch(model, features, labels)
        
        return model
    
    @time_it(collect_stats=True)
    def predict(self, model, data):
        """Fast prediction with timing statistics."""
        return model.predict(data)
    
    def _extract_features(self, data):
        # Feature extraction logic
        return data
    
    def _initialize_model(self):
        # Model initialization
        return None
    
    def _train_epoch(self, model, features, labels):
        # Training epoch logic
        return model

# Usage
pipeline = MLPipeline()

# Process data through pipeline
data = np.random.randn(1000, 50)
labels = np.random.randint(0, 2, 1000)

features = pipeline.preprocess(data)
model = pipeline.train(features, labels)

# Make multiple predictions to collect statistics
for _ in range(20):
    predictions = pipeline.predict(model, features[:100])

# Analyze prediction performance
pred_stats = get_timing_stats("predict")
print(f"Prediction stats:")
print(f"  Mean time: {pred_stats.mean_time:.6f}s")
print(f"  Throughput: {100/pred_stats.mean_time:.1f} samples/sec")
```

### Performance Comparison Framework

```python
from refunc.decorators import time_it, get_timing_stats, clear_timing_stats
import matplotlib.pyplot as plt

class PerformanceComparison:
    def __init__(self):
        self.results = {}
    
    def benchmark_function(self, name, func, *args, **kwargs):
        """Benchmark a function with given arguments."""
        
        # Clear previous stats
        clear_timing_stats(func.__name__)
        
        # Decorate function for timing
        timed_func = time_it(collect_stats=True)(func)
        
        # Run multiple times
        for _ in range(10):
            timed_func(*args, **kwargs)
        
        # Collect statistics
        stats = get_timing_stats(func.__name__)
        self.results[name] = stats
    
    def compare_algorithms(self):
        """Compare performance of different algorithms."""
        
        def bubble_sort(arr):
            n = len(arr)
            for i in range(n):
                for j in range(0, n-i-1):
                    if arr[j] > arr[j+1]:
                        arr[j], arr[j+1] = arr[j+1], arr[j]
            return arr
        
        def quick_sort(arr):
            if len(arr) <= 1:
                return arr
            pivot = arr[len(arr) // 2]
            left = [x for x in arr if x < pivot]
            middle = [x for x in arr if x == pivot]
            right = [x for x in arr if x > pivot]
            return quick_sort(left) + middle + quick_sort(right)
        
        # Test data
        test_data = [list(range(1000, 0, -1)) for _ in range(10)]
        
        # Benchmark algorithms
        self.benchmark_function("Bubble Sort", bubble_sort, test_data[0].copy())
        self.benchmark_function("Quick Sort", quick_sort, test_data[0].copy())
        self.benchmark_function("Python Sorted", sorted, test_data[0].copy())
    
    def plot_results(self):
        """Plot performance comparison."""
        names = list(self.results.keys())
        mean_times = [stats.mean_time for stats in self.results.values()]
        std_devs = [stats.std_dev for stats in self.results.values()]
        
        plt.figure(figsize=(10, 6))
        plt.bar(names, mean_times, yerr=std_devs, capsize=5)
        plt.ylabel('Execution Time (seconds)')
        plt.title('Algorithm Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Usage
comparison = PerformanceComparison()
comparison.compare_algorithms()
comparison.plot_results()
```

---

## See Also

- **[📈 Performance Guide](../guides/performance.md)** - Comprehensive performance monitoring guide
- **[📝 Logging](logging.md)** - Integration with logging framework
- **[⚠️ Exceptions](exceptions.md)** - Error handling in decorators
- **[📊 Math & Statistics](math_stats.md)** - Statistical analysis utilities
- **[🚀 Quick Start Guide](../guides/quickstart.md)** - Getting started
- **[💡 Examples](../examples/)** - More decorator examples
