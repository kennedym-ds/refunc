# Performance Monitoring Guide

_Comprehensive guide to monitoring performance in Refunc using decorators, profilers, and best practices._

## 🔍 Why Performance Monitoring Matters

Performance monitoring is essential for ML workflows where data volumes and computation costs can spike rapidly. With Refunc's decorators you can:

- Track execution time for critical sections
- Identify memory leaks and peak usage
- Monitor system resources across CPU, GPU, and I/O
- Validate inputs/outputs for consistent behavior
- Capture repeatable metrics for regression detection

## ⚙️ Tooling Overview

| Feature | Decorator / Helper | Use Case |
|---------|-------------------|----------|
| Execution timing | `time_it`, `time_it_async`, `timer`, `TimingProfiler` | Measure function/block runtime |
| Memory profiling | `memory_profile`, `memory_monitor`, `MemoryLeakDetector` | Track peak usage and leaks |
| System monitoring | `system_monitor`, `monitor_performance` | Observe CPU, GPU, disk, network |
| Validation | `validate_inputs`, `validate_outputs` | Guard against invalid data |
| Combined monitoring | `performance_monitor`, `performance_monitor_async`, `quick_monitor`, `full_monitor`, `debug_monitor` | One-stop decorator for all metrics |
| Statistics | `get_timing_stats`, `clear_timing_stats` | Aggregate metrics across runs |

> ℹ️ These APIs live inside `refunc.decorators`. Import modules or specific symbols as needed.

## 🚀 Quick Start Recipes

### 1. Time Critical Functions

```python
from refunc.decorators import time_it

@time_it(mode="wall_clock", collect_stats=True)
def preprocess_batch(batch):
    ...
```

- `mode`: choose between `"wall_clock"`, `"process_time"`, `"thread_time"`, or `"perf_counter"`.
- Use `collect_stats=True` to aggregate metrics via `get_timing_stats`.

#### Inspecting Aggregated Stats

```python
from refunc.decorators import get_timing_stats

stats = get_timing_stats("preprocess_batch")
print(f"Mean time: {stats.mean_time:.4f}s")
print(f"95th percentile: {stats.p95_time:.4f}s")
```

### 2. Track Memory Peaks

```python
from refunc.decorators import memory_profile

@memory_profile(track_peak=True, use_tracemalloc=True)
def materialize_features(df):
    ...
```

- `track_peak=True` surfaces the highest RSS usage observed.
- `use_tracemalloc=True` records allocation traces when Python's `tracemalloc` is available.

### 3. Monitor System Resources in One Shot

```python
from refunc.decorators import performance_monitor, MonitoringConfig

config = MonitoringConfig(
    enable_timing=True,
    enable_memory=True,
    enable_system=True,
    monitor_gpu=True,
    print_results=True,
)

@performance_monitor(config)
def train_epoch(state):
    ...
```

- Returns original function result by default.
- Set `return_results=True` in the config to get `(result, CombinedResult)` for programmatic analysis.

### 4. Async Workflows

```python
from refunc.decorators import performance_monitor_async

@performance_monitor_async(enable_system=False)
async def stream_predictions(queue):
    ...
```

Async wrappers collect metrics manually to avoid blocking event loops.

### 5. Context Manager for Ad-hoc Timing

```python
from refunc.decorators import monitor_performance

with monitor_performance("vectorized_pipeline", enable_system=False) as result:
    transformed = pipeline.fit_transform(dataset)

print(result.summary())
```

Use when decorating a function is not practical—for example within notebooks or scripts.

## 🧠 Validation + Monitoring

Combine validation with performance tracking to catch regressions early:

```python
from refunc.decorators import performance_monitor, MonitoringConfig, ValidatorBase
from refunc.exceptions import ValidationError

class ProbabilityValidator(ValidatorBase):
    def validate(self, value, context):
        if (value < 0).any() or (value > 1).any():
            raise ValidationError("Probabilities must be within [0, 1].")

config = MonitoringConfig(
    enable_timing=True,
    enable_memory=True,
    enable_validation=True,
    output_validators=ProbabilityValidator(),
    raise_on_validation_error=True,
)

@performance_monitor(config)
def predict_proba(model, data):
    return model.predict_proba(data)
```

## 📈 Regression Detection Workflow

1. Decorate functions with `time_it(collect_stats=True)`.
2. Run benchmark suite to populate stats.
3. Persist aggregated metrics (e.g., JSON/DB) after each CI run.
4. Compare new metrics against baseline—trigger alerts when `mean_time` grows beyond threshold.

Example helper:

```python
from refunc.decorators import get_timing_stats

def has_regressed(func_name: str, baseline_ms: float, tolerance: float = 0.25) -> bool:
    stats = get_timing_stats(func_name)
    if not stats:
        return False
    current_ms = stats.mean_time * 1000
    return current_ms > baseline_ms * (1 + tolerance)
```

## 🛠️ Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `tracemalloc` warning | Python < 3.4 or module disabled | remove `use_tracemalloc=True` or upgrade Python |
| GPU stats always zero | `psutil` / `GPUtil` missing or GPU not exposed | install extras (`pip install refunc[gpu]`) |
| High overhead from monitoring | Sampling interval too small or monitoring too many features | increase `system_sample_interval` or disable unused monitors |
| Validation errors stop pipeline | `raise_on_validation_error=True` | set to `False` and check `CombinedResult.validation_result` |

## 🧬 Best Practices Checklist

- ✅ Start with `quick_monitor` during exploratory work; switch to tailored `MonitoringConfig` for production.
- ✅ Capture both mean and tail latencies (`p95`, `p99`) before optimizing.
- ✅ Combine logging (e.g., `MLLogger`) with monitoring for traceable experiments.
- ✅ Record environment details (CPU/GPU model, RAM) when sharing performance numbers.
- ✅ Include monitoring decorators in regression tests to catch degradations early.

## 🔗 Related Docs

- [Decorators API Reference](../api/decorators.md)
- [Logging API Reference](../api/logging.md)
- [Quick Start Guide](quickstart.md)
- [Installation Guide](installation.md)

---

_Last updated: {{CURRENT_YEAR}}. Keep benchmarks reproducible by pinning dependencies and documenting hardware specs._
