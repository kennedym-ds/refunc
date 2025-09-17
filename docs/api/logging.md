# 📝 Logging API Reference

> **Advanced logging framework for ML workflows with experiment tracking, structured logging, and external integrations.**

## Overview

The logging module provides a comprehensive logging solution designed specifically for machine learning workflows. It includes structured logging, experiment tracking, progress monitoring, and integrations with popular ML platforms.

### Key Features

- **🎯 ML-Focused**: Designed specifically for ML experiment tracking and metrics logging
- **📊 Structured Logging**: JSON-compatible log entries with metadata
- **🌈 Rich Output**: Colored console output with progress bars and indicators
- **🔄 Progress Tracking**: Built-in progress tracking for training epochs and steps
- **🔗 Integrations**: MLflow, Weights & Biases, Prometheus, and more
- **⚡ Async Support**: Non-blocking logging for high-performance applications

## Quick Start

```python
from refunc.logging import MLLogger, experiment_context, progress_context

# Basic logger setup
logger = MLLogger("my_experiment")

# Simple logging
logger.info("Starting training process")
logger.metric("Training started", {"lr": 0.001, "batch_size": 32})

# Experiment tracking
with experiment_context("image_classification", "resnet_v1") as exp:
    logger.log_hyperparams({"lr": 0.001, "epochs": 100})
    
    for epoch in range(100):
        with progress_context(f"Epoch {epoch}", total=1000) as progress:
            # Training loop
            for step in range(1000):
                loss = train_step()
                progress.update({"loss": loss})
                
                if step % 100 == 0:
                    logger.log_metrics({"loss": loss, "step": step})
    
    logger.log_result("Final accuracy", 0.95)
```

---

## Core Classes

### MLLogger

The main logging class that provides structured logging with ML-specific features.

```python
class MLLogger:
    def __init__(
        self,
        name: str = "refunc",
        level: Union[int, str] = LogLevel.INFO,
        log_dir: Optional[Union[str, Path]] = None,
        experiment_tracking: bool = True,
        colored_output: bool = True,
        json_logging: bool = False,
        max_log_files: int = 10,
        max_file_size: str = "100MB"
    )
```

**Parameters:**

- `name`: Logger name (used in log messages and file names)
- `level`: Minimum log level to capture
- `log_dir`: Directory for log files (defaults to `./logs`)
- `experiment_tracking`: Enable experiment tracking features
- `colored_output`: Enable colored console output
- `json_logging`: Enable JSON-formatted log files
- `max_log_files`: Maximum number of rotating log files
- `max_file_size`: Maximum size per log file

**Example:**

```python
# Basic setup
logger = MLLogger("training_run")

# Advanced setup
logger = MLLogger(
    name="experiment_001",
    level="DEBUG",
    log_dir="./experiments/logs",
    experiment_tracking=True,
    colored_output=True,
    json_logging=True,
    max_log_files=5,
    max_file_size="50MB"
)
```

#### Basic Logging Methods

```python
# Standard log levels
logger.trace("Detailed debugging information")
logger.debug("Debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
logger.critical("Critical error")

# ML-specific levels
logger.metric("Training metric", {"accuracy": 0.92})
logger.experiment("Experiment event", {"stage": "validation"})
logger.progress("Progress update", {"completed": 75})
logger.result("Final result", {"test_accuracy": 0.95})
```

#### Experiment Tracking Methods

```python
# Hyperparameter logging
logger.log_hyperparams({
    "learning_rate": 0.001,
    "batch_size": 32,
    "optimizer": "adam",
    "model": "resnet50"
})

# Metrics logging
logger.log_metrics({
    "loss": 0.234,
    "accuracy": 0.92,
    "f1_score": 0.89
}, step=100)

# Artifact logging
logger.log_artifact("model.pkl", artifact_type="model")
logger.log_artifacts(["plot1.png", "plot2.png"], artifact_type="visualization")

# Tag management
logger.add_tags({"environment": "production", "version": "v1.2"})
logger.set_tag("experiment_type", "hyperparameter_tuning")

# Result logging
logger.log_result("final_accuracy", 0.945, metadata={"dataset": "test"})
```

#### Context Management

```python
# Start experiment
logger.start_experiment("image_classification", "experiment_001")

# End experiment
logger.end_experiment(status="completed", final_metrics={"accuracy": 0.95})

# Step tracking
logger.step()  # Increment step counter
logger.set_step(100)  # Set specific step

# Epoch tracking
logger.epoch()  # Increment epoch counter
logger.set_epoch(10)  # Set specific epoch

# Get current context
context = logger.get_experiment_context()
print(f"Experiment: {context.experiment_name}, Step: {context.step}")
```

### LogEntry

Structured log entry for ML workflows.

```python
@dataclass
class LogEntry:
    timestamp: float
    level: str
    message: str
    logger_name: str
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    step: Optional[int] = None
    epoch: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
```

**Methods:**

```python
entry = LogEntry(...)

# Serialization
entry_dict = entry.to_dict()
entry_json = entry.to_json()
```

### ExperimentContext

Context information for ML experiments.

```python
@dataclass
class ExperimentContext:
    experiment_id: str
    experiment_name: str
    run_id: str
    run_name: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "running"
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## Formatters

### ColoredFormatter

Provides colored console output with ML-specific formatting.

```python
from refunc.logging import ColoredFormatter

formatter = ColoredFormatter(
    show_time=True,
    show_level=True,
    show_logger=True,
    show_metrics=True,
    color_scheme="default"  # "default", "dark", "light"
)
```

**Features:**

- Color-coded log levels (ERROR=red, WARNING=yellow, INFO=blue, etc.)
- Highlighted metrics and key-value pairs
- Compact formatting for better readability
- Customizable color schemes

### JSONFormatter

Structured JSON output for log analysis and storage.

```python
from refunc.logging import JSONFormatter

formatter = JSONFormatter(
    include_extra=True,
    timestamp_format="iso",  # "iso", "unix", "readable"
    indent=None  # For pretty-printing, use 2 or 4
)
```

**Output Example:**

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "logger": "training",
  "message": "Epoch completed",
  "experiment_id": "exp_001",
  "step": 1000,
  "epoch": 5,
  "metrics": {"loss": 0.234, "accuracy": 0.92},
  "tags": {"model": "resnet50"}
}
```

### MLFormatter

Specialized formatter for ML workflows.

```python
from refunc.logging import MLFormatter

formatter = MLFormatter(
    show_step=True,
    show_epoch=True,
    show_metrics=True,
    metric_precision=4,
    compact_mode=False
)
```

---

## Progress Tracking

### ProgressTracker

Basic progress tracking for iterative processes.

```python
from refunc.logging import ProgressTracker

tracker = ProgressTracker(
    total=1000,
    description="Training",
    logger=logger
)

for i in range(1000):
    # Your processing here
    tracker.update(1, {"loss": current_loss})
    
    if i % 100 == 0:
        tracker.log_progress()

tracker.finish()
```

### TqdmProgressTracker

Progress tracking with tqdm integration.

```python
from refunc.logging import TqdmProgressTracker

with TqdmProgressTracker(total=1000, desc="Training") as progress:
    for i in range(1000):
        # Training step
        loss = train_step()
        progress.update(1, {"loss": loss, "lr": current_lr})
```

### EpochTracker

Specialized tracker for training epochs.

```python
from refunc.logging import EpochTracker

epoch_tracker = EpochTracker(
    total_epochs=100,
    steps_per_epoch=1000,
    logger=logger
)

for epoch in range(100):
    epoch_tracker.start_epoch(epoch)
    
    for step in range(1000):
        loss = train_step()
        epoch_tracker.update_step({"loss": loss})
    
    val_acc = validate()
    epoch_tracker.end_epoch({"val_accuracy": val_acc})
```

### Context Managers

```python
# Progress context
with progress_context("Training", total=1000) as progress:
    for i in range(1000):
        progress.update(1, {"step": i})

# Epoch context  
with epoch_context(total_epochs=10) as epochs:
    for epoch in epochs:
        # Training logic
        epochs.log({"accuracy": current_acc})
```

---

## Experiment Tracking

### ExperimentTracker

Advanced experiment tracking with metadata management.

```python
from refunc.logging import ExperimentTracker

tracker = ExperimentTracker(
    experiment_name="image_classification",
    run_name="resnet_baseline",
    log_dir="./experiments"
)

# Setup experiment
tracker.start_run()
tracker.log_params({"lr": 0.001, "batch_size": 32})
tracker.log_tags({"model": "resnet50", "dataset": "imagenet"})

# During training
for step in range(1000):
    metrics = {"loss": current_loss, "accuracy": current_acc}
    tracker.log_metrics(metrics, step=step)

# Artifacts
tracker.log_artifact("model.pkl")
tracker.log_artifact("training_plot.png", artifact_type="plot")

# Finish
tracker.end_run(status="completed")
```

### MLflow Integration

```python
from refunc.logging import MLflowIntegration

# Setup MLflow tracking
mlflow_integration = MLflowIntegration(
    tracking_uri="http://localhost:5000",
    experiment_name="my_experiment"
)

logger = MLLogger("training")
logger.add_integration(mlflow_integration)

# All logging will now be sent to MLflow
logger.log_hyperparams({"lr": 0.001})
logger.log_metrics({"loss": 0.5}, step=100)
```

### Weights & Biases Integration

```python
from refunc.logging import WandBIntegration

# Setup W&B tracking
wandb_integration = WandBIntegration(
    project="my_project",
    entity="my_team",
    config={"lr": 0.001, "batch_size": 32}
)

logger = MLLogger("training")
logger.add_integration(wandb_integration)

# Logging automatically syncs to W&B
logger.log_metrics({"accuracy": 0.92}, step=100)
```

---

## External Integrations

### Prometheus Integration

Export metrics to Prometheus for monitoring.

```python
from refunc.logging import PrometheusIntegration

prometheus = PrometheusIntegration(
    port=8000,
    namespace="ml_training"
)

logger = MLLogger("training")
logger.add_integration(prometheus)

# Metrics are automatically exported
logger.log_metrics({"loss": 0.5, "accuracy": 0.92})
```

### Slack Integration

Send important notifications to Slack.

```python
from refunc.logging import SlackIntegration

slack = SlackIntegration(
    webhook_url="https://hooks.slack.com/...",
    channel="#ml-alerts",
    error_notifications=True,
    completion_notifications=True
)

logger = MLLogger("training")
logger.add_integration(slack)

# Errors and completion messages sent to Slack
logger.error("Training failed!")
logger.log_result("Training completed", {"final_accuracy": 0.95})
```

### Integration Manager

Manage multiple integrations centrally.

```python
from refunc.logging import IntegrationsManager, integration_context

# Setup multiple integrations
manager = IntegrationsManager()
manager.add_integration("mlflow", MLflowIntegration(...))
manager.add_integration("wandb", WandBIntegration(...))
manager.add_integration("slack", SlackIntegration(...))

# Use with context manager
with integration_context(manager) as integrations:
    logger = MLLogger("training")
    logger.add_integrations(integrations)
    
    # All integrations receive logs
    logger.log_metrics({"accuracy": 0.95})
```

---

## Configuration

### Setup Functions

```python
from refunc.logging import setup_logging, get_logger

# Quick setup with defaults
setup_logging(level="INFO", colored=True)

# Advanced setup
setup_logging(
    level="DEBUG",
    log_dir="./logs",
    colored=True,
    json_logging=True,
    experiment_tracking=True,
    integrations={
        "mlflow": {"tracking_uri": "http://localhost:5000"},
        "slack": {"webhook_url": "https://hooks.slack.com/..."}
    }
)

# Get configured logger
logger = get_logger("my_experiment")
```

### Auto-Configuration

```python
from refunc.logging import auto_configure_integrations

# Automatically detect and configure available integrations
integrations = auto_configure_integrations()
logger = MLLogger("training")
logger.add_integrations(integrations)
```

---

## Advanced Usage

### Custom Log Levels

```python
from refunc.logging import LogLevel

# Using custom levels
logger.log(LogLevel.METRIC, "Custom metric", extra={"value": 0.95})
logger.log(LogLevel.EXPERIMENT, "Experiment milestone")
logger.log(LogLevel.RESULT, "Final result achieved")
```

### Structured Logging

```python
# Complex structured data
logger.info("Training step completed", 
    metrics={"loss": 0.234, "accuracy": 0.92},
    tags={"phase": "training", "model": "resnet"},
    artifacts=["checkpoint.pkl"],
    extra={"gpu_memory": "8.2GB", "batch_time": 0.15}
)
```

### Log Analysis

```python
# Access log entries
entries = logger.get_log_entries()
recent_errors = logger.get_log_entries(level="ERROR", last_n=10)

# Filter logs
training_logs = logger.get_log_entries(
    experiment_id="exp_001",
    start_time=start_timestamp,
    tags={"phase": "training"}
)

# Export logs
logger.export_logs("experiment_logs.json", format="json")
logger.export_logs("experiment_logs.csv", format="csv")
```

### Performance Optimization

```python
# Async logging for high-performance scenarios
from refunc.logging import AsyncHandler

async_handler = AsyncHandler(buffer_size=1000, flush_interval=5.0)
logger = MLLogger("high_performance")
logger.add_handler(async_handler)

# Buffered logging
from refunc.logging import BufferedHandler

buffered = BufferedHandler(buffer_size=100, auto_flush=True)
logger.add_handler(buffered)
```

---

## Error Handling

### Exception Logging

```python
try:
    risky_operation()
except Exception as e:
    logger.exception("Operation failed", 
        extra={"operation": "model_training", "step": current_step}
    )
    # Re-raise or handle as needed
```

### Graceful Degradation

```python
# Logger continues working even if integrations fail
try:
    logger.log_metrics({"accuracy": 0.95})
except Exception as e:
    logger.warning(f"Integration failed: {e}")
    # Local logging continues
```

---

## Best Practices

### 1. Structured Logging

```python
# Good: Structured data
logger.info("Training completed", 
    metrics={"accuracy": 0.95, "loss": 0.234},
    tags={"model": "resnet", "dataset": "imagenet"}
)

# Avoid: Unstructured strings
logger.info("Training completed with accuracy 0.95 and loss 0.234")
```

### 2. Consistent Experiment Tracking

```python
# Use experiment contexts for consistency
with experiment_context("image_classification", "run_001"):
    logger.log_hyperparams(params)
    
    for epoch in range(epochs):
        # Training logic
        logger.log_metrics(metrics, step=global_step)
    
    logger.log_result("final_accuracy", final_acc)
```

### 3. Appropriate Log Levels

```python
# Use appropriate levels
logger.debug("Loading data batch")           # Development info
logger.info("Starting training epoch")       # General info
logger.metric("Epoch completed", metrics)    # ML metrics
logger.warning("Low validation accuracy")    # Potential issues
logger.error("Training failed")              # Errors
logger.result("Training completed", result)  # Final results
```

### 4. Integration Management

```python
# Centralized integration setup
def setup_experiment_logging(experiment_name: str) -> MLLogger:
    logger = MLLogger(experiment_name)
    
    # Add integrations based on environment
    if os.getenv("MLFLOW_TRACKING_URI"):
        logger.add_integration(MLflowIntegration(...))
    
    if os.getenv("WANDB_API_KEY"):
        logger.add_integration(WandBIntegration(...))
    
    return logger
```

---

## Examples

### Complete Training Example

```python
from refunc.logging import MLLogger, experiment_context, progress_context

def train_model():
    logger = MLLogger("training")
    
    with experiment_context("image_classification", "resnet_v1") as exp:
        # Setup
        params = {"lr": 0.001, "batch_size": 32, "epochs": 10}
        logger.log_hyperparams(params)
        
        best_accuracy = 0
        
        for epoch in range(params["epochs"]):
            # Training phase
            with progress_context(f"Epoch {epoch}", total=1000) as progress:
                train_loss = 0
                
                for step in range(1000):
                    loss = train_step()
                    train_loss += loss
                    progress.update(1, {"loss": loss})
                    
                    if step % 100 == 0:
                        logger.log_metrics({"train_loss": loss}, step=step)
                
                avg_train_loss = train_loss / 1000
            
            # Validation phase
            val_accuracy = validate_model()
            
            # Log epoch results
            logger.log_metrics({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_accuracy": val_accuracy
            })
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                save_model("best_model.pkl")
                logger.log_artifact("best_model.pkl", artifact_type="model")
        
        # Final results
        logger.log_result("best_accuracy", best_accuracy)
        logger.info("Training completed successfully")

if __name__ == "__main__":
    train_model()
```

---

## See Also

- **[⚠️ Exceptions](exceptions.md)** - Error handling integration
- **[⚡ Decorators](decorators.md)** - Performance monitoring decorators  
- **[⚙️ Config](config.md)** - Configuration management
- **[🚀 Quick Start Guide](../guides/quickstart.md)** - Getting started
- **[💡 Examples](../examples/)** - More usage examples
