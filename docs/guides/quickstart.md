# Quick Start Guide

Get up and running with Refunc in 5 minutes! This guide will walk you through installation, basic setup, and your first example.

## 🚀 Installation

### Option 1: pip install (Recommended)

```bash
pip install refunc
```

### Option 2: From Source

```bash
git clone https://github.com/kennedym-ds/refunc.git
cd refunc
pip install -e .
```

### Option 3: Using the Setup Script

For automatic environment setup with virtual environment:

```bash
# Clone the repository
git clone https://github.com/kennedym-ds/refunc.git
cd refunc

# Run the automated setup script
python scripts/setup_venv.py --dev --force
```

This script will:

- Detect available Python versions
- Create a virtual environment
- Install all dependencies
- Set up development tools

## ⚡ Basic Usage

### 1. Performance Monitoring

Monitor execution time and memory usage:

```python
from refunc.decorators import time_it, memory_profile

@time_it()
@memory_profile(track_peak=True)
def train_model(data):
    # Your training code here
    import time
    time.sleep(1)  # Simulate training
    return "model_trained"

# Call your function
result = train_model([1, 2, 3, 4, 5])
# Output: Function 'train_model' took 1.002 seconds
# Output: Peak memory usage: 45.2 MB
```

### 2. Robust Error Handling

Add automatic retry functionality:

```python
from refunc.exceptions import retry_on_failure, ValidationError

@retry_on_failure(max_attempts=3, delay=1.0)
def unreliable_api_call():
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("API temporarily unavailable")
    return {"status": "success", "data": [1, 2, 3]}

# This will automatically retry up to 3 times
try:
    result = unreliable_api_call()
    print(f"Success: {result}")
except Exception as e:
    print(f"Failed after retries: {e}")
```

### 3. Advanced Logging

Set up ML experiment logging:

```python
from refunc.logging import MLLogger

# Initialize logger for your experiment
logger = MLLogger("experiment_001", log_dir="./logs")

# Log metrics
logger.log_metrics({
    "epoch": 1,
    "loss": 0.234,
    "accuracy": 0.891,
    "learning_rate": 0.001
})

# Log parameters
logger.log_params({
    "batch_size": 32,
    "model_type": "neural_network",
    "optimizer": "adam"
})

# Log artifacts
logger.log_artifact("model.pkl", "./models/my_model.pkl")

print("Experiment logged successfully!")
```

### 4. Statistical Analysis

Perform comprehensive statistical analysis:

```python
from refunc.math_stats import describe, test_normality
import numpy as np

# Generate sample data
data = np.random.normal(100, 15, 1000)

# Get comprehensive descriptive statistics
stats = describe(data)
print(stats.summary())

# Test for normality
normality_test = test_normality(data, method="shapiro")
print(f"\nNormality test: {normality_test.interpretation}")
print(f"P-value: {normality_test.p_value:.6f}")
```

### 5. Configuration Management

Manage your project configuration:

```python
from refunc.config import ConfigManager

# Initialize configuration
config = ConfigManager()

# Load configuration from file
config.load_from_file("config.yaml")

# Access configuration values
model_config = config.get("model", {})
data_path = config.get("data.path", "./data")

# Update configuration
config.set("training.epochs", 100)
config.set("training.batch_size", 32)

# Save updated configuration
config.save_to_file("updated_config.yaml")
```

## 🏗️ Complete Example: ML Workflow

Here's a complete example combining multiple Refunc features:

```python
from refunc.decorators import time_it, memory_profile
from refunc.exceptions import retry_on_failure, ValidationError
from refunc.logging import MLLogger
from refunc.math_stats import describe
import numpy as np

# Set up experiment logging
logger = MLLogger("ml_workflow_demo")

@time_it(logger=logger)
@memory_profile(track_peak=True)
@retry_on_failure(max_attempts=3, exceptions=(ValidationError,))
def load_and_analyze_data(data_path=None):
    """Load data and perform statistical analysis."""
    
    # Simulate data loading (replace with real data loading)
    if data_path is None:
        data = np.random.normal(50, 10, 1000)
    else:
        # Load your actual data here
        data = np.random.normal(50, 10, 1000)
    
    # Validate data
    if len(data) == 0:
        raise ValidationError("Dataset is empty")
    
    # Perform statistical analysis
    stats = describe(data)
    
    # Log metrics
    logger.log_metrics({
        "data_count": stats.count,
        "data_mean": stats.mean,
        "data_std": stats.std,
        "data_skewness": stats.skewness
    })
    
    print(f"Loaded {len(data)} samples")
    print(f"Mean: {stats.mean:.2f}, Std: {stats.std:.2f}")
    
    return data, stats

@time_it(logger=logger)
@retry_on_failure(max_attempts=2)
def train_model(data):
    """Train a simple model."""
    # Simulate model training
    import time
    time.sleep(0.5)  # Simulate training time
    
    # Log training metrics
    logger.log_metrics({
        "training_samples": len(data),
        "model_accuracy": 0.85,
        "training_loss": 0.234
    })
    
    model = {"type": "demo_model", "accuracy": 0.85}
    return model

def main():
    """Main workflow function."""
    try:
        # Load and analyze data
        data, stats = load_and_analyze_data()
        
        # Train model
        model = train_model(data)
        
        # Log final results
        logger.log_params({
            "workflow": "demo",
            "data_samples": len(data),
            "model_type": model["type"]
        })
        
        print("\n🎉 Workflow completed successfully!")
        print(f"Model accuracy: {model['accuracy']}")
        print(f"Check logs in: {logger.log_dir}")
        
    except Exception as e:
        logger.log_error(f"Workflow failed: {e}")
        print(f"❌ Workflow failed: {e}")

if __name__ == "__main__":
    main()
```

## 📁 File Structure

After running the examples, your project structure might look like:

```text
your_project/
├── main.py                 # Your main script
├── config.yaml            # Configuration file
├── logs/                   # Experiment logs
│   └── ml_workflow_demo/
│       ├── metrics.json
│       ├── params.json
│       └── artifacts/
└── models/                 # Saved models
    └── my_model.pkl
```

## 🎯 Next Steps

Now that you've seen the basics, explore more advanced features:

1. **[Installation Guide](installation.md)** - Detailed setup instructions
2. **[API Reference](../api/)** - Complete documentation for all modules
3. **[Examples](../examples/)** - More comprehensive examples and tutorials
4. **[Performance Monitoring](performance.md)** - Advanced monitoring techniques
5. **[Error Handling](error_handling.md)** - Robust error management strategies

## 🤔 Common Issues

### Import Errors

If you get import errors, make sure Refunc is properly installed:

```python
# Test installation
try:
    import refunc
    print(f"Refunc version: {refunc.__version__}")
except ImportError:
    print("Refunc not installed. Run: pip install refunc")
```

### Missing Dependencies

If you get dependency errors, install the development dependencies:

```bash
pip install "refunc[dev]"
```

### Virtual Environment Issues

If you have issues with virtual environments, use the setup script:

```bash
python scripts/setup_venv.py --help
```

## 📞 Getting Help

- 📖 [Full Documentation](../README.md)
- 🐛 [Report Issues](https://github.com/kennedym-ds/refunc/issues)
- 💬 [Join Discussions](https://github.com/kennedym-ds/refunc/discussions)

---

Ready to build robust ML workflows? Let's get started! 🚀