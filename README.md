# 🚀 Refunc - Reusable Functions for ML

A comprehensive ML utilities toolkit designed to provide a solid, reusable foundation for all your ML projects with professional development practices built-in!

## Core Features

### 📁 File Utilities

- Smart file detection and multi-format loading
- Directory scanning with pattern matching
- Intelligent caching system
- Model versioning and storage

### 📝 Advanced Logging

- ML-specific logger with experiment tracking
- Colored console output with progress bars
- Rotating file handlers with compression
- Integration hooks for MLflow/W&B

### ⚠️ Exception Framework

- Custom ML exception hierarchy
- Context-aware error messages
- Retry mechanisms with exponential backoff
- Automatic error recovery strategies

### ⚡ Performance Decorators

Key decorators included:

- `@time_it`: Execution timing
- `@memory_profile`: Memory usage tracking
- `@gpu_monitor`: GPU utilization
- `@cache_result`: Intelligent caching
- `@retry_on_failure`: Automatic retries
- `@log_execution`: Automatic logging
- `@validate_inputs`: Input validation
- `@profile_performance`: Combined monitoring

### 🔧 Development Tools

- Cross-platform setup scripts (Unix/Mac/Windows)
- Pre-commit hooks (black, isort, flake8, mypy)
- Automated testing with pytest
- CI/CD with GitHub Actions

## Repository Structure

```text
refunc/
├── README.md
├── setup.py
├── pyproject.toml
├── .pre-commit-config.yaml
├── scripts/
│ ├── setup_env.sh
│ ├── setup_env.bat
│ └── setup_env.ps1
├── requirements/
│ ├── base.txt
│ ├── dev.txt
│ └── test.txt
├── refunc/
│ ├── __init__.py
│ ├── utils/
│ ├── logging/
│ ├── exceptions/
│ ├── decorators/
│ └── config/
├── examples/
│ └── notebooks/
└── tests/
```

## Quick Usage Example

```python
from refunc import MLLogger, time_it, memory_profile, FileHandler
from refunc.exceptions import retry_on_failure

# Initialize logger
logger = MLLogger("experiment_001")

# Use decorators for monitoring
@time_it(logger=logger)
@memory_profile(track_peak=True)
@retry_on_failure(max_attempts=3)
def train_model(data_path):
    handler = FileHandler()
    data = handler.load_auto(data_path)
    
    # Training logic here
    logger.log_metrics({"loss": 0.5, "accuracy": 0.92})
    return model

# Smart file operations
handler = FileHandler()
all_csvs = handler.search_pattern("./data", "*.csv")
```

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/kennedym-ds/refunc.git
cd refunc

# Run setup (auto-detects OS)
./scripts/setup_env.sh # or setup_env.bat on Windows

# Activate environment
source venv/bin/activate

# Install pre-commit hooks
pre-commit install
```

## Production-Ready Features

- **Type hints** throughout for better IDE support
- **Comprehensive docstrings** with examples
- **Unit tests** with high coverage
- **Performance benchmarks** for critical paths
- **Thread-safe** operations
- **Minimal dependencies** for easy installation
- **Semantic versioning** for releases
