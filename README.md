# 🚀 Refunc - ML Utilities Toolkit

> **A comprehensive, production-ready ML utilities toolkit designed to accelerate machine learning development with robust, reusable components and professional development practices built-in.**

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![CI](https://github.com/kennedym-ds/refunc/workflows/CI/badge.svg)](https://github.com/kennedym-ds/refunc/actions/workflows/ci.yml)
[![Documentation Status](https://github.com/kennedym-ds/refunc/workflows/Deploy%20Documentation/badge.svg)](https://kennedym-ds.github.io/refunc/)
[![codecov](https://codecov.io/gh/kennedym-ds/refunc/branch/main/graph/badge.svg)](https://codecov.io/gh/kennedym-ds/refunc)
[![Security Rating](https://github.com/kennedym-ds/refunc/workflows/Security%20Scan/badge.svg)](https://github.com/kennedym-ds/refunc/actions/workflows/security.yml)

[![PyPI version](https://badge.fury.io/py/refunc.svg)](https://badge.fury.io/py/refunc)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/refunc)](https://pypi.org/project/refunc/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/kennedym-ds/refunc/graphs/commit-activity)
[![GitHub contributors](https://img.shields.io/github/contributors/kennedym-ds/refunc.svg)](https://GitHub.com/kennedym-ds/refunc/graphs/contributors/)

## 🎯 Overview

Refunc is a comprehensive ML utilities toolkit that provides essential building blocks for machine learning projects. From intelligent file handling and advanced logging to robust exception management and performance monitoring, Refunc eliminates boilerplate code and provides production-ready utilities that scale with your projects.

**🚀 [Quick Start Guide](docs/guides/quickstart.md)** | **📖 [Full Documentation](docs/README.md)** | **🔧 [Installation Guide](docs/guides/installation.md)**

## ✨ Key Features

### 🏗️ **Core Architecture**

- **Modular Design**: Independent utilities that work together seamlessly
- **Type Safety**: Comprehensive type hints throughout for better IDE support
- **Production Ready**: Thread-safe, tested, and optimized for real-world usage
- **Cross-Platform**: Full support for Windows, macOS, and Linux

### 📦 **Module Overview**

| Module | Purpose | Key Features |
|--------|---------|--------------|
| **🔧 Utils** | File operations & data handling | Auto-format detection, smart caching, batch processing |
| **📝 Logging** | ML-specific logging framework | Experiment tracking, colored output, metric logging |
| **⚠️ Exceptions** | Robust error handling | Custom ML exceptions, retry mechanisms, graceful recovery |
| **⚡ Decorators** | Performance monitoring | Timing, memory profiling, caching, input validation |
| **⚙️ Config** | Configuration management | YAML/JSON support, environment variables, validation |
| **📊 Math/Stats** | Statistical utilities | Hypothesis testing, bootstrapping, outlier detection |
| **🤖 ML** | Machine learning helpers | Model utilities, pipeline components, evaluation metrics |
| **🔬 Data Science** | Data analysis tools | Preprocessing, feature engineering, visualization helpers |

## 🏃‍♂️ Quick Start

### Installation

```bash
# Basic installation
pip install refunc

# Development installation
git clone https://github.com/kennedym-ds/refunc.git
cd refunc
pip install -e .
```

### Essential Usage

```python
from refunc import MLLogger, time_it, memory_profile, FileHandler
from refunc.exceptions import retry_on_failure
from refunc.math_stats import StatisticsEngine

# 1. Smart file operations
handler = FileHandler()
data = handler.load_auto("data.csv")  # Auto-detects format
all_data = handler.batch_load("./datasets/*.{csv,json}")

# 2. Performance monitoring
@time_it
@memory_profile(track_peak=True)
@retry_on_failure(max_attempts=3)
def train_model(data):
    # Your ML training code
    return model

# 3. Professional logging
logger = MLLogger("experiment_001")
logger.log_metrics({"accuracy": 0.95, "loss": 0.23})
logger.log_hyperparams({"lr": 0.001, "batch_size": 32})

# 4. Statistical analysis
stats = StatisticsEngine()
results = stats.hypothesis_test(data1, data2, test_type="t_test")
outliers = stats.detect_outliers(data, method="iqr")
```

### Data Science Toolkit in Action

```python
import pandas as pd
from refunc.data_science import DataCleaner

raw = pd.DataFrame(
    {
        "user": ["alice", "Bob ", "ALICE"],
        "age": ["32", "29", None],
        "signup": ["2025-08-01", "2025/08/02", "01-08-2025"],
    }
)

cleaner = DataCleaner()
clean, report = cleaner.clean_dataframe(raw)

print(report.summary())
# Data Cleaning Report
# ...

# Access the pandas accessor for quick checks
memory_report = clean.refunc.memory_usage_detailed()
print(memory_report[["column", "current_memory_mb"]])
```

## 🏗️ Architecture & Design

### Core Principles

- **🎯 Purpose-Built**: Designed specifically for ML workflows and common pain points
- **🔒 Reliability**: Comprehensive error handling with graceful degradation
- **⚡ Performance**: Optimized for speed with intelligent caching and lazy loading  
- **🧩 Modularity**: Use only what you need - no forced dependencies
- **📈 Scalability**: From prototypes to production environments

### Module Interactions

```mermaid
graph TB
    A[FileHandler] --> B[MLLogger]
    B --> C[Decorators]
    C --> D[Exceptions]
    E[Config] --> A
    E --> B
    F[Math/Stats] --> B
    G[ML] --> A
    G --> B
    H[Data Science] --> F
    H --> A
```

## 📖 Documentation

| Resource | Description |
|----------|-------------|
| **[📖 Main Documentation](docs/README.md)** | Complete documentation portal with navigation |
| **[🚀 Quick Start](docs/guides/quickstart.md)** | 5-minute getting started guide |
| **[🔧 Installation](docs/guides/installation.md)** | Detailed installation instructions |
| **[📚 API Reference](docs/api/)** | Complete API documentation for all modules |
| **[💡 Examples](docs/examples/)** | Practical usage examples and tutorials |
| **[🛠️ Contributing](docs/developer/contributing.md)** | Development guidelines and workflow |

### API Documentation

- **[⚠️ Exceptions Framework](docs/api/exceptions.md)** - Error handling and retry mechanisms
- **[📊 Math & Statistics](docs/api/math_stats.md)** - Statistical analysis and hypothesis testing
- **[📝 Logging](docs/api/logging.md)** - ML-specific logging and experiment tracking
- **[⚙️ Config](docs/api/config.md)** - Configuration management utilities
- **[⚡ Decorators](docs/api/decorators.md)** - Performance monitoring decorators
- **[🔧 Utils](docs/api/utils.md)** - File handling and data utilities

## 🚀 Repository Structure

```text
refunc/
├── 📁 docs/                     # 📖 Complete documentation
│   ├── README.md               # Documentation portal
│   ├── 📁 api/                 # API reference docs
│   ├── 📁 guides/              # User guides
│   ├── 📁 examples/            # Usage examples  
│   └── 📁 developer/           # Developer docs
├── 📁 refunc/                   # 🎯 Main package
│   ├── __init__.py
│   ├── 📁 utils/               # File & data utilities
│   ├── 📁 logging/             # ML logging framework
│   ├── 📁 exceptions/          # Exception handling
│   ├── 📁 decorators/          # Performance decorators
│   ├── 📁 config/              # Configuration management
│   ├── 📁 math_stats/          # Statistical utilities
│   ├── 📁 ml/                  # ML-specific helpers
│   └── 📁 data_science/        # Data analysis tools
├── 📁 scripts/                  # 🔧 Setup & utility scripts
├── 📁 requirements/             # 📦 Dependency definitions
├── 📁 tests/                    # ✅ Test suite
└── 📁 examples/                 # 💡 Usage examples
```

## 🛠️ Development

### Quick Setup

```bash
# Clone and setup development environment
git clone https://github.com/kennedym-ds/refunc.git
cd refunc

# Auto-detected setup (works on all platforms)
python scripts/setup_venv.py --dev

# Manual setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements/dev.txt
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### Quality Assurance

- **🧪 Testing**: Comprehensive test suite with pytest
- **📏 Code Style**: Black, isort, flake8, mypy for consistent formatting
- **🔍 Type Checking**: Full type hints with mypy validation
- **🚀 CI/CD**: GitHub Actions for automated testing and deployment
- **📦 Pre-commit**: Automated quality checks on every commit

## 📊 Project Status

### Current Release: v0.1.0

**Core Features Complete:**

- ✅ Exception handling framework with retry mechanisms
- ✅ Production logging & experiment tracking toolkit
- ✅ Performance monitoring decorators and profilers
- ✅ Data science cleaning, validation, and profiling suite
- ✅ File handling, configuration management, and math/stats engines

**Roadmap Highlights:**

- 🚧 Enhanced GPU monitoring and framework integrations
- 🚧 Advanced statistical methods and optimization tooling
- 🚧 Experiment dashboard with persistent metrics storage
- 🚧 Distributed execution helpers and cloud-native workflows

See our **[📋 Changelog](CHANGELOG.md)** for detailed release notes and roadmap.

## 🤝 Contributing

We welcome contributions! Whether it's bug reports, feature requests, or code contributions, please see our **[🛠️ Contributing Guide](docs/developer/contributing.md)** for details on:

- Development environment setup
- Code style and testing requirements  
- Pull request process
- Issue reporting guidelines

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **📖 [Documentation](docs/README.md)**
- **🐛 [Report Issues](https://github.com/kennedym-ds/refunc/issues)**
- **💡 [Feature Requests](https://github.com/kennedym-ds/refunc/issues)**
- **📧 [Contact](mailto:support@refunc.dev)**

---

*Built with ❤️ for the ML community. Star ⭐ this repo if you find it helpful!*
