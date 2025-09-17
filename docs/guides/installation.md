# Installation Guide

This guide provides comprehensive installation instructions for Refunc across different platforms and environments.

## 📋 Requirements

### System Requirements

- **Python**: 3.7 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 512MB RAM (2GB+ recommended for ML workflows)
- **Disk Space**: ~100MB for base installation, additional space for dependencies

### Python Version Support

Refunc supports the following Python versions:

- ✅ Python 3.7
- ✅ Python 3.8
- ✅ Python 3.9
- ✅ Python 3.10
- ✅ Python 3.11
- ✅ Python 3.12 (latest)

## 🚀 Installation Methods

### Method 1: pip Install (Recommended)

The easiest way to install Refunc:

```bash
pip install refunc
```

For the latest development version:

```bash
pip install git+https://github.com/kennedym-ds/refunc.git
```

### Method 2: conda Install

Using conda package manager:

```bash
conda install -c conda-forge refunc
```

### Method 3: From Source

For development or to get the latest features:

```bash
# Clone the repository
git clone https://github.com/kennedym-ds/refunc.git
cd refunc

# Install in development mode
pip install -e .
```

### Method 4: Automated Setup Script

For a complete development environment setup:

```bash
# Clone the repository
git clone https://github.com/kennedym-ds/refunc.git
cd refunc

# Run the automated setup script
python scripts/setup_venv.py
```

## 🔧 Installation Options

### Base Installation

Install only core dependencies:

```bash
pip install refunc
```

**Includes:**

- Core utilities and decorators
- Exception handling framework
- Basic logging functionality
- Configuration management
- Mathematical and statistical tools

### Development Installation

Install with development tools:

```bash
pip install "refunc[dev]"
```

**Additional includes:**

- Code formatting tools (black, isort)
- Linting tools (flake8, mypy)
- Pre-commit hooks
- Development utilities

### Testing Installation

Install with testing dependencies:

```bash
pip install "refunc[test]"
```

**Additional includes:**

- pytest and plugins
- Coverage reporting tools
- Performance benchmarking
- Mock testing utilities

### Documentation Installation

Install with documentation tools:

```bash
pip install "refunc[docs]"
```

**Additional includes:**

- Sphinx documentation generator
- Documentation themes
- Jupyter notebook support
- Documentation building tools

### Complete Installation

Install everything:

```bash
pip install "refunc[dev,test,docs,gpu]"
```

### GPU Support

For GPU monitoring capabilities:

```bash
pip install "refunc[gpu]"
```

**Additional includes:**

- GPUtil for GPU monitoring
- NVIDIA ML Python bindings
- GPU performance decorators

## 🖥️ Platform-Specific Instructions

### Windows

#### Using Command Prompt

```cmd
# Basic installation
pip install refunc

# With virtual environment
python -m venv refunc_env
refunc_env\Scripts\activate
pip install refunc
```

#### Using PowerShell

```powershell
# Basic installation
pip install refunc

# With virtual environment
python -m venv refunc_env
.\refunc_env\Scripts\Activate.ps1
pip install refunc
```

#### Using the Windows Setup Script

```cmd
# Clone and setup
git clone https://github.com/kennedym-ds/refunc.git
cd refunc
scripts\setup_env.bat
```

### macOS

#### Using Terminal

```bash
# Basic installation
pip3 install refunc

# With virtual environment
python3 -m venv refunc_env
source refunc_env/bin/activate
pip install refunc
```

#### Using Homebrew Python

```bash
# Install Python with Homebrew
brew install python

# Install Refunc
pip3 install refunc
```

#### Using the macOS Setup Script

```bash
# Clone and setup
git clone https://github.com/kennedym-ds/refunc.git
cd refunc
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

### Linux

#### Ubuntu/Debian

```bash
# Install Python and pip
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Install Refunc
pip3 install refunc
```

#### CentOS/RHEL/Fedora

```bash
# Install Python and pip
sudo yum install python3 python3-pip  # CentOS/RHEL
# or
sudo dnf install python3 python3-pip  # Fedora

# Install Refunc
pip3 install refunc
```

#### Using the Linux Setup Script

```bash
# Clone and setup
git clone https://github.com/kennedym-ds/refunc.git
cd refunc
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

## 🏗️ Virtual Environment Setup

### Using venv (Recommended)

```bash
# Create virtual environment
python -m venv refunc_env

# Activate (Linux/macOS)
source refunc_env/bin/activate

# Activate (Windows)
refunc_env\Scripts\activate

# Install Refunc
pip install refunc

# Deactivate when done
deactivate
```

### Using conda

```bash
# Create conda environment
conda create -n refunc_env python=3.11

# Activate environment
conda activate refunc_env

# Install Refunc
pip install refunc

# Deactivate when done
conda deactivate
```

### Using the Automated Setup Script

The setup script provides the most comprehensive environment setup:

```bash
# Interactive setup with options
python scripts/setup_venv.py

# Non-interactive setup with defaults
python scripts/setup_venv.py --auto --dev

# Custom environment name
python scripts/setup_venv.py --venv-name my_ml_env

# Force recreation of existing environment
python scripts/setup_venv.py --force

# Include all optional dependencies
python scripts/setup_venv.py --all
```

**Setup Script Features:**

- Automatic Python version detection
- Multiple Python version support
- Cross-platform compatibility
- Dependency conflict resolution
- Pre-commit hook installation
- Development tool configuration

## ✅ Verification

### Basic Installation Test

```python
# Test basic import
import refunc
print(f"Refunc version: {refunc.__version__}")

# Test core modules
from refunc import exceptions, decorators, logging
print("✅ Core modules imported successfully")

# Test functionality
from refunc.decorators import time_it

@time_it()
def test_function():
    return sum(range(1000))

result = test_function()
print(f"✅ Function execution test passed: {result}")
```

### Comprehensive Test

```python
# Run all module tests
from refunc.exceptions import retry_on_failure, ValidationError
from refunc.logging import MLLogger
from refunc.math_stats import describe
from refunc.config import ConfigManager

print("Testing all modules...")

# Test exceptions
@retry_on_failure(max_attempts=2)
def test_retry():
    return "success"

assert test_retry() == "success"
print("✅ Exceptions module working")

# Test logging
logger = MLLogger("test")
logger.log_metrics({"test": 1.0})
print("✅ Logging module working")

# Test statistics
import numpy as np
data = np.random.normal(0, 1, 100)
stats = describe(data)
assert stats.count == 100
print("✅ Math/stats module working")

# Test configuration
config = ConfigManager()
config.set("test.key", "value")
assert config.get("test.key") == "value"
print("✅ Configuration module working")

print("🎉 All tests passed!")
```

## 🔧 Development Setup

For contributing to Refunc:

```bash
# Clone the repository
git clone https://github.com/kennedym-ds/refunc.git
cd refunc

# Create development environment
python -m venv dev_env
source dev_env/bin/activate  # or dev_env\Scripts\activate on Windows

# Install in development mode with all dependencies
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
pre-commit run --all-files
```

## 🐛 Troubleshooting

### Common Issues

#### Import Error: No module named 'refunc'

**Solution:**

```bash
# Check if installed
pip list | grep refunc

# If not found, install
pip install refunc

# Check Python path
python -c "import sys; print(sys.path)"
```

#### Permission Denied (Windows)

**Solution:**

```cmd
# Run as administrator or use --user flag
pip install --user refunc
```

#### SSL Certificate Error

**Solution:**

```bash
# Use trusted hosts
pip install --trusted-host pypi.org --trusted-host pypi.python.org refunc

# Or upgrade certificates
pip install --upgrade certifi
```

#### Dependency Conflicts

**Solution:**

```bash
# Create clean virtual environment
python -m venv clean_env
source clean_env/bin/activate
pip install refunc

# Or use setup script
python scripts/setup_venv.py --force
```

#### Memory Issues During Installation

**Solution:**

```bash
# Disable cache
pip install --no-cache-dir refunc

# Increase swap space (Linux)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Platform-Specific Issues

#### Windows: Microsoft Visual C++ Build Tools Required

**Solution:**

```cmd
# Install Microsoft C++ Build Tools from:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Or use pre-compiled wheels
pip install --only-binary=all refunc
```

#### macOS: Command Line Tools Missing

**Solution:**

```bash
# Install Xcode command line tools
xcode-select --install

# Or install Xcode from App Store
```

#### Linux: Python Development Headers Missing

**Solution:**

```bash
# Ubuntu/Debian
sudo apt install python3-dev

# CentOS/RHEL
sudo yum install python3-devel

# Fedora
sudo dnf install python3-devel
```

## 📊 Performance Optimization

### Installation Performance

For faster installation:

```bash
# Use pip cache
pip install refunc

# Parallel installation
pip install --upgrade pip setuptools wheel
pip install refunc

# Use binary wheels only
pip install --only-binary=all refunc
```

### Runtime Performance

For optimal runtime performance:

```bash
# Install with performance dependencies
pip install "refunc[gpu]"

# Use compiled dependencies
pip install numpy scipy pandas --no-binary :none:
pip install refunc
```

## 🔄 Updating

### Update to Latest Version

```bash
# Update from PyPI
pip install --upgrade refunc

# Update from source
cd refunc
git pull origin main
pip install -e .
```

### Check Current Version

```python
import refunc
print(f"Current version: {refunc.__version__}")

# Check for updates
import subprocess
result = subprocess.run(["pip", "list", "--outdated"], capture_output=True, text=True)
if "refunc" in result.stdout:
    print("Update available!")
```

## 🗑️ Uninstallation

### Remove Refunc

```bash
# Uninstall package
pip uninstall refunc

# Remove configuration files (optional)
rm -rf ~/.refunc/  # Linux/macOS
# or manually delete %USERPROFILE%\.refunc\ on Windows

# Remove virtual environment
rm -rf refunc_env/  # or your env name
```

## 📞 Getting Help

If you encounter issues during installation:

1. 📖 Check this documentation
2. 🔍 Search [existing issues](https://github.com/kennedym-ds/refunc/issues)
3. 💡 Start a [discussion](https://github.com/kennedym-ds/refunc/discussions)
4. 🐛 [Report a bug](https://github.com/kennedym-ds/refunc/issues/new)

## 🎯 Next Steps

After successful installation:

1. Follow the [Quick Start Guide](quickstart.md)
2. Explore [API Reference](../api/)
3. Try [Examples and Tutorials](../examples/)
4. Set up [Development Environment](../developer/setup.md) (for contributors)

---

Happy coding with Refunc! 🚀
