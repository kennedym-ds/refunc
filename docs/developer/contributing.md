# Contributing to Refunc

Thank you for your interest in contributing to Refunc! This guide will help you get started with contributing to the project.

## 🎯 Ways to Contribute

There are many ways to contribute to Refunc:

- 🐛 **Bug Reports**: Report bugs and issues
- 💡 **Feature Requests**: Suggest new features or improvements
- 📝 **Documentation**: Improve documentation and examples
- 🔧 **Code Contributions**: Fix bugs, implement features, optimize performance
- 🧪 **Testing**: Add tests, improve test coverage
- 📚 **Examples**: Create tutorials and usage examples
- 🎨 **Design**: Improve user experience and interface design

## 🚀 Getting Started

### 1. Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/refunc.git
cd refunc
```

### 2. Set Up Development Environment

#### Option A: Using the Setup Script (Recommended)

```bash
# Automated setup with development dependencies
python scripts/setup_venv.py --dev --force

# Activate the environment
source venv/bin/activate  # Linux/macOS
# or venv\Scripts\activate  # Windows
```

#### Option B: Manual Setup

```bash
# Create virtual environment
python -m venv dev_env
source dev_env/bin/activate  # Linux/macOS
# or dev_env\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Setup

```bash
# Run tests
pytest

# Run linting
pre-commit run --all-files

# Check import
python -c "import refunc; print(f'Refunc v{refunc.__version__} ready!')"
```

## 📋 Development Workflow

### 1. Create a Branch

Create a descriptive branch name:

```bash
# Feature branches
git checkout -b feature/add-new-decorator
git checkout -b feature/improve-logging-performance

# Bug fix branches
git checkout -b bugfix/fix-memory-leak
git checkout -b bugfix/handle-edge-case

# Documentation branches
git checkout -b docs/update-api-reference
git checkout -b docs/add-examples
```

### 2. Make Changes

Follow these guidelines when making changes:

#### Code Style

We use automated code formatting and linting:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

```bash
# Format code
black refunc/ tests/

# Sort imports
isort refunc/ tests/

# Check linting
flake8 refunc/ tests/

# Type checking
mypy refunc/
```

#### Code Standards

- **Type Hints**: Add type hints to all functions and methods
- **Docstrings**: Follow Google/NumPy docstring format
- **Error Handling**: Use appropriate exception types from `refunc.exceptions`
- **Testing**: Add tests for new functionality
- **Performance**: Consider performance implications

#### Example Code Structure

```python
"""Module description.

Detailed description of what this module does.
"""

from typing import List, Optional, Union
import numpy as np

from .exceptions import ValidationError


def example_function(
    data: Union[List[float], np.ndarray],
    threshold: float = 0.5,
    normalize: bool = True
) -> np.ndarray:
    """
    Example function with proper documentation.
    
    Args:
        data: Input data as list or numpy array
        threshold: Threshold value for processing
        normalize: Whether to normalize the output
        
    Returns:
        Processed data as numpy array
        
    Raises:
        ValidationError: If data is empty or invalid
        
    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> result = example_function(data, threshold=0.3)
        >>> print(result.shape)
        (5,)
    """
    # Input validation
    if len(data) == 0:
        raise ValidationError("Data cannot be empty")
    
    # Convert to numpy array
    data_array = np.asarray(data)
    
    # Process data
    processed = data_array * threshold
    
    # Normalize if requested
    if normalize:
        processed = processed / np.max(processed)
    
    return processed
```

### 3. Write Tests

Add comprehensive tests for your changes:

```python
# tests/test_new_feature.py
import pytest
import numpy as np
from refunc.exceptions import ValidationError
from refunc.your_module import example_function


class TestExampleFunction:
    """Test suite for example_function."""
    
    def test_basic_functionality(self):
        """Test basic function operation."""
        data = [1, 2, 3, 4, 5]
        result = example_function(data)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)
    
    def test_threshold_parameter(self):
        """Test threshold parameter effect."""
        data = [2, 4, 6]
        result = example_function(data, threshold=0.5)
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(result, expected)
    
    def test_normalization(self):
        """Test normalization functionality."""
        data = [2, 4, 6]
        result = example_function(data, normalize=True)
        assert np.max(result) == 1.0
    
    def test_empty_data_raises_error(self):
        """Test that empty data raises ValidationError."""
        with pytest.raises(ValidationError, match="Data cannot be empty"):
            example_function([])
    
    def test_numpy_array_input(self):
        """Test with numpy array input."""
        data = np.array([1, 2, 3])
        result = example_function(data)
        assert isinstance(result, np.ndarray)
    
    @pytest.mark.parametrize("threshold", [0.1, 0.5, 1.0, 2.0])
    def test_various_thresholds(self, threshold):
        """Test function with various threshold values."""
        data = [1, 2, 3]
        result = example_function(data, threshold=threshold)
        expected_max = max(data) * threshold
        assert np.max(result) == expected_max
```

### 4. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=refunc --cov-report=html

# Run specific test file
pytest tests/test_new_feature.py

# Run with verbose output
pytest -v

# Run only failed tests
pytest --lf
```

### 5. Update Documentation

#### API Documentation

Update relevant API documentation in `docs/api/`:

```markdown
### new_function()

Description of the new function.

**Parameters:**
- `param1` (type): Description
- `param2` (type, optional): Description

**Returns:**
- `return_type`: Description

**Example:**
```python
from refunc.module import new_function
result = new_function(param1="value")
```

#### Add Examples

Create practical examples in `docs/examples/`:

```python
# Example: Using the new feature
from refunc.module import new_function

# Basic usage
result = new_function("input")
print(f"Result: {result}")

# Advanced usage
advanced_result = new_function(
    "complex_input",
    option1=True,
    option2="custom"
)
```

### 6. Commit Changes

Write clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "feat: add statistical outlier detection methods"
git commit -m "fix: resolve memory leak in logging module"
git commit -m "docs: update API reference for decorators"
git commit -m "test: add comprehensive tests for retry mechanism"

# Follow conventional commits format
git commit -m "type(scope): description"
```

**Commit Types:**

- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `style`: Code style changes
- `chore`: Maintenance tasks

### 7. Push and Create Pull Request

```bash
# Push your branch
git push origin feature/your-feature-name

# Create pull request on GitHub
```

## 🧪 Testing Guidelines

### Test Structure

```text
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures
├── test_exceptions.py          # Exception framework tests
├── test_decorators.py          # Decorator tests
├── test_logging.py             # Logging system tests
├── test_math_stats.py          # Statistics tests
├── test_config.py              # Configuration tests
├── test_utils.py               # Utility function tests
├── integration/                # Integration tests
│   ├── test_workflow.py
│   └── test_performance.py
└── fixtures/                   # Test data
    ├── sample_data.csv
    └── test_config.yaml
```

### Test Categories

#### Unit Tests

Test individual functions and methods:

```python
def test_individual_function():
    """Test a single function in isolation."""
    input_data = create_test_data()
    result = function_under_test(input_data)
    assert result == expected_result
```

#### Integration Tests

Test component interactions:

```python
def test_logging_with_decorators():
    """Test logging integration with decorators."""
    logger = MLLogger("test")
    
    @time_it(logger=logger)
    def test_function():
        return "result"
    
    result = test_function()
    assert result == "result"
    # Verify logging occurred
```

#### Performance Tests

Test performance requirements:

```python
import pytest

@pytest.mark.benchmark
def test_performance_requirement(benchmark):
    """Test that function meets performance requirements."""
    data = create_large_dataset()
    result = benchmark(function_under_test, data)
    assert len(result) == len(data)
```

### Test Fixtures

Create reusable test data:

```python
# conftest.py
import pytest
import numpy as np

@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return np.random.normal(0, 1, 1000)

@pytest.fixture
def ml_logger():
    """Provide ML logger for tests."""
    return MLLogger("test", log_dir="./test_logs")

@pytest.fixture
def temp_config_file(tmp_path):
    """Provide temporary configuration file."""
    config_file = tmp_path / "test_config.yaml"
    config_content = """
    model:
      type: "test_model"
      parameters:
        learning_rate: 0.01
    """
    config_file.write_text(config_content)
    return str(config_file)
```

## 📝 Documentation Standards

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """
    One-line summary of the function.
    
    Longer description explaining what the function does,
    its purpose, and any important details.
    
    Args:
        param1: Description of the first parameter
        param2: Description of the second parameter with default value
        
    Returns:
        Description of what the function returns
        
    Raises:
        SpecificError: When this specific error occurs
        AnotherError: When this other error occurs
        
    Example:
        Basic usage example:
        
        >>> result = example_function("test", 5)
        >>> print(result)
        True
        
        Advanced usage:
        
        >>> result = example_function(
        ...     param1="complex_case",
        ...     param2=20
        ... )
        >>> assert result is True
        
    Note:
        Any important notes about the function behavior,
        performance considerations, or usage recommendations.
    """
    # Implementation here
    return True
```
