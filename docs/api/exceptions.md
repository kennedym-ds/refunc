# Exceptions Framework

The `refunc.exceptions` module provides a comprehensive exception hierarchy designed specifically for machine learning workflows. It includes robust error handling, retry mechanisms, and context-aware error messages.

## Overview

The exceptions framework is organized into several categories:

- **Core Exceptions**: Base exceptions for general use
- **Data Exceptions**: Specific to data handling operations
- **Model Exceptions**: Related to ML model operations
- **Retry Mechanisms**: Automatic retry functionality with configurable strategies

## Core Exceptions

### RefuncError

The base exception class for all Refunc-specific errors.

```python
class RefuncError(Exception):
    """Base exception for all Refunc errors."""
```

**Usage:**

```python
from refunc.exceptions import RefuncError

try:
    # Some operation
    pass
except RefuncError as e:
    print(f"Refunc error occurred: {e}")
```

### ConfigurationError

Raised when there are configuration-related issues.

```python
class ConfigurationError(RefuncError):
    """Configuration-related errors."""
```

**Example:**

```python
from refunc.exceptions import ConfigurationError

# Raised when invalid configuration is provided
raise ConfigurationError("Missing required configuration key: 'model_path'")
```

### ValidationError

Raised when data or parameter validation fails.

```python
class ValidationError(RefuncError):
    """Data or parameter validation errors."""
```

**Example:**

```python
from refunc.exceptions import ValidationError

def validate_data(data):
    if data is None:
        raise ValidationError("Data cannot be None")
    if len(data) == 0:
        raise ValidationError("Data cannot be empty")
```

### OperationError

Raised when an operation fails to complete successfully.

```python
class OperationError(RefuncError):
    """Operation execution errors."""
```

### ResourceError

Raised when resource-related issues occur (memory, disk, network, etc.).

```python
class ResourceError(RefuncError):
    """Resource-related errors (memory, disk, network)."""
```

## Data Exceptions

### DataError

Base class for all data-related exceptions.

```python
class DataError(RefuncError):
    """Base class for data-related errors."""
```

### FileNotFoundError

Raised when a required file cannot be found.

```python
class FileNotFoundError(DataError):
    """File not found errors."""
```

**Example:**

```python
from refunc.exceptions import FileNotFoundError

def load_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
```

### UnsupportedFormatError

Raised when an unsupported file format is encountered.

```python
class UnsupportedFormatError(DataError):
    """Unsupported file format errors."""
```

### DataValidationError

Raised when data validation fails.

```python
class DataValidationError(DataError):
    """Data validation errors."""
```

### SchemaError

Raised when data schema validation fails.

```python
class SchemaError(DataError):
    """Data schema validation errors."""
```

### CorruptedDataError

Raised when data corruption is detected.

```python
class CorruptedDataError(DataError):
    """Data corruption errors."""
```

### EmptyDataError

Raised when empty data is encountered where it's not allowed.

```python
class EmptyDataError(DataError):
    """Empty data errors."""
```

## Model Exceptions

### ModelError

Base class for all model-related exceptions.

```python
class ModelError(RefuncError):
    """Base class for model-related errors."""
```

### ModelNotFoundError

Raised when a model cannot be found.

```python
class ModelNotFoundError(ModelError):
    """Model not found errors."""
```

### ModelLoadError

Raised when model loading fails.

```python
class ModelLoadError(ModelError):
    """Model loading errors."""
```

### ModelSaveError

Raised when model saving fails.

```python
class ModelSaveError(ModelError):
    """Model saving errors."""
```

### ModelTrainingError

Raised when model training fails.

```python
class ModelTrainingError(ModelError):
    """Model training errors."""
```

### ModelPredictionError

Raised when model prediction fails.

```python
class ModelPredictionError(ModelError):
    """Model prediction errors."""
```

### IncompatibleModelError

Raised when model compatibility issues occur.

```python
class IncompatibleModelError(ModelError):
    """Model compatibility errors."""
```

### ModelValidationError

Raised when model validation fails.

```python
class ModelValidationError(ModelError):
    """Model validation errors."""
```

## Retry Mechanisms

### RetryConfig

Configuration class for retry mechanisms.

```python
@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_attempts: int = 3
    delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 60.0
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
```

**Parameters:**

- `max_attempts`: Maximum number of retry attempts
- `delay`: Initial delay between retries (seconds)
- `backoff_factor`: Exponential backoff multiplier
- `max_delay`: Maximum delay between retries
- `exceptions`: Tuple of exception types to catch and retry

### RetryError

Exception raised when all retry attempts are exhausted.

```python
class RetryError(RefuncError):
    """Exception raised when all retry attempts are exhausted."""
```

### retry_on_failure

Decorator for automatic retry functionality.

```python
def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger: Optional[Any] = None
) -> Callable:
    """Decorator to retry function execution on failure."""
```

**Usage:**

```python
from refunc.exceptions import retry_on_failure, ValidationError

@retry_on_failure(
    max_attempts=3,
    delay=1.0,
    backoff_factor=2.0,
    exceptions=(ValidationError, ConnectionError)
)
def unreliable_operation():
    # Operation that might fail
    pass
```

**Advanced Example:**

```python
from refunc.exceptions import retry_on_failure, RetryConfig
import logging

logger = logging.getLogger(__name__)

@retry_on_failure(
    max_attempts=5,
    delay=0.5,
    backoff_factor=1.5,
    max_delay=10.0,
    exceptions=(ConnectionError, TimeoutError),
    logger=logger
)
def api_call():
    """Make an API call with automatic retries."""
    # API call implementation
    pass
```

### RetryableOperation

Context manager for retry operations.

```python
class RetryableOperation:
    """Context manager for retryable operations."""
```

**Usage:**

```python
from refunc.exceptions import RetryableOperation, RetryConfig

config = RetryConfig(max_attempts=3, delay=1.0)

with RetryableOperation(config) as retry:
    for attempt in retry:
        try:
            # Your operation here
            result = risky_operation()
            break  # Success, exit retry loop
        except SomeException as e:
            if attempt.is_last:
                raise  # Re-raise if last attempt
            print(f"Attempt {attempt.number} failed: {e}")
```

## Exception Hierarchy

```text
RefuncError
├── ConfigurationError
├── ValidationError
├── OperationError
├── ResourceError
├── DataError
│   ├── FileNotFoundError
│   ├── UnsupportedFormatError
│   ├── DataValidationError
│   ├── SchemaError
│   ├── CorruptedDataError
│   └── EmptyDataError
├── ModelError
│   ├── ModelNotFoundError
│   ├── ModelLoadError
│   ├── ModelSaveError
│   ├── ModelTrainingError
│   ├── ModelPredictionError
│   ├── IncompatibleModelError
│   └── ModelValidationError
└── RetryError
```

## Best Practices

### 1. Use Specific Exceptions

Always use the most specific exception type available:

```python
# Good
raise DataValidationError("Invalid data format: expected CSV, got JSON")

# Avoid
raise Exception("Something went wrong with data")
```

### 2. Provide Context

Include relevant context in exception messages:

```python
# Good
raise ModelLoadError(f"Failed to load model from {model_path}: {original_error}")

# Avoid
raise ModelLoadError("Model load failed")
```

### 3. Use Retry Mechanisms Wisely

Apply retries only to operations that can benefit from them:

```python
# Good - Network operations
@retry_on_failure(exceptions=(ConnectionError, TimeoutError))
def download_data():
    pass

# Avoid - Programming errors
@retry_on_failure()  # Don't retry programming errors
def calculate_mean(data):
    return sum(data) / len(data)  # Will fail on empty data
```

### 4. Chain Exceptions

Preserve the original exception when wrapping:

```python
try:
    risky_operation()
except SomeThirdPartyException as e:
    raise OperationError(f"Operation failed: {e}") from e
```

## See Also

- [Configuration Guide](../guides/configuration.md) - For handling ConfigurationError
- [Data Handling Guide](../guides/data_handling.md) - For data-related exceptions
- [Model Management Guide](../guides/model_management.md) - For model exceptions
- [Error Handling Guide](../guides/error_handling.md) - Best practices for error handling
