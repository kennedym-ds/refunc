"""
Exception handling module for refunc.

This module provides a comprehensive exception hierarchy for ML operations,
with retry mechanisms, context-aware error messages, and recovery strategies.
"""

# Core exceptions
from .core import (
    RefuncError,
    ConfigurationError,
    ValidationError,
    OperationError,
    ResourceError,
)

# Data-specific exceptions
from .data import (
    DataError,
    FileNotFoundError,
    UnsupportedFormatError,
    DataValidationError,
    SchemaError,
    CorruptedDataError,
    EmptyDataError,
)

# Model-specific exceptions
from .model import (
    ModelError,
    ModelNotFoundError,
    ModelLoadError,
    ModelSaveError,
    ModelTrainingError,
    ModelPredictionError,
    IncompatibleModelError,
    ModelValidationError,
)

# Retry mechanisms
from .retry import (
    RetryError,
    RetryConfig,
    retry_on_failure,
    RetryableOperation,
)

__all__ = [
    # Core exceptions
    "RefuncError",
    "ConfigurationError",
    "ValidationError",
    "OperationError",
    "ResourceError",
    # Data exceptions
    "DataError",
    "FileNotFoundError",
    "UnsupportedFormatError",
    "DataValidationError",
    "SchemaError",
    "CorruptedDataError",
    "EmptyDataError",
    # Model exceptions
    "ModelError",
    "ModelNotFoundError",
    "ModelLoadError",
    "ModelSaveError",
    "ModelTrainingError",
    "ModelPredictionError",
    "IncompatibleModelError",
    "ModelValidationError",
    # Retry mechanisms
    "RetryError",
    "RetryConfig",
    "retry_on_failure",
    "RetryableOperation",
]