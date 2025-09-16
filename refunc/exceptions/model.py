"""
Model-specific exception classes for the refunc library.

This module provides exceptions related to machine learning model
operations, training, prediction, and serialization.
"""

from typing import Any, Dict, List, Optional, Union
from .core import RefuncError, ResourceError, OperationError


class ModelError(RefuncError):
    """Base class for model-related errors."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a required model file is not found."""
    
    def __init__(self, model_path: str, model_type: Optional[str] = None):
        context = {"model_path": model_path}
        if model_type:
            context["model_type"] = model_type
            
        suggestion = "Check that the model file exists and the path is correct."
        
        super().__init__(
            f"Model not found: {model_path}",
            context=context,
            suggestion=suggestion
        )


class ModelLoadError(ModelError):
    """Raised when a model fails to load."""
    
    def __init__(
        self, 
        model_path: str, 
        original_error: Optional[Exception] = None,
        model_format: Optional[str] = None
    ):
        context = {"model_path": model_path}
        if model_format:
            context["model_format"] = model_format
            
        suggestion = "Ensure the model file is not corrupted and matches the expected format."
        
        super().__init__(
            f"Failed to load model: {model_path}",
            context=context,
            suggestion=suggestion,
            original_error=original_error
        )


class ModelSaveError(ModelError):
    """Raised when a model fails to save."""
    
    def __init__(
        self, 
        model_path: str, 
        original_error: Optional[Exception] = None,
        model_format: Optional[str] = None
    ):
        context = {"model_path": model_path}
        if model_format:
            context["model_format"] = model_format
            
        suggestion = "Check write permissions and available disk space."
        
        super().__init__(
            f"Failed to save model: {model_path}",
            context=context,
            suggestion=suggestion,
            original_error=original_error
        )


class ModelTrainingError(OperationError):
    """Raised when model training fails."""
    
    def __init__(
        self, 
        message: str,
        epoch: Optional[int] = None,
        batch: Optional[int] = None,
        metric_values: Optional[Dict[str, float]] = None,
        original_error: Optional[Exception] = None
    ):
        context = {}
        if epoch is not None:
            context["epoch"] = epoch
        if batch is not None:
            context["batch"] = batch
        if metric_values:
            context["metric_values"] = metric_values
            
        suggestion = "Check training data, model architecture, and hyperparameters."
        
        super().__init__(
            f"Model training failed: {message}",
            context=context,
            suggestion=suggestion,
            original_error=original_error
        )


class ModelPredictionError(OperationError):
    """Raised when model prediction fails."""
    
    def __init__(
        self, 
        message: str,
        input_shape: Optional[tuple] = None,
        expected_shape: Optional[tuple] = None,
        original_error: Optional[Exception] = None
    ):
        context = {}
        if input_shape:
            context["input_shape"] = input_shape
        if expected_shape:
            context["expected_shape"] = expected_shape
            
        suggestion = "Check input data format and preprocessing steps."
        
        super().__init__(
            f"Model prediction failed: {message}",
            context=context,
            suggestion=suggestion,
            original_error=original_error
        )


class IncompatibleModelError(ModelError):
    """Raised when a model is incompatible with the current operation."""
    
    def __init__(
        self, 
        message: str,
        required_version: Optional[str] = None,
        current_version: Optional[str] = None,
        model_type: Optional[str] = None
    ):
        context = {}
        if required_version:
            context["required_version"] = required_version
        if current_version:
            context["current_version"] = current_version
        if model_type:
            context["model_type"] = model_type
            
        suggestion = "Update the model or use a compatible version."
        
        super().__init__(
            f"Incompatible model: {message}",
            context=context,
            suggestion=suggestion
        )


class ModelValidationError(ModelError):
    """Raised when model validation fails."""
    
    def __init__(
        self, 
        message: str,
        validation_metric: Optional[str] = None,
        threshold: Optional[float] = None,
        actual_value: Optional[float] = None
    ):
        context = {}
        if validation_metric:
            context["validation_metric"] = validation_metric
        if threshold is not None:
            context["threshold"] = threshold
        if actual_value is not None:
            context["actual_value"] = actual_value
            
        suggestion = "Review model performance and validation criteria."
        
        super().__init__(
            f"Model validation failed: {message}",
            context=context,
            suggestion=suggestion
        )