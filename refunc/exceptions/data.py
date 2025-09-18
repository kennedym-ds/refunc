"""
Data-specific exception classes for the refunc library.

This module provides exceptions related to data loading, validation,
processing, and format handling.
"""

from typing import Any, Dict, List, Optional, Union
from .core import RefuncError, ValidationError


class DataError(RefuncError):
    """Base class for data-related errors."""
    pass


class FileNotFoundError(DataError):
    """Raised when a required file is not found."""
    
    def __init__(self, file_path: str, search_paths: Optional[List[str]] = None):
        context = {"file_path": file_path}
        if search_paths:
            context["search_paths"] = search_paths
        
        suggestion = "Check that the file path is correct and the file exists."
        if search_paths:
            suggestion += f" Searched in: {', '.join(search_paths)}"
            
        super().__init__(
            f"File not found: {file_path}",
            context=context,
            suggestion=suggestion
        )


class UnsupportedFormatError(DataError):
    """Raised when attempting to load an unsupported file format."""
    
    def __init__(self, file_path: str, supported_formats: List[str]):
        context = {
            "file_path": file_path,
            "supported_formats": supported_formats
        }
        
        suggestion = f"Use one of the supported formats: {', '.join(supported_formats)}"
        
        super().__init__(
            f"Unsupported file format for: {file_path}",
            context=context,
            suggestion=suggestion
        )


class DataValidationError(DataError, ValidationError):
    """Raised when data doesn't meet expected validation criteria."""
    
    def __init__(
        self, 
        message: str, 
        column: Optional[str] = None,
        row_index: Optional[int] = None,
        expected_type: Optional[type] = None,
        actual_type: Optional[type] = None,
        validation_rule: Optional[str] = None
    ):
        context = {}
        if column:
            context["column"] = column
        if row_index is not None:
            context["row_index"] = row_index
        if expected_type:
            context["expected_type"] = expected_type.__name__
        if actual_type:
            context["actual_type"] = actual_type.__name__
        if validation_rule:
            context["validation_rule"] = validation_rule
            
        super().__init__(message, context=context)


class SchemaError(DataError):
    """Raised when data doesn't match expected schema."""
    
    def __init__(
        self, 
        message: str,
        expected_columns: Optional[List[str]] = None,
        actual_columns: Optional[List[str]] = None,
        missing_columns: Optional[List[str]] = None,
        extra_columns: Optional[List[str]] = None
    ):
        context = {}
        if expected_columns:
            context["expected_columns"] = expected_columns
        if actual_columns:
            context["actual_columns"] = actual_columns
        if missing_columns:
            context["missing_columns"] = missing_columns
        if extra_columns:
            context["extra_columns"] = extra_columns
            
        suggestion = "Check that your data has the expected column structure."
        if missing_columns:
            suggestion += f" Missing columns: {', '.join(missing_columns)}"
        if extra_columns:
            suggestion += f" Extra columns: {', '.join(extra_columns)}"
            
        super().__init__(message, context=context, suggestion=suggestion)


class CorruptedDataError(DataError):
    """Raised when data appears to be corrupted or malformed."""
    
    def __init__(self, file_path: str, details: Optional[str] = None):
        context = {"file_path": file_path}
        if details:
            context["details"] = details
            
        suggestion = "Try re-downloading or regenerating the data file."
        
        super().__init__(
            f"Data appears to be corrupted: {file_path}",
            context=context,
            suggestion=suggestion
        )


class EmptyDataError(DataError):
    """Raised when expected data is empty."""
    
    def __init__(self, source: str, expected_min_size: Optional[int] = None):
        context = {"source": source}
        if expected_min_size:
            context["expected_min_size"] = expected_min_size
            
        suggestion = "Ensure the data source contains the expected data."
        
        super().__init__(
            f"Data source is empty: {source}",
            context=context,
            suggestion=suggestion
        )