"""
Core exception classes for the refunc library.

This module provides the base exception hierarchy for ML operations,
with context-aware error messages and structured error handling.
"""

from typing import Any, Dict, Optional, Union
import traceback
import sys
from datetime import datetime


class RefuncError(Exception):
    """Base exception class for all refunc errors.
    
    Provides enhanced error reporting with context information,
    timestamps, and optional suggestions for resolution.
    """
    
    def __init__(
        self, 
        message: str, 
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        self.message = message
        self.context = context or {}
        self.suggestion = suggestion
        self.original_error = original_error
        self.timestamp = datetime.now()
        self.traceback_info = traceback.format_exc() if original_error else None
        
        # Build comprehensive error message
        full_message = self._build_error_message()
        super().__init__(full_message)
    
    def _build_error_message(self) -> str:
        """Build a comprehensive error message with context."""
        lines = [f"RefuncError: {self.message}"]
        
        if self.context:
            lines.append("Context:")
            for key, value in self.context.items():
                # Handle different value types appropriately
                if isinstance(value, (list, dict)):
                    lines.append(f"  {key}: {value}")
                else:
                    lines.append(f"  {key}: {value}")
        
        if self.suggestion:
            lines.append(f"Suggestion: {self.suggestion}")
        
        if self.original_error:
            lines.append(f"Original error: {type(self.original_error).__name__}: {self.original_error}")
        
        lines.append(f"Timestamp: {self.timestamp.isoformat()}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "error_type": type(self).__name__,
            "message": self.message,
            "context": self.context,
            "suggestion": self.suggestion,
            "original_error": str(self.original_error) if self.original_error else None,
            "timestamp": self.timestamp.isoformat(),
            "traceback": self.traceback_info
        }


class ConfigurationError(RefuncError):
    """Raised when there's an issue with configuration or settings."""
    pass


class ValidationError(RefuncError):
    """Raised when input validation fails."""
    pass


class OperationError(RefuncError):
    """Raised when an operation fails during execution."""
    pass


class ResourceError(RefuncError):
    """Raised when a required resource is unavailable or insufficient."""
    pass