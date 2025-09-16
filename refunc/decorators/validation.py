"""
Validation decorators for input/output checking and data validation.

This module provides decorators for validating function inputs and outputs,
including type checking, value constraints, data shape validation, and custom validators.
"""

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Type, get_type_hints
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from ..exceptions import DataError, RefuncError


F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class ValidationResult:
    """Container for validation results."""
    
    function_name: str
    input_valid: bool
    output_valid: bool
    input_errors: List[str] = field(default_factory=list)
    output_errors: List[str] = field(default_factory=list)
    validation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ValidatorBase:
    """Base class for validators."""
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate a value. Should return True if valid, False otherwise."""
        raise NotImplementedError
    
    def get_error_message(self, value: Any, context: Optional[Dict[str, Any]] = None) -> str:
        """Get error message for invalid value."""
        return f"Validation failed for value: {value}"


class TypeValidator(ValidatorBase):
    """Validator for type checking."""
    
    def __init__(self, expected_type: Union[Type, tuple]):
        self.expected_type = expected_type
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> bool:
        return isinstance(value, self.expected_type)
    
    def get_error_message(self, value: Any, context: Optional[Dict[str, Any]] = None) -> str:
        actual_type = type(value).__name__
        expected_names = (
            self.expected_type.__name__ 
            if not isinstance(self.expected_type, tuple)
            else ', '.join(t.__name__ for t in self.expected_type)
        )
        return f"Expected type {expected_names}, got {actual_type}"


class RangeValidator(ValidatorBase):
    """Validator for numeric ranges."""
    
    def __init__(self, min_val: Optional[float] = None, max_val: Optional[float] = None):
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> bool:
        try:
            num_val = float(value)
            if self.min_val is not None and num_val < self.min_val:
                return False
            if self.max_val is not None and num_val > self.max_val:
                return False
            return True
        except (ValueError, TypeError):
            return False
    
    def get_error_message(self, value: Any, context: Optional[Dict[str, Any]] = None) -> str:
        range_desc = []
        if self.min_val is not None:
            range_desc.append(f"min: {self.min_val}")
        if self.max_val is not None:
            range_desc.append(f"max: {self.max_val}")
        return f"Value {value} not in range ({', '.join(range_desc)})"


class ShapeValidator(ValidatorBase):
    """Validator for array/dataframe shapes."""
    
    def __init__(self, expected_shape: Optional[tuple] = None, min_dims: Optional[int] = None, max_dims: Optional[int] = None):
        self.expected_shape = expected_shape
        self.min_dims = min_dims
        self.max_dims = max_dims
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> bool:
        # Get shape based on type
        shape = None
        if hasattr(value, 'shape'):
            shape = value.shape
        elif isinstance(value, (list, tuple)):
            shape = (len(value),)
        else:
            return False
        
        # Check exact shape if specified
        if self.expected_shape is not None:
            return shape == self.expected_shape
        
        # Check dimensions
        ndims = len(shape)
        if self.min_dims is not None and ndims < self.min_dims:
            return False
        if self.max_dims is not None and ndims > self.max_dims:
            return False
        
        return True
    
    def get_error_message(self, value: Any, context: Optional[Dict[str, Any]] = None) -> str:
        if hasattr(value, 'shape'):
            actual_shape = value.shape
        elif isinstance(value, (list, tuple)):
            actual_shape = (len(value),)
        else:
            actual_shape = "unknown"
        
        if self.expected_shape is not None:
            return f"Expected shape {self.expected_shape}, got {actual_shape}"
        else:
            return f"Shape {actual_shape} does not meet dimension constraints"


class DataFrameValidator(ValidatorBase):
    """Validator for pandas DataFrames."""
    
    def __init__(
        self,
        required_columns: Optional[List[str]] = None,
        min_rows: Optional[int] = None,
        max_rows: Optional[int] = None,
        column_types: Optional[Dict[str, type]] = None
    ):
        self.required_columns = required_columns or []
        self.min_rows = min_rows
        self.max_rows = max_rows
        self.column_types = column_types or {}
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> bool:
        if not isinstance(value, pd.DataFrame):
            return False
        
        # Check required columns
        if self.required_columns:
            missing_cols = set(self.required_columns) - set(value.columns)
            if missing_cols:
                return False
        
        # Check row count
        if self.min_rows is not None and len(value) < self.min_rows:
            return False
        if self.max_rows is not None and len(value) > self.max_rows:
            return False
        
        # Check column types
        for col, expected_type in self.column_types.items():
            if col in value.columns:
                if not value[col].dtype == expected_type:
                    return False
        
        return True
    
    def get_error_message(self, value: Any, context: Optional[Dict[str, Any]] = None) -> str:
        if not isinstance(value, pd.DataFrame):
            return f"Expected DataFrame, got {type(value).__name__}"
        
        errors = []
        
        # Check required columns
        if self.required_columns:
            missing_cols = set(self.required_columns) - set(value.columns)
            if missing_cols:
                errors.append(f"Missing columns: {missing_cols}")
        
        # Check row count
        if self.min_rows is not None and len(value) < self.min_rows:
            errors.append(f"Too few rows: {len(value)} < {self.min_rows}")
        if self.max_rows is not None and len(value) > self.max_rows:
            errors.append(f"Too many rows: {len(value)} > {self.max_rows}")
        
        # Check column types
        for col, expected_type in self.column_types.items():
            if col in value.columns:
                if not value[col].dtype == expected_type:
                    errors.append(f"Column {col}: expected {expected_type}, got {value[col].dtype}")
        
        return "; ".join(errors) if errors else "DataFrame validation failed"


class CustomValidator(ValidatorBase):
    """Validator using custom validation function."""
    
    def __init__(self, validation_func: Callable[[Any], bool], error_message: str = "Custom validation failed"):
        self.validation_func = validation_func
        self.error_message = error_message
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> bool:
        try:
            return self.validation_func(value)
        except Exception:
            return False
    
    def get_error_message(self, value: Any, context: Optional[Dict[str, Any]] = None) -> str:
        return self.error_message


def validate_inputs(
    validators: Dict[str, Union[ValidatorBase, List[ValidatorBase]]],
    raise_on_error: bool = True,
    return_result: bool = False
) -> Callable[[F], F]:
    """
    Decorator to validate function inputs.
    
    Args:
        validators: Dict mapping parameter names to validators
        raise_on_error: Whether to raise exception on validation error
        return_result: Whether to return ValidationResult
    
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        sig = inspect.signature(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            # Bind arguments to parameters
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            input_errors = []
            
            # Validate each parameter
            for param_name, param_validators in validators.items():
                if param_name not in bound_args.arguments:
                    continue
                
                value = bound_args.arguments[param_name]
                
                # Ensure validators is a list
                if not isinstance(param_validators, list):
                    param_validators = [param_validators]
                
                # Run all validators for this parameter
                for validator in param_validators:
                    if not validator.validate(value):
                        error_msg = f"Parameter '{param_name}': {validator.get_error_message(value)}"
                        input_errors.append(error_msg)
            
            validation_time = time.time() - start_time
            
            # Handle validation errors
            if input_errors:
                if raise_on_error:
                    raise DataError(
                        f"Input validation failed for {func.__name__}",
                        context={"errors": input_errors},
                        suggestion="Check function arguments against validation rules"
                    )
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Return with validation result if requested
            if return_result:
                validation_result = ValidationResult(
                    function_name=func.__name__,
                    input_valid=len(input_errors) == 0,
                    output_valid=True,  # Only validating inputs here
                    input_errors=input_errors,
                    validation_time=validation_time,
                    metadata={"validators_count": len(validators)}
                )
                return result, validation_result
            
            return result
        
        return wrapper  # type: ignore
    return decorator


def validate_outputs(
    validators: Union[ValidatorBase, List[ValidatorBase]],
    raise_on_error: bool = True,
    return_result: bool = False
) -> Callable[[F], F]:
    """
    Decorator to validate function outputs.
    
    Args:
        validators: Validator(s) for the return value
        raise_on_error: Whether to raise exception on validation error
        return_result: Whether to return ValidationResult
    
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            # Execute function
            result = func(*args, **kwargs)
            
            start_time = time.time()
            output_errors = []
            
            # Ensure validators is a list
            if not isinstance(validators, list):
                validator_list = [validators]
            else:
                validator_list = validators
            
            # Validate output
            for validator in validator_list:
                if not validator.validate(result):
                    error_msg = validator.get_error_message(result)
                    output_errors.append(error_msg)
            
            validation_time = time.time() - start_time
            
            # Handle validation errors
            if output_errors:
                if raise_on_error:
                    raise DataError(
                        f"Output validation failed for {func.__name__}",
                        context={"errors": output_errors},
                        suggestion="Check function return value against validation rules"
                    )
            
            # Return with validation result if requested
            if return_result:
                validation_result = ValidationResult(
                    function_name=func.__name__,
                    input_valid=True,  # Only validating outputs here
                    output_valid=len(output_errors) == 0,
                    output_errors=output_errors,
                    validation_time=validation_time,
                    metadata={"validators_count": len(validator_list)}
                )
                return result, validation_result
            
            return result
        
        return wrapper  # type: ignore
    return decorator


def validate_types(
    strict_mode: bool = True,
    check_return: bool = True,
    raise_on_error: bool = True
) -> Callable[[F], F]:
    """
    Decorator to validate function types based on type hints.
    
    Args:
        strict_mode: Whether to enforce strict type checking
        check_return: Whether to check return type
        raise_on_error: Whether to raise exception on type error
    
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        try:
            type_hints = get_type_hints(func)
        except (NameError, AttributeError):
            # No type hints available
            if strict_mode:
                raise RefuncError(f"No type hints found for function {func.__name__}")
            type_hints = {}
        
        sig = inspect.signature(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not type_hints:
                return func(*args, **kwargs)
            
            # Bind arguments to parameters
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            errors = []
            
            # Check input types
            for param_name, value in bound_args.arguments.items():
                if param_name in type_hints:
                    expected_type = type_hints[param_name]
                    
                    # Skip checking for complex generic types in non-strict mode
                    if not strict_mode and hasattr(expected_type, '__origin__'):
                        continue
                    
                    try:
                        if not isinstance(value, expected_type):
                            errors.append(f"Parameter '{param_name}': expected {expected_type}, got {type(value)}")
                    except TypeError:
                        # Handle complex types that can't be checked with isinstance
                        if strict_mode:
                            errors.append(f"Parameter '{param_name}': complex type checking failed")
            
            if errors and raise_on_error:
                raise DataError(
                    f"Type validation failed for {func.__name__}",
                    context={"errors": errors},
                    suggestion="Check function arguments match type hints"
                )
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Check return type
            if check_return and 'return' in type_hints:
                expected_return_type = type_hints['return']
                
                try:
                    if not isinstance(result, expected_return_type):
                        error_msg = f"Return value: expected {expected_return_type}, got {type(result)}"
                        if raise_on_error:
                            raise DataError(
                                f"Return type validation failed for {func.__name__}",
                                context={"error": error_msg},
                                suggestion="Check function return value matches type hint"
                            )
                except TypeError:
                    # Handle complex return types
                    if strict_mode and raise_on_error:
                        raise DataError(
                            f"Return type validation failed for {func.__name__}",
                            context={"error": "Complex return type checking failed"},
                            suggestion="Simplify return type hint or disable strict mode"
                        )
            
            return result
        
        return wrapper  # type: ignore
    return decorator


def validate_data_schema(
    input_schema: Optional[Dict[str, Any]] = None,
    output_schema: Optional[Dict[str, Any]] = None,
    raise_on_error: bool = True
) -> Callable[[F], F]:
    """
    Decorator to validate data against schemas.
    
    Args:
        input_schema: Schema for input validation (parameter name -> schema)
        output_schema: Schema for output validation  
        raise_on_error: Whether to raise exception on validation error
    
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        sig = inspect.signature(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            errors = []
            
            # Validate inputs
            if input_schema:
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                for param_name, schema in input_schema.items():
                    if param_name in bound_args.arguments:
                        value = bound_args.arguments[param_name]
                        validation_errors = _validate_against_schema(value, schema, f"Input '{param_name}'")
                        errors.extend(validation_errors)
            
            if errors and raise_on_error:
                raise DataError(
                    f"Input schema validation failed for {func.__name__}",
                    context={"errors": errors},
                    suggestion="Check input data matches expected schema"
                )
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Validate output
            if output_schema:
                validation_errors = _validate_against_schema(result, output_schema, "Output")
                if validation_errors and raise_on_error:
                    raise DataError(
                        f"Output schema validation failed for {func.__name__}",
                        context={"errors": validation_errors},
                        suggestion="Check output data matches expected schema"
                    )
            
            return result
        
        return wrapper  # type: ignore
    return decorator


def _validate_against_schema(value: Any, schema: Dict[str, Any], context: str) -> List[str]:
    """Validate a value against a schema. Returns list of error messages."""
    errors = []
    
    # Type validation
    if 'type' in schema:
        expected_type = schema['type']
        if not isinstance(value, expected_type):
            errors.append(f"{context}: expected type {expected_type.__name__}, got {type(value).__name__}")
            return errors  # Return early if type is wrong
    
    # DataFrame specific validations
    if isinstance(value, pd.DataFrame):
        if 'columns' in schema:
            expected_cols = set(schema['columns'])
            actual_cols = set(value.columns)
            missing_cols = expected_cols - actual_cols
            if missing_cols:
                errors.append(f"{context}: missing columns {missing_cols}")
        
        if 'min_rows' in schema and len(value) < schema['min_rows']:
            errors.append(f"{context}: too few rows ({len(value)} < {schema['min_rows']})")
        
        if 'max_rows' in schema and len(value) > schema['max_rows']:
            errors.append(f"{context}: too many rows ({len(value)} > {schema['max_rows']})")
    
    # Array specific validations
    if hasattr(value, 'shape'):
        if 'shape' in schema:
            expected_shape = schema['shape']
            if value.shape != expected_shape:
                errors.append(f"{context}: expected shape {expected_shape}, got {value.shape}")
        
        if 'min_dims' in schema and len(value.shape) < schema['min_dims']:
            errors.append(f"{context}: too few dimensions ({len(value.shape)} < {schema['min_dims']})")
    
    # Numeric range validations
    if isinstance(value, (int, float)):
        if 'min_value' in schema and value < schema['min_value']:
            errors.append(f"{context}: value too small ({value} < {schema['min_value']})")
        
        if 'max_value' in schema and value > schema['max_value']:
            errors.append(f"{context}: value too large ({value} > {schema['max_value']})")
    
    return errors


# Convenience aliases and factory functions
def type_check(expected_type: Union[Type, tuple]) -> TypeValidator:
    """Create a type validator."""
    return TypeValidator(expected_type)


def range_check(min_val: Optional[float] = None, max_val: Optional[float] = None) -> RangeValidator:
    """Create a range validator."""
    return RangeValidator(min_val, max_val)


def shape_check(expected_shape: Optional[tuple] = None, min_dims: Optional[int] = None, max_dims: Optional[int] = None) -> ShapeValidator:
    """Create a shape validator.""" 
    return ShapeValidator(expected_shape, min_dims, max_dims)


def dataframe_check(
    required_columns: Optional[List[str]] = None,
    min_rows: Optional[int] = None,
    max_rows: Optional[int] = None,
    column_types: Optional[Dict[str, type]] = None
) -> DataFrameValidator:
    """Create a DataFrame validator."""
    return DataFrameValidator(required_columns, min_rows, max_rows, column_types)


def custom_check(validation_func: Callable[[Any], bool], error_message: str = "Custom validation failed") -> CustomValidator:
    """Create a custom validator."""
    return CustomValidator(validation_func, error_message)