"""
Comprehensive tests for the refunc.exceptions module.

This test suite covers:
- Core exception hierarchy and behaviors
- Data-specific exceptions and error conditions
- Model-specific exceptions and edge cases
- Retry mechanisms and failure handling
- Exception chaining and context preservation
- Custom error messages and debugging information
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

# Import all exception classes
from refunc.exceptions import (
    # Core exceptions
    RefuncError,
    ConfigurationError,
    ValidationError,
    OperationError,
    ResourceError,
    
    # Data exceptions
    DataError,
    FileNotFoundError,
    UnsupportedFormatError,
    DataValidationError,
    SchemaError,
    CorruptedDataError,
    EmptyDataError,
    
    # Model exceptions
    ModelError,
    ModelNotFoundError,
    ModelLoadError,
    ModelSaveError,
    ModelTrainingError,
    ModelPredictionError,
    IncompatibleModelError,
    ModelValidationError,
    
    # Retry mechanisms
    RetryError,
    RetryConfig,
    retry_on_failure,
    RetryableOperation,
)


class TestCoreExceptions:
    """Test core exception classes and their behaviors."""
    
    def test_refunc_error_base_class(self):
        """Test RefuncError as the base exception class."""
        error = RefuncError("Test error message")
        assert "Test error message" in str(error)
        assert isinstance(error, Exception)
        
    def test_refunc_error_with_context(self):
        """Test RefuncError with additional context."""
        context = {"operation": "test_op", "file": "test.txt"}
        error = RefuncError("Test error", context=context)
        assert error.context == context
        assert "test_op" in str(error)
        
    def test_refunc_error_with_suggestion(self):
        """Test RefuncError with suggestion."""
        suggestion = "Try checking the file path"
        error = RefuncError("Test error", suggestion=suggestion)
        assert error.suggestion == suggestion
        assert suggestion in str(error)
        
    def test_refunc_error_with_original_error(self):
        """Test RefuncError with original error."""
        original = ValueError("Original error")
        error = RefuncError("Test error", original_error=original)
        assert error.original_error == original
        assert "ValueError" in str(error)
        
    def test_configuration_error(self):
        """Test ConfigurationError for configuration issues."""
        error = ConfigurationError("Invalid configuration")
        assert isinstance(error, RefuncError)
        assert "Invalid configuration" in str(error)
        
    def test_validation_error(self):
        """Test ValidationError for validation failures."""
        error = ValidationError("Validation failed")
        assert isinstance(error, RefuncError)
        assert "Validation failed" in str(error)
        
    def test_operation_error(self):
        """Test OperationError for operation failures."""
        error = OperationError("Operation failed")
        assert isinstance(error, RefuncError)
        assert "Operation failed" in str(error)
        
    def test_resource_error(self):
        """Test ResourceError for resource access issues."""
        error = ResourceError("Resource unavailable")
        assert isinstance(error, RefuncError)
        assert "Resource unavailable" in str(error)
        
    def test_refunc_error_to_dict(self):
        """Test RefuncError serialization to dictionary."""
        context = {"key": "value"}
        error = RefuncError("Test error", context=context, suggestion="Try this")
        error_dict = error.to_dict()
        
        assert error_dict["error_type"] == "RefuncError"
        assert error_dict["message"] == "Test error"
        assert error_dict["context"] == context
        assert error_dict["suggestion"] == "Try this"
        assert "timestamp" in error_dict


class TestDataExceptions:
    """Test data-specific exception classes."""
    
    def test_data_error_base(self):
        """Test DataError as base for data-related exceptions."""
        error = DataError("Data processing failed")
        assert isinstance(error, RefuncError)
        
    def test_file_not_found_error(self):
        """Test FileNotFoundError for missing files."""
        file_path = "/path/to/file.csv"
        error = FileNotFoundError(file_path)
        assert isinstance(error, DataError)
        assert file_path in str(error)
        assert error.context["file_path"] == file_path
        
    def test_file_not_found_error_with_search_paths(self):
        """Test FileNotFoundError with search paths."""
        file_path = "file.csv"
        search_paths = ["/path1", "/path2"]
        error = FileNotFoundError(file_path, search_paths=search_paths)
        assert error.context["search_paths"] == search_paths
        assert "path1" in str(error)
        
    def test_unsupported_format_error(self):
        """Test UnsupportedFormatError for format issues."""
        file_path = "file.xyz"
        supported_formats = ["csv", "json", "parquet"]
        error = UnsupportedFormatError(file_path, supported_formats)
        assert isinstance(error, DataError)
        assert file_path in str(error)
        assert error.context["supported_formats"] == supported_formats
        
    def test_data_validation_error(self):
        """Test DataValidationError for data validation failures."""
        error = DataValidationError("Invalid age", column="age", expected_type=int, actual_type=str)
        assert isinstance(error, ValidationError)
        assert isinstance(error, DataError)
        assert error.context["column"] == "age"
        assert error.context["expected_type"] == "int"
        assert error.context["actual_type"] == "str"
        
    def test_data_validation_error_with_row_index(self):
        """Test DataValidationError with row index."""
        error = DataValidationError("Invalid data", column="age", row_index=42)
        assert error.context["row_index"] == 42
        
    def test_schema_error(self):
        """Test SchemaError for schema mismatches."""
        expected_columns = ["name", "age", "email"]
        actual_columns = ["name", "age"]
        missing_columns = ["email"]
        
        error = SchemaError(
            "Schema mismatch", 
            expected_columns=expected_columns,
            actual_columns=actual_columns,
            missing_columns=missing_columns
        )
        assert isinstance(error, DataError)
        assert error.context["expected_columns"] == expected_columns
        assert error.context["missing_columns"] == missing_columns
        assert "email" in str(error)
        
    def test_corrupted_data_error(self):
        """Test CorruptedDataError for corrupted data."""
        file_path = "/path/to/corrupted.csv"
        details = "Checksum mismatch"
        error = CorruptedDataError(file_path, details=details)
        assert isinstance(error, DataError)
        assert error.context["file_path"] == file_path
        assert error.context["details"] == details
        
    def test_empty_data_error(self):
        """Test EmptyDataError for empty datasets."""
        source = "input.csv"
        expected_min_size = 100
        error = EmptyDataError(source, expected_min_size=expected_min_size)
        assert isinstance(error, DataError)
        assert error.context["source"] == source
        assert error.context["expected_min_size"] == expected_min_size


class TestModelExceptions:
    """Test model-specific exception classes."""
    
    def test_model_error_base(self):
        """Test ModelError as base for model-related exceptions."""
        error = ModelError("Model operation failed")
        assert isinstance(error, RefuncError)
        
    def test_model_not_found_error(self):
        """Test ModelNotFoundError for missing models."""
        model_path = "/models/model.pkl"
        model_type = "RandomForest"
        error = ModelNotFoundError(model_path, model_type=model_type)
        assert isinstance(error, ModelError)
        assert error.context["model_path"] == model_path
        assert error.context["model_type"] == model_type
        
    def test_model_load_error(self):
        """Test ModelLoadError for model loading failures."""
        model_path = "/models/model.pkl"
        original_error = ValueError("Pickle error")
        model_format = "pickle"
        
        error = ModelLoadError(
            model_path, 
            original_error=original_error,
            model_format=model_format
        )
        assert isinstance(error, ModelError)
        assert error.context["model_path"] == model_path
        assert error.context["model_format"] == model_format
        assert error.original_error == original_error
        
    def test_model_save_error(self):
        """Test ModelSaveError for model saving failures."""
        model_path = "/models/model.pkl"
        original_error = PermissionError("Permission denied")
        
        error = ModelSaveError(model_path, original_error=original_error)
        assert isinstance(error, ModelError)
        assert error.context["model_path"] == model_path
        assert error.original_error == original_error
        
    def test_model_training_error(self):
        """Test ModelTrainingError for training failures."""
        epoch = 5
        batch = 32
        metric_values = {"loss": 0.5, "accuracy": 0.8}
        
        error = ModelTrainingError(
            "Training diverged",
            epoch=epoch,
            batch=batch,
            metric_values=metric_values
        )
        assert isinstance(error, OperationError)
        assert error.context["epoch"] == epoch
        assert error.context["batch"] == batch
        assert error.context["metric_values"] == metric_values
        
    def test_model_prediction_error(self):
        """Test ModelPredictionError for prediction failures."""
        input_shape = (10, 5)
        expected_shape = (10, 3)
        
        error = ModelPredictionError(
            "Shape mismatch",
            input_shape=input_shape,
            expected_shape=expected_shape
        )
        assert isinstance(error, OperationError)
        assert error.context["input_shape"] == input_shape
        assert error.context["expected_shape"] == expected_shape
        
    def test_incompatible_model_error(self):
        """Test IncompatibleModelError for model compatibility issues."""
        required_version = "2.0"
        current_version = "1.0"
        model_type = "sklearn"
        
        error = IncompatibleModelError(
            "Version mismatch",
            required_version=required_version,
            current_version=current_version,
            model_type=model_type
        )
        assert isinstance(error, ModelError)
        assert error.context["required_version"] == required_version
        assert error.context["current_version"] == current_version
        assert error.context["model_type"] == model_type
        
    def test_model_validation_error(self):
        """Test ModelValidationError for model validation failures."""
        validation_metric = "accuracy"
        threshold = 0.8
        actual_value = 0.6
        
        error = ModelValidationError(
            "Validation failed",
            validation_metric=validation_metric,
            threshold=threshold,
            actual_value=actual_value
        )
        assert isinstance(error, ModelError)
        assert error.context["validation_metric"] == validation_metric
        assert error.context["threshold"] == threshold
        assert error.context["actual_value"] == actual_value


class TestRetryMechanisms:
    """Test retry mechanisms and configurations."""
    
    def test_retry_error(self):
        """Test RetryError for retry failures."""
        original_error = ValueError("Original error")
        attempts = 3
        total_time = 5.5
        
        error = RetryError(
            original_error=original_error,
            attempts=attempts,
            total_time=total_time,
            operation_name="test_operation"
        )
        assert isinstance(error, OperationError)
        assert error.context["attempts"] == attempts
        assert error.context["total_time_seconds"] == 5.5
        assert error.context["operation_name"] == "test_operation"
        assert error.original_error == original_error
        
    def test_retry_config_creation(self):
        """Test RetryConfig creation and validation."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=True
        )
        assert config.max_attempts == 5
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        
    def test_retry_config_defaults(self):
        """Test RetryConfig with default values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        
    def test_retry_config_should_retry(self):
        """Test RetryConfig.should_retry logic."""
        config = RetryConfig(max_attempts=3, retryable_exceptions=[ValueError])
        
        # Should retry ValueError
        assert config.should_retry(ValueError("test"), 1) is True
        assert config.should_retry(ValueError("test"), 2) is True
        
        # Should not retry after max attempts
        assert config.should_retry(ValueError("test"), 3) is False
        
        # Test with non-retryable exceptions
        config_with_non_retryable = RetryConfig(
            max_attempts=3, 
            retryable_exceptions=[Exception],
            non_retryable_exceptions=[TypeError]
        )
        assert config_with_non_retryable.should_retry(TypeError("test"), 1) is False
        
    def test_retry_config_calculate_delay(self):
        """Test RetryConfig.calculate_delay calculation."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
        
        assert config.calculate_delay(1) == 1.0  # 1.0 * 2^0
        assert config.calculate_delay(2) == 2.0  # 1.0 * 2^1
        assert config.calculate_delay(3) == 4.0  # 1.0 * 2^2
        
    def test_retry_config_calculate_delay_with_max(self):
        """Test RetryConfig.calculate_delay with max_delay."""
        config = RetryConfig(base_delay=10.0, max_delay=15.0, exponential_base=2.0, jitter=False)
        
        assert config.calculate_delay(1) == 10.0
        assert config.calculate_delay(2) == 15.0  # Capped at max_delay
        assert config.calculate_delay(3) == 15.0  # Capped at max_delay
        
    def test_retry_on_failure_decorator_success(self):
        """Test retry_on_failure decorator with successful operation."""
        call_count = 0
        
        @retry_on_failure(max_attempts=3)
        def successful_operation():
            nonlocal call_count
            call_count += 1
            return "success"
            
        result = successful_operation()
        assert result == "success"
        assert call_count == 1
        
    def test_retry_on_failure_decorator_eventual_success(self):
        """Test retry_on_failure decorator with eventual success."""
        call_count = 0
        
        @retry_on_failure(max_attempts=3, base_delay=0.01)
        def eventually_successful_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
            
        result = eventually_successful_operation()
        assert result == "success"
        assert call_count == 3
        
    def test_retry_on_failure_decorator_exhausted(self):
        """Test retry_on_failure decorator when attempts are exhausted."""
        call_count = 0
        
        @retry_on_failure(max_attempts=2, base_delay=0.01)
        def always_failing_operation():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")
            
        with pytest.raises(RetryError) as exc_info:
            always_failing_operation()
            
        assert call_count == 2
        assert exc_info.value.context["attempts"] == 2
        assert isinstance(exc_info.value.original_error, ValueError)
        
    def test_retry_on_failure_with_specific_exceptions(self):
        """Test retry_on_failure with specific exception types."""
        call_count = 0
        
        @retry_on_failure(
            max_attempts=3, 
            base_delay=0.01, 
            retryable_exceptions=[ValueError]
        )
        def operation_with_specific_error():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Retryable error")
            elif call_count == 2:
                raise TypeError("Non-retryable error")
            return "success"
            
        with pytest.raises(TypeError):
            operation_with_specific_error()
            
        assert call_count == 2  # Should stop at TypeError
        
    def test_retry_on_failure_with_callback(self):
        """Test retry_on_failure with callback function."""
        call_count = 0
        retry_calls = []
        
        def on_retry_callback(exception, attempt):
            retry_calls.append((str(exception), attempt))
        
        @retry_on_failure(
            max_attempts=3, 
            base_delay=0.01,
            on_retry=on_retry_callback
        )
        def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Failure {call_count}")
            return "success"
            
        result = failing_operation()
        assert result == "success"
        assert len(retry_calls) == 2
        assert "Failure 1" in retry_calls[0][0]
        assert retry_calls[0][1] == 1
        
    def test_retryable_operation_context_manager(self):
        """Test RetryableOperation as context manager."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        operation = RetryableOperation(config, "test_operation")
        
        # Test using the execute method for automatic retries
        attempt_count = 0
        def operation_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ValueError("Temporary failure")
            return "success"
        
        result = operation.execute(operation_func)
        assert result == "success"
        assert attempt_count == 2  # Should succeed on second attempt


class TestExceptionChaining:
    """Test exception chaining and context preservation."""
    
    def test_exception_chaining(self):
        """Test that exceptions properly chain."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise DataError("Data processing failed") from e
        except DataError as e:
            assert isinstance(e.__cause__, ValueError)
            assert str(e.__cause__) == "Original error"
            
    def test_context_preservation(self):
        """Test that context is preserved across exception handling."""
        context = {"file": "test.csv", "line": 42}
        
        try:
            raise RefuncError("Validation failed", context=context)
        except RefuncError as e:
            assert e.context == context
            
    def test_nested_exception_handling(self):
        """Test nested exception handling scenarios."""
        def level_3():
            raise ValueError("Level 3 error")
            
        def level_2():
            try:
                level_3()
            except ValueError as e:
                raise ModelTrainingError("Training failed at level 2") from e
                
        def level_1():
            try:
                level_2()
            except ModelTrainingError as e:
                raise OperationError("Operation failed at level 1") from e
                
        with pytest.raises(OperationError) as exc_info:
            level_1()
            
        # Check the exception chain
        assert isinstance(exc_info.value.__cause__, ModelTrainingError)
        assert isinstance(exc_info.value.__cause__.__cause__, ValueError)


class TestExceptionMessages:
    """Test exception message formatting and debugging information."""
    
    def test_detailed_error_messages(self):
        """Test that exceptions provide detailed error messages."""
        error = DataValidationError(
            "Value out of range",
            column="age",
            expected_type=int,
            actual_type=str
        )
        
        error_str = str(error)
        assert "age" in error_str
        assert "int" in error_str
        assert "str" in error_str
        
    def test_error_message_with_suggestions(self):
        """Test error messages with helpful suggestions."""
        error = UnsupportedFormatError(
            "file.xml",
            ["csv", "json", "parquet"]
        )
        
        error_str = str(error)
        assert "csv" in error_str
        assert "json" in error_str
        assert "parquet" in error_str
        
    def test_error_repr_consistency(self):
        """Test that exception repr is consistent."""
        error = ModelLoadError("/model.pkl")
        repr_str = repr(error)
        
        assert "ModelLoadError" in repr_str
        
    def test_timestamp_in_error(self):
        """Test that errors include timestamp information."""
        error = RefuncError("Test error")
        assert error.timestamp is not None
        assert "Timestamp" in str(error)


class TestExceptionUtilities:
    """Test utility functions and helper methods."""
    
    def test_exception_serialization(self):
        """Test exception serialization for logging/storage."""
        error = ModelTrainingError(
            "Training failed", 
            epoch=5, 
            metric_values={"loss": 0.5}
        )
        
        error_dict = error.to_dict()
        assert error_dict["error_type"] == "ModelTrainingError"
        assert "Training failed" in error_dict["message"]
        assert error_dict["context"]["epoch"] == 5
        assert error_dict["context"]["metric_values"] == {"loss": 0.5}
        assert "timestamp" in error_dict


class TestExceptionIntegration:
    """Test exception integration with other refunc components."""
    
    def test_exceptions_with_file_operations(self, temp_dir):
        """Test exceptions in file operation contexts."""
        non_existent_file = str(temp_dir / "does_not_exist.csv")
        
        error = FileNotFoundError(non_existent_file)
        assert isinstance(error, DataError)
        assert non_existent_file in str(error)
                
    def test_exception_logging_integration(self, capture_logs):
        """Test that exceptions integrate properly with logging."""
        import logging
        logger = logging.getLogger("refunc.test")
        
        try:
            raise DataError("Test error for logging")
        except DataError as e:
            logger.error("Caught exception: %s", e, exc_info=True)
            
        log_output = capture_logs.getvalue()
        assert "Test error for logging" in log_output


# Integration tests that cover complex scenarios
class TestComplexExceptionScenarios:
    """Test complex real-world exception scenarios."""
    
    def test_model_pipeline_failure_scenario(self):
        """Test a complex model pipeline failure scenario."""
        def simulate_model_pipeline():
            # Simulate data loading failure
            raise DataError("Data loading failed")
            
        def simulate_model_training():
            try:
                simulate_model_pipeline()
            except DataError as e:
                raise ModelTrainingError("Cannot train without data") from e
                
        def simulate_full_workflow():
            try:
                simulate_model_training()
            except ModelTrainingError as e:
                raise OperationError("Workflow failed") from e
                
        with pytest.raises(OperationError) as exc_info:
            simulate_full_workflow()
            
        # Verify the exception chain
        assert isinstance(exc_info.value.__cause__, ModelTrainingError)
        assert isinstance(exc_info.value.__cause__.__cause__, DataError)
        
    def test_retry_with_different_error_types(self):
        """Test retry mechanism with different types of errors."""
        attempt_count = 0
        
        @retry_on_failure(max_attempts=4, base_delay=0.01)
        def mixed_error_operation():
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count == 1:
                raise DataError("Data not ready")
            elif attempt_count == 2:
                raise ResourceError("Resource busy")
            elif attempt_count == 3:
                raise ModelError("Model not loaded")
            else:
                return "success"
                
        result = mixed_error_operation()
        assert result == "success"
        assert attempt_count == 4
        
    def test_exception_recovery_scenario(self):
        """Test exception recovery scenarios."""
        recovery_attempts = 0
        
        def attempt_operation_with_recovery():
            nonlocal recovery_attempts
            recovery_attempts += 1
            
            if recovery_attempts <= 2:
                if recovery_attempts == 1:
                    raise DataError("Initial data error")
                else:
                    raise ResourceError("Resource temporarily unavailable")
            
            return "recovered successfully"
            
        # Simulate recovery logic
        max_recovery_attempts = 3
        result = None
        for attempt in range(max_recovery_attempts):
            try:
                result = attempt_operation_with_recovery()
                break
            except (DataError, ResourceError) as e:
                if attempt == max_recovery_attempts - 1:
                    raise OperationError("Recovery failed after all attempts") from e
                # Continue to next attempt
                
        assert result == "recovered successfully"
        assert recovery_attempts == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])