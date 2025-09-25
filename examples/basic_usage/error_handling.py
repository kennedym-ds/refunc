#!/usr/bin/env python3
"""
Error Handling Examples - Refunc Exception Framework

This example demonstrates the comprehensive exception handling framework
including retry mechanisms, custom exceptions, and error recovery strategies
for robust ML workflows.

Key Features Demonstrated:
- Custom exception hierarchy
- Retry mechanisms and strategies
- Error recovery patterns
- Context-aware error messages
- Integration with logging
- Graceful degradation
"""

import os
import sys
import time
import random
from typing import Optional, List, Any

# Handle missing dependencies gracefully
try:
    from refunc.exceptions import (
        RefuncError, ValidationError, DataError, ModelError,
        retry_on_failure, RetryConfig, RetryableOperation,
        FileNotFoundError, ModelNotFoundError, DataValidationError
    )
    REFUNC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Refunc not fully installed: {e}")
    print("This example requires: pip install refunc")
    print("Showing example code structure...\n")
    REFUNC_AVAILABLE = False


def basic_exception_usage():
    """Demonstrate basic exception usage and hierarchy."""
    print("🔧 Basic Exception Usage")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Basic exception usage:
from refunc.exceptions import (
    RefuncError, ValidationError, DataError, ModelError
)

# Raise specific exceptions
try:
    if not data_valid:
        raise ValidationError("Data validation failed", 
                            context={"expected": "numeric", "got": "string"})
except ValidationError as e:
    print(f"Validation error: {e}")
    print(f"Context: {e.context}")

# Handle exception hierarchy
try:
    risky_operation()
except DataError as e:
    print(f"Data-specific error: {e}")
except RefuncError as e:
    print(f"General Refunc error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
        """)
        return
    
    print("🧪 Testing exception hierarchy...")
    
    # Demonstrate different exception types
    exceptions_to_test = [
        (ValidationError, "Invalid input format", {"field": "age", "value": -5}),
        (DataError, "Corrupted data file", {"file": "data.csv", "line": 42}),
        (ModelError, "Model training failed", {"model": "RandomForest", "error": "convergence"})
    ]
    
    for exception_class, message, context in exceptions_to_test:
        try:
            print(f"   Raising {exception_class.__name__}...")
            raise exception_class(message, context=context)
            
        except RefuncError as e:
            print(f"   ✓ Caught: {e.__class__.__name__}")
            print(f"     Message: {e}")
            print(f"     Context: {e.context}")
            print(f"     Timestamp: {e.timestamp}")
            print()
    
    # Demonstrate exception chaining
    try:
        print("🔗 Testing exception chaining...")
        try:
            # Simulate nested error
            raise ValueError("Original system error")
        except ValueError as original_error:
            # Wrap in Refunc exception
            raise DataError("Processing failed due to system error", 
                          context={"operation": "data_loading"}) from original_error
            
    except DataError as e:
        print(f"   ✓ Main error: {e}")
        print(f"   📎 Caused by: {e.__cause__}")
        print(f"   🔍 Context: {e.context}")


def retry_mechanism_examples():
    """Demonstrate retry mechanisms and strategies."""
    print("\n🔄 Retry Mechanisms")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Retry mechanism examples:
from refunc.exceptions import retry_on_failure, RetryConfig

# Simple retry decorator
@retry_on_failure(max_attempts=3, delay=1.0)
def unstable_operation():
    if random.random() < 0.7:  # 70% failure rate
        raise ConnectionError("Network timeout")
    return "Success!"

# Advanced retry configuration
retry_config = RetryConfig(
    max_attempts=5,
    delay=0.5,
    backoff_factor=2.0,
    max_delay=10.0,
    exceptions=(ConnectionError, TimeoutError)
)

@retry_on_failure(config=retry_config)
def flaky_api_call():
    return make_api_request()

# Conditional retry
@retry_on_failure(
    max_attempts=3,
    should_retry=lambda e: isinstance(e, (ConnectionError, TimeoutError))
)
def selective_retry_operation():
    return external_service_call()
        """)
        return
    
    # Simulate unstable operations for demonstration
    def simulate_flaky_network_call(success_rate: float = 0.3):
        """Simulate a network call that fails randomly."""
        if random.random() < success_rate:
            return {"status": "success", "data": "Retrieved data"}
        else:
            raise ConnectionError("Network timeout occurred")
    
    def simulate_flaky_file_operation(success_rate: float = 0.4):
        """Simulate a file operation that fails randomly."""
        if random.random() < success_rate:
            return "File processed successfully"
        else:
            raise FileNotFoundError("Temporary file system issue")
    
    # Test basic retry decorator
    print("🌐 Testing network operation with retries...")
    
    @retry_on_failure(max_attempts=3, delay=0.2)
    def network_operation():
        """Network operation with retry."""
        return simulate_flaky_network_call(success_rate=0.4)
    
    try:
        result = network_operation()
        print(f"   ✓ Success: {result}")
    except Exception as e:
        print(f"   ❌ Failed after retries: {e}")
    
    # Test retry with backoff
    print("\n💾 Testing file operation with exponential backoff...")
    
    @retry_on_failure(
        max_attempts=4, 
        delay=0.1, 
        backoff_factor=2.0,
        max_delay=1.0
    )
    def file_operation():
        """File operation with exponential backoff."""
        return simulate_flaky_file_operation(success_rate=0.3)
    
    start_time = time.time()
    try:
        result = file_operation()
        duration = time.time() - start_time
        print(f"   ✓ Success: {result} (took {duration:.2f}s)")
    except Exception as e:
        duration = time.time() - start_time
        print(f"   ❌ Failed after retries: {e} (took {duration:.2f}s)")
    
    # Test selective retry
    print("\n🎯 Testing selective retry (only specific exceptions)...")
    
    @retry_on_failure(
        max_attempts=3,
        delay=0.1,
        exceptions=(ConnectionError,)  # Only retry ConnectionError
    )
    def selective_operation():
        """Operation that retries only specific exceptions."""
        rand_val = random.random()
        if rand_val < 0.3:
            return "Success!"
        elif rand_val < 0.7:
            raise ConnectionError("Network issue (retryable)")
        else:
            raise ValueError("Data format error (not retryable)")
    
    for i in range(3):
        try:
            result = selective_operation()
            print(f"   ✓ Attempt {i+1}: {result}")
            break
        except ConnectionError as e:
            print(f"   🔄 Attempt {i+1}: {e} (will retry)")
        except ValueError as e:
            print(f"   ❌ Attempt {i+1}: {e} (no retry)")
            break


def custom_exception_patterns():
    """Demonstrate custom exception patterns for ML workflows."""
    print("\n🏗️ ML-Specific Exception Patterns")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# ML-specific exception patterns:
from refunc.exceptions import (
    ModelError, DataValidationError, ModelNotFoundError
)

def validate_model_input(data):
    '''Validate input data for model inference.'''
    if data is None:
        raise DataValidationError("Input data cannot be None")
    
    if len(data.shape) != 2:
        raise DataValidationError(
            "Input must be 2D array",
            context={
                "expected_dims": 2,
                "actual_dims": len(data.shape),
                "shape": data.shape
            }
        )

def load_trained_model(model_path):
    '''Load a trained model with proper error handling.'''
    if not os.path.exists(model_path):
        raise ModelNotFoundError(
            f"Model file not found: {model_path}",
            context={"path": model_path, "operation": "load"}
        )
    
    try:
        return pickle.load(open(model_path, 'rb'))
    except Exception as e:
        raise ModelError(
            "Failed to load model",
            context={"path": model_path, "error": str(e)}
        ) from e
        """)
        return
    
    # Simulate ML workflow with proper error handling
    def validate_training_data(X, y):
        """Validate training data with detailed error context."""
        if X is None or y is None:
            raise DataValidationError(
                "Training data cannot be None",
                context={"X_is_none": X is None, "y_is_none": y is None}
            )
        
        # Simulate data validation
        if hasattr(X, 'shape') and hasattr(y, 'shape'):
            if len(X.shape) != 2:
                raise DataValidationError(
                    "Features must be 2D array",
                    context={
                        "expected_dims": 2,
                        "actual_dims": len(X.shape),
                        "shape": X.shape
                    }
                )
            
            if X.shape[0] != y.shape[0]:
                raise DataValidationError(
                    "Mismatched sample sizes",
                    context={
                        "X_samples": X.shape[0],
                        "y_samples": y.shape[0]
                    }
                )
        
        return True
    
    def train_model_with_validation(X, y, model_type="random_forest"):
        """Train model with comprehensive error handling."""
        try:
            # Validate input data
            validate_training_data(X, y)
            
            # Simulate model training
            if model_type not in ["random_forest", "svm", "neural_network"]:
                raise ValidationError(
                    f"Unsupported model type: {model_type}",
                    context={
                        "supported_types": ["random_forest", "svm", "neural_network"],
                        "requested_type": model_type
                    }
                )
            
            # Simulate training success/failure
            if random.random() < 0.8:  # 80% success rate
                return {
                    "model_type": model_type,
                    "accuracy": 0.85 + random.random() * 0.1,
                    "status": "trained_successfully"
                }
            else:
                raise ModelError(
                    "Model training convergence failed",
                    context={
                        "model_type": model_type,
                        "iterations": 1000,
                        "convergence_threshold": 1e-6
                    }
                )
                
        except DataValidationError:
            raise  # Re-raise data validation errors
        except ValidationError:
            raise  # Re-raise validation errors
        except Exception as e:
            # Wrap unexpected errors
            raise ModelError(
                "Unexpected error during model training",
                context={"model_type": model_type, "original_error": str(e)}
            ) from e
    
    # Test the ML workflow
    print("🤖 Testing ML workflow with error handling...")
    
    # Test valid scenario
    try:
        # Simulate valid data (using lists to avoid numpy dependency)
        X_valid = [[1, 2], [3, 4], [5, 6]]  # 3 samples, 2 features
        y_valid = [0, 1, 0]  # 3 labels
        
        print("   📊 Training with valid data...")
        result = train_model_with_validation(X_valid, y_valid, "random_forest")
        print(f"   ✓ Success: {result['model_type']} (accuracy: {result['accuracy']:.3f})")
        
    except RefuncError as e:
        print(f"   ❌ Training failed: {e}")
        print(f"      Context: {e.context}")
    
    # Test invalid scenarios
    print("\n   🧪 Testing error scenarios...")
    
    error_scenarios = [
        (None, [1, 2, 3], "null_data"),
        ([[1, 2]], [1, 2, 3], "mismatched_sizes"),
        ([[1, 2], [3, 4]], [0, 1], "unsupported_model")
    ]
    
    for i, (X, y, scenario) in enumerate(error_scenarios):
        try:
            model_type = "invalid_model" if scenario == "unsupported_model" else "random_forest"
            print(f"      Scenario {i+1}: {scenario}")
            result = train_model_with_validation(X, y, model_type)
            print(f"      Unexpected success: {result}")
            
        except RefuncError as e:
            print(f"      ✓ Caught expected error: {e.__class__.__name__}")
            print(f"        Details: {e}")


def error_recovery_patterns():
    """Demonstrate error recovery and graceful degradation patterns."""
    print("\n🛡️ Error Recovery Patterns")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Error recovery patterns:
from refunc.exceptions import RefuncError, DataError

def robust_data_processor(data_sources):
    '''Process data with fallback strategies.'''
    results = []
    failed_sources = []
    
    for source in data_sources:
        try:
            # Primary processing
            data = load_data(source)
            processed = process_data(data)
            results.append(processed)
            
        except DataError as e:
            failed_sources.append((source, str(e)))
            
            # Fallback: try alternative processing
            try:
                backup_data = load_backup_data(source)
                processed = process_data_simple(backup_data)
                results.append(processed)
                
            except Exception:
                # Graceful degradation: use empty placeholder
                results.append(create_empty_placeholder())
    
    return {
        "results": results,
        "failed_sources": failed_sources,
        "success_rate": len(results) / len(data_sources)
    }
        """)
        return
    
    def load_data_source(source_id: str, reliability: float = 0.7):
        """Simulate loading data from various sources with different reliability."""
        if random.random() < reliability:
            return {
                "source_id": source_id,
                "data": [1, 2, 3, 4, 5],
                "quality": "high"
            }
        else:
            raise DataError(
                f"Failed to load data from source {source_id}",
                context={"source": source_id, "error_type": "connection_timeout"}
            )
    
    def process_data_with_fallback(source_id: str):
        """Process data with multiple fallback strategies."""
        strategies = [
            ("primary", lambda: load_data_source(source_id, 0.8)),
            ("backup", lambda: load_data_source(f"{source_id}_backup", 0.6)),
            ("cache", lambda: {"source_id": source_id, "data": [0, 0, 0], "quality": "cached"}),
            ("empty", lambda: {"source_id": source_id, "data": [], "quality": "empty"})
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                print(f"      Trying {strategy_name} strategy for {source_id}...")
                result = strategy_func()
                print(f"      ✓ Success with {strategy_name} strategy")
                result["strategy_used"] = strategy_name
                return result
                
            except Exception as e:
                print(f"      ❌ {strategy_name} failed: {e}")
                continue
        
        # All strategies failed
        raise DataError(
            f"All fallback strategies failed for {source_id}",
            context={"source": source_id, "strategies_tried": len(strategies)}
        )
    
    # Demonstrate robust data processing
    print("📊 Testing robust data processing with fallbacks...")
    
    data_sources = ["source_A", "source_B", "source_C", "source_D"]
    results = []
    failures = []
    
    for source in data_sources:
        print(f"   Processing {source}:")
        try:
            result = process_data_with_fallback(source)
            results.append(result)
            print(f"   ✓ {source}: {result['quality']} quality using {result['strategy_used']}")
            
        except Exception as e:
            failures.append((source, str(e)))
            print(f"   ❌ {source}: Complete failure - {e}")
    
    # Summary
    success_rate = len(results) / len(data_sources)
    print(f"\n📈 Processing Summary:")
    print(f"   Successful: {len(results)}/{len(data_sources)} ({success_rate:.1%})")
    print(f"   Failed: {len(failures)} sources")
    
    if results:
        strategies_used = {}
        for result in results:
            strategy = result.get('strategy_used', 'unknown')
            strategies_used[strategy] = strategies_used.get(strategy, 0) + 1
        
        print(f"   Strategies used: {strategies_used}")


def logging_integration():
    """Demonstrate integration with logging system."""
    print("\n📝 Logging Integration")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Logging integration example:
from refunc.exceptions import retry_on_failure, RefuncError
from refunc.logging import MLLogger

# Logger-aware retry decorator
logger = MLLogger("error_handling")

@retry_on_failure(max_attempts=3, logger=logger)
def monitored_operation():
    # Retry attempts are automatically logged
    return risky_operation()

# Exception logging with context
try:
    result = complex_ml_operation()
except RefuncError as e:
    logger.error(f"ML operation failed: {e}", extra=e.context)
    logger.metric("error_count", 1, extra={"error_type": e.__class__.__name__})
        """)
        return
    
    print("🔗 Error handling integrates seamlessly with:")
    print("   • MLLogger for automatic error tracking")
    print("   • Retry decorators with attempt logging")
    print("   • Context preservation in error messages")
    print("   • Metrics collection for error rates")
    print("   • Structured error reporting")
    
    print("\n💡 Integration benefits:")
    print("   • Automatic error categorization and counting")
    print("   • Retry attempt tracking and analysis")
    print("   • Error context preserved for debugging")
    print("   • Centralized error monitoring and alerting")
    print("   • Performance impact analysis of failures")


def main():
    """Run all error handling examples."""
    print("🚀 Refunc Error Handling Examples")
    print("=" * 60)
    
    if not REFUNC_AVAILABLE:
        print("ℹ️  Running in demo mode (showing code structure)")
    else:
        print("ℹ️  Running with full Refunc functionality")
    
    print()
    
    # Set random seed for reproducible examples
    random.seed(42)
    
    # Run examples
    basic_exception_usage()
    retry_mechanism_examples()
    custom_exception_patterns()
    error_recovery_patterns()
    logging_integration()
    
    print("\n✅ Error handling examples completed!")
    print("\n📖 Next steps:")
    print("- Integrate retry decorators into your workflows")
    print("- Create custom exceptions for domain-specific errors")
    print("- Implement fallback strategies for critical operations")
    print("- Check out performance_monitoring.py for decorator usage")


if __name__ == "__main__":
    main()