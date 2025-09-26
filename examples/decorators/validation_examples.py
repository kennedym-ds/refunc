#!/usr/bin/env python3
"""
Validation Examples - Refunc Decorators

This example demonstrates the comprehensive validation decorators including
input/output validation, type checking, data schema validation, and custom
validation patterns for robust ML workflows.

Key Features Demonstrated:
- Input and output validation decorators
- Type checking and constraints
- Data schema validation
- Range and shape validation
- Custom validation patterns
- Integration with error handling
"""

import os
import sys
import time
import random
from typing import List, Dict, Any, Optional, Union, Tuple

# Handle missing dependencies gracefully
try:
    from refunc.decorators import (
        validate_inputs, validate_outputs, validate_types,
        validate_data_schema, ValidationResult,
        type_check, range_check, shape_check, dataframe_check,
        TypeValidator, RangeValidator, ShapeValidator, CustomValidator
    )
    REFUNC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Refunc not fully installed: {e}")
    print("This example requires: pip install refunc")
    print("Showing example code structure...\n")
    REFUNC_AVAILABLE = False


def basic_validation_examples():
    """Demonstrate basic input/output validation."""
    print("✅ Basic Input/Output Validation")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Basic validation examples:
from refunc.decorators import validate_inputs, validate_outputs

# Input type validation
@validate_inputs(types={'data': list, 'threshold': float})
def process_data(data, threshold=0.5):
    filtered = [x for x in data if x > threshold]
    return filtered

# Output type validation
@validate_outputs(types=dict)
def compute_statistics(values):
    return {
        'mean': sum(values) / len(values),
        'count': len(values),
        'max': max(values),
        'min': min(values)
    }

# Combined input/output validation
@validate_inputs(types={'X': list, 'y': list})
@validate_outputs(types=dict)
def train_model(X, y):
    # Training logic
    return {'accuracy': 0.85, 'model_type': 'classifier'}
        """)
        return
    
    print("🔍 Testing basic validation patterns:")
    
    @validate_inputs(types={'data': list, 'multiplier': (int, float)})
    def multiply_data(data: List[float], multiplier: float = 2.0) -> List[float]:
        """Multiply all values in data by multiplier."""
        return [x * multiplier for x in data]
    
    @validate_outputs(types=dict)
    def compute_stats(values: List[float]) -> Dict[str, float]:
        """Compute basic statistics."""
        if not values:
            return {'count': 0, 'mean': 0.0, 'sum': 0.0}
        
        return {
            'count': len(values),
            'mean': sum(values) / len(values),
            'sum': sum(values),
            'min': min(values),
            'max': max(values)
        }
    
    @validate_inputs(types={'features': list, 'target': list})
    @validate_outputs(types=dict)
    def simple_linear_fit(features: List[float], target: List[float]) -> Dict[str, Any]:
        """Simple linear regression fit."""
        if len(features) != len(target):
            raise ValueError("Features and target must have same length")
        
        # Simple linear regression calculation
        n = len(features)
        sum_x = sum(features)
        sum_y = sum(target)
        sum_xy = sum(x * y for x, y in zip(features, target))
        sum_x2 = sum(x * x for x in features)
        
        # Calculate slope and intercept
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        return {
            'slope': slope,
            'intercept': intercept,
            'n_samples': n,
            'r_squared': 0.85  # Mock R²
        }
    
    # Test valid inputs
    print("   ✅ Testing valid inputs:")
    test_data = [1.0, 2.5, 3.7, 4.2, 5.1]
    
    # Test multiply_data
    result = multiply_data(test_data, 1.5)
    print(f"     Multiply result: {len(result)} values, first = {result[0]:.2f}")
    
    # Test compute_stats
    stats = compute_stats(test_data)
    print(f"     Stats: mean = {stats['mean']:.2f}, count = {stats['count']}")
    
    # Test linear fit
    features = [1.0, 2.0, 3.0, 4.0, 5.0]
    target = [2.1, 4.2, 6.1, 8.0, 10.1]  # Roughly y = 2x
    fit_result = simple_linear_fit(features, target)
    print(f"     Linear fit: slope = {fit_result['slope']:.2f}, intercept = {fit_result['intercept']:.2f}")
    
    # Test validation errors
    print("\n   ❌ Testing validation errors:")
    
    # Test invalid input type
    try:
        multiply_data("invalid_data", 2.0)  # Should fail: string instead of list
        print("     Unexpected success!")
    except Exception as e:
        print(f"     ✓ Caught input validation error: {type(e).__name__}")
    
    # Test invalid output (this would require modifying the function to return wrong type)
    try:
        # This should pass since compute_stats returns correct type
        stats = compute_stats([1, 2, 3])
        print(f"     ✓ Output validation passed: {type(stats).__name__}")
    except Exception as e:
        print(f"     Output validation error: {type(e).__name__}")


def type_constraint_examples():
    """Demonstrate type constraints and validation."""
    print("\n🏷️ Type Constraints and Validation")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Type constraint examples:
from refunc.decorators import validate_types, type_check

# Multiple allowed types
@validate_inputs(types={
    'data': (list, tuple),
    'weights': (list, tuple, type(None)),
    'normalize': bool
})
def weighted_average(data, weights=None, normalize=True):
    if weights is None:
        return sum(data) / len(data)
    
    weighted_sum = sum(d * w for d, w in zip(data, weights))
    total_weight = sum(weights) if normalize else len(weights)
    return weighted_sum / total_weight

# Strict type checking with custom messages
@type_check(
    data=list,
    model_type=str,
    params=dict,
    strict=True
)
def train_with_config(data, model_type, params):
    return f"Training {model_type} with {len(params)} parameters"
        """)
        return
    
    print("🏗️ Testing flexible type validation:")
    
    @validate_inputs(types={
        'values': (list, tuple),
        'operation': str,
        'default_value': (int, float, type(None))
    })
    def flexible_operation(values, operation: str, default_value=None):
        """Perform operation on values with flexible input types."""
        if operation == "sum":
            return sum(values)
        elif operation == "mean":
            return sum(values) / len(values) if values else (default_value or 0)
        elif operation == "max":
            return max(values) if values else default_value
        elif operation == "min":
            return min(values) if values else default_value
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    @validate_inputs(types={
        'model_config': dict,
        'data_source': (str, dict),
        'validation_split': float,
        'random_seed': (int, type(None))
    })
    def setup_training(model_config: Dict[str, Any], 
                      data_source: Union[str, Dict], 
                      validation_split: float = 0.2,
                      random_seed: Optional[int] = None) -> Dict[str, Any]:
        """Setup training configuration with flexible types."""
        config = {
            'model': model_config,
            'data': data_source,
            'validation_split': validation_split,
            'seed': random_seed,
            'setup_time': time.time()
        }
        return config
    
    # Test flexible types
    print("   🔄 Testing flexible type acceptance:")
    
    # Test with list
    result1 = flexible_operation([1, 2, 3, 4, 5], "mean")
    print(f"     List input: mean = {result1:.2f}")
    
    # Test with tuple
    result2 = flexible_operation((10, 20, 30), "sum")
    print(f"     Tuple input: sum = {result2}")
    
    # Test with None default
    result3 = flexible_operation([], "max", default_value=0)
    print(f"     Empty with default: max = {result3}")
    
    # Test complex configuration
    print("\n   ⚙️ Testing complex configuration validation:")
    model_config = {
        'type': 'random_forest',
        'n_estimators': 100,
        'max_depth': 10
    }
    
    # Test with string data source
    setup1 = setup_training(model_config, "path/to/data.csv", 0.3)
    print(f"     String data source: {setup1['data']}")
    
    # Test with dict data source
    data_dict = {'train': 'train.csv', 'test': 'test.csv'}
    setup2 = setup_training(model_config, data_dict, 0.2, 42)
    print(f"     Dict data source: {len(setup2['data'])} keys")
    
    # Test validation errors
    print("\n   ❌ Testing type validation errors:")
    
    try:
        flexible_operation("invalid", "sum")  # String instead of list/tuple
        print("     Unexpected success!")
    except Exception as e:
        print(f"     ✓ Caught type error: {type(e).__name__}")
    
    try:
        setup_training("invalid_config", "data.csv")  # String instead of dict
        print("     Unexpected success!")
    except Exception as e:
        print(f"     ✓ Caught config type error: {type(e).__name__}")


def range_and_shape_validation():
    """Demonstrate range and shape validation."""
    print("\n📏 Range and Shape Validation")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Range and shape validation examples:
from refunc.decorators import range_check, shape_check

# Range validation
@range_check(
    learning_rate=(0.0001, 1.0),
    epochs=(1, 1000),
    batch_size=(1, 10000)
)
def configure_training(learning_rate, epochs, batch_size):
    return {
        'lr': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size
    }

# Shape validation for arrays
@shape_check(
    X=(None, None),  # 2D array, any dimensions
    y=(None,),       # 1D array, any length
    weights=(None, None)  # 2D array matching X
)
def train_model(X, y, weights=None):
    return fit_model(X, y, weights)

# Combined range and shape validation
@range_check(threshold=(0.0, 1.0))
@shape_check(predictions=(None,), labels=(None,))
def evaluate_predictions(predictions, labels, threshold=0.5):
    return compute_metrics(predictions, labels, threshold)
        """)
        return
    
    print("📊 Testing range and shape validation:")
    
    @range_check(
        learning_rate=(0.0001, 1.0),
        dropout_rate=(0.0, 0.9),
        n_estimators=(1, 1000)
    )
    def create_model_config(learning_rate: float, 
                           dropout_rate: float = 0.1, 
                           n_estimators: int = 100) -> Dict[str, Any]:
        """Create model configuration with validated ranges."""
        return {
            'learning_rate': learning_rate,
            'dropout_rate': dropout_rate,
            'n_estimators': n_estimators,
            'timestamp': time.time()
        }
    
    # Mock shape validation (since we don't have numpy)
    @validate_inputs(types={'data': list, 'labels': list})
    def validate_data_shapes(data: List[List[float]], labels: List[int]) -> Dict[str, Any]:
        """Validate data shapes manually (mock shape validation)."""
        # Manual shape checking
        if not data:
            raise ValueError("Data cannot be empty")
        
        n_samples = len(data)
        n_features = len(data[0]) if data else 0
        
        if len(labels) != n_samples:
            raise ValueError(f"Shape mismatch: data has {n_samples} samples, "
                           f"labels has {len(labels)} samples")
        
        # Check all data rows have same number of features
        for i, row in enumerate(data):
            if len(row) != n_features:
                raise ValueError(f"Inconsistent feature count at row {i}: "
                               f"expected {n_features}, got {len(row)}")
        
        return {
            'n_samples': n_samples,
            'n_features': n_features,
            'shape_valid': True
        }
    
    # Test valid ranges
    print("   ✅ Testing valid ranges:")
    
    config1 = create_model_config(0.01, 0.2, 50)
    print(f"     Config 1: lr={config1['learning_rate']}, dropout={config1['dropout_rate']}")
    
    config2 = create_model_config(0.001, n_estimators=200)  # Use default dropout
    print(f"     Config 2: lr={config2['learning_rate']}, n_est={config2['n_estimators']}")
    
    # Test valid shapes
    print("\n   📐 Testing shape validation:")
    
    # Create mock training data
    training_data = [
        [1.0, 2.0, 3.0],  # 3 features
        [1.5, 2.5, 3.5],
        [2.0, 3.0, 4.0],
        [2.5, 3.5, 4.5]
    ]  # 4 samples, 3 features each
    training_labels = [0, 1, 0, 1]  # 4 labels
    
    shape_result = validate_data_shapes(training_data, training_labels)
    print(f"     Valid data: {shape_result['n_samples']} samples × {shape_result['n_features']} features")
    
    # Test range validation errors
    print("\n   ❌ Testing range validation errors:")
    
    try:
        create_model_config(2.0, 0.2, 50)  # learning_rate too high
        print("     Unexpected success!")
    except Exception as e:
        print(f"     ✓ Caught range error: {type(e).__name__}")
    
    try:
        create_model_config(0.01, 1.5, 50)  # dropout_rate too high
        print("     Unexpected success!")
    except Exception as e:
        print(f"     ✓ Caught range error: {type(e).__name__}")
    
    # Test shape validation errors
    print("\n   📐 Testing shape validation errors:")
    
    try:
        # Mismatched labels
        validate_data_shapes(training_data, [0, 1])  # Only 2 labels for 4 samples
        print("     Unexpected success!")
    except Exception as e:
        print(f"     ✓ Caught shape error: {type(e).__name__}")
    
    try:
        # Inconsistent feature count
        inconsistent_data = [
            [1.0, 2.0, 3.0],      # 3 features
            [1.5, 2.5],           # 2 features - inconsistent!
            [2.0, 3.0, 4.0]       # 3 features
        ]
        validate_data_shapes(inconsistent_data, [0, 1, 0])
        print("     Unexpected success!")
    except Exception as e:
        print(f"     ✓ Caught shape consistency error: {type(e).__name__}")


def custom_validation_patterns():
    """Demonstrate custom validation patterns."""
    print("\n🎨 Custom Validation Patterns")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Custom validation examples:
from refunc.decorators import CustomValidator, validate_inputs

# Custom business logic validator
class ModelConfigValidator(CustomValidator):
    def validate(self, value):
        if not isinstance(value, dict):
            return False, "Config must be a dictionary"
        
        required_keys = ['model_type', 'parameters']
        missing_keys = [k for k in required_keys if k not in value]
        if missing_keys:
            return False, f"Missing required keys: {missing_keys}"
        
        valid_models = ['random_forest', 'svm', 'neural_network']
        if value['model_type'] not in valid_models:
            return False, f"Invalid model type: {value['model_type']}"
        
        return True, "Valid"

# Use custom validator
@validate_inputs(validators={'config': ModelConfigValidator()})
def setup_ml_pipeline(config):
    return f"Setting up {config['model_type']} pipeline"

# Domain-specific validation
def validate_email(email):
    '''Custom email validation.'''
    return '@' in email and '.' in email.split('@')[1]

@validate_inputs(custom_validators={
    'email': validate_email,
    'age': lambda x: 0 <= x <= 150,
    'score': lambda x: 0.0 <= x <= 1.0
})
def create_user_profile(email, age, score):
    return {'email': email, 'age': age, 'score': score}
        """)
        return
    
    print("🏗️ Testing custom validation patterns:")
    
    # Custom validation functions
    def validate_model_config(config: Dict[str, Any]) -> Tuple[bool, str]:
        """Custom model configuration validator."""
        if not isinstance(config, dict):
            return False, "Config must be a dictionary"
        
        required_keys = ['model_type', 'parameters']
        missing_keys = [k for k in required_keys if k not in config]
        if missing_keys:
            return False, f"Missing required keys: {missing_keys}"
        
        valid_models = ['random_forest', 'svm', 'neural_network', 'xgboost']
        if config['model_type'] not in valid_models:
            return False, f"Invalid model type. Valid options: {valid_models}"
        
        if not isinstance(config['parameters'], dict):
            return False, "Parameters must be a dictionary"
        
        return True, "Valid configuration"
    
    def validate_data_quality(data: List[float]) -> Tuple[bool, str]:
        """Custom data quality validator."""
        if not data:
            return False, "Data cannot be empty"
        
        if len(data) < 10:
            return False, "Data must have at least 10 samples"
        
        # Check for too many missing values (represented as None)
        none_count = sum(1 for x in data if x is None)
        if none_count / len(data) > 0.5:
            return False, "Too many missing values (>50%)"
        
        # Check for extreme outliers (basic check)
        numeric_data = [x for x in data if x is not None]
        if numeric_data:
            mean_val = sum(numeric_data) / len(numeric_data)
            outliers = [x for x in numeric_data if abs(x - mean_val) > 3 * mean_val]
            if len(outliers) > len(numeric_data) * 0.1:
                return False, "Too many outliers detected"
        
        return True, "Data quality acceptable"
    
    # Functions using custom validation
    def setup_ml_experiment(config: Dict[str, Any], data: List[float]) -> Dict[str, Any]:
        """Setup ML experiment with custom validation."""
        # Validate config
        config_valid, config_msg = validate_model_config(config)
        if not config_valid:
            raise ValueError(f"Config validation failed: {config_msg}")
        
        # Validate data quality
        data_valid, data_msg = validate_data_quality(data)
        if not data_valid:
            raise ValueError(f"Data validation failed: {data_msg}")
        
        return {
            'experiment_id': f"exp_{int(time.time())}",
            'model_type': config['model_type'],
            'data_samples': len([x for x in data if x is not None]),
            'status': 'configured',
            'validation_passed': True
        }
    
    # Test valid configurations
    print("   ✅ Testing valid custom validation:")
    
    valid_configs = [
        {
            'model_type': 'random_forest',
            'parameters': {'n_estimators': 100, 'max_depth': 10}
        },
        {
            'model_type': 'neural_network',
            'parameters': {'hidden_layers': [64, 32], 'activation': 'relu'}
        }
    ]
    
    valid_data = [random.random() * 10 for _ in range(50)]  # 50 random samples
    
    for i, config in enumerate(valid_configs):
        try:
            result = setup_ml_experiment(config, valid_data)
            print(f"     Config {i+1}: {result['model_type']} - {result['status']}")
        except Exception as e:
            print(f"     Config {i+1} failed: {e}")
    
    # Test validation failures
    print("\n   ❌ Testing custom validation failures:")
    
    invalid_scenarios = [
        # Invalid model type
        {
            'config': {'model_type': 'invalid_model', 'parameters': {}},
            'data': valid_data,
            'scenario': 'invalid_model_type'
        },
        # Missing required keys
        {
            'config': {'model_type': 'random_forest'},  # Missing 'parameters'
            'data': valid_data,
            'scenario': 'missing_parameters'
        },
        # Bad data quality - too few samples
        {
            'config': valid_configs[0],
            'data': [1.0, 2.0],  # Only 2 samples
            'scenario': 'insufficient_data'
        },
        # Bad data quality - too many None values
        {
            'config': valid_configs[0],
            'data': [1.0, None, None, None, None, None, None, None, None, None, 2.0],
            'scenario': 'too_many_missing'
        }
    ]
    
    for scenario in invalid_scenarios:
        try:
            result = setup_ml_experiment(scenario['config'], scenario['data'])
            print(f"     {scenario['scenario']}: Unexpected success!")
        except Exception as e:
            print(f"     ✓ {scenario['scenario']}: {type(e).__name__}")
    
    # Demonstrate validation composition
    print("\n   🔗 Testing validation composition:")
    
    def validate_complete_pipeline(config: Dict[str, Any], 
                                 train_data: List[float], 
                                 test_data: List[float]) -> Dict[str, Any]:
        """Validate complete ML pipeline setup."""
        validations = []
        
        # Validate configuration
        config_valid, config_msg = validate_model_config(config)
        validations.append(('config', config_valid, config_msg))
        
        # Validate training data
        train_valid, train_msg = validate_data_quality(train_data)
        validations.append(('train_data', train_valid, train_msg))
        
        # Validate test data
        test_valid, test_msg = validate_data_quality(test_data)
        validations.append(('test_data', test_valid, test_msg))
        
        # Check data compatibility
        train_clean = [x for x in train_data if x is not None]
        test_clean = [x for x in test_data if x is not None]
        
        if train_clean and test_clean:
            train_range = max(train_clean) - min(train_clean)
            test_range = max(test_clean) - min(test_clean)
            
            # Simple range compatibility check
            if abs(train_range - test_range) > max(train_range, test_range) * 0.5:
                validations.append(('data_compatibility', False, 'Train/test data ranges too different'))
            else:
                validations.append(('data_compatibility', True, 'Data ranges compatible'))
        
        # Summarize validation results
        failed_validations = [v for v in validations if not v[1]]
        
        if failed_validations:
            failure_msg = "; ".join([f"{v[0]}: {v[2]}" for v in failed_validations])
            raise ValueError(f"Pipeline validation failed: {failure_msg}")
        
        return {
            'validation_summary': {v[0]: v[1] for v in validations},
            'pipeline_ready': True,
            'train_samples': len(train_clean),
            'test_samples': len(test_clean)
        }
    
    # Test complete pipeline validation
    train_data = [random.random() * 5 + 2 for _ in range(30)]  # Range ~2-7
    test_data = [random.random() * 5 + 2.5 for _ in range(20)]  # Range ~2.5-7.5
    
    try:
        pipeline_result = validate_complete_pipeline(valid_configs[0], train_data, test_data)
        print(f"     ✓ Complete validation passed: {pipeline_result['train_samples']} train, "
              f"{pipeline_result['test_samples']} test samples")
    except Exception as e:
        print(f"     ❌ Complete validation failed: {e}")


def main():
    """Run all validation examples."""
    print("🚀 Refunc Validation Examples")
    print("=" * 60)
    
    if not REFUNC_AVAILABLE:
        print("ℹ️  Running in demo mode (showing code structure)")
    else:
        print("ℹ️  Running with full Refunc functionality")
    
    print()
    
    # Set random seed for reproducible examples
    random.seed(42)
    
    # Run examples
    basic_validation_examples()
    type_constraint_examples()
    range_and_shape_validation()
    custom_validation_patterns()
    
    print("\n✅ Validation examples completed!")
    print("\n📖 Next steps:")
    print("- Implement validation in your ML functions")
    print("- Create custom validators for domain-specific logic")
    print("- Combine validation with performance monitoring")
    print("- Check out data_validation.py for data quality assessment")


if __name__ == "__main__":
    main()