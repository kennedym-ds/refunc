#!/usr/bin/env python3
"""
Model Training Examples - Refunc ML Workflows

This example demonstrates comprehensive model training workflows including
model management, hyperparameter optimization, training pipelines, and
model evaluation for production ML systems.

Key Features Demonstrated:
- Model registry and management
- Training pipeline orchestration
- Hyperparameter optimization
- Model evaluation and comparison
- Automated model selection
- Production deployment preparation
"""

import os
import sys
import time
import random
import json
from typing import List, Dict, Any, Optional, Tuple, Union

# Handle missing dependencies gracefully
try:
    from refunc.ml import (
        BaseModel, SklearnModel, ModelRegistry,
        ModelEvaluator, ModelComparator,
        HyperparameterOptimizer, AutoMLTrainer,
        optimize_hyperparameters, auto_train_models
    )
    from refunc.logging import MLLogger
    from refunc.decorators import time_it, memory_profile
    
    REFUNC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Refunc not fully installed: {e}")
    print("This example requires: pip install refunc")
    print("Showing example code structure...\n")
    REFUNC_AVAILABLE = False


def create_sample_ml_dataset():
    """Create sample dataset for ML training examples."""
    random.seed(42)
    
    # Generate synthetic classification dataset
    n_samples = 1000
    n_features = 10
    
    # Generate features with some correlations
    X = []
    y = []
    
    for i in range(n_samples):
        sample = []
        
        # Generate correlated features
        base_features = [random.gauss(0, 1) for _ in range(5)]
        
        # Add some derived features
        derived_features = [
            base_features[0] + base_features[1] + random.gauss(0, 0.1),  # Linear combination
            base_features[2] ** 2 + random.gauss(0, 0.1),  # Nonlinear
            abs(base_features[3]) + random.gauss(0, 0.1),  # Absolute value
            base_features[4] * base_features[0] + random.gauss(0, 0.1),  # Interaction
            random.gauss(0, 1)  # Pure noise
        ]
        
        sample = base_features + derived_features
        X.append(sample)
        
        # Generate target based on features with some noise
        target_score = (
            sample[0] * 0.3 + 
            sample[1] * 0.2 + 
            sample[5] * 0.1 -  # Derived feature
            sample[2] * 0.15 +
            random.gauss(0, 0.1)
        )
        
        # Convert to binary classification
        y.append(1 if target_score > 0 else 0)
    
    return X, y


def basic_model_training_examples():
    """Demonstrate basic model training workflows."""
    print("🤖 Basic Model Training")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Basic model training examples:
from refunc.ml import SklearnModel, ModelRegistry
from refunc.logging import MLLogger

# Initialize logging and registry
logger = MLLogger("model_training")
registry = ModelRegistry("./models")

# Create and train model
model = SklearnModel(
    model_type="RandomForestClassifier",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
)

# Train model
training_result = model.fit(X_train, y_train)
logger.log_metrics(training_result.metrics)

# Evaluate model
evaluation = model.evaluate(X_test, y_test)
print(f"Accuracy: {evaluation.accuracy:.3f}")

# Save to registry
registry.save_model(model, "random_forest_v1")
        """)
        return
    
    print("🏗️ Testing basic model training:")
    
    # Create sample dataset
    X, y = create_sample_ml_dataset()
    
    # Split into train/test (simple split)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"   📊 Dataset prepared:")
    print(f"     Training samples: {len(X_train)}")
    print(f"     Test samples: {len(X_test)}")
    print(f"     Features: {len(X_train[0])}")
    print(f"     Classes: {len(set(y))}")
    
    # Initialize components
    logger = MLLogger("model_training_demo")
    
    print(f"\n   🏁 Starting model training:")
    
    # Model configurations to test
    model_configs = [
        {
            "name": "Random Forest",
            "type": "RandomForestClassifier",
            "params": {
                "n_estimators": 50,
                "max_depth": 8,
                "random_state": 42
            }
        },
        {
            "name": "Logistic Regression",
            "type": "LogisticRegression", 
            "params": {
                "max_iter": 1000,
                "random_state": 42
            }
        },
        {
            "name": "SVM",
            "type": "SVC",
            "params": {
                "kernel": "rbf",
                "C": 1.0,
                "random_state": 42
            }
        }
    ]
    
    # Train and evaluate models
    training_results = {}
    
    for config in model_configs:
        model_name = config["name"]
        print(f"\n     🎯 Training {model_name}:")
        
        # Simulate model training
        start_time = time.time()
        
        # Mock training process
        training_time = random.uniform(0.1, 0.3)  # Simulate training time
        time.sleep(training_time)
        
        # Mock model evaluation
        # Generate realistic performance metrics
        base_accuracy = 0.75 + random.uniform(0, 0.15)  # 75-90% accuracy
        accuracy = min(0.95, max(0.70, base_accuracy))
        
        precision = accuracy + random.uniform(-0.05, 0.05)
        recall = accuracy + random.uniform(-0.05, 0.05)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        # Ensure realistic bounds
        precision = min(1.0, max(0.5, precision))
        recall = min(1.0, max(0.5, recall))
        f1_score = min(1.0, max(0.5, f1_score))
        
        # Training metrics
        training_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "training_time": training_time,
            "n_features": len(X_train[0]),
            "n_samples": len(X_train)
        }
        
        training_results[model_name] = training_metrics
        
        # Log training progress
        logger.log_params({
            "model_type": config["type"],
            "hyperparameters": config["params"],
            "dataset_size": len(X_train)
        })
        
        logger.log_metrics(training_metrics)
        
        print(f"       Training time: {training_time:.2f}s")
        print(f"       Accuracy: {accuracy:.3f}")
        print(f"       Precision: {precision:.3f}")
        print(f"       Recall: {recall:.3f}")
        print(f"       F1-score: {f1_score:.3f}")
    
    # Model comparison
    print(f"\n   📊 Model Comparison Summary:")
    
    # Sort models by F1-score
    sorted_models = sorted(training_results.items(), 
                          key=lambda x: x[1]['f1_score'], 
                          reverse=True)
    
    print(f"     {'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'Training Time':<15}")
    print(f"     {'-'*20} {'-'*10} {'-'*10} {'-'*15}")
    
    for model_name, metrics in sorted_models:
        print(f"     {model_name:<20} {metrics['accuracy']:<10.3f} "
              f"{metrics['f1_score']:<10.3f} {metrics['training_time']:<15.2f}s")
    
    # Best model selection
    best_model_name, best_metrics = sorted_models[0]
    print(f"\n   🏆 Best Model: {best_model_name}")
    print(f"     Performance: {best_metrics['f1_score']:.3f} F1-score")
    print(f"     Efficiency: {best_metrics['training_time']:.2f}s training time")
    
    # Model registry simulation
    print(f"\n   💾 Model Registry:")
    registry_entries = []
    
    for model_name, metrics in training_results.items():
        model_id = f"{model_name.lower().replace(' ', '_')}_v1"
        registry_entry = {
            "model_id": model_id,
            "model_name": model_name,
            "version": "1.0",
            "metrics": metrics,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "trained"
        }
        registry_entries.append(registry_entry)
        
        print(f"     ✓ Registered: {model_id}")
    
    print(f"     Total models in registry: {len(registry_entries)}")


def hyperparameter_optimization_examples():
    """Demonstrate hyperparameter optimization workflows."""
    print("\n🎛️ Hyperparameter Optimization")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Hyperparameter optimization examples:
from refunc.ml import HyperparameterOptimizer, optimize_hyperparameters

# Grid search optimization
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10]
}

optimizer = HyperparameterOptimizer(
    model_type="RandomForestClassifier",
    param_grid=param_grid,
    cv_folds=5,
    scoring="f1_weighted"
)

best_params = optimizer.optimize(X_train, y_train)
print(f"Best parameters: {best_params.params}")
print(f"Best score: {best_params.score:.3f}")

# Random search optimization
random_params = {
    "n_estimators": (10, 500),
    "max_depth": (1, 20),
    "learning_rate": (0.01, 1.0)
}

random_optimizer = HyperparameterOptimizer(
    model_type="XGBClassifier",
    param_distributions=random_params,
    n_iter=50,
    cv_folds=3
)

best_random = random_optimizer.optimize(X_train, y_train)
        """)
        return
    
    print("🔍 Testing hyperparameter optimization:")
    
    # Create sample dataset
    X, y = create_sample_ml_dataset()
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"   📊 Optimization dataset: {len(X_train)} samples, {len(X_train[0])} features")
    
    # Define hyperparameter search spaces
    optimization_scenarios = [
        {
            "name": "Random Forest Grid Search",
            "model_type": "RandomForestClassifier",
            "search_type": "grid",
            "param_space": {
                "n_estimators": [25, 50, 100],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5]
            }
        },
        {
            "name": "Logistic Regression Random Search",
            "model_type": "LogisticRegression",
            "search_type": "random",
            "param_space": {
                "C": (0.01, 10.0),
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"]
            }
        }
    ]
    
    optimization_results = {}
    
    for scenario in optimization_scenarios:
        scenario_name = scenario["name"]
        print(f"\n   🎯 {scenario_name}:")
        
        param_space = scenario["param_space"]
        search_type = scenario["search_type"]
        
        print(f"     Search type: {search_type}")
        print(f"     Parameter space: {len(param_space)} parameters")
        
        # Simulate optimization process
        start_time = time.time()
        
        if search_type == "grid":
            # Calculate total combinations for grid search
            total_combinations = 1
            for param_values in param_space.values():
                if isinstance(param_values, list):
                    total_combinations *= len(param_values)
            
            print(f"     Total combinations: {total_combinations}")
            
            # Simulate grid search
            best_score = 0
            best_params = {}
            
            # Test a few random combinations
            n_combinations_tested = min(total_combinations, 8)
            
            for i in range(n_combinations_tested):
                # Generate random parameter combination
                test_params = {}
                for param_name, param_values in param_space.items():
                    if isinstance(param_values, list):
                        test_params[param_name] = random.choice(param_values)
                
                # Simulate model training and cross-validation
                cv_score = 0.7 + random.uniform(0, 0.2)  # Random score between 0.7-0.9
                
                if cv_score > best_score:
                    best_score = cv_score
                    best_params = test_params.copy()
                
                print(f"       Combination {i+1}: CV score = {cv_score:.3f}")
        
        else:  # random search
            # Simulate random search
            n_iterations = 10
            best_score = 0
            best_params = {}
            
            print(f"     Random iterations: {n_iterations}")
            
            for i in range(n_iterations):
                # Generate random parameters
                test_params = {}
                for param_name, param_range in param_space.items():
                    if isinstance(param_range, tuple):
                        # Continuous parameter
                        test_params[param_name] = random.uniform(param_range[0], param_range[1])
                    elif isinstance(param_range, list):
                        # Categorical parameter
                        test_params[param_name] = random.choice(param_range)
                
                # Simulate model training
                cv_score = 0.7 + random.uniform(0, 0.2)
                
                if cv_score > best_score:
                    best_score = cv_score
                    best_params = test_params.copy()
                
                print(f"       Iteration {i+1}: CV score = {cv_score:.3f}")
        
        optimization_time = time.time() - start_time
        
        # Store results
        optimization_results[scenario_name] = {
            "best_params": best_params,
            "best_score": best_score,
            "optimization_time": optimization_time,
            "search_type": search_type
        }
        
        print(f"     ✅ Optimization completed in {optimization_time:.2f}s")
        print(f"     Best CV score: {best_score:.3f}")
        print(f"     Best parameters: {best_params}")
    
    # Compare optimization approaches
    print(f"\n   📊 Optimization Comparison:")
    
    print(f"     {'Method':<30} {'Best Score':<12} {'Time (s)':<10} {'Efficiency':<12}")
    print(f"     {'-'*30} {'-'*12} {'-'*10} {'-'*12}")
    
    for method_name, results in optimization_results.items():
        efficiency = results['best_score'] / results['optimization_time']
        print(f"     {method_name:<30} {results['best_score']:<12.3f} "
              f"{results['optimization_time']:<10.2f} {efficiency:<12.3f}")
    
    # Advanced optimization strategies
    print(f"\n   🚀 Advanced Optimization Strategies:")
    
    # Bayesian optimization simulation
    print(f"     🎯 Bayesian Optimization (simulated):")
    print(f"       - Uses probabilistic model to guide search")
    print(f"       - Balances exploration vs exploitation")
    print(f"       - Typically 2-5x more efficient than grid search")
    print(f"       - Best for expensive model training")
    
    # Multi-objective optimization
    print(f"     ⚖️ Multi-objective Optimization:")
    print(f"       - Optimize multiple metrics simultaneously")
    print(f"       - Example: accuracy vs inference time")
    print(f"       - Results in Pareto frontier of solutions")
    print(f"       - Allows business trade-off decisions")
    
    # Early stopping
    print(f"     ⏹️ Early Stopping:")
    print(f"       - Stop poor configurations early")
    print(f"       - Save 30-70% of optimization time")
    print(f"       - Based on learning curves")
    print(f"       - Particularly effective for neural networks")


def model_evaluation_examples():
    """Demonstrate comprehensive model evaluation."""
    print("\n📊 Model Evaluation")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Model evaluation examples:
from refunc.ml import ModelEvaluator, ModelComparator

# Single model evaluation
evaluator = ModelEvaluator(task_type="classification")
evaluation = evaluator.evaluate(
    model=trained_model,
    X_test=X_test,
    y_test=y_test,
    metrics=["accuracy", "precision", "recall", "f1", "auc"]
)

print(evaluation.summary())
print(f"Confusion Matrix:\\n{evaluation.confusion_matrix}")

# Model comparison
comparator = ModelComparator(task_type="classification")
models = {"rf": rf_model, "svm": svm_model, "lr": lr_model}
comparison = comparator.compare(models, X_test, y_test)

print(comparison.ranking_table())
print(f"Best model: {comparison.best_model}")

# Cross-validation evaluation
cv_results = evaluator.cross_validate(
    model=best_model,
    X=X_train,
    y=y_train,
    cv_folds=5,
    metrics=["accuracy", "f1"]
)
        """)
        return
    
    print("📈 Testing model evaluation:")
    
    # Create sample dataset and trained models
    X, y = create_sample_ml_dataset()
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"   📊 Evaluation dataset: {len(X_test)} test samples")
    
    # Simulate trained models with different characteristics
    model_profiles = {
        "High Precision Model": {
            "accuracy": 0.87,
            "precision": 0.92,
            "recall": 0.81,
            "training_time": 45.2,
            "inference_time": 0.003,
            "model_size_mb": 12.5
        },
        "Balanced Model": {
            "accuracy": 0.85,
            "precision": 0.85,
            "recall": 0.86,
            "training_time": 28.7,
            "inference_time": 0.002,
            "model_size_mb": 8.3
        },
        "High Recall Model": {
            "accuracy": 0.82,
            "precision": 0.78,
            "recall": 0.94,
            "training_time": 67.1,
            "inference_time": 0.005,
            "model_size_mb": 18.9
        },
        "Fast Model": {
            "accuracy": 0.79,
            "precision": 0.80,
            "recall": 0.77,
            "training_time": 12.3,
            "inference_time": 0.001,
            "model_size_mb": 3.2
        }
    }
    
    # Detailed evaluation for each model
    print(f"\n   🔍 Detailed Model Evaluation:")
    
    evaluation_results = {}
    
    for model_name, profile in model_profiles.items():
        print(f"\n     {model_name}:")
        
        # Core performance metrics  
        accuracy = profile["accuracy"]
        precision = profile["precision"]
        recall = profile["recall"]
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        # Generate confusion matrix (simplified)
        n_test = len(y_test)
        n_positive = sum(y_test)
        n_negative = n_test - n_positive
        
        # True positives, false positives, etc.
        tp = int(n_positive * recall)
        fn = n_positive - tp
        fp = int(tp / precision - tp) if precision > 0 else 0
        tn = n_negative - fp
        
        confusion_matrix = {
            "tp": tp, "fp": fp,
            "fn": fn, "tn": tn
        }
        
        print(f"       Accuracy: {accuracy:.3f}")
        print(f"       Precision: {precision:.3f}")
        print(f"       Recall: {recall:.3f}")
        print(f"       F1-Score: {f1_score:.3f}")
        
        # Confusion matrix
        print(f"       Confusion Matrix:")
        print(f"         Predicted:    0    1")
        print(f"         Actual 0:   {tn:3d}  {fp:3d}")
        print(f"         Actual 1:   {fn:3d}  {tp:3d}")
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall  # Same as recall
        
        print(f"       Specificity: {specificity:.3f}")
        print(f"       Sensitivity: {sensitivity:.3f}")
        
        # Performance characteristics
        print(f"       Training Time: {profile['training_time']:.1f}s")
        print(f"       Inference Time: {profile['inference_time']:.3f}s")
        print(f"       Model Size: {profile['model_size_mb']:.1f} MB")
        
        # Store comprehensive results
        evaluation_results[model_name] = {
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "specificity": specificity
            },
            "confusion_matrix": confusion_matrix,
            "performance": {
                "training_time": profile["training_time"],
                "inference_time": profile["inference_time"],
                "model_size_mb": profile["model_size_mb"]
            }
        }
    
    # Model comparison and ranking
    print(f"\n   🏆 Model Comparison and Ranking:")
    
    # Create comparison table
    print(f"     {'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'Inference':<12} {'Size (MB)':<10}")
    print(f"     {'-'*20} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")
    
    # Sort by F1-score
    sorted_models = sorted(evaluation_results.items(), 
                          key=lambda x: x[1]['metrics']['f1_score'], 
                          reverse=True)
    
    for model_name, results in sorted_models:
        metrics = results['metrics']
        perf = results['performance']
        
        print(f"     {model_name:<20} {metrics['accuracy']:<10.3f} "
              f"{metrics['f1_score']:<10.3f} {perf['inference_time']:<12.3f} "
              f"{perf['model_size_mb']:<10.1f}")
    
    # Multi-criteria analysis
    print(f"\n   ⚖️ Multi-Criteria Model Selection:")
    
    # Define different selection criteria
    selection_criteria = {
        "Best Performance": lambda x: x[1]['metrics']['f1_score'],
        "Fastest Inference": lambda x: -x[1]['performance']['inference_time'],  # Negative for ascending
        "Smallest Model": lambda x: -x[1]['performance']['model_size_mb'],
        "Most Balanced": lambda x: min(x[1]['metrics']['precision'], x[1]['metrics']['recall'])
    }
    
    for criterion_name, criterion_func in selection_criteria.items():
        best_model = max(evaluation_results.items(), key=criterion_func)
        model_name, model_data = best_model
        
        print(f"     {criterion_name}: {model_name}")
        
        if criterion_name == "Best Performance":
            print(f"       F1-Score: {model_data['metrics']['f1_score']:.3f}")
        elif criterion_name == "Fastest Inference":
            print(f"       Inference Time: {model_data['performance']['inference_time']:.3f}s")
        elif criterion_name == "Smallest Model":
            print(f"       Model Size: {model_data['performance']['model_size_mb']:.1f} MB")
        elif criterion_name == "Most Balanced":
            prec = model_data['metrics']['precision']
            rec = model_data['metrics']['recall']
            print(f"       Precision/Recall: {prec:.3f}/{rec:.3f}")
    
    # Cross-validation simulation
    print(f"\n   🔄 Cross-Validation Analysis:")
    
    # Simulate 5-fold CV for best model
    best_model_name = sorted_models[0][0]
    best_model_profile = model_profiles[best_model_name]
    
    print(f"     Model: {best_model_name}")
    print(f"     Cross-validation folds: 5")
    
    # Generate CV scores with some variance
    base_accuracy = best_model_profile["accuracy"]
    cv_scores = []
    
    for fold in range(5):
        # Add some realistic variance to CV scores
        fold_score = base_accuracy + random.gauss(0, 0.02)  # ±2% std dev
        fold_score = max(0.5, min(0.99, fold_score))  # Realistic bounds
        cv_scores.append(fold_score)
        
        print(f"       Fold {fold + 1}: {fold_score:.3f}")
    
    # CV statistics
    cv_mean = sum(cv_scores) / len(cv_scores)
    cv_std = (sum((score - cv_mean) ** 2 for score in cv_scores) / (len(cv_scores) - 1)) ** 0.5
    
    print(f"     CV Mean: {cv_mean:.3f} ± {cv_std:.3f}")
    print(f"     CV Range: [{min(cv_scores):.3f}, {max(cv_scores):.3f}]")
    
    # Model stability assessment
    if cv_std < 0.02:
        stability = "Highly stable"
    elif cv_std < 0.05:
        stability = "Stable"
    else:
        stability = "Variable performance"
    
    print(f"     Model Stability: {stability}")


def production_deployment_examples():
    """Demonstrate production deployment preparation."""
    print("\n🚀 Production Deployment")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Production deployment examples:
from refunc.ml import ModelRegistry, ProductionValidator

# Model validation for production
validator = ProductionValidator()
validation_report = validator.validate_model(
    model=best_model,
    validation_data=(X_val, y_val),
    performance_thresholds={
        "accuracy": 0.85,
        "inference_time": 0.1,
        "memory_usage": 500
    }
)

if validation_report.is_production_ready:
    # Deploy to registry
    registry = ModelRegistry("./production_models")
    model_version = registry.deploy_model(
        model=best_model,
        metadata={
            "version": "1.0.0",
            "training_data_hash": data_hash,
            "performance_metrics": metrics
        }
    )
    
    # Setup monitoring
    monitor = ModelMonitor(model_version)
    monitor.start_monitoring()
    
    print(f"Model deployed as version {model_version}")
else:
    print("Model failed production validation")
    print(validation_report.issues)
        """)
        return
    
    print("🏭 Testing production deployment workflow:")
    
    # Simulate production validation checklist
    print("   ✅ Production Readiness Checklist:")
    
    # Best model from previous examples
    best_model_metrics = {
        "accuracy": 0.87,
        "precision": 0.92,
        "recall": 0.81,
        "f1_score": 0.86,
        "inference_time": 0.003,
        "model_size_mb": 12.5,
        "training_time": 45.2
    }
    
    # Production requirements
    production_requirements = {
        "min_accuracy": 0.80,
        "max_inference_time": 0.010,  # 10ms
        "max_model_size_mb": 50.0,
        "min_precision": 0.75,
        "min_recall": 0.70
    }
    
    print(f"     📊 Performance Validation:")
    
    validation_results = {}
    
    # Check each requirement
    for metric, threshold in production_requirements.items():
        if metric.startswith("min_"):
            actual_metric = metric[4:]  # Remove "min_" prefix
            actual_value = best_model_metrics.get(actual_metric, 0)
            passed = actual_value >= threshold
            operator = ">="
        elif metric.startswith("max_"):
            actual_metric = metric[4:]  # Remove "max_" prefix  
            actual_value = best_model_metrics.get(actual_metric, float('inf'))
            passed = actual_value <= threshold
            operator = "<="
        else:
            continue
        
        validation_results[metric] = {
            "passed": passed,
            "actual": actual_value,
            "threshold": threshold,
            "operator": operator
        }
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"       {actual_metric}: {actual_value:.3f} {operator} {threshold:.3f} - {status}")
    
    # Overall validation status
    all_passed = all(result["passed"] for result in validation_results.values())
    overall_status = "✅ PRODUCTION READY" if all_passed else "❌ NEEDS IMPROVEMENT"
    
    print(f"     Overall Status: {overall_status}")
    
    if all_passed:
        # Production deployment simulation
        print(f"\n   🚀 Deployment Process:")
        
        # Model versioning
        model_version = "1.0.0"
        deployment_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"     📦 Model Packaging:")
        print(f"       Version: {model_version}")
        print(f"       Timestamp: {deployment_timestamp}")
        print(f"       Model Size: {best_model_metrics['model_size_mb']:.1f} MB")
        
        # Deployment metadata
        deployment_metadata = {
            "model_id": "high_precision_classifier_v1",
            "version": model_version,
            "deployment_timestamp": deployment_timestamp,
            "performance_metrics": best_model_metrics,
            "validation_results": validation_results,
            "training_dataset_hash": "abc123def456",
            "feature_schema_version": "1.2",
            "dependencies": [
                "scikit-learn==1.0.2",
                "numpy==1.21.0",
                "pandas==1.3.0"
            ]
        }
        
        print(f"     📋 Deployment Metadata:")
        for key, value in deployment_metadata.items():
            if isinstance(value, dict) or isinstance(value, list):
                print(f"       {key}: {type(value).__name__} with {len(value)} items")
            else:
                print(f"       {key}: {value}")
        
        # Production environment setup
        print(f"\n   🌐 Production Environment:")
        
        production_config = {
            "infrastructure": {
                "cpu_cores": 4,
                "memory_gb": 16,
                "storage_gb": 100,
                "container": "Docker",
                "orchestration": "Kubernetes"
            },
            "scaling": {
                "min_replicas": 2,
                "max_replicas": 10,
                "target_cpu_utilization": 70,
                "auto_scaling": True
            },
            "monitoring": {
                "metrics_collection": True,
                "drift_detection": True,
                "performance_alerts": True,
                "logging_level": "INFO"
            },
            "data_pipeline": {
                "preprocessing": "standardization",
                "feature_validation": True,
                "batch_size": 1000,
                "real_time_inference": True
            }
        }
        
        for category, settings in production_config.items():
            print(f"     {category.title()}:")
            for setting, value in settings.items():
                print(f"       {setting}: {value}")
        
        # Monitoring and alerting setup
        print(f"\n   📊 Monitoring Setup:")
        
        monitoring_metrics = [
            "prediction_latency",
            "prediction_throughput", 
            "model_accuracy",
            "data_drift_score",
            "feature_importance_stability",
            "error_rate",
            "system_resource_usage"
        ]
        
        alert_thresholds = {
            "prediction_latency": "> 50ms",
            "prediction_throughput": "< 100 req/s",
            "model_accuracy": "< 0.82",
            "data_drift_score": "> 0.3",
            "error_rate": "> 1%"
        }
        
        print(f"     Monitored Metrics: {len(monitoring_metrics)}")
        for metric in monitoring_metrics:
            threshold = alert_thresholds.get(metric, "Custom threshold")
            print(f"       {metric}: {threshold}")
        
        # Deployment rollout strategy
        print(f"\n   📈 Rollout Strategy:")
        
        rollout_phases = [
            {"phase": "Canary", "traffic": "5%", "duration": "24 hours"},
            {"phase": "Blue-Green", "traffic": "50%", "duration": "48 hours"},
            {"phase": "Full Rollout", "traffic": "100%", "duration": "Ongoing"}
        ]
        
        for phase_info in rollout_phases:
            print(f"     {phase_info['phase']}: {phase_info['traffic']} traffic for {phase_info['duration']}")
        
        print(f"\n   ✅ Model successfully prepared for production deployment!")
        
    else:
        # Improvement recommendations
        print(f"\n   🔧 Improvement Recommendations:")
        
        failed_checks = [metric for metric, result in validation_results.items() if not result["passed"]]
        
        for failed_metric in failed_checks:
            result = validation_results[failed_metric]
            actual = result["actual"]
            threshold = result["threshold"]
            
            if failed_metric == "max_inference_time":
                print(f"     ⚡ Optimize inference speed:")
                print(f"       Current: {actual:.3f}s, Target: ≤{threshold:.3f}s")
                print(f"       Suggestions: Model quantization, feature selection, ensemble pruning")
            
            elif failed_metric == "min_accuracy":
                print(f"     📈 Improve model accuracy:")
                print(f"       Current: {actual:.3f}, Target: ≥{threshold:.3f}")
                print(f"       Suggestions: More training data, hyperparameter tuning, ensemble methods")
            
            elif failed_metric == "max_model_size_mb":
                print(f"     💾 Reduce model size:")
                print(f"       Current: {actual:.1f}MB, Target: ≤{threshold:.1f}MB")
                print(f"       Suggestions: Model compression, knowledge distillation, pruning")


def main():
    """Run all model training examples."""
    print("🚀 Refunc Model Training Examples")
    print("=" * 60)
    
    if not REFUNC_AVAILABLE:
        print("ℹ️  Running in demo mode (showing code structure)")
    else:
        print("ℹ️  Running with full Refunc functionality")
    
    print()
    
    # Set random seed for reproducible examples
    random.seed(42)
    
    # Run examples
    basic_model_training_examples()
    hyperparameter_optimization_examples()
    model_evaluation_examples()
    production_deployment_examples()
    
    print("\n✅ Model training examples completed!")
    print("\n📖 Next steps:")
    print("- Implement model training pipelines in your ML projects")
    print("- Set up hyperparameter optimization for better performance")
    print("- Establish model evaluation and comparison frameworks")
    print("- Check out experiment_tracking.py for comprehensive logging")


if __name__ == "__main__":
    main()