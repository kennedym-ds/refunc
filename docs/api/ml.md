# Machine Learning Module API Reference

The `refunc.ml` module provides comprehensive machine learning functionality including model management, evaluation metrics, feature engineering, and training pipelines to support end-to-end ML workflows.

---

## Overview

The ML module offers four core components:

- **Model Management**: Model registry, versioning, persistence, and metadata tracking
- **Model Evaluation**: Comprehensive metrics, comparison, and performance analysis
- **Feature Engineering**: Selection, importance analysis, dimensionality reduction
- **Training & Optimization**: Hyperparameter tuning, AutoML, and pipeline automation

**Key Features:**

- Support for classification, regression, and clustering tasks
- Model registry with versioning and metadata tracking
- Automated hyperparameter optimization with grid/random search
- Comprehensive model evaluation and comparison framework
- Feature selection and dimensionality reduction utilities
- AutoML capabilities for rapid model development

---

## Quick Start

```python
from refunc.ml import (
    SklearnModel, ModelRegistry, ModelEvaluator, ModelComparator,
    FeatureSelector, auto_train_models, optimize_hyperparameters
)

# Model management with registry
registry = ModelRegistry("./models")
model = SklearnModel(RandomForestClassifier(), name="rf_classifier")
registry.register(model)

# Automated model training and comparison
results = auto_train_models(X_train, y_train, task_type='classification')
best_model = results['best_model']

# Feature engineering
selector = FeatureSelector(method='rfe')
X_selected = selector.fit_transform(X_train, y_train, n_features=10)

# Model evaluation and comparison
evaluator = ModelEvaluator(task_type='classification')
eval_result = evaluator.evaluate(best_model, X_test, y_test)
print(eval_result.summary())
```

---

## Model Management

### BaseModel

Abstract base class for all models with standardized interface.

```python
from refunc.ml import BaseModel

class CustomModel(BaseModel):
    def __init__(self, name=None):
        super().__init__(name)
        # Custom initialization
        
    def fit(self, X, y=None, **kwargs):
        # Custom fitting logic
        self._is_fitted = True
        return self
        
    def predict(self, X, **kwargs):
        # Custom prediction logic
        return predictions

# Usage
model = CustomModel(name="my_model")
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Save and load models
model.save("model.pkl")
loaded_model = CustomModel.load("model.pkl")
```

**Key Methods:**

- `fit(X, y=None, **kwargs)`: Fit model to training data
- `predict(X, **kwargs)`: Make predictions on input data
- `save(path)`: Save model to file with pickle
- `load(path)`: Class method to load model from file
- `is_fitted`: Property to check if model is trained

### SklearnModel

Wrapper for scikit-learn models with enhanced functionality.

```python
from refunc.ml import SklearnModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC

# Classification model
rf_model = SklearnModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    name="random_forest_classifier"
)

# Regression model
gb_model = SklearnModel(
    GradientBoostingRegressor(n_estimators=100),
    name="gradient_boosting_regressor"
)

# Train models
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)

# Probability predictions (if supported)
if hasattr(rf_model._model, 'predict_proba'):
    probabilities = rf_model.predict_proba(X_test)
```

**Features:**

- Automatic metadata extraction from sklearn models
- Support for both classification and regression
- Probability prediction support for compatible models
- Consistent interface across different sklearn algorithms

### ModelMetadata

Comprehensive metadata tracking for models.

```python
from refunc.ml import ModelMetadata
from datetime import datetime

# Automatic metadata creation
metadata = ModelMetadata(
    name="production_model",
    version="2.1.0",
    description="Production RandomForest with optimized hyperparameters",
    model_type="sklearn.RandomForestClassifier",
    parameters={
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5
    },
    metrics={
        "accuracy": 0.95,
        "f1_score": 0.93,
        "roc_auc": 0.97
    },
    tags=["production", "optimized", "v2"]
)

# Access metadata
print(f"Model: {metadata.name} v{metadata.version}")
print(f"Created: {metadata.created_at}")
print(f"Metrics: {metadata.metrics}")
```

### ModelRegistry

Centralized registry for managing multiple models with versioning.

```python
from refunc.ml import ModelRegistry, SklearnModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Initialize registry
registry = ModelRegistry(storage_path="./model_registry")

# Register models
rf_model = SklearnModel(RandomForestClassifier(), name="random_forest")
gb_model = SklearnModel(GradientBoostingClassifier(), name="gradient_boost")

registry.register(rf_model)
registry.register(gb_model)

# List all models
models = registry.list_models()
print(f"Registered models: {models}")

# Retrieve models
retrieved_rf = registry.get("random_forest")
retrieved_gb = registry.get("gradient_boost")

# Remove models
registry.remove("old_model")

# Global registry functions
from refunc.ml import register_model, get_model, list_models

# Use global registry
register_model(rf_model, "global_rf")
model = get_model("global_rf")
all_models = list_models()
```

**Features:**

- Persistent storage with automatic file management
- Model versioning and metadata tracking
- Thread-safe operations for concurrent access
- Global registry instance for convenience
- Automatic model loading on demand

---

## Model Evaluation

### ModelEvaluator

Comprehensive model evaluation engine with task-specific metrics.

```python
from refunc.ml import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator(task_type='classification')  # or 'regression', 'auto'

# Evaluate a model
result = evaluator.evaluate(
    model=trained_model,
    X_test=X_test,
    y_test=y_test,
    model_name="RandomForest"
)

# Print evaluation summary
print(result.summary())

# Access specific metrics
print(f"Accuracy: {result.metrics['accuracy']:.4f}")
print(f"F1 Score: {result.metrics['f1']:.4f}")
print(f"ROC AUC: {result.metrics.get('roc_auc', 'N/A')}")

# Access additional results
if result.confusion_matrix is not None:
    print("Confusion Matrix:")
    print(result.confusion_matrix)

if result.classification_report:
    print("Classification Report:")
    print(result.classification_report)
```

**Classification Metrics:**

- Accuracy, Precision, Recall, F1-Score
- ROC AUC (for binary classification with probabilities)
- Confusion Matrix
- Detailed Classification Report

**Regression Metrics:**

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Residual statistics (mean, standard deviation)

### EvaluationResult

Container for comprehensive evaluation results.

```python
from refunc.ml import EvaluationResult

# Example result structure
result = EvaluationResult(
    model_name="RandomForest",
    task_type="classification",
    metrics={
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.93,
        "f1": 0.93,
        "roc_auc": 0.97
    },
    predictions=predictions_array,
    probabilities=probabilities_array,
    confusion_matrix=confusion_matrix,
    classification_report=classification_report_str
)

# Format and display results
print(result.summary())

# Access individual components
accuracy = result.metrics['accuracy']
cm = result.confusion_matrix
predictions = result.predictions
```

### ModelComparator

Compare multiple models on the same dataset with ranking.

```python
from refunc.ml import ModelComparator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Initialize comparator
comparator = ModelComparator(task_type='classification')

# Train and add models for comparison
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100),
    'SVM': SVC(probability=True)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    comparator.add_model(model, X_test, y_test, name)

# Get comparison results
comparison_df = comparator.compare_models(primary_metric='f1')
print(comparison_df)

# Find best model
best_model_name = comparator.best_model(metric='f1')
print(f"Best model: {best_model_name}")

# Convenience function for quick comparison
from refunc.ml.evaluation import compare_models

comparison_df = compare_models(
    models={'RF': rf_model, 'GB': gb_model, 'SVM': svm_model},
    X_test=X_test,
    y_test=y_test,
    task_type='classification'
)
```

**Features:**

- Automatic task type detection
- Ranking by any available metric
- Support for both classification and regression
- Comprehensive comparison tables
- Best model identification

---

## Feature Engineering

### FeatureSelector

Advanced feature selection with multiple algorithms.

```python
from refunc.ml import FeatureSelector

# Univariate feature selection
univariate_selector = FeatureSelector(method='univariate')
X_selected = univariate_selector.fit_transform(X_train, y_train, k=10)
selected_features = univariate_selector.selected_features_

# Recursive Feature Elimination (RFE)
from sklearn.ensemble import RandomForestClassifier

rfe_selector = FeatureSelector(method='rfe')
X_selected = rfe_selector.fit_transform(
    X_train, y_train,
    estimator=RandomForestClassifier(n_estimators=50),
    n_features=15
)

# Model-based feature selection
model_selector = FeatureSelector(method='model_based')
X_selected = model_selector.fit_transform(
    X_train, y_train,
    estimator=RandomForestClassifier(n_estimators=50)
)

# Transform test data
X_test_selected = model_selector.transform(X_test)

print(f"Selected features: {model_selector.selected_features_}")
print(f"Original shape: {X_train.shape}, Selected shape: {X_selected.shape}")
```

**Selection Methods:**

- **Univariate**: Statistical tests for feature-target relationships
- **RFE**: Recursive elimination with model feedback
- **Model-based**: Feature importance from tree-based models

### DimensionalityReducer

Dimensionality reduction for high-dimensional data.

```python
from refunc.ml import DimensionalityReducer

# Principal Component Analysis (PCA)
pca_reducer = DimensionalityReducer(method='pca', n_components=50)
X_reduced = pca_reducer.fit_transform(X_train)

# Transform test data
X_test_reduced = pca_reducer.transform(X_test)

# Explained variance (for PCA)
if hasattr(pca_reducer.reducer, 'explained_variance_ratio_'):
    explained_var = pca_reducer.reducer.explained_variance_ratio_
    cumulative_var = explained_var.cumsum()
    print(f"Cumulative explained variance: {cumulative_var[-1]:.3f}")

print(f"Original shape: {X_train.shape}, Reduced shape: {X_reduced.shape}")
```

### FeatureEngineer

Automated feature engineering and transformation.

```python
from refunc.ml import FeatureEngineer

# Create feature engineer
engineer = FeatureEngineer()

# Add polynomial features
engineer.add_polynomial_features(degree=2, include_bias=False)

# Apply transformations
X_engineered = engineer.fit_transform(X_train)
X_test_engineered = engineer.transform(X_test)

print(f"Original features: {X_train.shape[1]}")
print(f"Engineered features: {X_engineered.shape[1]}")
```

### Convenience Functions

Quick feature engineering operations.

```python
from refunc.ml import select_best_features, reduce_dimensions, engineer_features

# Quick feature selection
X_selected, selected_feature_names = select_best_features(
    X_train, y_train, 
    method='rfe',
    k=20
)

# Quick dimensionality reduction
X_reduced = reduce_dimensions(
    X_train,
    method='pca',
    n_components=30
)

# Quick feature engineering
X_engineered = engineer_features(
    X_train,
    polynomial_degree=2
)
```

---

## Training & Optimization

### HyperparameterOptimizer

Advanced hyperparameter optimization with multiple search strategies.

```python
from refunc.ml import HyperparameterOptimizer
from sklearn.ensemble import RandomForestClassifier

# Initialize optimizer
optimizer = HyperparameterOptimizer(
    method='random',  # or 'grid'
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42
)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Optimize hyperparameters
best_model = optimizer.optimize(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    X=X_train,
    y=y_train,
    n_iter=50  # for random search
)

# Get optimization results
results = optimizer.get_results()
print(f"Best parameters: {results['best_params']}")
print(f"Best cross-validation score: {results['best_score']:.4f}")

# Access detailed CV results
cv_results = results['cv_results']
```

**Optimization Methods:**

- **Grid Search**: Exhaustive search over parameter grid
- **Random Search**: Random sampling from parameter distributions
- Configurable cross-validation and scoring metrics
- Parallel processing support

### AutoMLTrainer

Automated machine learning with model selection and optimization.

```python
from refunc.ml import AutoMLTrainer

# Initialize AutoML trainer
trainer = AutoMLTrainer(
    task_type='classification',  # or 'regression'
    cv=5,
    scoring='f1_weighted',
    random_state=42,
    max_models=8
)

# Automatic training and model selection
results = trainer.train(X_train, y_train)

# Get training results
print(f"Best model: {type(results['best_model']).__name__}")
print(f"Best cross-validation score: {results['best_score']:.4f}")
print(f"Total models trained: {results['model_count']}")

# Access all model results
for model_info in results['all_models']:
    print(f"{model_info['name']}: {model_info['cv_score_mean']:.4f} ± {model_info['cv_score_std']:.4f}")

# Make predictions with best model
best_model = results['best_model']
predictions = trainer.predict(X_test)

# Or use the best model directly
predictions = best_model.predict(X_test)
```

**Default Models (Classification):**

- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine
- Logistic Regression

**Default Models (Regression):**

- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regression
- Ridge Regression

### Training Convenience Functions

Quick training and optimization operations.

```python
from refunc.ml import optimize_hyperparameters, auto_train_models

# Quick hyperparameter optimization
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None]
}

best_model = optimize_hyperparameters(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    X=X_train,
    y=y_train,
    method='grid',
    cv=5,
    scoring='f1_weighted'
)

# Quick AutoML training
results = auto_train_models(
    X=X_train,
    y=y_train,
    task_type='classification',
    cv=5,
    scoring='f1_weighted',
    max_models=5
)

best_model = results['best_model']
```

---

## Advanced Usage

### Custom Model Pipeline

```python
from refunc.ml import (
    SklearnModel, ModelRegistry, ModelEvaluator,
    FeatureSelector, DimensionalityReducer
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

class MLPipeline:
    def __init__(self, registry_path="./models"):
        self.registry = ModelRegistry(registry_path)
        self.evaluator = ModelEvaluator(task_type='auto')
        
    def create_preprocessing_pipeline(self):
        """Create preprocessing pipeline."""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selector', FeatureSelector(method='rfe')),
            ('dim_reducer', DimensionalityReducer(method='pca', n_components=50))
        ])
    
    def train_and_evaluate_model(self, X_train, y_train, X_test, y_test, model_name):
        """Train, register, and evaluate a model."""
        # Preprocessing
        preprocessor = self.create_preprocessing_pipeline()
        
        # Full pipeline with model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train pipeline
        pipeline.fit(X_train, y_train)
        
        # Wrap and register model
        sklearn_model = SklearnModel(pipeline, name=model_name)
        sklearn_model._is_fitted = True  # Mark as fitted
        self.registry.register(sklearn_model)
        
        # Evaluate model
        result = self.evaluator.evaluate(pipeline, X_test, y_test, model_name)
        
        return result

# Usage
pipeline_manager = MLPipeline()
result = pipeline_manager.train_and_evaluate_model(
    X_train, y_train, X_test, y_test, "preprocessed_rf"
)
print(result.summary())
```

### Model Comparison with Cross-Validation

```python
from refunc.ml import ModelComparator, HyperparameterOptimizer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def compare_optimized_models(X, y, test_size=0.2, random_state=42):
    """Compare multiple optimized models with proper validation."""
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Define models and their parameter grids
    models_config = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=random_state),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=random_state),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=random_state),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
    }
    
    # Initialize comparator
    comparator = ModelComparator(task_type='classification')
    
    # Optimize and add each model
    for name, config in models_config.items():
        print(f"Optimizing {name}...")
        
        optimizer = HyperparameterOptimizer(
            method='random',
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
            scoring='f1_weighted',
            random_state=random_state
        )
        
        best_model = optimizer.optimize(
            config['model'], config['params'], X_train, y_train, n_iter=20
        )
        
        comparator.add_model(best_model, X_test, y_test, name)
    
    # Compare models
    comparison_df = comparator.compare_models(primary_metric='f1')
    best_model_name = comparator.best_model('f1')
    
    return comparison_df, best_model_name

# Usage
comparison_results, best_model = compare_optimized_models(X, y)
print(comparison_results)
print(f"Best model: {best_model}")
```

### Feature Engineering Pipeline

```python
from refunc.ml import FeatureSelector, DimensionalityReducer, FeatureEngineer

class AdvancedFeatureEngineer:
    def __init__(self):
        self.feature_selector = None
        self.dim_reducer = None
        self.feature_engineer = None
        self.selected_features_ = None
        
    def fit(self, X, y, selection_method='rfe', n_features=None, 
            reduce_dimensions=True, n_components=None, 
            polynomial_features=False, poly_degree=2):
        """Fit comprehensive feature engineering pipeline."""
        
        # Step 1: Feature Engineering (create new features)
        if polynomial_features:
            self.feature_engineer = FeatureEngineer()
            self.feature_engineer.add_polynomial_features(degree=poly_degree)
            X_engineered = self.feature_engineer.fit_transform(X)
        else:
            X_engineered = X
            
        # Step 2: Feature Selection
        if n_features:
            self.feature_selector = FeatureSelector(method=selection_method)
            X_selected = self.feature_selector.fit_transform(
                X_engineered, y, n_features=n_features
            )
            self.selected_features_ = self.feature_selector.selected_features_
        else:
            X_selected = X_engineered
            
        # Step 3: Dimensionality Reduction
        if reduce_dimensions:
            if n_components is None:
                n_components = min(50, X_selected.shape[1] // 2)
                
            self.dim_reducer = DimensionalityReducer(
                method='pca', n_components=n_components
            )
            X_final = self.dim_reducer.fit_transform(X_selected)
        else:
            X_final = X_selected
            
        return X_final
    
    def transform(self, X):
        """Transform new data using fitted pipeline."""
        X_transformed = X
        
        if self.feature_engineer:
            X_transformed = self.feature_engineer.transform(X_transformed)
            
        if self.feature_selector:
            X_transformed = self.feature_selector.transform(X_transformed)
            
        if self.dim_reducer:
            X_transformed = self.dim_reducer.transform(X_transformed)
            
        return X_transformed
    
    def fit_transform(self, X, y, **kwargs):
        """Fit pipeline and transform data."""
        return self.fit(X, y, **kwargs)

# Usage
engineer = AdvancedFeatureEngineer()
X_train_transformed = engineer.fit(
    X_train, y_train,
    selection_method='rfe',
    n_features=30,
    reduce_dimensions=True,
    n_components=20,
    polynomial_features=True,
    poly_degree=2
)

X_test_transformed = engineer.transform(X_test)
print(f"Original: {X_train.shape} -> Transformed: {X_train_transformed.shape}")
```

---

## Error Handling

The ML module provides comprehensive error handling with descriptive messages.

### Common Exceptions

```python
from refunc.exceptions import RefuncError, ValidationError

try:
    # Model prediction without fitting
    model = SklearnModel(RandomForestClassifier())
    predictions = model.predict(X_test)
except ValueError as e:
    print(f"Model not fitted: {e}")

try:
    # Invalid optimization method
    optimizer = HyperparameterOptimizer(method='invalid_method')
except ValueError as e:
    print(f"Invalid method: {e}")

try:
    # Model not found in registry
    model = registry.get("nonexistent_model")
except KeyError as e:
    print(f"Model not found: {e}")

try:
    # Invalid task type
    evaluator = ModelEvaluator(task_type='invalid_task')
    result = evaluator.evaluate(model, X_test, y_test)
except ValidationError as e:
    print(f"Validation error: {e}")
```

### Robust Model Training

```python
def robust_model_training(X, y, models_config, max_retries=3):
    """Robust model training with error handling."""
    successful_models = []
    failed_models = []
    
    for name, config in models_config.items():
        for attempt in range(max_retries):
            try:
                print(f"Training {name} (attempt {attempt + 1})")
                
                # Train model
                model = config['model']
                model.fit(X, y)
                
                # Validate model
                test_pred = model.predict(X[:10])  # Quick validation
                
                successful_models.append((name, model))
                print(f"✓ {name} trained successfully")
                break
                
            except Exception as e:
                print(f"✗ {name} failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    failed_models.append((name, str(e)))
    
    return successful_models, failed_models

# Usage with error handling
models = {
    'RF': {'model': RandomForestClassifier()},
    'GB': {'model': GradientBoostingClassifier()},
    'SVM': {'model': SVC()}
}

successful, failed = robust_model_training(X_train, y_train, models)
print(f"Successful models: {len(successful)}")
print(f"Failed models: {len(failed)}")
```

---

## Best Practices

### 1. Model Lifecycle Management

```python
# Organize models with proper naming and versioning
registry = ModelRegistry("./production_models")

# Use descriptive names and versions
model_name = f"fraud_detection_rf_v{version}"
rf_model = SklearnModel(
    RandomForestClassifier(n_estimators=200, random_state=42),
    name=model_name
)

# Add comprehensive metadata
rf_model.metadata.description = "Production fraud detection model with optimized hyperparameters"
rf_model.metadata.version = "2.1.0"
rf_model.metadata.tags = ["production", "fraud", "optimized"]
rf_model.metadata.parameters = model.get_params()

# Register with evaluation metrics
registry.register(rf_model)
```

### 2. Comprehensive Model Evaluation

```python
def comprehensive_evaluation(model, X_test, y_test, model_name):
    """Perform comprehensive model evaluation."""
    evaluator = ModelEvaluator(task_type='auto')
    
    # Basic evaluation
    result = evaluator.evaluate(model, X_test, y_test, model_name)
    
    # Additional analysis for classification
    if result.task_type == 'classification':
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': range(len(model.feature_importances_)),
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("Top 10 important features:")
            print(importance_df.head(10))
    
    return result

# Use comprehensive evaluation
result = comprehensive_evaluation(trained_model, X_test, y_test, "production_model")
print(result.summary())
```

### 3. Efficient Feature Engineering

```python
# Cache expensive feature engineering operations
from functools import lru_cache

class CachedFeatureEngineer:
    def __init__(self):
        self.cache = {}
    
    @lru_cache(maxsize=128)
    def _cached_polynomial_features(self, X_hash, degree):
        """Cache polynomial feature transformation."""
        engineer = FeatureEngineer()
        engineer.add_polynomial_features(degree=degree)
        return engineer.fit_transform(X)
    
    def smart_feature_selection(self, X, y, max_features=100):
        """Intelligent feature selection with multiple methods."""
        results = {}
        
        # Try different selection methods
        methods = ['univariate', 'rfe', 'model_based']
        
        for method in methods:
            try:
                selector = FeatureSelector(method=method)
                X_selected = selector.fit_transform(X, y, k=min(max_features, X.shape[1]))
                
                # Quick model evaluation to assess feature quality
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import cross_val_score
                
                quick_model = RandomForestClassifier(n_estimators=10, random_state=42)
                scores = cross_val_score(quick_model, X_selected, y, cv=3)
                
                results[method] = {
                    'score': scores.mean(),
                    'features': selector.selected_features_,
                    'n_features': X_selected.shape[1]
                }
                
            except Exception as e:
                print(f"Method {method} failed: {e}")
                continue
        
        # Return best method
        best_method = max(results.keys(), key=lambda k: results[k]['score'])
        return results[best_method], best_method

# Usage
engineer = CachedFeatureEngineer()
best_features, method = engineer.smart_feature_selection(X_train, y_train)
print(f"Best method: {method}, Score: {best_features['score']:.4f}")
```

### 4. Automated Model Selection

```python
def automated_model_selection(X, y, test_size=0.2, random_state=42):
    """Automated model selection with comprehensive evaluation."""
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Feature engineering
    engineer = AdvancedFeatureEngineer()
    X_train_eng = engineer.fit(X_train, y_train, n_features=20)
    X_test_eng = engineer.transform(X_test)
    
    # AutoML training
    trainer = AutoMLTrainer(
        task_type='auto',
        cv=5,
        random_state=random_state,
        max_models=6
    )
    
    results = trainer.train(X_train_eng, y_train)
    
    # Evaluate best model
    best_model = results['best_model']
    evaluator = ModelEvaluator(task_type='auto')
    evaluation = evaluator.evaluate(best_model, X_test_eng, y_test, "best_automl")
    
    return {
        'best_model': best_model,
        'feature_engineer': engineer,
        'evaluation': evaluation,
        'all_results': results
    }

# Complete automated pipeline
pipeline_results = automated_model_selection(X, y)
print(pipeline_results['evaluation'].summary())
```

---

## Examples

### Complete ML Workflow

```python
from refunc.ml import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd

class MLWorkflow:
    def __init__(self, registry_path="./ml_workflow"):
        self.registry = ModelRegistry(registry_path)
        self.comparator = ModelComparator(task_type='auto')
        self.results = {}
        
    def run_complete_workflow(self, X, y, test_size=0.2):
        """Run complete ML workflow from data to production model."""
        print("🚀 Starting complete ML workflow...")
        
        # 1. Data splitting
        print("📊 Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 2. Feature engineering
        print("🔧 Engineering features...")
        engineer = AdvancedFeatureEngineer()
        X_train_eng = engineer.fit(
            X_train, y_train,
            selection_method='rfe',
            n_features=15,
            reduce_dimensions=True,
            n_components=10
        )
        X_test_eng = engineer.transform(X_test)
        
        # 3. AutoML training
        print("🤖 Training models with AutoML...")
        automl_results = auto_train_models(
            X_train_eng, y_train,
            task_type='classification',
            max_models=5
        )
        
        # 4. Manual model optimization
        print("⚙️ Optimizing selected models...")
        from sklearn.ensemble import RandomForestClassifier
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5, 10]
        }
        
        optimized_rf = optimize_hyperparameters(
            RandomForestClassifier(random_state=42),
            param_grid,
            X_train_eng, y_train,
            method='random',
            cv=5
        )
        
        # 5. Model comparison
        print("📈 Comparing models...")
        models_to_compare = {
            'AutoML_Best': automl_results['best_model'],
            'Optimized_RF': optimized_rf
        }
        
        for name, model in models_to_compare.items():
            self.comparator.add_model(model, X_test_eng, y_test, name)
        
        comparison_df = self.comparator.compare_models('f1')
        best_model_name = self.comparator.best_model('f1')
        
        # 6. Model registration
        print("📝 Registering best model...")
        best_model = models_to_compare[best_model_name]
        
        wrapped_model = SklearnModel(best_model, name=f"production_{best_model_name}")
        wrapped_model.metadata.description = f"Best performing model: {best_model_name}"
        wrapped_model.metadata.version = "1.0.0"
        wrapped_model.metadata.tags = ["production", "optimized", "automated"]
        
        self.registry.register(wrapped_model)
        
        # 7. Final evaluation
        print("🎯 Final evaluation...")
        evaluator = ModelEvaluator(task_type='classification')
        final_evaluation = evaluator.evaluate(
            best_model, X_test_eng, y_test, f"final_{best_model_name}"
        )
        
        self.results = {
            'feature_engineer': engineer,
            'comparison_df': comparison_df,
            'best_model_name': best_model_name,
            'best_model': best_model,
            'final_evaluation': final_evaluation,
            'automl_results': automl_results
        }
        
        print("✅ Workflow completed!")
        return self.results
    
    def get_production_model(self):
        """Get the production-ready model with preprocessing."""
        if not self.results:
            raise ValueError("Workflow not completed yet")
            
        return {
            'preprocessor': self.results['feature_engineer'],
            'model': self.results['best_model'],
            'evaluation': self.results['final_evaluation']
        }

# Generate sample data
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_redundant=5, n_clusters_per_class=1, random_state=42
)

# Run complete workflow
workflow = MLWorkflow()
results = workflow.run_complete_workflow(X, y)

# Print results
print("\n📋 Workflow Results:")
print(f"Best model: {results['best_model_name']}")
print(f"Model comparison:")
print(results['comparison_df'])
print(f"\nFinal evaluation:")
print(results['final_evaluation'].summary())

# Get production model
production_setup = workflow.get_production_model()
```

### Model Monitoring and Drift Detection

```python
class ModelMonitor:
    def __init__(self, model_registry):
        self.registry = model_registry
        self.performance_history = []
        
    def monitor_model_performance(self, model_name, X_new, y_new):
        """Monitor model performance on new data."""
        model = self.registry.get(model_name)
        evaluator = ModelEvaluator(task_type='auto')
        
        # Evaluate on new data
        result = evaluator.evaluate(model, X_new, y_new, f"{model_name}_monitoring")
        
        # Store performance history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'model_name': model_name,
            'metrics': result.metrics,
            'data_size': len(X_new)
        })
        
        return result
    
    def detect_performance_drift(self, model_name, threshold=0.05):
        """Detect performance drift in model."""
        model_history = [
            entry for entry in self.performance_history 
            if entry['model_name'] == model_name
        ]
        
        if len(model_history) < 2:
            return False, "Insufficient history for drift detection"
        
        # Compare recent performance with baseline
        baseline_metrics = model_history[0]['metrics']
        recent_metrics = model_history[-1]['metrics']
        
        drift_detected = False
        drift_details = {}
        
        for metric in baseline_metrics:
            if metric in recent_metrics:
                baseline_value = baseline_metrics[metric]
                recent_value = recent_metrics[metric]
                change = abs(recent_value - baseline_value)
                
                if change > threshold:
                    drift_detected = True
                    drift_details[metric] = {
                        'baseline': baseline_value,
                        'recent': recent_value,
                        'change': change
                    }
        
        return drift_detected, drift_details

# Usage
monitor = ModelMonitor(registry)

# Simulate monitoring over time
for i in range(5):
    # Simulate new data (with potential drift)
    X_new, y_new = make_classification(
        n_samples=100, n_features=20, 
        random_state=42 + i  # Different random state simulates drift
    )
    
    result = monitor.monitor_model_performance("production_model", X_new, y_new)
    drift_detected, drift_info = monitor.detect_performance_drift("production_model")
    
    print(f"Week {i+1}: F1 Score = {result.metrics.get('f1', 'N/A'):.4f}")
    if drift_detected:
        print(f"⚠️ Performance drift detected: {drift_info}")
```

---

## See Also

- **[⚠️ Exceptions](exceptions.md)** - Error handling for ML operations
- **[📝 Logging](logging.md)** - Logging integration for ML workflows
- **[📊 Data Science](data_science.md)** - Data analysis and preprocessing utilities
- **[🚀 Quick Start Guide](../guides/quickstart.md)** - Getting started with ML workflows
- **[💡 Examples](../examples/)** - More ML examples and use cases
