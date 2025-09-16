"""
Machine Learning utilities package.

This module provides comprehensive machine learning functionality including
model management, evaluation metrics, feature engineering, and training pipelines
to support end-to-end ML workflows.
"""

# Model management
from .models import BaseModel, SklearnModel, ModelRegistry

# Model evaluation
from .evaluation import ModelEvaluator, ModelComparator

# Feature engineering
from .features import (
    FeatureSelector, DimensionalityReducer, FeatureEngineer,
    select_best_features, reduce_dimensions, engineer_features
)

# Training and optimization
from .training import (
    HyperparameterOptimizer, AutoMLTrainer,
    optimize_hyperparameters, auto_train_models
)

__all__ = [
    # Model management
    'BaseModel', 'SklearnModel', 'ModelRegistry',
    
    # Model evaluation
    'ModelEvaluator', 'ModelComparator',
    
    # Feature engineering
    'FeatureSelector', 'DimensionalityReducer', 'FeatureEngineer',
    'select_best_features', 'reduce_dimensions', 'engineer_features',
    
    # Training and optimization
    'HyperparameterOptimizer', 'AutoMLTrainer',
    'optimize_hyperparameters', 'auto_train_models'
]

__version__ = "0.1.0"