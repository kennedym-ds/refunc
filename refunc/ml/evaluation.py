"""
Machine Learning model evaluation utilities.

This module provides comprehensive model evaluation functionality including
metrics calculation, model comparison, cross-validation, and performance analysis
for classification, regression, and clustering tasks.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)

from ..exceptions import RefuncError, ValidationError
from ..logging import get_logger


@dataclass
class EvaluationResult:
    """Container for model evaluation results."""
    
    model_name: str
    task_type: str  # 'classification', 'regression', 'clustering'
    metrics: Dict[str, float]
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None
    
    def summary(self) -> str:
        """Get formatted summary of evaluation results."""
        lines = [
            f"Evaluation Results: {self.model_name}",
            "=" * 40,
            f"Task Type: {self.task_type}",
            "",
            "Metrics:"
        ]
        
        for metric, value in self.metrics.items():
            lines.append(f"  {metric}: {value:.4f}")
            
        return "\n".join(lines)


class ModelEvaluator:
    """Comprehensive model evaluation engine."""
    
    def __init__(self, task_type: str = 'auto'):
        """
        Initialize model evaluator.
        
        Args:
            task_type: Type of ML task ('classification', 'regression', 'clustering', 'auto')
        """
        self.task_type = task_type
        self.logger = get_logger(__name__)
        
    def evaluate(
        self,
        model,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        model_name: str = "Model"
    ) -> EvaluationResult:
        """
        Evaluate a model on test data.
        
        Args:
            model: Trained model with predict method
            X_test: Test features
            y_test: Test targets
            model_name: Name for the model
            
        Returns:
            EvaluationResult object
        """
        # Make predictions
        predictions = model.predict(X_test)
        
        # Determine task type if auto
        task_type = self._determine_task_type(y_test, predictions)
        
        # Calculate metrics based on task type
        if task_type == 'classification':
            return self._evaluate_classification(
                model, X_test, y_test, predictions, model_name
            )
        elif task_type == 'regression':
            return self._evaluate_regression(
                y_test, predictions, model_name
            )
        else:
            raise ValidationError(f"Unsupported task type: {task_type}")
            
    def _determine_task_type(self, y_true, y_pred) -> str:
        """Automatically determine the task type."""
        if self.task_type != 'auto':
            return self.task_type
            
        # Check if targets are continuous or discrete
        y_true_array = np.array(y_true)
        
        # If integer types and small number of unique values, likely classification
        if (np.issubdtype(y_true_array.dtype, np.integer) and 
            len(np.unique(y_true_array)) < 20):
            return 'classification'
        else:
            return 'regression'
            
    def _evaluate_classification(
        self,
        model,
        X_test,
        y_test,
        predictions,
        model_name: str
    ) -> EvaluationResult:
        """Evaluate classification model."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_test, predictions)
        metrics['precision'] = precision_score(y_test, predictions, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_test, predictions, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_test, predictions, average='weighted', zero_division=0)
        
        # ROC AUC if binary classification and probabilities available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(X_test)
                if len(np.unique(y_test)) == 2:  # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_test, probabilities[:, 1])
            except Exception:
                pass
                
        # Confusion matrix and classification report
        cm = confusion_matrix(y_test, predictions)
        report = str(classification_report(y_test, predictions))
        
        return EvaluationResult(
            model_name=model_name,
            task_type='classification',
            metrics=metrics,
            predictions=predictions,
            probabilities=probabilities,
            confusion_matrix=cm,
            classification_report=report
        )
        
    def _evaluate_regression(
        self,
        y_test,
        predictions,
        model_name: str
    ) -> EvaluationResult:
        """Evaluate regression model."""
        metrics = {}
        
        # Basic regression metrics
        metrics['mse'] = mean_squared_error(y_test, predictions)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_test, predictions)
        metrics['r2'] = r2_score(y_test, predictions)
        
        # Additional metrics
        residuals = np.array(y_test) - predictions
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        
        return EvaluationResult(
            model_name=model_name,
            task_type='regression',
            metrics=metrics,
            predictions=predictions
        )


class ModelComparator:
    """Compare multiple models on the same dataset."""
    
    def __init__(self, task_type: str = 'auto'):
        """
        Initialize model comparator.
        
        Args:
            task_type: Type of ML task
        """
        self.task_type = task_type
        self.evaluator = ModelEvaluator(task_type)
        self.results: List[EvaluationResult] = []
        
    def add_model(
        self,
        model,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        model_name: str
    ) -> EvaluationResult:
        """
        Add a model to comparison.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name for the model
            
        Returns:
            EvaluationResult for the model
        """
        result = self.evaluator.evaluate(model, X_test, y_test, model_name)
        self.results.append(result)
        return result
        
    def compare_models(self, primary_metric: Optional[str] = None) -> pd.DataFrame:
        """
        Compare all added models.
        
        Args:
            primary_metric: Primary metric for ranking
            
        Returns:
            DataFrame with comparison results
        """
        if not self.results:
            raise ValueError("No models added for comparison")
            
        # Create comparison dataframe
        comparison_data = []
        for result in self.results:
            row: Dict[str, Any] = {'model_name': result.model_name, 'task_type': result.task_type}
            row.update(result.metrics)
            comparison_data.append(row)
            
        df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric if specified
        if primary_metric and primary_metric in df.columns:
            ascending = primary_metric in ['mse', 'rmse', 'mae']  # Lower is better
            df = df.sort_values(primary_metric, ascending=ascending)
            
        return df
        
    def best_model(self, metric: Optional[str] = None) -> Optional[str]:
        """
        Get the name of the best performing model.
        
        Args:
            metric: Metric to use for selection
            
        Returns:
            Name of best model
        """
        if not self.results:
            raise ValueError("No models to compare")
            
        if metric is None:
            # Default metrics
            if self.results[0].task_type == 'classification':
                metric = 'f1'
            else:
                metric = 'r2'
                
        best_score = None
        best_model = None
        
        for result in self.results:
            if metric not in result.metrics:
                continue
                
            score = result.metrics[metric]
            
            # Determine if higher or lower is better
            higher_is_better = metric not in ['mse', 'rmse', 'mae']
            
            if (best_score is None or 
                (higher_is_better and score > best_score) or
                (not higher_is_better and score < best_score)):
                best_score = score
                best_model = result.model_name
                
        return best_model


# Convenience function
def compare_models(
    models: Dict[str, Any],
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.Series],
    task_type: str = 'auto'
) -> pd.DataFrame:
    """
    Compare multiple models.
    
    Args:
        models: Dictionary of {name: model} pairs
        X_test: Test features
        y_test: Test targets
        task_type: Type of ML task
        
    Returns:
        DataFrame with comparison results
    """
    comparator = ModelComparator(task_type)
    
    for name, model in models.items():
        comparator.add_model(model, X_test, y_test, name)
        
    return comparator.compare_models()