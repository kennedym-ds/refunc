"""ML training utilities for refunc package."""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import warnings


class HyperparameterOptimizer:
    """Hyperparameter optimization utilities."""
    
    def __init__(self, 
                 method: str = 'grid',
                 cv: int = 5,
                 scoring: Optional[str] = None,
                 n_jobs: int = -1,
                 random_state: Optional[int] = None):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            method: Optimization method ('grid' or 'random')
            cv: Number of cross-validation folds
            scoring: Scoring metric to optimize
            n_jobs: Number of parallel jobs
            random_state: Random state for reproducibility
        """
        self.method = method
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.optimizer: Optional[Union[GridSearchCV, RandomizedSearchCV]] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        
    def optimize(self,
                estimator: BaseEstimator,
                param_grid: Dict[str, List[Any]],
                X: Union[np.ndarray, pd.DataFrame],
                y: Union[np.ndarray, pd.Series],
                n_iter: int = 100) -> BaseEstimator:
        """
        Optimize hyperparameters for given estimator.
        
        Args:
            estimator: Model to optimize
            param_grid: Parameter search space
            X: Training features
            y: Training targets
            n_iter: Number of iterations for random search
            
        Returns:
            Best fitted estimator
        """
        if self.method == 'grid':
            self.optimizer = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                return_train_score=True
            )
        elif self.method == 'random':
            self.optimizer = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                return_train_score=True
            )
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")
            
        self.optimizer.fit(X, y)
        self.best_params_ = self.optimizer.best_params_
        self.best_score_ = self.optimizer.best_score_
        
        return self.optimizer.best_estimator_
    
    def get_results(self) -> Dict[str, Any]:
        """Get optimization results."""
        if self.optimizer is None:
            raise ValueError("Optimizer has not been fitted yet")
            
        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'cv_results': self.optimizer.cv_results_
        }


class AutoMLTrainer:
    """Automated machine learning trainer."""
    
    def __init__(self, 
                 task_type: str = 'classification',
                 cv: int = 5,
                 scoring: Optional[str] = None,
                 random_state: Optional[int] = None,
                 max_models: int = 10):
        """
        Initialize AutoML trainer.
        
        Args:
            task_type: Type of ML task ('classification' or 'regression')
            cv: Number of cross-validation folds
            scoring: Scoring metric
            random_state: Random state for reproducibility
            max_models: Maximum number of models to try
        """
        self.task_type = task_type
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.max_models = max_models
        self.models_: List[Dict[str, Any]] = []
        self.best_model_: Optional[BaseEstimator] = None
        self.best_score_: Optional[float] = None
        
    def _get_default_models(self) -> List[Tuple[str, BaseEstimator, Dict[str, List[Any]]]]:
        """Get default models and parameter grids."""
        if self.task_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.svm import SVC
            from sklearn.linear_model import LogisticRegression
            
            models = [
                ('rf', RandomForestClassifier(random_state=self.random_state), {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5, 10]
                }),
                ('gb', GradientBoostingClassifier(random_state=self.random_state), {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }),
                ('svm', SVC(random_state=self.random_state), {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }),
                ('lr', LogisticRegression(random_state=self.random_state), {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],  # Use only l2 penalty for compatibility
                    'solver': ['lbfgs']  # lbfgs only supports l2 or None penalty
                })
            ]
        else:  # regression
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.svm import SVR
            from sklearn.linear_model import LinearRegression, Ridge
            
            models = [
                ('rf', RandomForestRegressor(random_state=self.random_state), {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5, 10]
                }),
                ('gb', GradientBoostingRegressor(random_state=self.random_state), {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }),
                ('svr', SVR(), {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }),
                ('ridge', Ridge(random_state=self.random_state), {
                    'alpha': [0.1, 1, 10, 100]
                })
            ]
            
        return models[:self.max_models]
    
    def train(self,
              X: Union[np.ndarray, pd.DataFrame],
              y: Union[np.ndarray, pd.Series],
              models: Optional[List[Tuple[str, BaseEstimator, Dict[str, List[Any]]]]] = None) -> Dict[str, Any]:
        """
        Train multiple models and select the best one.
        
        Args:
            X: Training features
            y: Training targets
            models: Custom models to try (optional)
            
        Returns:
            Training results summary
        """
        if models is None:
            models = self._get_default_models()
            
        self.models_ = []
        best_score = float('-inf')
        
        for name, model, param_grid in models:
            try:
                # Calculate parameter space size
                param_space_size = 1
                for param_values in param_grid.values():
                    param_space_size *= len(param_values)
                
                # Choose search method and iterations based on parameter space size
                if param_space_size <= 10:
                    # Use grid search for small parameter spaces
                    search_method = 'grid'
                    n_iter = 10  # Default value, won't be used for grid search
                else:
                    # Use random search for larger parameter spaces
                    search_method = 'random'
                    n_iter = min(20, param_space_size)  # Don't exceed total combinations
                
                # Optimize hyperparameters
                optimizer = HyperparameterOptimizer(
                    method=search_method,
                    cv=self.cv,
                    scoring=self.scoring,
                    random_state=self.random_state
                )
                
                if search_method == 'grid':
                    best_model = optimizer.optimize(model, param_grid, X, y)
                else:
                    best_model = optimizer.optimize(model, param_grid, X, y, n_iter=n_iter)
                
                # Cross-validate best model
                cv_scores = cross_val_score(
                    best_model, X, y, 
                    cv=self.cv, 
                    scoring=self.scoring
                )
                
                mean_score = float(np.mean(cv_scores))
                std_score = float(np.std(cv_scores))
                
                model_info = {
                    'name': name,
                    'model': best_model,
                    'best_params': optimizer.best_params_,
                    'cv_score_mean': mean_score,
                    'cv_score_std': std_score,
                    'cv_scores': cv_scores.tolist()
                }
                
                self.models_.append(model_info)
                
                if mean_score > best_score:
                    best_score = mean_score
                    self.best_model_ = best_model
                    self.best_score_ = mean_score
                    
            except Exception as e:
                warnings.warn(f"Failed to train model {name}: {str(e)}")
                continue
        
        return self.get_results()
    
    def get_results(self) -> Dict[str, Any]:
        """Get training results summary."""
        if not self.models_:
            raise ValueError("No models have been trained yet")
            
        # Sort models by performance
        sorted_models = sorted(
            self.models_, 
            key=lambda x: x['cv_score_mean'], 
            reverse=True
        )
        
        return {
            'best_model': self.best_model_,
            'best_score': self.best_score_,
            'all_models': sorted_models,
            'model_count': len(self.models_)
        }
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the best model."""
        if self.best_model_ is None:
            raise ValueError("No best model available. Train models first.")
        # Type ignore for BaseEstimator predict method
        return self.best_model_.predict(X)  # type: ignore


# Convenience functions
def optimize_hyperparameters(
    estimator: BaseEstimator,
    param_grid: Dict[str, List[Any]],
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    method: str = 'grid',
    cv: int = 5,
    scoring: Optional[str] = None
) -> Dict[str, Any]:
    """Optimize hyperparameters for given estimator."""
    optimizer = HyperparameterOptimizer(method=method, cv=cv, scoring=scoring)
    optimizer.optimize(estimator, param_grid, X, y)
    if optimizer.best_params_ is None:
        raise RuntimeError("Hyperparameter optimization failed to find best parameters")
    return optimizer.best_params_


def auto_train_models(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    task_type: str = 'classification',
    cv: int = 5,
    scoring: Optional[str] = None,
    max_models: int = 5
) -> Dict[str, Any]:
    """Automatically train and compare multiple models."""
    trainer = AutoMLTrainer(
        task_type=task_type,
        cv=cv,
        scoring=scoring,
        max_models=max_models
    )
    return trainer.train(X, y)