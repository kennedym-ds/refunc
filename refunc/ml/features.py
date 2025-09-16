"""
Feature engineering and selection utilities.

This module provides comprehensive feature engineering functionality including
feature selection, importance analysis, dimensionality reduction, and automated
feature generation to improve model performance and interpretability.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier

from ..exceptions import RefuncError, ValidationError
from ..logging import get_logger


class FeatureSelectionMethod(Enum):
    """Enumeration of feature selection methods."""
    UNIVARIATE = "univariate"
    RFE = "rfe"
    MODEL_BASED = "model_based"


class DimensionalityReductionMethod(Enum):
    """Enumeration of dimensionality reduction methods."""
    PCA = "pca"


@dataclass
class FeatureImportanceResult:
    """Container for feature importance analysis results."""
    feature_names: List[str]
    importance_scores: np.ndarray
    method: str
    selected_features: Optional[List[str]] = None


class FeatureSelector:
    """Feature selection utilities."""
    
    def __init__(self, method: str = 'univariate'):
        self.method = method
        self.selector = None
        self.selected_features_ = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], **kwargs):
        """Fit feature selector."""
        if self.method == 'univariate':
            k = kwargs.get('k', 10)
            self.selector = SelectKBest(k=k)
        elif self.method == 'rfe':
            estimator = kwargs.get('estimator', RandomForestClassifier(n_estimators=10))
            n_features = kwargs.get('n_features', 10)
            self.selector = RFE(estimator, n_features_to_select=n_features)
        elif self.method == 'model_based':
            estimator = kwargs.get('estimator', RandomForestClassifier(n_estimators=10))
            self.selector = SelectFromModel(estimator)
            
        if self.selector is not None:
            self.selector.fit(X, y)
            
            if isinstance(X, pd.DataFrame):
                self.selected_features_ = X.columns[self.selector.get_support()].tolist()
            else:
                self.selected_features_ = [f"feature_{i}" for i in range(X.shape[1]) if self.selector.get_support()[i]]
            
        return self
        
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform data using fitted selector."""
        if self.selector is None:
            raise ValueError("FeatureSelector must be fitted first")
        result = self.selector.transform(X)
        return np.asarray(result)
        
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], **kwargs):
        """Fit selector and transform data."""
        return self.fit(X, y, **kwargs).transform(X)


class DimensionalityReducer:
    """Dimensionality reduction utilities."""
    
    def __init__(self, method: str = 'pca', n_components: Optional[int] = None):
        self.method = method
        self.n_components = n_components
        self.reducer = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], **kwargs):
        """Fit dimensionality reducer."""
        if self.method == 'pca':
            self.reducer = PCA(n_components=self.n_components)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        self.reducer.fit(X)
        return self
        
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform data using fitted reducer."""
        if self.reducer is None:
            raise ValueError("DimensionalityReducer must be fitted first")
        return self.reducer.transform(X)
        
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> np.ndarray:
        """Fit reducer and transform data."""
        return self.fit(X, **kwargs).transform(X)


class FeatureEngineer:
    """Automated feature engineering utilities."""
    
    def __init__(self):
        self.transformers = []
        
    def add_polynomial_features(self, degree: int = 2, include_bias: bool = False):
        """Add polynomial feature transformation."""
        transformer = PolynomialFeatures(degree=degree, include_bias=include_bias)
        self.transformers.append(('polynomial', transformer))
        return self
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None):
        """Fit all transformers."""
        for name, transformer in self.transformers:
            transformer.fit(X)
        return self
        
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Transform data using all fitted transformers."""
        result = X
        for name, transformer in self.transformers:
            result = transformer.transform(result)
        return result
        
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None):
        """Fit transformers and transform data."""
        return self.fit(X, y).transform(X)


# Convenience functions
def select_best_features(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    method: str = 'univariate',
    k: int = 10
) -> Tuple[Union[np.ndarray, pd.DataFrame], List[str]]:
    """Select best features using specified method."""
    selector = FeatureSelector(method)
    X_selected = selector.fit_transform(X, y, k=k)
    selected_features = selector.selected_features_ if selector.selected_features_ else []
    return X_selected, selected_features


def reduce_dimensions(
    X: Union[np.ndarray, pd.DataFrame],
    method: str = 'pca',
    n_components: Optional[int] = None
) -> np.ndarray:
    """Reduce data dimensionality using specified method."""
    reducer = DimensionalityReducer(method, n_components)
    return reducer.fit_transform(X)


def engineer_features(
    X: Union[np.ndarray, pd.DataFrame],
    polynomial_degree: int = 2
) -> Union[np.ndarray, pd.DataFrame]:
    """Engineer new features from existing ones."""
    engineer = FeatureEngineer()
    engineer.add_polynomial_features(degree=polynomial_degree)
    return engineer.fit_transform(X)