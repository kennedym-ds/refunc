"""
Data transformation and preprocessing pipelines.

This module provides composable data transformation utilities including
preprocessing pipelines, feature engineering, data cleaning,
and transformation chains with logging and error handling.
"""

import warnings
from typing import List, Dict, Any, Optional, Union, Callable, Tuple, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import copy
from datetime import datetime

from ..exceptions import RefuncError, ValidationError
from ..logging import get_logger


class TransformationType(Enum):
    """Types of data transformations."""
    CLEANING = "cleaning"
    SCALING = "scaling"
    ENCODING = "encoding"
    IMPUTATION = "imputation"
    FEATURE_SELECTION = "feature_selection"
    FEATURE_ENGINEERING = "feature_engineering"
    CUSTOM = "custom"


class ScalingMethod(Enum):
    """Scaling methods."""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    POWER = "power"


class ImputationMethod(Enum):
    """Imputation methods."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    KNN = "knn"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"


@dataclass
class TransformationResult:
    """Result of a data transformation."""
    success: bool
    data: Optional[pd.DataFrame]
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    transformation_name: str
    execution_time: float
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Get summary of transformation result."""
        status = "✅ SUCCESS" if self.success else "❌ FAILED"
        shape_change = f"{self.original_shape} → {self.final_shape}"
        
        lines = [
            f"Transformation: {self.transformation_name}",
            f"Status: {status}",
            f"Shape Change: {shape_change}",
            f"Execution Time: {self.execution_time:.3f}s"
        ]
        
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
        
        return " | ".join(lines)


@dataclass
class PipelineResult:
    """Result of a transformation pipeline."""
    success: bool
    original_data: pd.DataFrame
    final_data: Optional[pd.DataFrame]
    step_results: List[TransformationResult]
    total_execution_time: float
    pipeline_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def summary(self) -> str:
        """Get summary of pipeline execution."""
        successful_steps = sum(1 for result in self.step_results if result.success)
        total_steps = len(self.step_results)
        
        lines = [
            f"Pipeline: {self.pipeline_name}",
            f"Status: {'✅ SUCCESS' if self.success else '❌ FAILED'}",
            f"Steps: {successful_steps}/{total_steps} successful",
            f"Total Time: {self.total_execution_time:.3f}s",
            f"Shape: {self.original_data.shape} → {self.final_data.shape if self.final_data is not None else 'N/A'}"
        ]
        
        return " | ".join(lines)
    
    def get_failed_steps(self) -> List[TransformationResult]:
        """Get list of failed transformation steps."""
        return [result for result in self.step_results if not result.success]


class BaseTransformer(ABC):
    """Base class for data transformers."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"transformer.{name}")
        self.fitted = False
        self._fit_metadata = {}
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'BaseTransformer':
        """Fit the transformer to the data."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        pass
    
    def fit_transform(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit the transformer and transform the data."""
        return self.fit(data, target).transform(data)
    
    def _validate_fitted(self) -> None:
        """Check if transformer is fitted."""
        if not self.fitted:
            raise ValidationError(f"Transformer '{self.name}' must be fitted before transform")


class MissingValueImputer(BaseTransformer):
    """Impute missing values using various strategies."""
    
    def __init__(
        self,
        method: ImputationMethod = ImputationMethod.MEAN,
        columns: Optional[List[str]] = None,
        fill_value: Any = None,
        n_neighbors: int = 5
    ):
        super().__init__(f"imputer_{method.value}")
        self.method = method
        self.columns = columns
        self.fill_value = fill_value
        self.n_neighbors = n_neighbors
        self._imputers = {}
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'MissingValueImputer':
        """Fit imputation strategy to data."""
        columns_to_impute = self.columns or data.columns.tolist()
        
        for col in columns_to_impute:
            if col not in data.columns:
                self.logger.warning(f"Column '{col}' not found in data")
                continue
            
            col_data = data[col]
            
            if col_data.isnull().sum() == 0:
                continue  # No missing values
            
            if self.method == ImputationMethod.KNN:
                # Use KNN imputer for numeric data
                if pd.api.types.is_numeric_dtype(col_data):
                    imputer = KNNImputer(n_neighbors=self.n_neighbors)
                    # Fit on all numeric columns for KNN
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    imputer.fit(data[numeric_cols])
                    self._imputers[col] = ('knn', imputer, numeric_cols.tolist())
                else:
                    # Fall back to mode for non-numeric
                    self._imputers[col] = ('mode', col_data.mode().iloc[0] if len(col_data.mode()) > 0 else None, None)
            
            elif self.method == ImputationMethod.CONSTANT:
                self._imputers[col] = ('constant', self.fill_value, None)
            
            elif self.method in [ImputationMethod.FORWARD_FILL, ImputationMethod.BACKWARD_FILL]:
                self._imputers[col] = (self.method.value, None, None)
            
            else:
                # Statistical imputation
                if pd.api.types.is_numeric_dtype(col_data):
                    if self.method == ImputationMethod.MEAN:
                        fill_value = col_data.mean()
                    elif self.method == ImputationMethod.MEDIAN:
                        fill_value = col_data.median()
                    else:  # mode
                        fill_value = col_data.mode().iloc[0] if len(col_data.mode()) > 0 else col_data.mean()
                else:
                    # Use mode for categorical data
                    fill_value = col_data.mode().iloc[0] if len(col_data.mode()) > 0 else None
                
                self._imputers[col] = (self.method.value, fill_value, None)
        
        self.fitted = True
        self._fit_metadata = {
            'columns_fitted': list(self._imputers.keys()),
            'missing_counts_before': data[columns_to_impute].isnull().sum().to_dict()
        }
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply imputation to data."""
        self._validate_fitted()
        
        result = data.copy()
        
        for col, (method, fill_value, extra_cols) in self._imputers.items():
            if col not in result.columns:
                continue
            
            if method == 'knn' and extra_cols:
                # Apply KNN imputation
                available_cols = [c for c in extra_cols if c in result.columns]
                if available_cols:
                    imputed_values = fill_value.transform(result[available_cols])
                    col_idx = available_cols.index(col)
                    result[col] = imputed_values[:, col_idx]
            
            elif method == 'constant':
                result[col] = result[col].fillna(fill_value)
            
            elif method == 'ffill':
                result[col] = result[col].fillna(method='ffill')
            
            elif method == 'bfill':
                result[col] = result[col].fillna(method='bfill')
            
            else:
                # Statistical imputation
                result[col] = result[col].fillna(fill_value)
        
        return result


class DataScaler(BaseTransformer):
    """Scale numeric data using various methods."""
    
    def __init__(
        self,
        method: ScalingMethod = ScalingMethod.STANDARD,
        columns: Optional[List[str]] = None
    ):
        super().__init__(f"scaler_{method.value}")
        self.method = method
        self.columns = columns
        self._scalers = {}
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'DataScaler':
        """Fit scaling parameters to data."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        columns_to_scale = self.columns or numeric_columns.tolist()
        
        # Filter to only numeric columns
        columns_to_scale = [col for col in columns_to_scale if col in numeric_columns]
        
        for col in columns_to_scale:
            if col not in data.columns:
                continue
            
            # Create appropriate scaler
            if self.method == ScalingMethod.STANDARD:
                scaler = StandardScaler()
            elif self.method == ScalingMethod.MINMAX:
                scaler = MinMaxScaler()
            elif self.method == ScalingMethod.ROBUST:
                scaler = RobustScaler()
            elif self.method == ScalingMethod.POWER:
                scaler = PowerTransformer()
            else:
                raise ValueError(f"Unknown scaling method: {self.method}")
            
            # Fit scaler
            col_data = data[[col]].dropna()
            if len(col_data) > 0:
                scaler.fit(col_data)
                self._scalers[col] = scaler
        
        self.fitted = True
        self._fit_metadata = {
            'columns_fitted': list(self._scalers.keys()),
            'scaling_method': self.method.value
        }
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling to data."""
        self._validate_fitted()
        
        result = data.copy()
        
        for col, scaler in self._scalers.items():
            if col not in result.columns:
                continue
            
            # Apply scaling only to non-null values
            mask = result[col].notna()
            if mask.any():
                result.loc[mask, col] = scaler.transform(result.loc[mask, [col]]).flatten()
        
        return result


class OutlierRemover(BaseTransformer):
    """Remove outliers using statistical methods."""
    
    def __init__(
        self,
        method: str = 'iqr',
        threshold: float = 1.5,
        columns: Optional[List[str]] = None
    ):
        super().__init__(f"outlier_remover_{method}")
        self.method = method
        self.threshold = threshold
        self.columns = columns
        self._outlier_bounds = {}
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'OutlierRemover':
        """Compute outlier bounds."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        columns_to_check = self.columns or numeric_columns.tolist()
        
        for col in columns_to_check:
            if col not in data.columns or col not in numeric_columns:
                continue
            
            col_data = data[col].dropna()
            
            if self.method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.threshold * IQR
                upper_bound = Q3 + self.threshold * IQR
            
            elif self.method == 'zscore':
                mean = col_data.mean()
                std = col_data.std()
                lower_bound = mean - self.threshold * std
                upper_bound = mean + self.threshold * std
            
            else:
                raise ValueError(f"Unknown outlier detection method: {self.method}")
            
            self._outlier_bounds[col] = (lower_bound, upper_bound)
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from data."""
        self._validate_fitted()
        
        result = data.copy()
        outlier_mask = pd.Series([False] * len(result), index=result.index)
        
        for col, (lower_bound, upper_bound) in self._outlier_bounds.items():
            if col not in result.columns:
                continue
            
            col_outliers = (result[col] < lower_bound) | (result[col] > upper_bound)
            outlier_mask |= col_outliers
        
        # Remove rows with outliers
        result = result[~outlier_mask]
        
        removed_count = outlier_mask.sum()
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} outlier rows")
        
        return result


class CategoricalEncoder(BaseTransformer):
    """Encode categorical variables."""
    
    def __init__(
        self,
        method: str = 'onehot',
        columns: Optional[List[str]] = None,
        drop_first: bool = True
    ):
        super().__init__(f"encoder_{method}")
        self.method = method
        self.columns = columns
        self.drop_first = drop_first
        self._encodings = {}
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'CategoricalEncoder':
        """Fit encoding mappings."""
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        columns_to_encode = self.columns or categorical_columns.tolist()
        
        for col in columns_to_encode:
            if col not in data.columns or col not in categorical_columns:
                continue
            
            if self.method == 'label':
                # Label encoding
                unique_values = data[col].dropna().unique()
                encoding_map = {val: idx for idx, val in enumerate(unique_values)}
                self._encodings[col] = ('label', encoding_map)
            
            elif self.method == 'onehot':
                # One-hot encoding (store column names for later)
                unique_values = sorted(data[col].dropna().unique())
                if self.drop_first and len(unique_values) > 1:
                    unique_values = unique_values[1:]  # Drop first category
                
                encoded_columns = [f"{col}_{val}" for val in unique_values]
                self._encodings[col] = ('onehot', encoded_columns, unique_values)
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply encoding to data."""
        self._validate_fitted()
        
        result = data.copy()
        
        for col, encoding_info in self._encodings.items():
            if col not in result.columns:
                continue
            
            method = encoding_info[0]
            
            if method == 'label':
                encoding_map = encoding_info[1]
                result[col] = result[col].map(encoding_map)
            
            elif method == 'onehot':
                encoded_columns, unique_values = encoding_info[1], encoding_info[2]
                
                # Create one-hot encoded columns
                for i, val in enumerate(unique_values):
                    encoded_col_name = encoded_columns[i]
                    result[encoded_col_name] = (result[col] == val).astype(int)
                
                # Drop original column
                result = result.drop(columns=[col])
        
        return result


class TransformationPipeline:
    """Composable data transformation pipeline."""
    
    def __init__(self, name: str = "pipeline"):
        self.name = name
        self.steps = []
        self.fitted = False
        self.logger = get_logger(f"pipeline.{name}")
    
    def add_step(self, transformer: BaseTransformer) -> 'TransformationPipeline':
        """Add a transformation step to the pipeline."""
        self.steps.append(transformer)
        return self
    
    def add_imputation(
        self,
        method: ImputationMethod = ImputationMethod.MEAN,
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> 'TransformationPipeline':
        """Add imputation step."""
        imputer = MissingValueImputer(method=method, columns=columns, **kwargs)
        return self.add_step(imputer)
    
    def add_scaling(
        self,
        method: ScalingMethod = ScalingMethod.STANDARD,
        columns: Optional[List[str]] = None
    ) -> 'TransformationPipeline':
        """Add scaling step."""
        scaler = DataScaler(method=method, columns=columns)
        return self.add_step(scaler)
    
    def add_outlier_removal(
        self,
        method: str = 'iqr',
        threshold: float = 1.5,
        columns: Optional[List[str]] = None
    ) -> 'TransformationPipeline':
        """Add outlier removal step."""
        outlier_remover = OutlierRemover(method=method, threshold=threshold, columns=columns)
        return self.add_step(outlier_remover)
    
    def add_encoding(
        self,
        method: str = 'onehot',
        columns: Optional[List[str]] = None,
        drop_first: bool = True
    ) -> 'TransformationPipeline':
        """Add categorical encoding step."""
        encoder = CategoricalEncoder(method=method, columns=columns, drop_first=drop_first)
        return self.add_step(encoder)
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'TransformationPipeline':
        """Fit all transformers in the pipeline."""
        current_data = data.copy()
        
        for transformer in self.steps:
            self.logger.info(f"Fitting transformer: {transformer.name}")
            transformer.fit(current_data, target)
            current_data = transformer.transform(current_data)
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> PipelineResult:
        """Apply all transformations in the pipeline."""
        if not self.fitted:
            raise ValidationError("Pipeline must be fitted before transform")
        
        start_time = datetime.now()
        current_data = data.copy()
        original_data = data.copy()
        step_results = []
        success = True
        
        for transformer in self.steps:
            step_start_time = datetime.now()
            
            try:
                original_shape = current_data.shape
                transformed_data = transformer.transform(current_data)
                final_shape = transformed_data.shape
                
                execution_time = (datetime.now() - step_start_time).total_seconds()
                
                step_result = TransformationResult(
                    success=True,
                    data=transformed_data,
                    original_shape=original_shape,
                    final_shape=final_shape,
                    transformation_name=transformer.name,
                    execution_time=execution_time
                )
                
                current_data = transformed_data
                self.logger.info(f"Successfully applied {transformer.name}: {original_shape} → {final_shape}")
                
            except Exception as e:
                execution_time = (datetime.now() - step_start_time).total_seconds()
                
                step_result = TransformationResult(
                    success=False,
                    data=None,
                    original_shape=current_data.shape,
                    final_shape=current_data.shape,
                    transformation_name=transformer.name,
                    execution_time=execution_time,
                    errors=[str(e)]
                )
                
                success = False
                self.logger.error(f"Failed to apply {transformer.name}: {str(e)}")
            
            step_results.append(step_result)
            
            if not step_result.success:
                break
        
        total_execution_time = (datetime.now() - start_time).total_seconds()
        
        return PipelineResult(
            success=success,
            original_data=original_data,
            final_data=current_data if success else None,
            step_results=step_results,
            total_execution_time=total_execution_time,
            pipeline_name=self.name
        )
    
    def fit_transform(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> PipelineResult:
        """Fit the pipeline and transform the data."""
        self.fit(data, target)
        return self.transform(data)


class CustomTransformer(BaseTransformer):
    """Custom transformer that wraps a user-defined function."""
    
    def __init__(
        self,
        name: str,
        transform_func: Callable[[pd.DataFrame], pd.DataFrame],
        fit_func: Optional[Callable[[pd.DataFrame, Optional[pd.Series]], Any]] = None
    ):
        super().__init__(name)
        self.transform_func = transform_func
        self.fit_func = fit_func
        self.fit_params = None
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'CustomTransformer':
        """Fit custom transformer."""
        if self.fit_func:
            self.fit_params = self.fit_func(data, target)
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply custom transformation."""
        self._validate_fitted()
        return self.transform_func(data)


# Convenience functions
def create_basic_pipeline() -> TransformationPipeline:
    """Create a basic preprocessing pipeline."""
    return (TransformationPipeline("basic_preprocessing")
            .add_imputation(ImputationMethod.MEDIAN)
            .add_scaling(ScalingMethod.STANDARD)
            .add_encoding(method='onehot'))


def create_robust_pipeline() -> TransformationPipeline:
    """Create a robust preprocessing pipeline with outlier handling."""
    return (TransformationPipeline("robust_preprocessing")
            .add_outlier_removal(method='iqr', threshold=1.5)
            .add_imputation(ImputationMethod.MEDIAN)
            .add_scaling(ScalingMethod.ROBUST)
            .add_encoding(method='onehot', drop_first=True))


def apply_quick_preprocessing(
    data: pd.DataFrame,
    target: Optional[pd.Series] = None,
    include_outlier_removal: bool = False
) -> pd.DataFrame:
    """Apply quick preprocessing to data."""
    if include_outlier_removal:
        pipeline = create_robust_pipeline()
    else:
        pipeline = create_basic_pipeline()
    
    result = pipeline.fit_transform(data, target)
    
    if not result.success:
        raise RefuncError(f"Preprocessing failed: {[r.errors for r in result.get_failed_steps()]}")
    
    return result.final_data