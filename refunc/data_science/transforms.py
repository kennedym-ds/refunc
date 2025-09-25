"""
Data transformation and preprocessing pipelines.

This module provides composable data transformation utilities including
preprocessing pipelines, feature engineering, data cleaning,
and transformation chains with logging and error handling.
"""

import warnings
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Union,
    Callable,
    Tuple,
    Type,
    Sequence,
    Literal,
    cast,
)
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
from pandas.errors import MergeError


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
                # Get original dtype to preserve it if needed
                original_dtype = result[col].dtype
                scaled_values = scaler.transform(result.loc[mask, [col]]).flatten()
                
                # For integer columns, convert to float to allow fractional values
                if pd.api.types.is_integer_dtype(original_dtype):
                    result[col] = result[col].astype(float)
                
                result.loc[mask, col] = scaled_values
        
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
                
                # Create one-hot encoded columns efficiently using pd.concat
                onehot_data = {}
                for i, val in enumerate(unique_values):
                    encoded_col_name = encoded_columns[i]
                    onehot_data[encoded_col_name] = (result[col] == val).astype(int)
                
                # Add all one-hot columns at once to avoid fragmentation
                onehot_df = pd.DataFrame(onehot_data, index=result.index)
                result = pd.concat([result, onehot_df], axis=1)
                
                # Drop original column
                result = result.drop(columns=[col])
        
        return result


MergeHow = Literal[
    "left",
    "right",
    "outer",
    "inner",
    "cross",
    "left_anti",
    "right_anti",
]

MergeValidate = Literal[
    "one_to_one",
    "1:1",
    "one_to_many",
    "1:m",
    "many_to_one",
    "m:1",
    "many_to_many",
    "m:m",
]


JoinSourceProvider = Union[
    Sequence[pd.DataFrame],
    Callable[[pd.DataFrame], Sequence[pd.DataFrame]],
    Callable[[], Sequence[pd.DataFrame]],
]


@dataclass(frozen=True)
class PipelineJoinConfig:
    """Configuration for joining additional DataFrames within a pipeline."""

    sources: JoinSourceProvider
    columns: Optional[Sequence[str]] = None
    how: MergeHow = "inner"
    validate: Optional[MergeValidate] = None
    indicator: Union[bool, str] = False
    sort: bool = False
    suffix_template: str = "_df{index}"


class TransformationPipeline:
    """Composable data transformation pipeline."""
    
    def __init__(
        self,
        name: str = "pipeline",
        *,
        join_config: Optional[PipelineJoinConfig] = None,
    ):
        self.name = name
        self.steps = []
        self.fitted = False
        self.logger = get_logger(f"pipeline.{name}")
        self._join_config = join_config
    
    def add_step(self, transformer: BaseTransformer) -> 'TransformationPipeline':
        """Add a transformation step to the pipeline."""
        self.steps.append(transformer)
        return self
    
    def set_join_config(self, join_config: PipelineJoinConfig) -> 'TransformationPipeline':
        """Attach a join configuration that runs before transformation steps."""
        self._join_config = join_config
        return self

    def _resolve_join_sources(self, data: pd.DataFrame) -> Sequence[pd.DataFrame]:
        if self._join_config is None:
            return []
        sources = self._join_config.sources
        if callable(sources):
            callable_sources = cast(Callable[..., Sequence[pd.DataFrame]], sources)
            try:
                resolved_candidate = callable_sources(data)
            except TypeError:
                resolved_candidate = callable_sources()
        else:
            resolved_candidate = sources
        resolved_sequence = cast(Sequence[pd.DataFrame], resolved_candidate)
        resolved_list = list(resolved_sequence)
        if not resolved_list:
            raise ValidationError("Join configuration requires at least one DataFrame source")
        for index, frame in enumerate(resolved_list):
            if not isinstance(frame, pd.DataFrame):
                raise ValidationError(f"Join source at position {index} is not a pandas DataFrame")
        return resolved_list

    def _apply_join_if_configured(self, data: pd.DataFrame) -> pd.DataFrame:
        if self._join_config is None:
            return data
        additional_frames = self._resolve_join_sources(data)
        join_inputs = [data, *additional_frames]
        joined = join_dataframes_on_common_columns(
            join_inputs,
            columns=self._join_config.columns,
            how=self._join_config.how,
            validate=self._join_config.validate,
            indicator=self._join_config.indicator,
            sort=self._join_config.sort,
            suffix_template=self._join_config.suffix_template,
        )
        self.logger.info(
            "Applied pre-step join for pipeline '%s' with %d additional frame(s)"
            % (self.name, len(additional_frames))
        )
        return joined
    
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
        current_data = self._apply_join_if_configured(current_data)
        
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
        
        if self._join_config is not None:
            join_start_time = datetime.now()
            try:
                joined_data = self._apply_join_if_configured(current_data)
                join_result = TransformationResult(
                    success=True,
                    data=joined_data,
                    original_shape=current_data.shape,
                    final_shape=joined_data.shape,
                    transformation_name="join_dataframes",
                    execution_time=(datetime.now() - join_start_time).total_seconds(),
                )
                current_data = joined_data
                self.logger.info(
                    "Join step completed: %s → %s"
                    % (join_result.original_shape, join_result.final_shape)
                )
            except ValidationError as error:
                execution_time = (datetime.now() - join_start_time).total_seconds()
                join_result = TransformationResult(
                    success=False,
                    data=None,
                    original_shape=current_data.shape,
                    final_shape=current_data.shape,
                    transformation_name="join_dataframes",
                    execution_time=execution_time,
                    errors=[str(error)],
                )
                success = False
                self.logger.error(f"Join step failed: {error}")
            step_results.append(join_result)
            if not success:
                total_execution_time = (datetime.now() - start_time).total_seconds()
                return PipelineResult(
                    success=False,
                    original_data=original_data,
                    final_data=None,
                    step_results=step_results,
                    total_execution_time=total_execution_time,
                    pipeline_name=self.name,
                )
        
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
JOIN_LOGGER = get_logger("transforms.join")


def join_dataframes_on_common_columns(
    dataframes: Sequence[pd.DataFrame],
    *,
    columns: Optional[Sequence[str]] = None,
    how: MergeHow = "inner",
    validate: Optional[MergeValidate] = None,
    indicator: Union[bool, str] = False,
    sort: bool = False,
    suffix_template: str = "_df{index}",
) -> pd.DataFrame:
    """Join a sequence of DataFrames on shared column names."""

    if not dataframes:
        raise ValidationError("At least one DataFrame is required for join operations")

    total_frames = len(dataframes)
    for index, dataframe in enumerate(dataframes):
        if not isinstance(dataframe, pd.DataFrame):
            raise ValidationError(f"Item at position {index} is not a pandas DataFrame")

    if total_frames == 1:
        return dataframes[0].copy()

    if columns is not None:
        join_columns = list(columns)
        missing_columns: Dict[int, List[str]] = {}
        for index, dataframe in enumerate(dataframes):
            absent = [column for column in join_columns if column not in dataframe.columns]
            if absent:
                missing_columns[index] = absent
        if missing_columns:
            raise ValidationError(f"Join columns missing from DataFrames: {missing_columns}")
    else:
        common = set(dataframes[0].columns)
        for dataframe in dataframes[1:]:
            common &= set(dataframe.columns)
        if not common:
            raise ValidationError("No shared column names found across all DataFrames")
        join_columns = sorted(common)

    JOIN_LOGGER.info(
        f"Joining {total_frames} dataframes on columns {join_columns} using {how} join"
    )

    result = dataframes[0].copy()
    last_index = total_frames - 1
    for index in range(1, total_frames):
        dataframe = dataframes[index]
        indicator_flag: Union[bool, str] = indicator if indicator and index == last_index else False
        try:
            suffix = suffix_template.format(index=index)
        except (KeyError, IndexError, ValueError) as error:
            raise ValidationError(f"Unable to generate suffix for index {index}: {error}") from error

        try:
            result = pd.merge(
                result,
                dataframe,
                on=join_columns,
                how=how,
                suffixes=("", suffix),
                validate=validate,
                indicator=indicator_flag,
                sort=sort,
            )
        except MergeError as error:
            raise ValidationError(f"Join failed on columns {join_columns}: {error}") from error

    return result


def create_basic_pipeline(
    *,
    join_config: Optional[PipelineJoinConfig] = None,
) -> TransformationPipeline:
    """Create a basic preprocessing pipeline."""
    return (TransformationPipeline("basic_preprocessing", join_config=join_config)
            .add_imputation(ImputationMethod.MEDIAN)
            .add_scaling(ScalingMethod.STANDARD)
            .add_encoding(method='onehot'))


def create_robust_pipeline(
    *,
    join_config: Optional[PipelineJoinConfig] = None,
) -> TransformationPipeline:
    """Create a robust preprocessing pipeline with outlier handling."""
    return (TransformationPipeline("robust_preprocessing", join_config=join_config)
            .add_outlier_removal(method='iqr', threshold=1.5)
            .add_imputation(ImputationMethod.MEDIAN)
            .add_scaling(ScalingMethod.ROBUST)
            .add_encoding(method='onehot', drop_first=True))


def apply_quick_preprocessing(
    data: pd.DataFrame,
    target: Optional[pd.Series] = None,
    include_outlier_removal: bool = False,
    *,
    join_config: Optional[PipelineJoinConfig] = None,
    join_sources: Optional[JoinSourceProvider] = None,
    join_columns: Optional[Sequence[str]] = None,
    join_how: MergeHow = "inner",
    join_validate: Optional[MergeValidate] = None,
    join_indicator: Union[bool, str] = False,
    join_sort: bool = False,
    join_suffix_template: str = "_df{index}",
) -> pd.DataFrame:
    """Apply quick preprocessing to data.

    Parameters
    ----------
    data:
        Primary DataFrame to transform.
    target:
        Optional target series for supervised scenarios.
    include_outlier_removal:
        Set to ``True`` to include outlier removal prior to imputation.
    join_config:
        Explicit pipeline join configuration evaluated before transformation steps.
    join_sources:
        Optional shorthand for providing additional DataFrames to join with ``data``.
        Ignored when ``join_config`` is supplied directly.
    join_columns:
        Column names used for the join when ``join_sources`` is provided.
    join_how:
        Join method applied to the pipeline input when ``join_sources`` is provided.
    join_validate:
        Optional pandas validation rule enforced during the join.
    join_indicator:
        Whether to append the pandas merge indicator column on the final join operation.
    join_sort:
        When ``True`` the join keys are sorted before returning from the join step.
    join_suffix_template:
        Template applied to duplicate column names contributed by additional DataFrames.

    Returns
    -------
    pd.DataFrame
        The transformed result after pipeline execution.
    """

    computed_join_config = join_config
    if computed_join_config is None and join_sources is not None:
        computed_join_config = PipelineJoinConfig(
            sources=join_sources,
            columns=join_columns,
            how=join_how,
            validate=join_validate,
            indicator=join_indicator,
            sort=join_sort,
            suffix_template=join_suffix_template,
        )

    if include_outlier_removal:
        pipeline = create_robust_pipeline(join_config=computed_join_config)
    else:
        pipeline = create_basic_pipeline(join_config=computed_join_config)
    
    result = pipeline.fit_transform(data, target)
    
    if not result.success:
        raise RefuncError(f"Preprocessing failed: {[r.errors for r in result.get_failed_steps()]}")
    
    if result.final_data is None:
        raise RefuncError("Pipeline succeeded but returned no data")
    
    return result.final_data