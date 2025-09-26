"""
Pandas extensions and enhanced DataFrame functionality.

This module provides custom pandas extensions, accessor methods,
and enhanced DataFrame operations for improved data manipulation
and analysis capabilities.
"""

import warnings
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
import numpy as np
import pandas as pd
from pandas.api.extensions import register_dataframe_accessor, register_series_accessor
import matplotlib.pyplot as plt
import seaborn as sns

from ..exceptions import RefuncError, ValidationError
from ..logging import get_logger
from .validation import DataValidator, ValidationReport
from .profiling import DataProfiler, DatasetProfile, ProfileType
from .cleaning import DataCleaner, CleaningReport


@register_dataframe_accessor("refunc")
class RefuncDataFrameAccessor:
    """Custom DataFrame accessor providing enhanced functionality."""
    
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj
        self.logger = get_logger("refunc_accessor")
    
    def profile(self, detailed: bool = True, name: str = "Dataset") -> DatasetProfile:
        """
        Generate comprehensive data profile.
        
        Args:
            detailed: Whether to generate detailed profile
            name: Name for the dataset
            
        Returns:
            DatasetProfile with comprehensive analysis
        """
        profiler = DataProfiler()
        profile_type = ProfileType.DETAILED if detailed else ProfileType.BASIC
        return profiler.profile_dataframe(self._obj, name=name, profile_type=profile_type)
    
    def validate(self, schema=None, strict: bool = False) -> ValidationReport:
        """
        Validate DataFrame data quality.
        
        Args:
            schema: Optional schema for validation
            strict: Whether to treat warnings as errors
            
        Returns:
            ValidationReport with results
        """
        validator = DataValidator(strict_mode=strict)
        return validator.validate_dataframe(self._obj, schema=schema)
    
    def clean(self, aggressive: bool = False) -> Tuple[pd.DataFrame, CleaningReport]:
        """
        Clean DataFrame data.
        
        Args:
            aggressive: Whether to apply aggressive cleaning
            
        Returns:
            Tuple of (cleaned_dataframe, cleaning_report)
        """
        cleaner = DataCleaner(aggressive_cleaning=aggressive)
        return cleaner.clean_dataframe(self._obj)
    
    def quick_clean(self, aggressive: bool = False) -> pd.DataFrame:
        """
        Quick DataFrame cleaning.
        
        Args:
            aggressive: Whether to apply aggressive cleaning
            
        Returns:
            Cleaned DataFrame
        """
        cleaned_df, _ = self.clean(aggressive=aggressive)
        return cleaned_df
    
    def memory_usage_detailed(self) -> pd.DataFrame:
        """
        Get detailed memory usage information.
        
        Returns:
            DataFrame with memory usage details
        """
        memory_info = []
        
        for col in self._obj.columns:
            col_data = self._obj[col]
            base_memory = col_data.memory_usage(deep=True)
            
            # Calculate potential savings
            if col_data.dtype == 'object':
                # Try categorical conversion
                try:
                    categorical_memory = col_data.astype('category').memory_usage(deep=True)
                    categorical_savings = base_memory - categorical_memory
                except (ValueError, TypeError, AttributeError) as e:
                    self.logger.debug(f"Categorical conversion failed for column '{col}': {e}")
                    categorical_memory = base_memory
                    categorical_savings = 0
            else:
                categorical_memory = base_memory
                categorical_savings = 0
            
            # Check for nullable integer optimization
            if pd.api.types.is_integer_dtype(col_data):
                try:
                    nullable_memory = col_data.astype('Int64').memory_usage(deep=True)
                    nullable_savings = base_memory - nullable_memory
                except (ValueError, TypeError, AttributeError) as e:
                    self.logger.debug(f"Nullable integer conversion failed for column '{col}': {e}")
                    nullable_memory = base_memory
                    nullable_savings = 0
            else:
                nullable_memory = base_memory
                nullable_savings = 0
            
            memory_info.append({
                'column': col,
                'dtype': str(col_data.dtype),
                'current_memory_mb': base_memory / 1024 / 1024,
                'categorical_memory_mb': categorical_memory / 1024 / 1024,
                'categorical_savings_mb': categorical_savings / 1024 / 1024,
                'nullable_memory_mb': nullable_memory / 1024 / 1024,
                'nullable_savings_mb': nullable_savings / 1024 / 1024,
                'non_null_count': col_data.count(),
                'null_count': col_data.isnull().sum(),
                'unique_count': col_data.nunique()
            })
        
        return pd.DataFrame(memory_info)
    
    def optimize_memory(self, categorical_threshold: float = 0.5) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.
        
        Args:
            categorical_threshold: Threshold for converting to categorical
            
        Returns:
            Memory-optimized DataFrame
        """
        df_optimized = self._obj.copy()
        
        for col in df_optimized.columns:
            col_data = df_optimized[col]
            
            # Optimize numeric types
            if pd.api.types.is_integer_dtype(col_data):
                # Try smaller integer types
                if col_data.min() >= 0:
                    if col_data.max() <= 255:
                        df_optimized[col] = col_data.astype('uint8')
                    elif col_data.max() <= 65535:
                        df_optimized[col] = col_data.astype('uint16')
                    elif col_data.max() <= 4294967295:
                        df_optimized[col] = col_data.astype('uint32')
                else:
                    if col_data.min() >= -128 and col_data.max() <= 127:
                        df_optimized[col] = col_data.astype('int8')
                    elif col_data.min() >= -32768 and col_data.max() <= 32767:
                        df_optimized[col] = col_data.astype('int16')
                    elif col_data.min() >= -2147483648 and col_data.max() <= 2147483647:
                        df_optimized[col] = col_data.astype('int32')
            
            elif pd.api.types.is_float_dtype(col_data):
                # Try float32 if precision allows
                try:
                    float32_data = col_data.astype('float32')
                    if np.allclose(col_data.dropna(), float32_data.dropna(), equal_nan=True):
                        df_optimized[col] = float32_data
                except (ValueError, TypeError, AttributeError) as e:
                    self.logger.debug(f"Float32 optimization failed for column '{col}': {e}")
            
            elif col_data.dtype == 'object':
                # Convert to categorical if low cardinality
                unique_ratio = col_data.nunique() / len(col_data)
                if unique_ratio < categorical_threshold:
                    df_optimized[col] = col_data.astype('category')
        
        original_memory = self._obj.memory_usage(deep=True).sum()
        optimized_memory = df_optimized.memory_usage(deep=True).sum()
        savings = original_memory - optimized_memory
        
        self.logger.info(f"Memory optimization: {original_memory/1024/1024:.1f}MB → {optimized_memory/1024/1024:.1f}MB "
                        f"(saved {savings/1024/1024:.1f}MB, {savings/original_memory*100:.1f}%)")
        
        return df_optimized
    
    def missing_patterns(self) -> pd.DataFrame:
        """
        Analyze missing data patterns.
        
        Returns:
            DataFrame with missing data pattern analysis
        """
        missing_data = self._obj.isnull()
        
        # Count missing values per column
        missing_counts = missing_data.sum()
        missing_percentages = (missing_counts / len(self._obj) * 100).round(2)
        
        # Find missing patterns
        patterns = missing_data.value_counts().head(10)
        
        # Create summary
        summary_data = []
        for col in self._obj.columns:
            summary_data.append({
                'column': col,
                'missing_count': missing_counts[col],
                'missing_percentage': missing_percentages[col],
                'data_type': str(self._obj[col].dtype)
            })
        
        return pd.DataFrame(summary_data).sort_values('missing_percentage', ascending=False)
    
    def correlation_heatmap(self, method: str = 'pearson', figsize: Tuple[int, int] = (10, 8), **kwargs):
        """
        Generate correlation heatmap for numeric columns.
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
            figsize: Figure size
            **kwargs: Additional arguments for seaborn heatmap
        """
        numeric_df = self._obj.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            raise ValidationError("Need at least 2 numeric columns for correlation analysis")
        
        correlation_matrix = numeric_df.corr(method=method)
        
        plt.figure(figsize=figsize)
        
        # Default heatmap parameters
        heatmap_kwargs = {
            'annot': True,
            'cmap': 'coolwarm',
            'center': 0,
            'square': True,
            'linewidths': 0.5
        }
        heatmap_kwargs.update(kwargs)
        
        sns.heatmap(correlation_matrix, **heatmap_kwargs)
        plt.title(f'{method.title()} Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def distribution_plots(self, columns: Optional[List[str]] = None, figsize: Tuple[int, int] = (15, 10)):
        """
        Generate distribution plots for numeric columns.
        
        Args:
            columns: Specific columns to plot (None for all numeric)
            figsize: Figure size
        """
        numeric_columns = self._obj.select_dtypes(include=[np.number]).columns
        
        if columns:
            numeric_columns = [col for col in columns if col in numeric_columns]
        
        if len(numeric_columns) == 0:
            raise ValidationError("No numeric columns found for distribution plots")
        
        n_cols = min(3, len(numeric_columns))
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, col in enumerate(numeric_columns):
            if i < len(axes):
                self._obj[col].hist(bins=30, alpha=0.7, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for i in range(len(numeric_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def outlier_analysis(self, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Analyze outliers in numeric columns.
        
        Args:
            method: Outlier detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier analysis
        """
        numeric_columns = self._obj.select_dtypes(include=[np.number]).columns
        outlier_info = []
        
        for col in numeric_columns:
            col_data = self._obj[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            if method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            elif method == 'zscore':
                mean = col_data.mean()
                std = col_data.std()
                z_scores = abs((col_data - mean) / std)
                outliers = col_data[z_scores > threshold]
            
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            outlier_info.append({
                'column': col,
                'total_values': len(col_data),
                'outlier_count': len(outliers),
                'outlier_percentage': len(outliers) / len(col_data) * 100,
                'min_outlier': outliers.min() if len(outliers) > 0 else None,
                'max_outlier': outliers.max() if len(outliers) > 0 else None
            })
        
        return pd.DataFrame(outlier_info)
    
    def sample_balanced(self, target_column: str, n_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Create a balanced sample based on target column.
        
        Args:
            target_column: Column to balance on
            n_samples: Number of samples per class (None for min class size)
            
        Returns:
            Balanced sample DataFrame
        """
        if target_column not in self._obj.columns:
            raise ValidationError(f"Target column '{target_column}' not found")
        
        # Get class counts
        class_counts = self._obj[target_column].value_counts()
        
        if n_samples is None:
            n_samples = class_counts.min()
        
        # Sample from each class
        balanced_dfs = []
        for class_value in class_counts.index:
            class_data = self._obj[self._obj[target_column] == class_value]
            if len(class_data) >= n_samples:
                sampled_data = class_data.sample(n=n_samples, random_state=42)
            else:
                sampled_data = class_data
            balanced_dfs.append(sampled_data)
        
        return pd.concat(balanced_dfs, ignore_index=True)
    
    def export_summary(self, filename: str, include_plots: bool = True):
        """
        Export comprehensive data summary to file.
        
        Args:
            filename: Output filename (supports .html, .txt)
            include_plots: Whether to include visualization plots
        """
        # Generate profile
        profile = self.profile(detailed=True)
        validation = self.validate()
        
        if filename.endswith('.html'):
            self._export_html_summary(filename, profile, validation, include_plots)
        else:
            self._export_text_summary(filename, profile, validation)
    
    def _export_html_summary(self, filename: str, profile: DatasetProfile, validation: ValidationReport, include_plots: bool):
        """Export HTML summary report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Summary Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Summary Report</h1>
                <p>Generated on: {profile.creation_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Dataset Overview</h2>
                <div class="metric">Rows: {profile.shape[0]:,}</div>
                <div class="metric">Columns: {profile.shape[1]}</div>
                <div class="metric">Memory: {profile.memory_usage/1024/1024:.1f} MB</div>
                <div class="metric">Quality Score: {profile.data_quality_score:.2%}</div>
            </div>
            
            <div class="section">
                <h2>Data Quality</h2>
                <div class="metric">Valid: {'✅' if validation.is_valid else '❌'}</div>
                <div class="metric">Issues: {validation.total_issues}</div>
                <div class="metric">Missing: {profile.missing_percentage:.1f}%</div>
                <div class="metric">Duplicates: {profile.duplicate_percentage:.1f}%</div>
            </div>
            
            <div class="section">
                <h2>Column Information</h2>
                <table>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Non-Null</th>
                        <th>Unique</th>
                        <th>Missing %</th>
                    </tr>
        """
        
        for col_name, col_profile in profile.columns.items():
            html_content += f"""
                    <tr>
                        <td>{col_name}</td>
                        <td>{col_profile.dtype}</td>
                        <td>{col_profile.total_count - col_profile.null_count:,}</td>
                        <td>{col_profile.unique_count:,}</td>
                        <td>{col_profile.null_percentage:.1f}%</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _export_text_summary(self, filename: str, profile: DatasetProfile, validation: ValidationReport):
        """Export text summary report."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(profile.summary())
            f.write("\n\n" + "="*80 + "\n")
            f.write(validation.summary())


@register_series_accessor("refunc")
class RefuncSeriesAccessor:
    """Custom Series accessor providing enhanced functionality."""
    
    def __init__(self, pandas_obj: pd.Series):
        self._obj = pandas_obj
        self.logger = get_logger("refunc_series_accessor")
    
    def outliers(self, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
        """
        Detect outliers in the series.
        
        Args:
            method: Outlier detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean Series indicating outliers
        """
        if not pd.api.types.is_numeric_dtype(self._obj):
            raise ValidationError("Outlier detection requires numeric data")
        
        data = self._obj.dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (self._obj < lower_bound) | (self._obj > upper_bound)
        
        elif method == 'zscore':
            mean = data.mean()
            std = data.std()
            z_scores = abs((self._obj - mean) / std)
            outliers = z_scores > threshold
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return outliers
    
    def remove_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
        """
        Remove outliers from the series.
        
        Args:
            method: Outlier detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Series with outliers removed
        """
        outlier_mask = self.outliers(method=method, threshold=threshold)
        return self._obj[~outlier_mask]
    
    def normalize(self, method: str = 'minmax') -> pd.Series:
        """
        Normalize the series.
        
        Args:
            method: Normalization method ('minmax', 'zscore', 'robust')
            
        Returns:
            Normalized Series
        """
        if not pd.api.types.is_numeric_dtype(self._obj):
            raise ValidationError("Normalization requires numeric data")
        
        if method == 'minmax':
            min_val = self._obj.min()
            max_val = self._obj.max()
            return (self._obj - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean = self._obj.mean()
            std = self._obj.std()
            return (self._obj - mean) / std
        
        elif method == 'robust':
            median = self._obj.median()
            mad = (self._obj - median).abs().median()
            return (self._obj - median) / mad
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def entropy(self) -> float:
        """
        Calculate Shannon entropy of the series.
        
        Returns:
            Entropy value
        """
        value_counts = self._obj.value_counts()
        probabilities = value_counts / len(self._obj)
        return -np.sum(probabilities * np.log2(probabilities))
    
    def pattern_frequency(self, pattern: str) -> int:
        """
        Count occurrences of a pattern in string series.
        
        Args:
            pattern: Regular expression pattern
            
        Returns:
            Number of matches
        """
        if self._obj.dtype != 'object':
            raise ValidationError("Pattern matching requires string data")
        
        return self._obj.astype(str).str.contains(pattern, regex=True, na=False).sum()


# Utility functions for DataFrame enhancement
def merge_on_fuzzy(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    threshold: float = 0.8,
    how: str = 'inner'
) -> pd.DataFrame:
    """
    Merge DataFrames using fuzzy string matching.
    
    Args:
        left: Left DataFrame
        right: Right DataFrame
        left_on: Column name in left DataFrame
        right_on: Column name in right DataFrame
        threshold: Similarity threshold (0-1)
        how: Type of merge
        
    Returns:
        Merged DataFrame
    """
    try:
        from fuzzywuzzy import fuzz, process
    except ImportError:
        raise ImportError("fuzzywuzzy library required for fuzzy matching: pip install fuzzywuzzy")
    
    # Create mapping of fuzzy matches
    left_values = left[left_on].dropna().unique()
    right_values = right[right_on].dropna().unique()
    
    matches = {}
    for left_val in left_values:
        match = process.extractOne(left_val, right_values, scorer=fuzz.ratio)
        if match and match[1] >= threshold * 100:
            matches[left_val] = match[0]
    
    # Create temporary columns for merging
    left_temp = left.copy()
    left_temp['_merge_key'] = left_temp[left_on].map(matches)
    
    right_temp = right.copy()
    right_temp['_merge_key'] = right_temp[right_on]
    
    # Perform merge
    result = pd.merge(left_temp, right_temp, on='_merge_key', how=how, suffixes=('', '_right'))
    result = result.drop(columns=['_merge_key'])
    
    return result


def pivot_advanced(
    df: pd.DataFrame,
    index: Union[str, List[str]],
    columns: str,
    values: Optional[str] = None,
    aggfunc: Union[str, Callable] = 'mean',
    fill_value: Any = 0,
    normalize: bool = False
) -> pd.DataFrame:
    """
    Advanced pivot table with additional options.
    
    Args:
        df: DataFrame to pivot
        index: Column(s) to use as index
        columns: Column to use as columns
        values: Column to aggregate (None for all numeric)
        aggfunc: Aggregation function
        fill_value: Value to fill missing entries
        normalize: Whether to normalize values
        
    Returns:
        Pivoted DataFrame
    """
    if values is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValidationError("No numeric columns found for aggregation")
        values = numeric_cols.tolist()
    
    pivot_table = pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
        fill_value=fill_value
    )
    
    if normalize:
        pivot_table = pivot_table.div(pivot_table.sum(axis=1), axis=0)
    
    return pivot_table