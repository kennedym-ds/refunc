"""
Data profiling and statistical analysis utilities.

This module provides comprehensive data profiling capabilities including
statistical summaries, distribution analysis, correlation studies,
and automated insights generation.
"""

import warnings
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

from ..exceptions import RefuncError, ValidationError
from ..math_stats.statistics import StatisticsEngine
from ..math_stats.distributions import DistributionAnalyzer


class ProfileType(Enum):
    """Types of data profiling."""
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


class InsightType(Enum):
    """Types of automated insights."""
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"
    OUTLIER = "outlier"
    TREND = "trend"
    PATTERN = "pattern"
    QUALITY = "quality"


@dataclass
class ColumnProfile:
    """Profile information for a single column."""
    name: str
    dtype: str
    total_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    
    # Numeric statistics
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # Categorical statistics
    top_values: Optional[Dict[str, int]] = None
    value_counts: Optional[Dict[str, int]] = None
    
    # Distribution information
    distribution_type: Optional[str] = None
    distribution_params: Optional[Dict[str, float]] = None
    distribution_fit_score: Optional[float] = None
    
    # Additional insights
    is_constant: bool = False
    is_unique: bool = False
    has_outliers: bool = False
    outlier_count: int = 0
    
    def summary(self) -> str:
        """Get formatted summary of column profile."""
        lines = [
            f"Column: {self.name}",
            f"Type: {self.dtype}",
            f"Count: {self.total_count:,} ({self.null_count:,} null, {self.null_percentage:.1f}%)",
            f"Unique: {self.unique_count:,} ({self.unique_percentage:.1f}%)"
        ]
        
        if self.is_constant:
            lines.append("⚠️  Constant values detected")
        if self.is_unique:
            lines.append("🔑 All values are unique")
        if self.has_outliers:
            lines.append(f"📊 {self.outlier_count} outliers detected")
        
        if self.mean is not None:
            lines.extend([
                f"Mean: {self.mean:.3f}",
                f"Median: {self.median:.3f}",
                f"Std: {self.std:.3f}",
                f"Range: [{self.min_value:.3f}, {self.max_value:.3f}]"
            ])
            
            if self.skewness is not None:
                lines.append(f"Skewness: {self.skewness:.3f}")
            if self.kurtosis is not None:
                lines.append(f"Kurtosis: {self.kurtosis:.3f}")
        
        if self.top_values:
            lines.append("Top values:")
            for value, count in list(self.top_values.items())[:5]:
                lines.append(f"  '{value}': {count}")
        
        if self.distribution_type:
            lines.append(f"Best fit distribution: {self.distribution_type} (score: {self.distribution_fit_score:.3f})")
        
        return "\n".join(lines)


@dataclass
class DatasetProfile:
    """Complete dataset profile."""
    name: str
    shape: Tuple[int, int]
    memory_usage: int
    creation_time: datetime
    profile_type: ProfileType
    
    # Column profiles
    columns: Dict[str, ColumnProfile] = field(default_factory=dict)
    
    # Dataset-level statistics
    total_missing_cells: int = 0
    missing_percentage: float = 0.0
    duplicate_rows: int = 0
    duplicate_percentage: float = 0.0
    
    # Data types summary
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    datetime_columns: List[str] = field(default_factory=list)
    boolean_columns: List[str] = field(default_factory=list)
    
    # Correlation matrix (for numeric columns)
    correlation_matrix: Optional[pd.DataFrame] = None
    high_correlations: List[Tuple[str, str, float]] = field(default_factory=list)
    
    # Automated insights
    insights: List[Dict[str, Any]] = field(default_factory=list)
    
    # Quality metrics
    data_quality_score: float = 0.0
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    
    def summary(self) -> str:
        """Get formatted summary of dataset profile."""
        rows, cols = self.shape
        lines = [
            f"Dataset Profile: {self.name}",
            "=" * 50,
            f"Shape: {rows:,} rows × {cols} columns",
            f"Memory Usage: {self.memory_usage / 1024 / 1024:.1f} MB",
            f"Profile Type: {self.profile_type.value.title()}",
            f"Created: {self.creation_time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Data Quality Metrics:",
            f"  Overall Quality: {self.data_quality_score:.2%}",
            f"  Completeness: {self.completeness_score:.2%}",
            f"  Consistency: {self.consistency_score:.2%}",
            "",
            "Data Overview:",
            f"  Missing Cells: {self.total_missing_cells:,} ({self.missing_percentage:.1f}%)",
            f"  Duplicate Rows: {self.duplicate_rows:,} ({self.duplicate_percentage:.1f}%)",
            "",
            "Column Types:",
            f"  Numeric: {len(self.numeric_columns)}",
            f"  Categorical: {len(self.categorical_columns)}",
            f"  DateTime: {len(self.datetime_columns)}",
            f"  Boolean: {len(self.boolean_columns)}"
        ]
        
        if self.high_correlations:
            lines.extend([
                "",
                "High Correlations (|r| > 0.7):",
            ])
            for col1, col2, corr in self.high_correlations[:5]:
                lines.append(f"  {col1} ↔ {col2}: {corr:.3f}")
        
        if self.insights:
            lines.extend([
                "",
                f"Automated Insights ({len(self.insights)}):"
            ])
            for insight in self.insights[:5]:
                lines.append(f"  • {insight['message']}")
        
        return "\n".join(lines)
    
    def get_column_profile(self, column_name: str) -> Optional[ColumnProfile]:
        """Get profile for a specific column."""
        return self.columns.get(column_name)
    
    def get_insights_by_type(self, insight_type: InsightType) -> List[Dict[str, Any]]:
        """Get insights filtered by type."""
        return [insight for insight in self.insights if insight.get('type') == insight_type.value]


class DataProfiler:
    """Comprehensive data profiling engine."""
    
    def __init__(self, enable_plots: bool = True):
        """
        Initialize data profiler.
        
        Args:
            enable_plots: Whether to generate visualization plots
        """
        self.enable_plots = enable_plots
        self.stats_engine = StatisticsEngine()
        self.dist_analyzer = DistributionAnalyzer()
        
        # Configure matplotlib for headless operation if needed
        if not enable_plots:
            plt.ioff()
    
    def profile_dataframe(
        self,
        df: pd.DataFrame,
        name: str = "Dataset",
        profile_type: ProfileType = ProfileType.DETAILED,
        sample_size: Optional[int] = None
    ) -> DatasetProfile:
        """
        Generate comprehensive profile of a DataFrame.
        
        Args:
            df: DataFrame to profile
            name: Dataset name for the profile
            profile_type: Level of profiling detail
            sample_size: Optional sample size for large datasets
            
        Returns:
            DatasetProfile with comprehensive analysis
        """
        # Sample data if needed for performance
        if sample_size and len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
            warnings.warn(f"Using sample of {sample_size} rows for profiling")
        else:
            df_sample = df
        
        # Initialize profile
        profile = DatasetProfile(
            name=name,
            shape=df.shape,
            memory_usage=df.memory_usage(deep=True).sum(),
            creation_time=datetime.now(),
            profile_type=profile_type
        )
        
        # Analyze dataset-level statistics
        self._analyze_dataset_level(df_sample, profile)
        
        # Profile each column
        for col in df_sample.columns:
            column_profile = self._profile_column(df_sample[col], profile_type)
            profile.columns[col] = column_profile
        
        # Categorize columns by type
        self._categorize_columns(profile)
        
        # Generate correlation analysis for numeric columns
        if profile_type in [ProfileType.DETAILED, ProfileType.COMPREHENSIVE]:
            self._analyze_correlations(df_sample, profile)
        
        # Generate automated insights
        if profile_type == ProfileType.COMPREHENSIVE:
            self._generate_insights(df_sample, profile)
        
        # Calculate quality scores
        self._calculate_quality_scores(profile)
        
        return profile
    
    def _analyze_dataset_level(self, df: pd.DataFrame, profile: DatasetProfile) -> None:
        """Analyze dataset-level statistics."""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        
        profile.total_missing_cells = missing_cells
        profile.missing_percentage = (missing_cells / total_cells * 100) if total_cells > 0 else 0
        
        # Check for duplicate rows
        duplicates = df.duplicated()
        profile.duplicate_rows = duplicates.sum()
        profile.duplicate_percentage = (profile.duplicate_rows / len(df) * 100) if len(df) > 0 else 0
    
    def _profile_column(self, series: pd.Series, profile_type: ProfileType) -> ColumnProfile:
        """Profile a single column."""
        col_name = series.name
        total_count = len(series)
        null_count = series.isnull().sum()
        non_null_series = series.dropna()
        
        # Basic statistics
        profile = ColumnProfile(
            name=str(col_name) if col_name is not None else "unknown",
            dtype=str(series.dtype),
            total_count=total_count,
            null_count=null_count,
            null_percentage=(null_count / total_count * 100) if total_count > 0 else 0,
            unique_count=series.nunique(),
            unique_percentage=(series.nunique() / total_count * 100) if total_count > 0 else 0
        )
        
        # Check for special cases
        profile.is_constant = series.nunique() <= 1
        profile.is_unique = series.nunique() == total_count and null_count == 0
        
        # Numeric column analysis
        if pd.api.types.is_numeric_dtype(series) and len(non_null_series) > 0:
            self._analyze_numeric_column(non_null_series, profile, profile_type)
        
        # Categorical column analysis
        elif isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == 'object':
            self._analyze_categorical_column(non_null_series, profile, profile_type)
        
        # DateTime column analysis
        elif pd.api.types.is_datetime64_any_dtype(series):
            self._analyze_datetime_column(non_null_series, profile, profile_type)
        
        return profile
    
    def _analyze_numeric_column(self, series: pd.Series, profile: ColumnProfile, profile_type: ProfileType) -> None:
        """Analyze numeric column."""
        # Basic statistics using our StatisticsEngine
        basic_stats = self.stats_engine.describe(series)
        
        profile.mean = basic_stats.mean
        profile.median = basic_stats.median
        profile.std = basic_stats.std
        profile.min_value = basic_stats.min
        profile.max_value = basic_stats.max
        profile.q1 = basic_stats.q1
        profile.q3 = basic_stats.q3
        profile.skewness = basic_stats.skewness
        profile.kurtosis = basic_stats.kurtosis
        
        # Outlier detection
        outliers = self._detect_outliers(series)
        profile.has_outliers = len(outliers) > 0
        profile.outlier_count = len(outliers)
        
        # Distribution analysis for detailed profiling
        if profile_type in [ProfileType.DETAILED, ProfileType.COMPREHENSIVE]:
            self._analyze_distribution(series, profile)
    
    def _analyze_categorical_column(self, series: pd.Series, profile: ColumnProfile, profile_type: ProfileType) -> None:
        """Analyze categorical column."""
        value_counts = series.value_counts()
        
        # Top values (limit to 20 for performance)
        profile.top_values = value_counts.head(20).to_dict()
        
        if profile_type in [ProfileType.DETAILED, ProfileType.COMPREHENSIVE]:
            profile.value_counts = value_counts.to_dict()
    
    def _analyze_datetime_column(self, series: pd.Series, profile: ColumnProfile, profile_type: ProfileType) -> None:
        """Analyze datetime column."""
        if len(series) > 0:
            profile.min_value = series.min().timestamp()
            profile.max_value = series.max().timestamp()
            
            # Calculate time span
            time_span = series.max() - series.min()
            profile.mean = time_span.total_seconds() / 2  # Approximate mean as midpoint
    
    def _analyze_distribution(self, series: pd.Series, profile: ColumnProfile) -> None:
        """Analyze distribution of numeric data."""
        try:
            # Use our DistributionAnalyzer - convert to numpy array to ensure compatibility
            comparison = self.dist_analyzer.fit_best_distribution(np.array(series.values))
            
            if comparison and hasattr(comparison, 'best_fit') and comparison.best_fit:
                best_fit = comparison.best_fit
                profile.distribution_type = best_fit.distribution_name
                profile.distribution_params = best_fit.parameters
                profile.distribution_fit_score = best_fit.aic  # Use AIC as fit score
        
        except Exception as e:
            warnings.warn(f"Distribution analysis failed for {profile.name}: {str(e)}")
    
    def _detect_outliers(self, series: pd.Series) -> List[int]:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return outliers.index.tolist()
    
    def _categorize_columns(self, profile: DatasetProfile) -> None:
        """Categorize columns by data type."""
        for col_name, col_profile in profile.columns.items():
            if 'int' in col_profile.dtype or 'float' in col_profile.dtype:
                profile.numeric_columns.append(col_name)
            elif 'datetime' in col_profile.dtype:
                profile.datetime_columns.append(col_name)
            elif 'bool' in col_profile.dtype:
                profile.boolean_columns.append(col_name)
            else:
                profile.categorical_columns.append(col_name)
    
    def _analyze_correlations(self, df: pd.DataFrame, profile: DatasetProfile) -> None:
        """Analyze correlations between numeric columns."""
        if len(profile.numeric_columns) < 2:
            return
        
        try:
            # Calculate correlation matrix using pandas
            numeric_df = df[profile.numeric_columns]
            correlation_matrix = numeric_df.corr()
            profile.correlation_matrix = correlation_matrix
            
            # Find high correlations
            for i in range(len(profile.numeric_columns)):
                for j in range(i + 1, len(profile.numeric_columns)):
                    col1 = profile.numeric_columns[i]
                    col2 = profile.numeric_columns[j]
                    try:
                        corr_value = correlation_matrix.iloc[i, j]
                        if pd.notna(corr_value) and isinstance(corr_value, (int, float)) and abs(corr_value) > 0.7:
                            profile.high_correlations.append((col1, col2, float(corr_value)))
                    except (IndexError, KeyError, ValueError, TypeError) as e:
                        warnings.warn(f"Failed to process correlation between '{col1}' and '{col2}': {e}")
                        continue
            
            # Sort by absolute correlation value
            profile.high_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        except Exception as e:
            warnings.warn(f"Correlation analysis failed: {str(e)}")
    
    def _generate_insights(self, df: pd.DataFrame, profile: DatasetProfile) -> None:
        """Generate automated insights about the data."""
        insights = []
        
        # Data quality insights
        if profile.missing_percentage > 10:
            insights.append({
                'type': InsightType.QUALITY.value,
                'severity': 'warning',
                'message': f"High missing data rate: {profile.missing_percentage:.1f}% of cells are empty",
                'recommendation': "Consider data imputation or investigate data collection process"
            })
        
        if profile.duplicate_percentage > 5:
            insights.append({
                'type': InsightType.QUALITY.value,
                'severity': 'warning',
                'message': f"Significant duplicate rows: {profile.duplicate_percentage:.1f}% of data",
                'recommendation': "Review and remove duplicate entries"
            })
        
        # Column-specific insights
        for col_name, col_profile in profile.columns.items():
            # Constant columns
            if col_profile.is_constant:
                insights.append({
                    'type': InsightType.PATTERN.value,
                    'severity': 'info',
                    'message': f"Column '{col_name}' has constant values",
                    'recommendation': "Consider removing this column as it provides no information"
                })
            
            # High cardinality categorical columns
            if col_name in profile.categorical_columns and col_profile.unique_percentage > 95:
                insights.append({
                    'type': InsightType.PATTERN.value,
                    'severity': 'info',
                    'message': f"Column '{col_name}' has very high cardinality ({col_profile.unique_percentage:.1f}% unique)",
                    'recommendation': "Consider if this should be treated as an identifier rather than categorical"
                })
            
            # Outlier insights
            if col_profile.has_outliers and col_profile.outlier_count > len(df) * 0.05:
                insights.append({
                    'type': InsightType.OUTLIER.value,
                    'severity': 'warning',
                    'message': f"Column '{col_name}' has many outliers ({col_profile.outlier_count} values)",
                    'recommendation': "Investigate outliers - they may indicate data quality issues or important patterns"
                })
        
        # Correlation insights
        if profile.high_correlations:
            for col1, col2, corr in profile.high_correlations[:3]:
                insights.append({
                    'type': InsightType.CORRELATION.value,
                    'severity': 'info',
                    'message': f"Strong correlation between '{col1}' and '{col2}' (r={corr:.3f})",
                    'recommendation': "Consider feature selection to avoid multicollinearity in modeling"
                })
        
        # Distribution insights
        for col_name in profile.numeric_columns:
            col_profile = profile.columns[col_name]
            if col_profile.skewness and abs(col_profile.skewness) > 2:
                skew_direction = "right" if col_profile.skewness > 0 else "left"
                insights.append({
                    'type': InsightType.DISTRIBUTION.value,
                    'severity': 'info',
                    'message': f"Column '{col_name}' is highly {skew_direction}-skewed (skewness={col_profile.skewness:.2f})",
                    'recommendation': "Consider data transformation (log, square root) to normalize distribution"
                })
        
        profile.insights = insights
    
    def _calculate_quality_scores(self, profile: DatasetProfile) -> None:
        """Calculate data quality scores."""
        # Completeness score
        profile.completeness_score = max(0, 1 - profile.missing_percentage / 100)
        
        # Consistency score (based on outliers and data type issues)
        outlier_penalties = sum(1 for col in profile.columns.values() if col.has_outliers)
        max_outlier_penalty = len(profile.columns)
        consistency_penalty = outlier_penalties / max_outlier_penalty if max_outlier_penalty > 0 else 0
        profile.consistency_score = max(0, 1 - consistency_penalty * 0.2)  # Outliers reduce score by up to 20%
        
        # Overall quality score (weighted average)
        profile.data_quality_score = (
            profile.completeness_score * 0.5 +
            profile.consistency_score * 0.3 +
            (1 - profile.duplicate_percentage / 100) * 0.2
        )
    
    def generate_report(self, profile: DatasetProfile, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive text report."""
        report_lines = [
            profile.summary(),
            "\n" + "=" * 80 + "\n",
            "COLUMN PROFILES",
            "=" * 80
        ]
        
        for col_name, col_profile in profile.columns.items():
            report_lines.extend([
                "\n" + "-" * 50,
                col_profile.summary()
            ])
        
        if profile.insights:
            report_lines.extend([
                "\n" + "=" * 80,
                "AUTOMATED INSIGHTS",
                "=" * 80
            ])
            
            for i, insight in enumerate(profile.insights, 1):
                severity_icon = {
                    'info': 'ℹ️',
                    'warning': '⚠️',
                    'error': '❌',
                    'critical': '🚨'
                }.get(insight['severity'], '•')
                
                report_lines.extend([
                    f"\n{i}. {severity_icon} {insight['message']}",
                    f"   💡 {insight['recommendation']}"
                ])
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
    
    def export_profile(self, profile: DatasetProfile, output_file: str, format: str = 'json') -> None:
        """Export profile to file."""
        if format.lower() == 'json':
            # Convert profile to JSON-serializable format
            profile_dict = {
                'name': profile.name,
                'shape': profile.shape,
                'memory_usage': profile.memory_usage,
                'creation_time': profile.creation_time.isoformat(),
                'profile_type': profile.profile_type.value,
                'total_missing_cells': profile.total_missing_cells,
                'missing_percentage': profile.missing_percentage,
                'duplicate_rows': profile.duplicate_rows,
                'duplicate_percentage': profile.duplicate_percentage,
                'numeric_columns': profile.numeric_columns,
                'categorical_columns': profile.categorical_columns,
                'datetime_columns': profile.datetime_columns,
                'boolean_columns': profile.boolean_columns,
                'high_correlations': profile.high_correlations,
                'insights': profile.insights,
                'data_quality_score': profile.data_quality_score,
                'completeness_score': profile.completeness_score,
                'consistency_score': profile.consistency_score,
                'columns': {}
            }
            
            # Convert column profiles
            for col_name, col_profile in profile.columns.items():
                profile_dict['columns'][col_name] = {
                    'name': col_profile.name,
                    'dtype': col_profile.dtype,
                    'total_count': col_profile.total_count,
                    'null_count': col_profile.null_count,
                    'null_percentage': col_profile.null_percentage,
                    'unique_count': col_profile.unique_count,
                    'unique_percentage': col_profile.unique_percentage,
                    'mean': col_profile.mean,
                    'median': col_profile.median,
                    'std': col_profile.std,
                    'min_value': col_profile.min_value,
                    'max_value': col_profile.max_value,
                    'q1': col_profile.q1,
                    'q3': col_profile.q3,
                    'skewness': col_profile.skewness,
                    'kurtosis': col_profile.kurtosis,
                    'top_values': col_profile.top_values,
                    'distribution_type': col_profile.distribution_type,
                    'distribution_params': col_profile.distribution_params,
                    'distribution_fit_score': col_profile.distribution_fit_score,
                    'is_constant': col_profile.is_constant,
                    'is_unique': col_profile.is_unique,
                    'has_outliers': col_profile.has_outliers,
                    'outlier_count': col_profile.outlier_count
                }
            
            with open(output_file, 'w') as f:
                json.dump(profile_dict, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Convenience functions
def profile_dataframe(
    df: pd.DataFrame,
    name: str = "Dataset",
    detailed: bool = True
) -> DatasetProfile:
    """Profile a DataFrame with default settings."""
    profiler = DataProfiler()
    profile_type = ProfileType.DETAILED if detailed else ProfileType.BASIC
    return profiler.profile_dataframe(df, name=name, profile_type=profile_type)


def quick_profile(df: pd.DataFrame) -> str:
    """Generate a quick text summary of DataFrame."""
    profiler = DataProfiler()
    profile = profiler.profile_dataframe(df, profile_type=ProfileType.BASIC)
    return profile.summary()


def compare_profiles(profile1: DatasetProfile, profile2: DatasetProfile) -> Dict[str, Any]:
    """Compare two dataset profiles."""
    comparison = {
        'shape_change': {
            'rows': profile2.shape[0] - profile1.shape[0],
            'columns': profile2.shape[1] - profile1.shape[1]
        },
        'quality_change': {
            'data_quality': profile2.data_quality_score - profile1.data_quality_score,
            'completeness': profile2.completeness_score - profile1.completeness_score,
            'consistency': profile2.consistency_score - profile1.consistency_score
        },
        'missing_data_change': profile2.missing_percentage - profile1.missing_percentage,
        'duplicate_change': profile2.duplicate_percentage - profile1.duplicate_percentage
    }
    
    # Column-level changes
    common_columns = set(profile1.columns.keys()) & set(profile2.columns.keys())
    column_changes = {}
    
    for col in common_columns:
        col1 = profile1.columns[col]
        col2 = profile2.columns[col]
        
        column_changes[col] = {
            'null_percentage_change': col2.null_percentage - col1.null_percentage,
            'unique_count_change': col2.unique_count - col1.unique_count
        }
        
        if col1.mean is not None and col2.mean is not None:
            column_changes[col]['mean_change'] = col2.mean - col1.mean
    
    comparison['column_changes'] = column_changes
    
    return comparison