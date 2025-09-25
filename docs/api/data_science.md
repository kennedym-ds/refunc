# Data Science Module API Reference

The `refunc.data_science` module provides comprehensive data science utilities for pandas DataFrames, including data validation, profiling, transformation pipelines, cleaning operations, and enhanced DataFrame functionality through custom pandas extensions.

## Module Overview

```python
from refunc.data_science import (
    # Validation
    DataValidator, DataSchema, ValidationReport, ValidationIssue,
    validate_dataframe, quick_validate, create_schema_from_dataframe,
    
    # Profiling
    DataProfiler, DatasetProfile, ColumnProfile,
    profile_dataframe, quick_profile, compare_profiles,
    
    # Transformations
    TransformationPipeline, PipelineJoinConfig, BaseTransformer,
    MissingValueImputer, DataScaler, OutlierRemover, CategoricalEncoder,
    create_basic_pipeline, create_robust_pipeline, apply_quick_preprocessing,
    join_dataframes_on_common_columns,
    
    # Cleaning
    DataCleaner, CleaningResult, CleaningReport,
    quick_clean, clean_with_report, remove_duplicates_advanced,
    
    # Extensions (automatically registered)
    RefuncDataFrameAccessor, RefuncSeriesAccessor,
    merge_on_fuzzy, pivot_advanced
)
```

## Data Validation

### DataValidator

Comprehensive data validation engine for quality assessment and constraint checking.

```python
class DataValidator:
    def __init__(self, strict_mode: bool = False)
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        schema: Optional[DataSchema] = None,
        rules: Optional[List[str]] = None
    ) -> ValidationReport
    
    def add_custom_rule(
        self, 
        name: str, 
        rule_function: Callable[[pd.DataFrame], List[ValidationIssue]]
    ) -> None
```

**Built-in Validation Rules:**

- `missing_values`: Check for missing data patterns
- `duplicates`: Detect duplicate rows
- `data_types`: Validate data type consistency
- `outliers`: Detect statistical outliers using IQR method
- `consistency`: Check data formatting consistency
- `completeness`: Assess overall data completeness

**Example Usage:**

```python
import pandas as pd
from refunc.data_science import DataValidator, DataSchema

# Create sample data
df = pd.DataFrame({
    'id': [1, 2, 3, 2],  # Duplicate
    'name': ['Alice', 'Bob', None, 'Charlie'],  # Missing value
    'age': [25, 30, 150, 35],  # Outlier
    'email': ['alice@test.com', 'invalid-email', 'bob@test.com', 'charlie@test.com']
})

# Basic validation
validator = DataValidator()
report = validator.validate_dataframe(df)

print(report.summary())
# Output:
# Data Validation Report
# ==============================
# Overall Status: ❌ INVALID
# Quality Score: 65.23%
# Quality Level: FAIR
# Total Issues: 4

# Access specific issues
for issue in report.issues:
    print(f"{issue.severity.value}: {issue.message}")

# Schema-based validation
schema = DataSchema(
    columns={
        'id': {'dtype': 'int64', 'unique': True},
        'name': {'dtype': 'object', 'nullable': False},
        'age': {'dtype': 'int64', 'min': 0, 'max': 120},
        'email': {'dtype': 'object', 'format': 'email'}
    },
    required_columns=['id', 'name', 'age'],
    primary_key=['id'],
    constraints=[
        {'type': 'range', 'column': 'age', 'min': 0, 'max': 120}
    ]
)

schema_report = validator.validate_dataframe(df, schema=schema)
```

### ValidationReport

Comprehensive validation results with quality metrics and issue details.

```python
@dataclass
class ValidationReport:
    is_valid: bool
    quality_score: float
    quality_level: DataQualityLevel
    total_issues: int
    issues_by_severity: Dict[ValidationSeverity, int]
    issues: List[ValidationIssue]
    dataset_stats: Dict[str, Any]
    column_stats: Dict[str, Dict[str, Any]]
    
    def summary(self) -> str
    def get_issues(
        self, 
        severity: Optional[ValidationSeverity] = None,
        column: Optional[str] = None
    ) -> List[ValidationIssue]
```

### DataSchema

Schema definition for structured data validation.

```python
@dataclass
class DataSchema:
    columns: Dict[str, Dict[str, Any]]
    required_columns: List[str] = field(default_factory=list)
    optional_columns: List[str] = field(default_factory=list)
    primary_key: Optional[List[str]] = None
    foreign_keys: Dict[str, str] = field(default_factory=dict)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
```

**Constraint Types:**

- `range`: Min/max value constraints for numeric columns
- `custom`: Custom validation expressions using pandas `.eval()`

**Example Schema:**

```python
schema = DataSchema(
    columns={
        'user_id': {
            'dtype': 'int64',
            'nullable': False,
            'unique': True
        },
        'username': {
            'dtype': 'object',
            'nullable': False,
            'max_length': 50
        },
        'age': {
            'dtype': 'int64',
            'nullable': True,
            'min': 13,
            'max': 120
        },
        'email': {
            'dtype': 'object',
            'nullable': False,
            'format': 'email'
        }
    },
    required_columns=['user_id', 'username', 'email'],
    primary_key=['user_id'],
    constraints=[
        {
            'type': 'range',
            'column': 'age',
            'min': 13,
            'max': 120
        },
        {
            'type': 'custom',
            'condition': 'age > 0 or age.isnull()'
        }
    ]
)
```

### Convenience Functions

```python
def validate_dataframe(
    df: pd.DataFrame,
    schema: Optional[DataSchema] = None,
    strict: bool = False
) -> ValidationReport

def quick_validate(df: pd.DataFrame) -> bool

def create_schema_from_dataframe(df: pd.DataFrame) -> DataSchema
```

## Data Profiling

### DataProfiler

Comprehensive data profiling engine for statistical analysis and insights generation.

```python
class DataProfiler:
    def __init__(self, enable_plots: bool = True)
    
    def profile_dataframe(
        self,
        df: pd.DataFrame,
        name: str = "Dataset",
        profile_type: ProfileType = ProfileType.DETAILED,
        sample_size: Optional[int] = None
    ) -> DatasetProfile
    
    def generate_report(
        self, 
        profile: DatasetProfile, 
        output_file: Optional[str] = None
    ) -> str
    
    def export_profile(
        self, 
        profile: DatasetProfile, 
        output_file: str, 
        format: str = 'json'
    ) -> None
```

**Profile Types:**

- `ProfileType.BASIC`: Essential statistics only
- `ProfileType.DETAILED`: Comprehensive analysis with correlations
- `ProfileType.COMPREHENSIVE`: Full analysis with automated insights

**Example Usage:**

```python
from refunc.data_science import DataProfiler, ProfileType

# Create sample dataset
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'customer_id': range(1000),
    'age': np.random.normal(35, 10, 1000),
    'income': np.random.exponential(50000, 1000),
    'purchases': np.random.poisson(5, 1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000),
    'registration_date': pd.date_range('2020-01-01', periods=1000, freq='D')
})

# Generate comprehensive profile
profiler = DataProfiler()
profile = profiler.profile_dataframe(df, name="Customer Data", profile_type=ProfileType.COMPREHENSIVE)

print(profile.summary())
# Output:
# Dataset Profile: Customer Data
# ==================================================
# Shape: 1,000 rows × 6 columns
# Memory Usage: 0.5 MB
# Profile Type: Comprehensive
# Created: 2024-01-15 10:30:45
# 
# Data Quality Metrics:
#   Overall Quality: 98.50%
#   Completeness: 100.00%
#   Consistency: 97.00%

# Access column profiles
age_profile = profile.get_column_profile('age')
print(age_profile.summary())

# Get automated insights
quality_insights = profile.get_insights_by_type(InsightType.QUALITY)
correlation_insights = profile.get_insights_by_type(InsightType.CORRELATION)

# Export detailed report
profiler.generate_report(profile, "customer_data_report.txt")
profiler.export_profile(profile, "customer_data_profile.json")
```

### DatasetProfile

Complete dataset profile with comprehensive statistics and insights.

```python
@dataclass
class DatasetProfile:
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
    
    # Correlation analysis
    correlation_matrix: Optional[pd.DataFrame] = None
    high_correlations: List[Tuple[str, str, float]] = field(default_factory=list)
    
    # Automated insights
    insights: List[Dict[str, Any]] = field(default_factory=list)
    
    # Quality metrics
    data_quality_score: float = 0.0
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    
    def summary(self) -> str
    def get_column_profile(self, column_name: str) -> Optional[ColumnProfile]
    def get_insights_by_type(self, insight_type: InsightType) -> List[Dict[str, Any]]
```

### ColumnProfile

Detailed profile information for individual columns.

```python
@dataclass
class ColumnProfile:
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
    
    def summary(self) -> str
```

### Profile Comparison

```python
def compare_profiles(profile1: DatasetProfile, profile2: DatasetProfile) -> Dict[str, Any]
```

**Example:**

```python
# Compare profiles from different time periods
profile_before = profiler.profile_dataframe(df_january, name="January Data")
profile_after = profiler.profile_dataframe(df_february, name="February Data")

comparison = compare_profiles(profile_before, profile_after)
print(f"Shape change: {comparison['shape_change']}")
print(f"Quality change: {comparison['quality_change']}")
print(f"Missing data change: {comparison['missing_data_change']:.2f}%")
```

### Profile Convenience Functions

```python
def profile_dataframe(
    df: pd.DataFrame,
    name: str = "Dataset",
    detailed: bool = True
) -> DatasetProfile

def quick_profile(df: pd.DataFrame) -> str
```

## Data Transformations

### TransformationPipeline

Composable data transformation pipeline with logging and error handling. Pipelines can now perform an optional pre-step join by supplying a `PipelineJoinConfig`, allowing the primary DataFrame to merge with auxiliary inputs before any transformation steps run.

```python
class TransformationPipeline:
    def __init__(self, name: str = "pipeline")
    
    def add_step(self, transformer: BaseTransformer) -> 'TransformationPipeline'
    
    # Convenience methods
    def add_imputation(
        self,
        method: ImputationMethod = ImputationMethod.MEAN,
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> 'TransformationPipeline'
    
    def add_scaling(
        self,
        method: ScalingMethod = ScalingMethod.STANDARD,
        columns: Optional[List[str]] = None
    ) -> 'TransformationPipeline'
    
    def add_outlier_removal(
        self,
        method: str = 'iqr',
        threshold: float = 1.5,
        columns: Optional[List[str]] = None
    ) -> 'TransformationPipeline'
    
    def add_encoding(
        self,
        method: str = 'onehot',
        columns: Optional[List[str]] = None,
        drop_first: bool = True
    ) -> 'TransformationPipeline'
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'TransformationPipeline'
    def transform(self, data: pd.DataFrame) -> PipelineResult
    def fit_transform(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> PipelineResult
```

#### PipelineJoinConfig

```python
@dataclass(frozen=True)
class PipelineJoinConfig:
    sources: JoinSourceProvider
    columns: Optional[Sequence[str]] = None
    how: MergeHow = "inner"
    validate: Optional[MergeValidate] = None
    indicator: Union[bool, str] = False
    sort: bool = False
    suffix_template: str = "_df{index}"
```

- `sources`: Sequence (or callable returning a sequence) of additional `DataFrame` objects to merge with the pipeline input.
- `columns`: Specific column names to join on. When omitted, all sources must share at least one common column name.
- `how`: Pandas merge strategy (`"inner"`, `"left"`, `"outer"`, etc.).
- `validate`: Optional pandas validation rule such as `"1:1"` or `"1:m"` to enforce join cardinality.
- `indicator`: Append merge provenance information using pandas' indicator column.
- `suffix_template`: Template applied to duplicate column names contributed by additional sources (receives the source index).

```python
from refunc.data_science import (
    PipelineJoinConfig,
    TransformationPipeline,
    join_dataframes_on_common_columns,
)

reference_frame = pd.read_parquet("lookups/country_codes.parquet")

join_config = PipelineJoinConfig(
    sources=[reference_frame],
    columns=["country_code"],
    how="left",
    validate="m:1",
    indicator="country_match",
)

pipeline = (TransformationPipeline("customer_enrichment", join_config=join_config)
            .add_imputation()
            .add_scaling())
```

**Example Usage:**

```python
from refunc.data_science import (
    TransformationPipeline, ImputationMethod, ScalingMethod
)

# Create preprocessing pipeline
pipeline = (TransformationPipeline("preprocessing")
           .add_outlier_removal(method='iqr', threshold=1.5)
           .add_imputation(ImputationMethod.MEDIAN)
           .add_scaling(ScalingMethod.STANDARD)
           .add_encoding(method='onehot', drop_first=True))

# Attach a join configuration later if needed
pipeline.set_join_config(join_config)

# Fit and transform data
result = pipeline.fit_transform(df, target=target_series)

if result.success:
    transformed_data = result.final_data
    print(f"Pipeline completed in {result.total_execution_time:.3f}s")
    print(f"Shape: {result.original_data.shape} → {result.final_data.shape}")
else:
    # Handle failed steps
    for failed_step in result.get_failed_steps():
        print(f"Failed: {failed_step.transformation_name}")
        print(f"Errors: {failed_step.errors}")
```

### Individual Transformers

#### MissingValueImputer

```python
class MissingValueImputer(BaseTransformer):
    def __init__(
        self,
        method: ImputationMethod = ImputationMethod.MEAN,
        columns: Optional[List[str]] = None,
        fill_value: Any = None,
        n_neighbors: int = 5
    )
```

**Imputation Methods:**

- `ImputationMethod.MEAN`: Mean value for numeric columns
- `ImputationMethod.MEDIAN`: Median value for numeric columns
- `ImputationMethod.MODE`: Most frequent value
- `ImputationMethod.CONSTANT`: User-specified constant value
- `ImputationMethod.KNN`: K-nearest neighbors imputation
- `ImputationMethod.FORWARD_FILL`: Forward fill (time series)
- `ImputationMethod.BACKWARD_FILL`: Backward fill (time series)

#### DataScaler

```python
class DataScaler(BaseTransformer):
    def __init__(
        self,
        method: ScalingMethod = ScalingMethod.STANDARD,
        columns: Optional[List[str]] = None
    )
```

**Scaling Methods:**

- `ScalingMethod.STANDARD`: Z-score normalization (mean=0, std=1)
- `ScalingMethod.MINMAX`: Min-max scaling to [0, 1] range
- `ScalingMethod.ROBUST`: Robust scaling using median and IQR
- `ScalingMethod.POWER`: Power transformation (Yeo-Johnson)

#### OutlierRemover

```python
class OutlierRemover(BaseTransformer):
    def __init__(
        self,
        method: str = 'iqr',
        threshold: float = 1.5,
        columns: Optional[List[str]] = None
    )
```

**Outlier Detection Methods:**

- `iqr`: Interquartile range method
- `zscore`: Z-score method

#### CategoricalEncoder

```python
class CategoricalEncoder(BaseTransformer):
    def __init__(
        self,
        method: str = 'onehot',
        columns: Optional[List[str]] = None,
        drop_first: bool = True
    )
```

**Encoding Methods:**

- `onehot`: One-hot encoding with dummy variables
- `label`: Label encoding with integer mapping

#### CustomTransformer

```python
class CustomTransformer(BaseTransformer):
    def __init__(
        self,
        name: str,
        transform_func: Callable[[pd.DataFrame], pd.DataFrame],
        fit_func: Optional[Callable[[pd.DataFrame, Optional[pd.Series]], Any]] = None
    )
```

### Pipeline Results

```python
@dataclass
class PipelineResult:
    success: bool
    original_data: pd.DataFrame
    final_data: Optional[pd.DataFrame]
    step_results: List[TransformationResult]
    total_execution_time: float
    pipeline_name: str
    timestamp: datetime
    
    def summary(self) -> str
    def get_failed_steps(self) -> List[TransformationResult]
```

### Pipeline Convenience Functions

```python
def create_basic_pipeline(
    *,
    join_config: Optional[PipelineJoinConfig] = None,
) -> TransformationPipeline

def create_robust_pipeline(
    *,
    join_config: Optional[PipelineJoinConfig] = None,
) -> TransformationPipeline

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
) -> pd.DataFrame
```

**Example:**

```python
# Quick preprocessing with reference data join
lookup_df = pd.read_csv("lookups/channel_labels.csv")

cleaned_data = apply_quick_preprocessing(
    df,
    include_outlier_removal=True,
    join_sources=[lookup_df],
    join_columns=["channel_id"],
    join_how="left",
    join_validate="m:1",
)

# Or use predefined pipelines
robust_pipeline = create_robust_pipeline(join_config=join_config)
result = robust_pipeline.fit_transform(df)
```

### Joining Utilities

#### join_dataframes_on_common_columns

```python
def join_dataframes_on_common_columns(
    dataframes: Sequence[pd.DataFrame],
    *,
    columns: Optional[Sequence[str]] = None,
    how: MergeHow = "inner",
    validate: Optional[MergeValidate] = None,
    indicator: Union[bool, str] = False,
    sort: bool = False,
    suffix_template: str = "_df{index}",
) -> pd.DataFrame
```

Utility helper that powers the pipeline join flow. The function joins any number of `DataFrame` objects on shared column names, automatically templating suffixes for overlapping columns and optionally surfacing pandas' `_merge` indicator on the final merge step.

```python
sales_with_targets = join_dataframes_on_common_columns(
    [regional_sales_df, targets_df, territories_df],
    columns=["region_id"],
    how="left",
    validate="m:1",
    indicator=True,
    suffix_template="_src{index}",
)
```

## Data Cleaning

### DataCleaner

Comprehensive data cleaning engine with automated quality improvements.

```python
class DataCleaner:
    def __init__(self, aggressive_cleaning: bool = False)
    
    def clean_dataframe(
        self,
        df: pd.DataFrame,
        operations: Optional[List[CleaningOperation]] = None,
        column_types: Optional[Dict[str, str]] = None
    ) -> Tuple[pd.DataFrame, CleaningReport]
```

**Cleaning Operations:**

- `CleaningOperation.REMOVE_DUPLICATES`: Remove duplicate rows
- `CleaningOperation.STANDARDIZE_TEXT`: Standardize text formatting
- `CleaningOperation.FIX_DATA_TYPES`: Auto-detect and fix data types
- `CleaningOperation.CLEAN_NUMERIC`: Clean numeric data (infinities, precision errors)
- `CleaningOperation.STANDARDIZE_DATES`: Parse and standardize date formats
- `CleaningOperation.REMOVE_OUTLIERS`: Remove statistical outliers (aggressive mode)
- `CleaningOperation.NORMALIZE_STRINGS`: Normalize string formatting
- `CleaningOperation.VALIDATE_FORMATS`: Validate common formats (email, phone, URL)

**Example Usage:**

```python
from refunc.data_science import DataCleaner, CleaningOperation

# Create messy data
df_messy = pd.DataFrame({
    'id': [1, 2, 3, 2, 4],  # Duplicates
    'name': ['  Alice  ', 'BOB', 'charlie', '  Alice  ', 'Diana'],  # Inconsistent formatting
    'age': ['25', '30.0', 'unknown', '25', '35'],  # Mixed types
    'email': ['alice@test.com', 'invalid-email', 'charlie@test.com', 'alice@test.com', 'diana@test.com'],
    'phone': ['(555) 123-4567', '555.987.6543', '+1-555-111-2222', '(555) 123-4567', '555-444-3333'],
    'score': [85.5, 92.0, float('inf'), 85.5, 78.2]  # Contains infinity
})

# Comprehensive cleaning
cleaner = DataCleaner(aggressive_cleaning=True)
cleaned_df, report = cleaner.clean_dataframe(df_messy)

print(report.summary())
# Output:
# Data Cleaning Report
# ==============================
# Shape Change: (5, 6) → (4, 6)
# Total Changes: 12
# Operations: 7/7 successful
# Quality Improvement: 45.20% → 89.50%
# Execution Time: 0.025s

# Check specific operations
for result in report.operations_performed:
    if result.changes_made > 0:
        print(f"{result.operation.value}: {result.changes_made} changes")
```

### CleaningReport

```python
@dataclass
class CleaningReport:
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    total_changes: int
    operations_performed: List[CleaningResult]
    execution_time: float
    data_quality_before: float
    data_quality_after: float
    
    def summary(self) -> str
```

### Advanced Cleaning Functions

```python
def remove_duplicates_advanced(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first',
    ignore_case: bool = True
) -> pd.DataFrame

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame

def detect_encoding_issues(df: pd.DataFrame) -> Dict[str, List[str]]
```

**Example:**

```python
# Advanced duplicate removal with case-insensitive matching
df_no_dups = remove_duplicates_advanced(
    df, 
    subset=['name', 'email'], 
    ignore_case=True
)

# Standardize column names to snake_case
df_clean_cols = standardize_column_names(df)

# Detect encoding issues
encoding_issues = detect_encoding_issues(df)
for column, issues in encoding_issues.items():
    print(f"{column}: {len(issues)} encoding issues found")
```

### Cleaning Convenience Functions

```python
def quick_clean(df: pd.DataFrame, aggressive: bool = False) -> pd.DataFrame

def clean_with_report(df: pd.DataFrame, aggressive: bool = False) -> Tuple[pd.DataFrame, CleaningReport]
```

## Pandas Extensions

### DataFrame Accessor (.refunc)

The `refunc` accessor is automatically registered for all pandas DataFrames, providing enhanced functionality through the `.refunc` namespace.

```python
# Automatic registration - no import needed
df.refunc.profile()  # Data profiling
df.refunc.validate()  # Data validation
df.refunc.clean()    # Data cleaning
```

#### Profiling and Validation Methods

```python
df.refunc.profile(detailed: bool = True, name: str = "Dataset") -> DatasetProfile
df.refunc.validate(schema=None, strict: bool = False) -> ValidationReport
df.refunc.clean(aggressive: bool = False) -> Tuple[pd.DataFrame, CleaningReport]
df.refunc.quick_clean(aggressive: bool = False) -> pd.DataFrame
```

#### Memory Optimization

```python
df.refunc.memory_usage_detailed() -> pd.DataFrame
df.refunc.optimize_memory(categorical_threshold: float = 0.5) -> pd.DataFrame
```

**Example:**

```python
# Analyze memory usage
memory_info = df.refunc.memory_usage_detailed()
print(memory_info[['column', 'current_memory_mb', 'categorical_savings_mb']])

# Optimize memory usage
df_optimized = df.refunc.optimize_memory()
print(f"Memory reduced from {df.memory_usage(deep=True).sum()/1024/1024:.1f}MB "
      f"to {df_optimized.memory_usage(deep=True).sum()/1024/1024:.1f}MB")
```

#### Missing Data Analysis

```python
df.refunc.missing_patterns() -> pd.DataFrame
```

**Example:**

```python
missing_info = df.refunc.missing_patterns()
print(missing_info.head())
#     column  missing_count  missing_percentage data_type
# 0     age             15               15.0     int64
# 1    name              8                8.0    object
```

#### Visualization Methods

```python
df.refunc.correlation_heatmap(
    method: str = 'pearson', 
    figsize: Tuple[int, int] = (10, 8), 
    **kwargs
)

df.refunc.distribution_plots(
    columns: Optional[List[str]] = None, 
    figsize: Tuple[int, int] = (15, 10)
)
```

#### Outlier Analysis

```python
df.refunc.outlier_analysis(
    method: str = 'iqr', 
    threshold: float = 1.5
) -> pd.DataFrame
```

#### Balanced Sampling

```python
df.refunc.sample_balanced(
    target_column: str, 
    n_samples: Optional[int] = None
) -> pd.DataFrame
```

**Example:**

```python
# Create balanced sample for machine learning
balanced_df = df.refunc.sample_balanced('target_class', n_samples=1000)
print(balanced_df['target_class'].value_counts())
```

#### Report Export

```python
df.refunc.export_summary(filename: str, include_plots: bool = True)
```

**Example:**

```python
# Export comprehensive HTML report
df.refunc.export_summary('data_summary.html', include_plots=True)

# Export text summary
df.refunc.export_summary('data_summary.txt', include_plots=False)
```

### Series Accessor (.refunc)

Enhanced functionality for pandas Series through the `.refunc` namespace.

```python
series.refunc.outliers(method: str = 'iqr', threshold: float = 1.5) -> pd.Series
series.refunc.remove_outliers(method: str = 'iqr', threshold: float = 1.5) -> pd.Series
series.refunc.normalize(method: str = 'minmax') -> pd.Series
series.refunc.entropy() -> float
series.refunc.pattern_frequency(pattern: str) -> int
```

**Example:**

```python
# Detect outliers
outlier_mask = df['price'].refunc.outliers(method='iqr', threshold=2.0)
print(f"Found {outlier_mask.sum()} outliers")

# Remove outliers
clean_prices = df['price'].refunc.remove_outliers()

# Normalize data
normalized_prices = df['price'].refunc.normalize(method='zscore')

# Calculate entropy (information content)
entropy = df['category'].refunc.entropy()
print(f"Category entropy: {entropy:.3f}")

# Pattern matching in text
email_count = df['email'].refunc.pattern_frequency(r'.*@gmail\.com$')
print(f"Gmail addresses: {email_count}")
```

### Enhanced DataFrame Operations

#### Fuzzy Merging

```python
def merge_on_fuzzy(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    threshold: float = 0.8,
    how: str = 'inner'
) -> pd.DataFrame
```

**Example:**

```python
from refunc.data_science import merge_on_fuzzy

# Merge datasets with fuzzy string matching
companies_df = pd.DataFrame({
    'company': ['Apple Inc.', 'Microsoft Corp', 'Google LLC'],
    'founded': [1976, 1975, 1998]
})

stocks_df = pd.DataFrame({
    'company_name': ['Apple Inc', 'Microsoft Corporation', 'Alphabet Inc'],
    'ticker': ['AAPL', 'MSFT', 'GOOGL']
})

merged = merge_on_fuzzy(
    companies_df, stocks_df,
    left_on='company', right_on='company_name',
    threshold=0.7
)
```

#### Advanced Pivot Tables

```python
def pivot_advanced(
    df: pd.DataFrame,
    index: Union[str, List[str]],
    columns: str,
    values: Optional[str] = None,
    aggfunc: Union[str, Callable] = 'mean',
    fill_value: Any = 0,
    normalize: bool = False
) -> pd.DataFrame
```

**Example:**

```python
from refunc.data_science import pivot_advanced

# Create advanced pivot table with normalization
pivot_table = pivot_advanced(
    sales_df,
    index=['region', 'quarter'],
    columns='product',
    values='revenue',
    aggfunc='sum',
    normalize=True  # Normalize by row
)
```

## Integration Examples

### Complete Data Science Workflow

```python
import pandas as pd
from refunc.data_science import *

# Load raw data
df = pd.read_csv('raw_data.csv')

# 1. Initial data exploration
print("=== Initial Data Profile ===")
initial_profile = df.refunc.profile(detailed=True, name="Raw Data")
print(initial_profile.summary())

# 2. Data validation
print("\n=== Data Validation ===")
validation_report = df.refunc.validate()
print(validation_report.summary())

# Print validation issues
for issue in validation_report.issues:
    if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
        print(f"❌ {issue}")

# 3. Data cleaning
print("\n=== Data Cleaning ===")
cleaned_df, cleaning_report = df.refunc.clean(aggressive=True)
print(cleaning_report.summary())

# 4. Memory optimization
print("\n=== Memory Optimization ===")
optimized_df = cleaned_df.refunc.optimize_memory()
original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
optimized_memory = optimized_df.memory_usage(deep=True).sum() / 1024 / 1024
print(f"Memory usage: {original_memory:.1f}MB → {optimized_memory:.1f}MB")

# 5. Data transformation pipeline
print("\n=== Data Transformation ===")
pipeline = (create_robust_pipeline()
           .add_imputation(ImputationMethod.KNN, n_neighbors=5)
           .add_scaling(ScalingMethod.ROBUST))

# Separate features and target
feature_columns = [col for col in optimized_df.columns if col != 'target']
X = optimized_df[feature_columns]
y = optimized_df['target'] if 'target' in optimized_df.columns else None

# Transform features
transformation_result = pipeline.fit_transform(X, y)

if transformation_result.success:
    X_transformed = transformation_result.final_data
    print(f"Transformation completed: {X.shape} → {X_transformed.shape}")
    
    # 6. Final validation
    print("\n=== Final Quality Check ===")
    final_validation = X_transformed.refunc.validate()
    print(f"Final data quality: {'✅ Valid' if final_validation.is_valid else '❌ Invalid'}")
    print(f"Quality score: {final_validation.quality_score:.2%}")
    
    # 7. Export comprehensive report
    print("\n=== Exporting Reports ===")
    X_transformed.refunc.export_summary('final_data_report.html')
    
    # Export transformation metadata
    with open('transformation_log.txt', 'w') as f:
        f.write("Data Science Pipeline Report\n")
        f.write("="*50 + "\n\n")
        f.write("Initial Profile:\n")
        f.write(initial_profile.summary() + "\n\n")
        f.write("Cleaning Report:\n")
        f.write(cleaning_report.summary() + "\n\n")
        f.write("Transformation Summary:\n")
        f.write(transformation_result.summary() + "\n\n")
        f.write("Final Validation:\n")
        f.write(final_validation.summary())
    
    print("✅ Data science pipeline completed successfully!")
    
else:
    print("❌ Transformation failed:")
    for failed_step in transformation_result.get_failed_steps():
        print(f"  - {failed_step.transformation_name}: {failed_step.errors}")
```

### Schema-Driven Data Processing

```python
# Define data schema for consistent processing
user_schema = DataSchema(
    columns={
        'user_id': {'dtype': 'int64', 'nullable': False, 'unique': True},
        'username': {'dtype': 'object', 'nullable': False, 'max_length': 50},
        'email': {'dtype': 'object', 'nullable': False, 'format': 'email'},
        'age': {'dtype': 'int64', 'nullable': True, 'min': 13, 'max': 120},
        'registration_date': {'dtype': 'datetime64[ns]', 'nullable': False},
        'subscription_type': {'dtype': 'category', 'nullable': False},
        'monthly_spend': {'dtype': 'float64', 'nullable': True, 'min': 0}
    },
    required_columns=['user_id', 'username', 'email', 'registration_date'],
    primary_key=['user_id'],
    constraints=[
        {'type': 'range', 'column': 'age', 'min': 13, 'max': 120},
        {'type': 'range', 'column': 'monthly_spend', 'min': 0, 'max': 10000},
        {'type': 'custom', 'condition': 'registration_date <= @pd.Timestamp.now()'}
    ]
)

def process_user_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process user data according to schema."""
    
    # 1. Schema validation
    validator = DataValidator(strict_mode=True)
    validation_report = validator.validate_dataframe(df, schema=user_schema)
    
    if not validation_report.is_valid:
        print("❌ Schema validation failed:")
        for issue in validation_report.get_issues(severity=ValidationSeverity.ERROR):
            print(f"  - {issue}")
        raise ValidationError("Data does not conform to schema")
    
    # 2. Type conversion based on schema
    column_types = {
        col: details['dtype'] for col, details in user_schema.columns.items()
    }
    
    cleaner = DataCleaner()
    cleaned_df, _ = cleaner.clean_dataframe(df, column_types=column_types)
    
    # 3. Apply business rules
    processed_df = cleaned_df.copy()
    
    # Convert subscription types to category with specific order
    subscription_order = ['free', 'basic', 'premium', 'enterprise']
    processed_df['subscription_type'] = pd.Categorical(
        processed_df['subscription_type'], 
        categories=subscription_order, 
        ordered=True
    )
    
    # Create derived features
    processed_df['account_age_days'] = (
        pd.Timestamp.now() - processed_df['registration_date']
    ).dt.days
    
    processed_df['spend_per_age'] = (
        processed_df['monthly_spend'] / processed_df['age']
    ).fillna(0)
    
    return processed_df

# Usage
processed_users = process_user_data(raw_user_df)
```

### Automated Data Quality Monitoring

```python
class DataQualityMonitor:
    """Monitor data quality over time."""
    
    def __init__(self, baseline_df: pd.DataFrame):
        self.baseline_profile = profile_dataframe(baseline_df, name="Baseline")
        self.thresholds = {
            'quality_score': 0.8,
            'missing_percentage': 10.0,
            'duplicate_percentage': 5.0
        }
    
    def check_quality(self, df: pd.DataFrame, name: str = "Current") -> Dict[str, Any]:
        """Check data quality against baseline."""
        current_profile = profile_dataframe(df, name=name)
        comparison = compare_profiles(self.baseline_profile, current_profile)
        
        # Quality checks
        alerts = []
        
        if current_profile.data_quality_score < self.thresholds['quality_score']:
            alerts.append(f"Quality score below threshold: {current_profile.data_quality_score:.2%}")
        
        if current_profile.missing_percentage > self.thresholds['missing_percentage']:
            alerts.append(f"Missing data above threshold: {current_profile.missing_percentage:.1f}%")
        
        if current_profile.duplicate_percentage > self.thresholds['duplicate_percentage']:
            alerts.append(f"Duplicates above threshold: {current_profile.duplicate_percentage:.1f}%")
        
        # Check for significant changes
        if abs(comparison['missing_data_change']) > 5.0:
            alerts.append(f"Significant missing data change: {comparison['missing_data_change']:+.1f}%")
        
        quality_change = comparison['quality_change']['data_quality']
        if quality_change < -0.1:
            alerts.append(f"Quality degradation: {quality_change:+.2%}")
        
        return {
            'profile': current_profile,
            'comparison': comparison,
            'alerts': alerts,
            'quality_score': current_profile.data_quality_score,
            'passed': len(alerts) == 0
        }

# Usage
monitor = DataQualityMonitor(baseline_data)

# Check new data batch
result = monitor.check_quality(new_batch_df, name="Batch_2024_01")

if result['alerts']:
    print("🚨 Data Quality Alerts:")
    for alert in result['alerts']:
        print(f"  - {alert}")
else:
    print(f"✅ Data quality check passed (score: {result['quality_score']:.2%})")
```

This comprehensive documentation covers all aspects of the `refunc.data_science` module, providing users with detailed information about data validation, profiling, transformation, cleaning, and the enhanced pandas functionality through custom extensions.
 
 
