"""
Data Science Tools Package.

This package provides comprehensive data science utilities including:
- Data validation and quality assessment
- Data profiling and statistical analysis  
- Data transformation and preprocessing pipelines
- Data cleaning and quality improvement
- Enhanced pandas functionality and extensions

The package integrates with the math_stats package to provide
statistical analysis capabilities and offers a complete toolkit
for data science workflows.
"""

from .validation import (
    DataValidator,
    DataSchema,
    ValidationReport,
    ValidationIssue,
    DataQualityLevel,
    ValidationSeverity,
    validate_dataframe,
    quick_validate,
    create_schema_from_dataframe
)

from .profiling import (
    DataProfiler,
    DatasetProfile,
    ColumnProfile,
    ProfileType,
    InsightType,
    profile_dataframe,
    quick_profile,
    compare_profiles
)

from .transforms import (
    TransformationPipeline,
    BaseTransformer,
    MissingValueImputer,
    DataScaler,
    OutlierRemover,
    CategoricalEncoder,
    CustomTransformer,
    TransformationResult,
    PipelineResult,
    TransformationType,
    ScalingMethod,
    ImputationMethod,
    create_basic_pipeline,
    create_robust_pipeline,
    apply_quick_preprocessing
)

from .cleaning import (
    DataCleaner,
    CleaningResult,
    CleaningReport,
    CleaningOperation,
    quick_clean,
    clean_with_report,
    remove_duplicates_advanced,
    standardize_column_names,
    detect_encoding_issues
)

from .extensions import (
    RefuncDataFrameAccessor,
    RefuncSeriesAccessor,
    merge_on_fuzzy,
    pivot_advanced
)


__all__ = [
    # Validation
    'DataValidator',
    'DataSchema', 
    'ValidationReport',
    'ValidationIssue',
    'DataQualityLevel',
    'ValidationSeverity',
    'validate_dataframe',
    'quick_validate',
    'create_schema_from_dataframe',
    
    # Profiling
    'DataProfiler',
    'DatasetProfile',
    'ColumnProfile', 
    'ProfileType',
    'InsightType',
    'profile_dataframe',
    'quick_profile',
    'compare_profiles',
    
    # Transforms
    'TransformationPipeline',
    'BaseTransformer',
    'MissingValueImputer',
    'DataScaler',
    'OutlierRemover', 
    'CategoricalEncoder',
    'CustomTransformer',
    'TransformationResult',
    'PipelineResult',
    'TransformationType',
    'ScalingMethod',
    'ImputationMethod',
    'create_basic_pipeline',
    'create_robust_pipeline',
    'apply_quick_preprocessing',
    
    # Cleaning
    'DataCleaner',
    'CleaningResult',
    'CleaningReport',
    'CleaningOperation',
    'quick_clean',
    'clean_with_report',
    'remove_duplicates_advanced', 
    'standardize_column_names',
    'detect_encoding_issues',
    
    # Extensions
    'RefuncDataFrameAccessor',
    'RefuncSeriesAccessor',
    'merge_on_fuzzy',
    'pivot_advanced'
]


# Package metadata
__version__ = "1.0.0"
__author__ = "Refunc Development Team"
__description__ = "Comprehensive data science utilities for pandas DataFrames"


def get_version() -> str:
    """Get package version."""
    return __version__


def list_capabilities() -> dict:
    """List all data science capabilities."""
    return {
        'validation': {
            'description': 'Data quality validation and schema checking',
            'key_features': [
                'Comprehensive data quality assessment',
                'Schema validation and constraint checking',
                'Missing value and outlier detection',
                'Data consistency analysis'
            ]
        },
        'profiling': {
            'description': 'Statistical analysis and data profiling',
            'key_features': [
                'Automated statistical summaries',
                'Distribution analysis and fitting',
                'Correlation analysis',
                'Automated insights generation'
            ]
        },
        'transforms': {
            'description': 'Data transformation and preprocessing pipelines',
            'key_features': [
                'Composable transformation pipelines',
                'Multiple imputation strategies',
                'Data scaling and normalization',
                'Categorical encoding'
            ]
        },
        'cleaning': {
            'description': 'Data cleaning and quality improvement',
            'key_features': [
                'Duplicate removal',
                'Data type correction',
                'Text standardization',
                'Format validation'
            ]
        },
        'extensions': {
            'description': 'Enhanced pandas functionality',
            'key_features': [
                'Custom DataFrame/Series accessors',
                'Memory optimization tools',
                'Advanced merging capabilities',
                'Visualization helpers'
            ]
        }
    }


# Auto-import extensions to register accessors
try:
    import pandas as pd
    # Import extensions to register the accessors
    from . import extensions
    
    # Verify accessors are registered
    if hasattr(pd.DataFrame, 'refunc') and hasattr(pd.Series, 'refunc'):
        pass  # Successfully registered
    else:
        import warnings
        warnings.warn("Failed to register pandas accessors", ImportWarning)
        
except ImportError as e:
    import warnings
    warnings.warn(f"Could not register pandas extensions: {e}", ImportWarning)