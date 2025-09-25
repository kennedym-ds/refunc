"""
Performance benchmarks for refunc data science utilities.

This module contains benchmarks for data processing, validation, transformation,
and analysis operations to ensure they maintain good performance characteristics.
"""

import pytest
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import time


# Import data science utilities for benchmarking
try:
    from refunc.data_science import (
        cleaning, transforms, validation, profiling
    )
    from refunc.math_stats import StatisticsEngine
    DATA_SCIENCE_AVAILABLE = True
except ImportError:
    DATA_SCIENCE_AVAILABLE = False
    pytestmark = pytest.mark.skip("refunc data science utilities not available")


@pytest.fixture
def sample_dataset():
    """Create sample dataset for benchmarking."""
    np.random.seed(42)
    n_samples = 10000
    
    return pd.DataFrame({
        'id': range(n_samples),
        'numeric_1': np.random.normal(100, 15, n_samples),
        'numeric_2': np.random.exponential(2, n_samples),
        'categorical': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'binary': np.random.choice([0, 1], n_samples),
        'with_nulls': np.random.choice([1, 2, 3, np.nan], n_samples),
        'outliers': np.concatenate([
            np.random.normal(50, 10, n_samples - 50),
            np.random.normal(500, 50, 50)  # Outliers
        ]),
        'text': [f"text_value_{i}" for i in range(n_samples)],
        'datetime': pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    })


@pytest.fixture  
def large_dataset():
    """Create larger dataset for stress testing."""
    np.random.seed(42)
    n_samples = 100000
    
    return pd.DataFrame({
        'feature_1': np.random.random(n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'feature_3': np.random.exponential(1, n_samples),
        'target': np.random.choice([0, 1], n_samples),
        'group': np.random.choice(range(100), n_samples)
    })


@pytest.mark.skipif(not DATA_SCIENCE_AVAILABLE, reason="refunc data science utilities not available")
class TestDataCleaningPerformance:
    """Benchmark tests for data cleaning operations."""
    
    def test_null_value_detection(self, benchmark, sample_dataset):
        """Benchmark null value detection."""
        
        def detect_nulls():
            return cleaning.detect_missing_values(sample_dataset)
        
        null_info = benchmark(detect_nulls)
        assert isinstance(null_info, dict)
        assert 'with_nulls' in null_info
    
    def test_outlier_detection_zscore(self, benchmark, sample_dataset):
        """Benchmark Z-score outlier detection."""
        
        def detect_outliers():
            return cleaning.detect_outliers(
                sample_dataset['outliers'], 
                method='zscore',
                threshold=3.0
            )
        
        outliers = benchmark(detect_outliers)
        assert len(outliers) > 0  # Should detect some outliers
    
    def test_outlier_detection_iqr(self, benchmark, sample_dataset):
        """Benchmark IQR outlier detection."""
        
        def detect_outliers_iqr():
            return cleaning.detect_outliers(
                sample_dataset['outliers'],
                method='iqr',
                threshold=1.5
            )
        
        outliers = benchmark(detect_outliers_iqr)
        assert len(outliers) >= 0
    
    def test_data_deduplication(self, benchmark):
        """Benchmark data deduplication."""
        
        # Create dataset with duplicates
        base_data = pd.DataFrame({
            'id': range(1000),
            'value': np.random.random(1000)
        })
        
        # Add duplicates
        duplicated_data = pd.concat([base_data, base_data.iloc[:100]], ignore_index=True)
        
        def remove_duplicates():
            return cleaning.remove_duplicates(
                duplicated_data,
                subset=['id'],
                keep='first'
            )
        
        clean_data = benchmark(remove_duplicates)
        assert len(clean_data) == 1000  # Should remove 100 duplicates
    
    def test_missing_value_imputation(self, benchmark, sample_dataset):
        """Benchmark missing value imputation."""
        
        def impute_missing():
            return cleaning.impute_missing_values(
                sample_dataset['with_nulls'],
                method='mean'
            )
        
        imputed_data = benchmark(impute_missing)
        assert not imputed_data.isna().any()


@pytest.mark.skipif(not DATA_SCIENCE_AVAILABLE, reason="refunc data science utilities not available")
class TestDataTransformationPerformance:
    """Benchmark tests for data transformation operations."""
    
    def test_standard_scaling(self, benchmark, sample_dataset):
        """Benchmark standard scaling transformation."""
        
        numeric_data = sample_dataset[['numeric_1', 'numeric_2']].copy()
        
        def standard_scale():
            return transforms.standard_scale(numeric_data)
        
        scaled_data = benchmark(standard_scale)
        assert scaled_data.shape == numeric_data.shape
        # Check if standardized (mean ≈ 0, std ≈ 1)
        assert abs(scaled_data.mean().mean()) < 0.1
    
    def test_min_max_scaling(self, benchmark, sample_dataset):
        """Benchmark min-max scaling transformation."""
        
        numeric_data = sample_dataset[['numeric_1', 'numeric_2']].copy()
        
        def min_max_scale():
            return transforms.min_max_scale(numeric_data, feature_range=(0, 1))
        
        scaled_data = benchmark(min_max_scale)
        assert scaled_data.shape == numeric_data.shape
        assert scaled_data.min().min() >= 0
        assert scaled_data.max().max() <= 1
    
    def test_one_hot_encoding(self, benchmark, sample_dataset):
        """Benchmark one-hot encoding."""
        
        categorical_data = sample_dataset[['categorical']].copy()
        
        def one_hot_encode():
            return transforms.one_hot_encode(categorical_data, columns=['categorical'])
        
        encoded_data = benchmark(one_hot_encode)
        assert encoded_data.shape[0] == categorical_data.shape[0]
        assert encoded_data.shape[1] > categorical_data.shape[1]  # More columns after encoding
    
    def test_feature_binning(self, benchmark, sample_dataset):
        """Benchmark feature binning/discretization."""
        
        numeric_data = sample_dataset['numeric_1'].copy()
        
        def bin_features():
            return transforms.bin_numeric_feature(
                numeric_data,
                n_bins=10,
                strategy='equal_width'
            )
        
        binned_data = benchmark(bin_features)
        assert len(binned_data) == len(numeric_data)
        assert binned_data.nunique() <= 10
    
    def test_log_transformation(self, benchmark, sample_dataset):
        """Benchmark log transformation."""
        
        # Use exponential data which is suitable for log transform
        exp_data = sample_dataset['numeric_2'].copy()
        exp_data = exp_data[exp_data > 0]  # Ensure positive values
        
        def log_transform():
            return transforms.log_transform(exp_data, method='natural')
        
        log_data = benchmark(log_transform)
        assert len(log_data) == len(exp_data)
        assert not log_data.isna().any()


@pytest.mark.skipif(not DATA_SCIENCE_AVAILABLE, reason="refunc data science utilities not available")
class TestDataValidationPerformance:
    """Benchmark tests for data validation operations."""
    
    def test_data_type_validation(self, benchmark, sample_dataset):
        """Benchmark data type validation."""
        
        expected_types = {
            'id': 'int64',
            'numeric_1': 'float64', 
            'categorical': 'object',
            'binary': 'int64'
        }
        
        def validate_types():
            return validation.validate_data_types(sample_dataset, expected_types)
        
        validation_result = benchmark(validate_types)
        assert isinstance(validation_result, dict)
    
    def test_range_validation(self, benchmark, sample_dataset):
        """Benchmark range validation."""
        
        def validate_ranges():
            results = {}
            results['id'] = validation.validate_range(
                sample_dataset['id'], min_val=0, max_val=10000
            )
            results['binary'] = validation.validate_range(
                sample_dataset['binary'], min_val=0, max_val=1
            )
            return results
        
        range_results = benchmark(validate_ranges)
        assert all(isinstance(result, bool) for result in range_results.values())
    
    def test_schema_validation(self, benchmark, sample_dataset):
        """Benchmark schema validation."""
        
        schema = {
            'required_columns': ['id', 'numeric_1', 'categorical'],
            'column_types': {
                'id': 'int64',
                'numeric_1': 'float64',
                'categorical': 'object'
            },
            'constraints': {
                'id': {'min': 0, 'unique': True},
                'categorical': {'allowed_values': ['A', 'B', 'C', 'D']}
            }
        }
        
        def validate_schema():
            return validation.validate_dataframe_schema(sample_dataset, schema)
        
        schema_result = benchmark(validate_schema)
        assert isinstance(schema_result, dict)
    
    def test_data_quality_assessment(self, benchmark, sample_dataset):
        """Benchmark comprehensive data quality assessment."""
        
        def assess_quality():
            return validation.assess_data_quality(sample_dataset)
        
        quality_report = benchmark(assess_quality)
        assert isinstance(quality_report, dict)
        assert 'missing_values' in quality_report
        assert 'data_types' in quality_report


@pytest.mark.skipif(not DATA_SCIENCE_AVAILABLE, reason="refunc data science utilities not available")
class TestDataProfilingPerformance:
    """Benchmark tests for data profiling operations."""
    
    def test_basic_profiling(self, benchmark, sample_dataset):
        """Benchmark basic data profiling."""
        
        def profile_data():
            return profiling.profile_dataframe(sample_dataset)
        
        profile_result = benchmark(profile_data)
        assert isinstance(profile_result, dict)
        assert len(profile_result) > 0
    
    def test_statistical_summary(self, benchmark, sample_dataset):
        """Benchmark statistical summary generation."""
        
        def generate_stats():
            return profiling.generate_statistical_summary(
                sample_dataset.select_dtypes(include=[np.number])
            )
        
        stats_summary = benchmark(generate_stats)
        assert isinstance(stats_summary, pd.DataFrame)
        assert 'mean' in stats_summary.index
        assert 'std' in stats_summary.index
    
    def test_correlation_analysis(self, benchmark, sample_dataset):
        """Benchmark correlation analysis."""
        
        numeric_data = sample_dataset.select_dtypes(include=[np.number])
        
        def compute_correlations():
            return profiling.compute_correlation_matrix(
                numeric_data,
                method='pearson'
            )
        
        correlation_matrix = benchmark(compute_correlations)
        assert isinstance(correlation_matrix, pd.DataFrame)
        assert correlation_matrix.shape[0] == correlation_matrix.shape[1]
    
    def test_distribution_analysis(self, benchmark, sample_dataset):
        """Benchmark distribution analysis."""
        
        def analyze_distributions():
            results = {}
            for col in ['numeric_1', 'numeric_2']:
                results[col] = profiling.analyze_distribution(
                    sample_dataset[col]
                )
            return results
        
        dist_analysis = benchmark(analyze_distributions)
        assert len(dist_analysis) == 2
        assert all(isinstance(result, dict) for result in dist_analysis.values())


@pytest.mark.skipif(not DATA_SCIENCE_AVAILABLE, reason="refunc data science utilities not available")
class TestStatisticsEnginePerformance:
    """Benchmark tests for StatisticsEngine."""
    
    def test_descriptive_statistics(self, benchmark, sample_dataset):
        """Benchmark descriptive statistics computation."""
        
        engine = StatisticsEngine()
        numeric_data = sample_dataset['numeric_1'].values
        
        def compute_descriptive_stats():
            return engine.descriptive_statistics(numeric_data)
        
        stats = benchmark(compute_descriptive_stats)
        assert 'mean' in stats
        assert 'std' in stats
        assert 'median' in stats
    
    def test_hypothesis_testing(self, benchmark, sample_dataset):
        """Benchmark hypothesis testing."""
        
        engine = StatisticsEngine()
        sample1 = sample_dataset[sample_dataset['binary'] == 0]['numeric_1'].values
        sample2 = sample_dataset[sample_dataset['binary'] == 1]['numeric_1'].values
        
        def run_hypothesis_test():
            return engine.t_test_independent(sample1, sample2)
        
        test_result = benchmark(run_hypothesis_test)
        assert 'statistic' in test_result
        assert 'p_value' in test_result
    
    def test_confidence_intervals(self, benchmark, sample_dataset):
        """Benchmark confidence interval computation."""
        
        engine = StatisticsEngine()
        data = sample_dataset['numeric_1'].values
        
        def compute_confidence_interval():
            return engine.confidence_interval(data, confidence_level=0.95)
        
        ci_result = benchmark(compute_confidence_interval)
        assert 'lower' in ci_result
        assert 'upper' in ci_result
        assert 'confidence_level' in ci_result


@pytest.mark.skipif(not DATA_SCIENCE_AVAILABLE, reason="refunc data science utilities not available")
class TestScalabilityBenchmarks:
    """Scalability benchmark tests with larger datasets."""
    
    def test_large_dataset_cleaning(self, benchmark, large_dataset):
        """Test cleaning operations on large dataset."""
        
        def clean_large_data():
            # Add some nulls to test cleaning
            data_with_nulls = large_dataset.copy()
            null_indices = np.random.choice(len(data_with_nulls), size=1000, replace=False)
            data_with_nulls.loc[null_indices, 'feature_1'] = np.nan
            
            return cleaning.impute_missing_values(
                data_with_nulls['feature_1'],
                method='median'
            )
        
        result = benchmark.pedantic(clean_large_data, rounds=1, iterations=1)
        assert not result.isna().any()
    
    def test_large_dataset_transformation(self, benchmark, large_dataset):
        """Test transformation operations on large dataset."""
        
        def transform_large_data():
            numeric_cols = ['feature_1', 'feature_2', 'feature_3']
            return transforms.standard_scale(large_dataset[numeric_cols])
        
        result = benchmark.pedantic(transform_large_data, rounds=1, iterations=1)
        assert result.shape == large_dataset[['feature_1', 'feature_2', 'feature_3']].shape
    
    def test_large_dataset_profiling(self, benchmark, large_dataset):
        """Test profiling operations on large dataset."""
        
        def profile_large_data():
            return profiling.generate_statistical_summary(
                large_dataset.select_dtypes(include=[np.number])
            )
        
        result = benchmark.pedantic(profile_large_data, rounds=1, iterations=1)
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) >= 4  # At least 4 numeric features