"""
Comprehensive tests for the refunc.math_stats module.

This test suite covers:
- Statistical analysis and descriptive statistics
- Hypothesis testing and statistical tests
- Distribution fitting and analysis
- Optimization algorithms and methods
- Numerical integration and differentiation
- Root finding and interpolation
- Special mathematical functions
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional
import warnings

# Import all math_stats components that actually exist
from refunc.math_stats import (
    # Statistics classes and functions
    StatisticsEngine,
    StatTestResult,
    DescriptiveStats,
    StatTestType,
    describe,
    compare_groups,
    bootstrap_ci,
    detect_outliers,
    
    # Distribution classes and functions  
    DistributionAnalyzer,
    DistributionFit,
    DistributionComparison,
    DistributionFamily,
    fit_distribution,
    find_best_distribution,
    
    # Optimization classes and functions
    Optimizer,
    OptimizationResult,
    OptimizationBounds,
    OptimizationConstraint,
    OptimizationMethod,
    ConstraintType,
    minimize_function,
    find_minimum_scalar,
    
    # Numerical classes and functions
    NumericalIntegrator,
    IntegrationResult,
    IntegrationMethod,
    integrate_function,
    numerical_derivative
)

# Import the problematic functions with different names to avoid pytest conflicts
from refunc.math_stats import test_normality as check_normality_fn
from refunc.math_stats import test_correlation as check_correlation_fn


class TestStatistics:
    """Test statistical analysis functionality."""
    
    def test_describe_basic_functionality(self, sample_numpy_arrays):
        """Test basic descriptive statistics calculation."""
        data = sample_numpy_arrays['small']
        stats = describe(data)
        
        assert isinstance(stats, DescriptiveStats)
        assert stats.count == len(data)
        assert stats.mean == pytest.approx(np.mean(data), rel=1e-5)
        assert stats.std == pytest.approx(np.std(data, ddof=1), rel=1e-5)
        assert stats.min == np.min(data)
        assert stats.max == np.max(data)
        
    def test_describe_with_pandas_series(self, sample_dataframe):
        """Test descriptive statistics with pandas Series."""
        series = sample_dataframe['numeric']
        stats = describe(series)
        
        assert isinstance(stats, DescriptiveStats)
        assert stats.count == len(series)
        assert not np.isnan(stats.mean)
        assert not np.isnan(stats.std)
        
    def test_describe_with_missing_values(self):
        """Test descriptive statistics with missing values."""
        data = [1, 2, np.nan, 4, 5]
        stats = describe(data)
        
        assert stats.count == 4  # Excluding NaN
        assert stats.mean == pytest.approx(3.0, rel=1e-5)
        
    def test_describe_empty_data(self):
        """Test descriptive statistics with empty data."""
        with pytest.raises(ValueError):
            describe([])
            
    def test_describe_single_value(self):
        """Test descriptive statistics with single value."""
        stats = describe([5.0])
        assert stats.count == 1
        assert stats.mean == 5.0
        assert stats.std == 0.0
        
    def test_test_normality_shapiro(self, sample_numpy_arrays):
        """Test normality testing with Shapiro-Wilk test."""
        # Use normal data
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 50)
        
        result = check_normality_fn(normal_data, method="shapiro")
        assert isinstance(result, StatTestResult)
        assert result.test_name == "shapiro"
        assert hasattr(result, 'statistic')
        assert hasattr(result, 'p_value')
        
    def test_test_normality_ks(self, sample_numpy_arrays):
        """Test normality testing with Kolmogorov-Smirnov test."""
        data = sample_numpy_arrays['medium'][:, 0]
        result = check_normality_fn(data, method="ks")
        
        assert isinstance(result, StatTestResult)
        assert result.test_name == "ks"
        
    def test_test_normality_invalid_method(self):
        """Test normality testing with invalid method."""
        data = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError):
            check_normality_fn(data, method="invalid_method")
            
    def test_test_correlation_pearson(self, sample_numpy_arrays):
        """Test correlation testing with Pearson method."""
        data = sample_numpy_arrays['medium']
        x, y = data[:, 0], data[:, 1]
        
        result = check_correlation_fn(x, y, method="pearson")
        assert isinstance(result, StatTestResult)
        assert result.test_name == "pearson"
        assert hasattr(result, 'statistic')
        assert hasattr(result, 'p_value')
        
    def test_test_correlation_spearman(self, sample_numpy_arrays):
        """Test correlation testing with Spearman method."""
        data = sample_numpy_arrays['medium']
        x, y = data[:, 0], data[:, 1]
        
        result = check_correlation_fn(x, y, method="spearman")
        assert isinstance(result, StatTestResult)
        assert result.test_name == "spearman"
        
    def test_compare_groups_basic(self, sample_numpy_arrays):
        """Test basic group comparison."""
        data = sample_numpy_arrays['medium']
        group1, group2 = data[:25, 0], data[25:50, 0]
        
        result = compare_groups(group1, group2)
        assert isinstance(result, StatTestResult)
        assert hasattr(result, 'statistic')
        assert hasattr(result, 'p_value')
        
    def test_bootstrap_ci(self, sample_numpy_arrays):
        """Test bootstrap confidence intervals."""
        data = sample_numpy_arrays['small']
        
        ci = bootstrap_ci(data, statistic_func=np.mean)
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] <= np.mean(data) <= ci[1]
        
    def test_bootstrap_ci_custom_statistic(self, sample_numpy_arrays):
        """Test bootstrap confidence intervals with custom statistic."""
        data = sample_numpy_arrays['small']
        
        def custom_stat(x):
            return np.median(x)
        
        ci = bootstrap_ci(data, statistic_func=custom_stat)
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        
    def test_detect_outliers_iqr(self, sample_numpy_arrays):
        """Test outlier detection using IQR method."""
        data = sample_numpy_arrays['small']
        # Add some outliers
        data_with_outliers = np.concatenate([data, [100, -100]])
        
        outliers, indices = detect_outliers(data_with_outliers, method="iqr")
        assert isinstance(outliers, np.ndarray)
        assert isinstance(indices, np.ndarray)
        assert len(outliers) >= 2  # Should detect the added outliers
        
    def test_detect_outliers_zscore(self, sample_numpy_arrays):
        """Test outlier detection using z-score method."""
        data = sample_numpy_arrays['small']
        data_with_outliers = np.concatenate([data, [100, -100]])
        
        outliers, indices = detect_outliers(data_with_outliers, method="zscore", threshold=2.0)
        assert isinstance(outliers, np.ndarray)
        assert isinstance(indices, np.ndarray)
        
    def test_statistics_engine_creation(self):
        """Test StatisticsEngine creation and configuration."""
        engine = StatisticsEngine()
        assert engine is not None
        
        # Test with custom configuration
        engine_custom = StatisticsEngine(confidence_level=0.99)
        assert engine_custom is not None
        assert engine_custom.confidence_level == 0.99


class TestDistributions:
    """Test distribution fitting and analysis functionality."""
    
    def test_fit_distribution_normal(self, sample_numpy_arrays):
        """Test fitting normal distribution."""
        # Generate normal data
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)
        
        fit = fit_distribution(normal_data, distribution='normal')
        assert isinstance(fit, DistributionFit)
        assert fit.distribution_name == 'normal'
        assert 'loc' in fit.parameters or 'mu' in fit.parameters
        assert 'scale' in fit.parameters or 'sigma' in fit.parameters
        
    def test_fit_distribution_exponential(self):
        """Test fitting exponential distribution."""
        np.random.seed(42)
        exp_data = np.random.exponential(2.0, 100)
        
        fit = fit_distribution(exp_data, distribution='exponential')
        assert isinstance(fit, DistributionFit)
        assert fit.distribution_name == 'exponential'
        
    def test_find_best_distribution(self, sample_numpy_arrays):
        """Test finding best distribution fit."""
        data = sample_numpy_arrays['small']
        
        comparison = find_best_distribution(data, distributions=['normal', 'exponential', 'uniform'])
        assert isinstance(comparison, DistributionComparison)
        assert len(comparison.fits) == 3
        assert comparison.best_fit is not None
        
    def test_distribution_analyzer_creation(self):
        """Test DistributionAnalyzer creation."""
        analyzer = DistributionAnalyzer()
        assert analyzer is not None
        
    def test_distribution_families(self):
        """Test distribution family enumeration."""
        assert DistributionFamily.CONTINUOUS
        assert DistributionFamily.DISCRETE


class TestOptimization:
    """Test optimization algorithms and methods."""
    
    def test_minimize_function_simple_quadratic(self):
        """Test minimizing simple quadratic function."""
        def quadratic(x):
            return (x[0] - 2)**2 + (x[1] - 3)**2
        
        result = minimize_function(quadratic, x0=[0, 0])
        assert isinstance(result, OptimizationResult)
        assert result.success
        assert result.x[0] == pytest.approx(2.0, abs=1e-3)
        assert result.x[1] == pytest.approx(3.0, abs=1e-3)
        
    def test_minimize_function_with_bounds(self):
        """Test minimization with bounds."""
        def objective(x):
            return x[0]**2 + x[1]**2
        
        bounds = OptimizationBounds(lower=[-1, -1], upper=[1, 1])
        result = minimize_function(objective, x0=[0.5, 0.5], bounds=bounds)
        
        assert isinstance(result, OptimizationResult)
        assert result.success
        assert -1 <= result.x[0] <= 1
        assert -1 <= result.x[1] <= 1
        
    def test_find_minimum_scalar(self):
        """Test scalar function minimization."""
        def scalar_func(x):
            return (x - 3)**2 + 5
        
        result = find_minimum_scalar(scalar_func, bounds=(-10, 10))
        assert isinstance(result, OptimizationResult)
        assert result.x == pytest.approx(3.0, abs=1e-3)
        assert result.fun == pytest.approx(5.0, abs=1e-3)
        
    def test_optimizer_creation(self):
        """Test Optimizer class creation."""
        optimizer = Optimizer()
        assert optimizer is not None
        
    def test_optimization_methods_enum(self):
        """Test optimization method enumeration."""
        assert OptimizationMethod.NELDER_MEAD
        assert OptimizationMethod.L_BFGS_B  # Correct enum value
        
    def test_constraint_types_enum(self):
        """Test constraint type enumeration."""
        assert ConstraintType.EQUALITY
        assert ConstraintType.INEQUALITY


class TestNumerical:
    """Test numerical analysis functionality."""
    
    def test_integrate_function_simple(self):
        """Test numerical integration of simple function."""
        def simple_func(x):
            return x**2
        
        result = integrate_function(simple_func, 0, 1)
        assert isinstance(result, IntegrationResult)
        assert result.value == pytest.approx(1/3, abs=1e-3)
        
    def test_integrate_function_complex(self):
        """Test numerical integration of complex function."""
        def complex_func(x):
            return np.sin(x) * np.exp(-x)
        
        result = integrate_function(complex_func, 0, np.pi)
        assert isinstance(result, IntegrationResult)
        assert result.error is not None or hasattr(result, 'error')
        
    def test_integrate_function_with_method(self):
        """Test numerical integration with specific method."""
        def func(x):
            return np.cos(x)
        
        result = integrate_function(func, 0, np.pi/2, method="simpson")
        assert isinstance(result, IntegrationResult)
        assert result.method == "simpson"
        
    def test_numerical_derivative(self):
        """Test numerical differentiation."""
        def func(x):
            return x**3
        
        # Derivative should be 3*x^2, so at x=2: derivative = 12
        derivative = numerical_derivative(func, 2.0)
        assert derivative == pytest.approx(12.0, abs=1e-3)
        
    def test_numerical_derivative_array(self):
        """Test numerical differentiation with array input."""
        x = np.linspace(0, 2*np.pi, 100)
        y = np.sin(x)
        
        # This would need a specific implementation for array derivatives
        # For now, just test that we can compute derivatives at individual points
        x_point = x[50]  # Middle point
        def sin_func(t):
            return np.sin(t)
        
        dy_dx = numerical_derivative(sin_func, x_point)
        expected = np.cos(x_point)  # Derivative of sin(x) is cos(x)
        assert dy_dx == pytest.approx(expected, abs=1e-3)
        
    def test_numerical_integrator_creation(self):
        """Test NumericalIntegrator creation."""
        integrator = NumericalIntegrator()
        assert integrator is not None


class TestMathStatsEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_statistics_with_inf_values(self):
        """Test statistics calculation with infinite values."""
        data = [1, 2, np.inf, 4, 5]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stats = describe(data)
            # Should handle infinite values gracefully
            assert stats is not None
            
    def test_empty_data_handling(self):
        """Test handling of empty data arrays."""
        with pytest.raises(ValueError):
            describe([])
            
    def test_single_point_data(self):
        """Test handling of single data point."""
        single_point = [5.0]
        
        stats = describe(single_point)
        assert stats.count == 1
        assert stats.std == 0.0


class TestMathStatsIntegration:
    """Test integration between different math_stats components."""
    
    def test_statistics_and_distribution_workflow(self, sample_numpy_arrays):
        """Test workflow combining statistics and distribution analysis."""
        data = sample_numpy_arrays['medium'][:, 0]
        
        # First, get descriptive statistics
        stats = describe(data)
        assert stats is not None
        
        # Test normality
        normality_result = check_normality_fn(data)
        assert isinstance(normality_result, StatTestResult)
        
        # Fit distributions
        comparison = find_best_distribution(data, distributions=['normal', 'uniform'])
        assert isinstance(comparison, DistributionComparison)
        
    def test_optimization_and_numerical_workflow(self):
        """Test workflow combining optimization and numerical methods."""
        # Define a function to minimize
        def objective(x):
            return (x[0] - 1)**2 + (x[1] - 2)**2
        
        # Find minimum
        opt_result = minimize_function(objective, x0=[0, 0])
        assert opt_result.success
        
        # Verify using numerical derivative at optimum
        def grad_check(x):
            return objective([x, opt_result.x[1]])
        
        derivative = numerical_derivative(grad_check, opt_result.x[0])
        assert abs(derivative) < 1e-2  # Should be near zero at optimum
        
    def test_statistical_hypothesis_testing_workflow(self, sample_numpy_arrays):
        """Test complete statistical hypothesis testing workflow."""
        data = sample_numpy_arrays['medium']
        group1, group2 = data[:25, 0], data[25:50, 0]
        
        # Test normality of both groups
        norm1 = check_normality_fn(group1)
        norm2 = check_normality_fn(group2)
        
        # Compare groups
        comparison = compare_groups(group1, group2)
        assert isinstance(comparison, StatTestResult)
        
        # Calculate confidence intervals
        ci1 = bootstrap_ci(group1, statistic_func=np.mean)
        ci2 = bootstrap_ci(group2, statistic_func=np.mean)
        
        assert isinstance(ci1, tuple)
        assert isinstance(ci2, tuple)


class TestMathStatsPerformance:
    """Test performance characteristics of math_stats functions."""
    
    @pytest.mark.slow
    def test_large_dataset_statistics(self):
        """Test statistics calculation on large dataset."""
        large_data = np.random.normal(0, 1, 100000)
        
        stats = describe(large_data)
        assert stats is not None
        assert abs(stats.mean) < 0.1  # Should be close to 0
        assert abs(stats.std - 1.0) < 0.1  # Should be close to 1
        
    @pytest.mark.slow
    def test_complex_optimization_performance(self):
        """Test optimization performance on complex function."""
        def complex_objective(x):
            # Multi-modal function
            return np.sum([np.sin(xi) * np.cos(xi**2) for xi in x]) + 0.1 * np.sum(x**2)
        
        result = minimize_function(complex_objective, x0=[1, 1, 1])
        assert isinstance(result, OptimizationResult)
        
    @pytest.mark.slow
    def test_high_precision_integration(self):
        """Test high-precision numerical integration."""
        def precise_func(x):
            return np.exp(-x**2)  # Gaussian-like
        
        result = integrate_function(precise_func, -3, 3)
        assert isinstance(result, IntegrationResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])