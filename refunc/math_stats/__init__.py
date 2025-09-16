"""
Mathematical and statistical utilities for ML workflows.

This package provides comprehensive mathematical and statistical tools including:
- Descriptive statistics and hypothesis testing
- Distribution fitting and analysis  
- Numerical optimization algorithms
- Numerical integration and differentiation
- Root finding and interpolation
- Special mathematical functions

Main components:
- StatisticsEngine: Comprehensive statistical analysis
- DistributionAnalyzer: Distribution fitting and comparison
- Optimizer: Multi-method optimization engine
- NumericalIntegrator: Integration methods
- RootFinder: Equation solving utilities

Example:
    Basic statistical analysis:
    
    >>> from refunc.math_stats import describe, test_normality
    >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> stats = describe(data)
    >>> print(stats.summary())
    >>> 
    >>> normality_result = test_normality(data)
    >>> print(normality_result.summary())
    
    Distribution fitting:
    
    >>> from refunc.math_stats import find_best_distribution
    >>> comparison = find_best_distribution(data)
    >>> print(comparison.summary())
    
    Optimization:
    
    >>> from refunc.math_stats import minimize_function
    >>> result = minimize_function(lambda x: x**2 + 2*x + 1, x0=[0])
    >>> print(result.summary)
    
    Numerical integration:
    
    >>> from refunc.math_stats import integrate_function
    >>> result = integrate_function(lambda x: x**2, 0, 1)
    >>> print(result.summary())
"""

# Statistics module
from .statistics import (
    StatisticsEngine,
    StatTestResult,
    DescriptiveStats,
    StatTestType,
    describe,
    test_normality,
    test_correlation,
    compare_groups,
    bootstrap_ci,
    detect_outliers
)

# Distributions module
from .distributions import (
    DistributionAnalyzer,
    DistributionFit,
    DistributionComparison,
    DistributionFamily,
    BayesianDistributionAnalyzer,
    fit_distribution,
    find_best_distribution,
    generate_from_distribution
)

# Optimization module
from .optimization import (
    Optimizer,
    OptimizationResult,
    OptimizationBounds,
    OptimizationConstraint,
    OptimizationMethod,
    ConstraintType,
    ObjectiveFunction,
    minimize_function,
    find_minimum_scalar,
    global_minimize,
    multi_start_minimize
)

# Numerical module
from .numerical import (
    NumericalIntegrator,
    NumericalDifferentiator,
    RootFinder,
    Interpolator,
    SpecialFunctions,
    IntegrationResult,
    RootFindingResult,
    IntegrationMethod,
    InterpolationMethod,
    integrate_function,
    find_function_root,
    create_interpolator,
    numerical_derivative
)

__all__ = [
    # Statistics classes and functions
    'StatisticsEngine',
    'StatTestResult',
    'DescriptiveStats',
    'StatTestType',
    'describe',
    'test_normality',
    'test_correlation',
    'compare_groups',
    'bootstrap_ci',
    'detect_outliers',
    
    # Distribution classes and functions
    'DistributionAnalyzer',
    'DistributionFit',
    'DistributionComparison',
    'DistributionFamily',
    'BayesianDistributionAnalyzer',
    'fit_distribution',
    'find_best_distribution',
    'generate_from_distribution',
    
    # Optimization classes and functions
    'Optimizer',
    'OptimizationResult',
    'OptimizationBounds',
    'OptimizationConstraint',
    'OptimizationMethod',
    'ConstraintType',
    'ObjectiveFunction',
    'minimize_function',
    'find_minimum_scalar',
    'global_minimize',
    'multi_start_minimize',
    
    # Numerical classes and functions
    'NumericalIntegrator',
    'NumericalDifferentiator',
    'RootFinder',
    'Interpolator',
    'SpecialFunctions',
    'IntegrationResult',
    'RootFindingResult',
    'IntegrationMethod',
    'InterpolationMethod',
    'integrate_function',
    'find_function_root',
    'create_interpolator',
    'numerical_derivative'
]

# Version info
__version__ = "0.1.0"
__author__ = "kennedym-ds"
__description__ = "Mathematical and statistical utilities for ML workflows"