#!/usr/bin/env python3
"""
Statistical Analysis Examples - Refunc Math/Stats

This example demonstrates comprehensive statistical analysis capabilities
including descriptive statistics, hypothesis testing, distribution analysis,
and advanced mathematical computations for data science workflows.

Key Features Demonstrated:
- Descriptive statistics and data profiling
- Hypothesis testing and statistical inference
- Distribution fitting and analysis
- Correlation and regression analysis
- Numerical optimization and integration
- Statistical modeling and validation
"""

import os
import sys
import time
import random
import math
from typing import List, Dict, Any, Optional, Tuple, Union

# Handle missing dependencies gracefully
try:
    from refunc.math_stats import (
        describe, test_normality, test_correlation, compare_groups,
        bootstrap_ci, detect_outliers, find_best_distribution,
        minimize_function, integrate_function, StatisticsEngine,
        DistributionAnalyzer, Optimizer
    )
    REFUNC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Refunc not fully installed: {e}")
    print("This example requires: pip install refunc")
    print("Showing example code structure...\n")
    REFUNC_AVAILABLE = False


def generate_sample_data():
    """Generate sample datasets for statistical analysis."""
    random.seed(42)
    
    datasets = {
        'normal_data': [random.gauss(50, 15) for _ in range(200)],
        'skewed_data': [random.expovariate(0.1) for _ in range(200)],
        'bimodal_data': (
            [random.gauss(30, 5) for _ in range(100)] + 
            [random.gauss(70, 8) for _ in range(100)]
        ),
        'uniform_data': [random.uniform(0, 100) for _ in range(200)],
        'experiment_group_a': [random.gauss(75, 12) for _ in range(50)],
        'experiment_group_b': [random.gauss(82, 10) for _ in range(48)],
        'time_series': [
            50 + 10 * math.sin(i * 0.1) + random.gauss(0, 3) 
            for i in range(100)
        ],
        'correlation_x': [i + random.gauss(0, 5) for i in range(50)],
        'correlation_y': [2 * i + 10 + random.gauss(0, 8) for i in range(50)]
    }
    
    return datasets


def descriptive_statistics_examples():
    """Demonstrate descriptive statistics and data profiling."""
    print("📊 Descriptive Statistics")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Descriptive statistics examples:
from refunc.math_stats import describe, StatisticsEngine

# Basic descriptive statistics
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
stats = describe(data)
print(f"Mean: {stats.mean}")
print(f"Median: {stats.median}")
print(f"Standard deviation: {stats.std}")
print(f"Skewness: {stats.skewness}")
print(f"Kurtosis: {stats.kurtosis}")

# Comprehensive statistics engine
engine = StatisticsEngine()
detailed_stats = engine.analyze(data)
print(detailed_stats.summary())

# Multiple dataset comparison
datasets = {'group_a': data_a, 'group_b': data_b}
comparison = engine.compare_datasets(datasets)
        """)
        return
    
    print("📈 Testing descriptive statistics:")
    
    datasets = generate_sample_data()
    
    # Analyze different types of distributions
    analysis_datasets = {
        'Normal Distribution': datasets['normal_data'],
        'Skewed Distribution': datasets['skewed_data'][:50],  # Truncate for display
        'Bimodal Distribution': datasets['bimodal_data'],
        'Uniform Distribution': datasets['uniform_data']
    }
    
    print("   📊 Dataset characteristics:")
    
    for name, data in analysis_datasets.items():
        print(f"\n   {name}:")
        
        # Calculate basic statistics manually
        n = len(data)
        mean_val = sum(data) / n
        
        # Variance and standard deviation
        variance = sum((x - mean_val) ** 2 for x in data) / (n - 1)
        std_dev = math.sqrt(variance)
        
        # Median
        sorted_data = sorted(data)
        if n % 2 == 0:
            median_val = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
        else:
            median_val = sorted_data[n//2]
        
        # Quartiles
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = sorted_data[q1_idx]
        q3 = sorted_data[q3_idx]
        iqr = q3 - q1
        
        # Skewness (simplified Pearson's moment coefficient)
        skewness = sum(((x - mean_val) / std_dev) ** 3 for x in data) / n
        
        # Range
        data_range = max(data) - min(data)
        
        print(f"     Count: {n}")
        print(f"     Mean: {mean_val:.2f}")
        print(f"     Median: {median_val:.2f}")
        print(f"     Std Dev: {std_dev:.2f}")
        print(f"     Range: {data_range:.2f}")
        print(f"     IQR: {iqr:.2f}")
        print(f"     Skewness: {skewness:.3f}")
        
        # Quartile summary
        print(f"     Quartiles: Q1={q1:.1f}, Q2={median_val:.1f}, Q3={q3:.1f}")
        
        # Distribution shape assessment
        if abs(skewness) < 0.5:
            shape = "approximately symmetric"
        elif skewness > 0.5:
            shape = "right-skewed (positive skew)"
        else:
            shape = "left-skewed (negative skew)"
        
        print(f"     Shape: {shape}")
    
    # Comparative analysis
    print(f"\n   🔍 Comparative Analysis:")
    
    # Compare means
    normal_mean = sum(datasets['normal_data']) / len(datasets['normal_data'])
    skewed_mean = sum(datasets['skewed_data']) / len(datasets['skewed_data'])
    
    print(f"     Normal vs Skewed mean difference: {abs(normal_mean - skewed_mean):.2f}")
    
    # Compare variability
    normal_std = math.sqrt(sum((x - normal_mean) ** 2 for x in datasets['normal_data']) / (len(datasets['normal_data']) - 1))
    skewed_std = math.sqrt(sum((x - skewed_mean) ** 2 for x in datasets['skewed_data']) / (len(datasets['skewed_data']) - 1))
    
    print(f"     Variability ratio (normal/skewed): {normal_std / skewed_std:.2f}")
    
    # Outlier detection example
    print(f"\n   🎯 Outlier Detection:")
    
    for name, data in [('Normal', datasets['normal_data']), ('Skewed', datasets['skewed_data'][:100])]:
        sorted_data = sorted(data)
        n = len(data)
        q1 = sorted_data[n//4]
        q3 = sorted_data[3*n//4]
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        outlier_pct = (len(outliers) / len(data)) * 100
        
        print(f"     {name} data: {len(outliers)} outliers ({outlier_pct:.1f}%)")
        if outliers:
            print(f"       Range: [{min(outliers):.1f}, {max(outliers):.1f}]")
            print(f"       Bounds: [{lower_bound:.1f}, {upper_bound:.1f}]")


def hypothesis_testing_examples():
    """Demonstrate hypothesis testing and statistical inference."""
    print("\n🧪 Hypothesis Testing")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Hypothesis testing examples:
from refunc.math_stats import test_normality, compare_groups, bootstrap_ci

# Test for normality
data = [random.gauss(50, 10) for _ in range(100)]
normality_result = test_normality(data)
print(f"Is normal: {normality_result.is_normal}")
print(f"P-value: {normality_result.p_value}")

# Compare two groups (t-test)
group_a = [random.gauss(50, 10) for _ in range(30)]
group_b = [random.gauss(55, 12) for _ in range(32)]
comparison_result = compare_groups(group_a, group_b)
print(f"Significant difference: {comparison_result.is_significant}")
print(f"Effect size: {comparison_result.effect_size}")

# Bootstrap confidence intervals
ci_result = bootstrap_ci(data, statistic='mean', n_bootstrap=1000)
print(f"95% CI: [{ci_result.lower:.2f}, {ci_result.upper:.2f}]")
        """)
        return
    
    print("🔬 Testing statistical hypotheses:")
    
    datasets = generate_sample_data()
    
    # Test 1: Normality Testing
    print("   📊 Normality Testing:")
    
    test_datasets = {
        'Normal Data': datasets['normal_data'],
        'Skewed Data': datasets['skewed_data'][:100],
        'Uniform Data': datasets['uniform_data']
    }
    
    for name, data in test_datasets.items():
        # Simplified normality test using skewness and kurtosis
        n = len(data)
        mean_val = sum(data) / n
        std_dev = math.sqrt(sum((x - mean_val) ** 2 for x in data) / (n - 1))
        
        # Skewness test
        skewness = sum(((x - mean_val) / std_dev) ** 3 for x in data) / n
        
        # Kurtosis test
        kurtosis = sum(((x - mean_val) / std_dev) ** 4 for x in data) / n - 3
        
        # Simple normality assessment (rules of thumb)
        is_normal_skew = abs(skewness) < 1.0
        is_normal_kurt = abs(kurtosis) < 1.0
        is_normal = is_normal_skew and is_normal_kurt
        
        print(f"     {name}:")
        print(f"       Skewness: {skewness:.3f} ({'✓' if is_normal_skew else '✗'} normal)")
        print(f"       Kurtosis: {kurtosis:.3f} ({'✓' if is_normal_kurt else '✗'} normal)")
        print(f"       Overall: {'✓ Likely normal' if is_normal else '✗ Non-normal'}")
    
    # Test 2: Two-Sample Comparison
    print(f"\n   ⚖️ Two-Sample Comparison:")
    
    group_a = datasets['experiment_group_a']
    group_b = datasets['experiment_group_b']
    
    # Calculate sample statistics
    n_a, n_b = len(group_a), len(group_b)
    mean_a = sum(group_a) / n_a
    mean_b = sum(group_b) / n_b
    
    # Sample standard deviations
    var_a = sum((x - mean_a) ** 2 for x in group_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in group_b) / (n_b - 1)
    std_a = math.sqrt(var_a)
    std_b = math.sqrt(var_b)
    
    # Pooled standard error for independent t-test
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_se = math.sqrt(pooled_var * (1/n_a + 1/n_b))
    
    # T-statistic
    t_stat = (mean_a - mean_b) / pooled_se if pooled_se > 0 else 0
    
    # Degrees of freedom
    df = n_a + n_b - 2
    
    # Effect size (Cohen's d)
    cohens_d = (mean_a - mean_b) / math.sqrt(pooled_var) if pooled_var > 0 else 0
    
    print(f"     Group A: n={n_a}, mean={mean_a:.2f}, std={std_a:.2f}")
    print(f"     Group B: n={n_b}, mean={mean_b:.2f}, std={std_b:.2f}")
    print(f"     Difference: {mean_a - mean_b:.2f}")
    print(f"     T-statistic: {t_stat:.3f}")
    print(f"     Degrees of freedom: {df}")
    print(f"     Effect size (Cohen's d): {cohens_d:.3f}")
    
    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect_interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interpretation = "small"
    elif abs(cohens_d) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    print(f"     Effect size interpretation: {effect_interpretation}")
    
    # Simplified significance test (|t| > 2 approximation for p < 0.05)
    is_significant = abs(t_stat) > 2.0
    print(f"     Statistically significant: {'✓ Yes' if is_significant else '✗ No'} (α = 0.05)")
    
    # Test 3: Confidence Intervals (Bootstrap simulation)
    print(f"\n   🎯 Confidence Intervals (Bootstrap):")
    
    # Simple bootstrap for mean
    bootstrap_means = []
    n_bootstrap = 1000
    
    for _ in range(n_bootstrap):
        # Bootstrap sample with replacement
        bootstrap_sample = [random.choice(group_a) for _ in range(len(group_a))]
        bootstrap_mean = sum(bootstrap_sample) / len(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)
    
    # Calculate confidence interval (2.5th and 97.5th percentiles)
    bootstrap_means.sort()
    ci_lower = bootstrap_means[int(0.025 * n_bootstrap)]
    ci_upper = bootstrap_means[int(0.975 * n_bootstrap)]
    
    print(f"     Original mean: {mean_a:.2f}")
    print(f"     Bootstrap samples: {n_bootstrap}")
    print(f"     95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"     CI width: {ci_upper - ci_lower:.2f}")
    
    # Margin of error
    margin_of_error = (ci_upper - ci_lower) / 2
    print(f"     Margin of error: ±{margin_of_error:.2f}")


def correlation_regression_analysis():
    """Demonstrate correlation and regression analysis."""
    print("\n📈 Correlation and Regression")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Correlation and regression examples:
from refunc.math_stats import test_correlation, linear_regression

# Correlation analysis
x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_data = [2.1, 4.0, 5.8, 8.2, 9.9, 12.1, 13.8, 16.2, 18.0, 19.9]

correlation_result = test_correlation(x_data, y_data)
print(f"Correlation coefficient: {correlation_result.correlation:.3f}")
print(f"P-value: {correlation_result.p_value:.3f}")
print(f"Significant: {correlation_result.is_significant}")

# Linear regression
regression_result = fit_linear_regression(x_data, y_data)
print(f"Slope: {regression_result.slope:.3f}")
print(f"Intercept: {regression_result.intercept:.3f}")
print(f"R-squared: {regression_result.r_squared:.3f}")
        """)
        return
    
    print("🔗 Testing correlation and regression:")
    
    datasets = generate_sample_data()
    
    # Correlation Analysis
    print("   📊 Correlation Analysis:")
    
    x_data = datasets['correlation_x']
    y_data = datasets['correlation_y']
    
    # Calculate Pearson correlation coefficient
    n = len(x_data)
    mean_x = sum(x_data) / n
    mean_y = sum(y_data) / n
    
    # Calculate correlation components
    sum_xy = sum((x_data[i] - mean_x) * (y_data[i] - mean_y) for i in range(n))
    sum_x2 = sum((x - mean_x) ** 2 for x in x_data)
    sum_y2 = sum((y - mean_y) ** 2 for y in y_data)
    
    # Pearson correlation coefficient
    correlation = sum_xy / math.sqrt(sum_x2 * sum_y2) if (sum_x2 * sum_y2) > 0 else 0
    
    print(f"     Sample size: {n}")
    print(f"     Pearson correlation (r): {correlation:.3f}")
    
    # Correlation strength interpretation
    abs_corr = abs(correlation)
    if abs_corr < 0.1:
        strength = "negligible"
    elif abs_corr < 0.3:
        strength = "weak"
    elif abs_corr < 0.5:
        strength = "moderate"
    elif abs_corr < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    
    direction = "positive" if correlation > 0 else "negative"
    print(f"     Correlation strength: {strength} {direction}")
    
    # Coefficient of determination
    r_squared = correlation ** 2
    print(f"     R-squared: {r_squared:.3f} ({r_squared*100:.1f}% variance explained)")
    
    # Linear Regression Analysis
    print(f"\n   📈 Linear Regression Analysis:")
    
    # Calculate regression coefficients
    # Slope (beta_1)
    slope = sum_xy / sum_x2 if sum_x2 > 0 else 0
    
    # Intercept (beta_0)
    intercept = mean_y - slope * mean_x
    
    print(f"     Regression equation: y = {intercept:.2f} + {slope:.2f}x")
    print(f"     Slope: {slope:.3f}")
    print(f"     Intercept: {intercept:.3f}")
    
    # Predictions and residuals
    predictions = [intercept + slope * x for x in x_data]
    residuals = [y_data[i] - predictions[i] for i in range(n)]
    
    # Sum of squares
    ss_res = sum(r ** 2 for r in residuals)  # Residual sum of squares
    ss_tot = sum((y - mean_y) ** 2 for y in y_data)  # Total sum of squares
    
    # R-squared (coefficient of determination)
    r_squared_reg = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f"     R-squared: {r_squared_reg:.3f}")
    print(f"     Residual sum of squares: {ss_res:.2f}")
    
    # Standard error of estimate
    se_estimate = math.sqrt(ss_res / (n - 2)) if n > 2 else 0
    print(f"     Standard error of estimate: {se_estimate:.3f}")
    
    # Residual analysis
    mean_residual = sum(residuals) / len(residuals)
    residual_std = math.sqrt(sum((r - mean_residual) ** 2 for r in residuals) / (len(residuals) - 1))
    
    print(f"\n   🔍 Residual Analysis:")
    print(f"     Mean residual: {mean_residual:.3f} (should be ≈ 0)")
    print(f"     Residual std dev: {residual_std:.3f}")
    print(f"     Min residual: {min(residuals):.2f}")
    print(f"     Max residual: {max(residuals):.2f}")
    
    # Check for outliers in residuals (|residual| > 2 * se)
    outlier_threshold = 2 * se_estimate
    outliers = [i for i, r in enumerate(residuals) if abs(r) > outlier_threshold]
    
    print(f"     Outliers (|residual| > 2SE): {len(outliers)}")
    if outliers:
        print(f"       Outlier indices: {outliers[:5]}")  # Show first 5
    
    # Multiple correlation scenarios
    print(f"\n   🔄 Multiple Correlation Scenarios:")
    
    # Test different correlation patterns
    correlation_tests = [
        ("Strong Positive", [i for i in range(20)], [2*i + random.gauss(0, 1) for i in range(20)]),
        ("Weak Negative", [i for i in range(20)], [-0.3*i + 10 + random.gauss(0, 3) for i in range(20)]),
        ("No Correlation", [i for i in range(20)], [random.gauss(10, 3) for _ in range(20)])
    ]
    
    for name, x_test, y_test in correlation_tests:
        n_test = len(x_test)
        mean_x_test = sum(x_test) / n_test
        mean_y_test = sum(y_test) / n_test
        
        sum_xy_test = sum((x_test[i] - mean_x_test) * (y_test[i] - mean_y_test) for i in range(n_test))
        sum_x2_test = sum((x - mean_x_test) ** 2 for x in x_test)
        sum_y2_test = sum((y - mean_y_test) ** 2 for y in y_test)
        
        corr_test = sum_xy_test / math.sqrt(sum_x2_test * sum_y2_test) if (sum_x2_test * sum_y2_test) > 0 else 0
        
        print(f"     {name}: r = {corr_test:.3f}")


def distribution_analysis_examples():
    """Demonstrate distribution fitting and analysis."""
    print("\n🎲 Distribution Analysis")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Distribution analysis examples:
from refunc.math_stats import find_best_distribution, fit_distribution

# Find best fitting distribution
data = [random.gauss(50, 10) for _ in range(200)]
distribution_result = find_best_distribution(data)
print(f"Best distribution: {distribution_result.best_distribution}")
print(f"Parameters: {distribution_result.parameters}")
print(f"Goodness of fit: {distribution_result.goodness_of_fit}")

# Fit specific distribution
normal_fit = fit_distribution(data, 'normal')
print(f"Normal distribution: μ={normal_fit.mu:.2f}, σ={normal_fit.sigma:.2f}")

# Generate samples from fitted distribution
generated_samples = distribution_result.generate_samples(100)
        """)
        return
    
    print("📊 Testing distribution analysis:")
    
    datasets = generate_sample_data()
    
    # Analyze different distribution types
    distribution_datasets = {
        'Normal': datasets['normal_data'],
        'Exponential': datasets['skewed_data'][:100],
        'Bimodal': datasets['bimodal_data'],
        'Uniform': datasets['uniform_data']
    }
    
    print("   🔍 Distribution Fitting Analysis:")
    
    for name, data in distribution_datasets.items():
        print(f"\n     {name} Distribution:")
        
        n = len(data)
        mean_val = sum(data) / n
        variance = sum((x - mean_val) ** 2 for x in data) / (n - 1)
        std_dev = math.sqrt(variance)
        
        # Basic distribution parameters
        print(f"       Sample size: {n}")
        print(f"       Mean: {mean_val:.2f}")
        print(f"       Std Dev: {std_dev:.2f}")
        
        # Test for normality (simplified)
        skewness = sum(((x - mean_val) / std_dev) ** 3 for x in data) / n
        kurtosis = sum(((x - mean_val) / std_dev) ** 4 for x in data) / n - 3
        
        print(f"       Skewness: {skewness:.3f}")
        print(f"       Kurtosis: {kurtosis:.3f}")
        
        # Distribution shape assessment
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            likely_distribution = "Normal"
            fit_quality = "Good"
        elif skewness > 1.0:
            likely_distribution = "Right-skewed (Exponential/Gamma)"
            fit_quality = "Moderate"
        elif abs(kurtosis) > 2.0:
            likely_distribution = "Heavy-tailed or Multimodal"
            fit_quality = "Poor normal fit"
        else:
            likely_distribution = "Unknown"
            fit_quality = "Requires further analysis"
        
        print(f"       Likely distribution: {likely_distribution}")
        print(f"       Normal fit quality: {fit_quality}")
        
        # Fit normal distribution parameters
        if name != 'Uniform':  # Skip uniform for normal fitting
            # Maximum likelihood estimates for normal distribution
            mle_mu = mean_val
            mle_sigma = std_dev
            
            print(f"       Normal fit: N({mle_mu:.2f}, {mle_sigma:.2f}²)")
            
            # Calculate log-likelihood for normal distribution
            log_likelihood = -n/2 * math.log(2 * math.pi) - n * math.log(mle_sigma) - sum((x - mle_mu)**2 for x in data) / (2 * mle_sigma**2)
            
            # AIC (Akaike Information Criterion)
            aic = 2 * 2 - 2 * log_likelihood  # 2 parameters for normal distribution
            
            print(f"       Log-likelihood: {log_likelihood:.2f}")
            print(f"       AIC: {aic:.2f}")
    
    # Distribution comparison
    print(f"\n   ⚖️ Distribution Comparison:")
    
    # Compare normal vs exponential data
    normal_data = datasets['normal_data'][:100]
    exponential_data = datasets['skewed_data'][:100]
    
    # Kolmogorov-Smirnov test simulation (simplified)
    # Sort both datasets
    normal_sorted = sorted(normal_data)
    exp_sorted = sorted(exponential_data)
    
    # Create empirical CDFs
    n_normal = len(normal_sorted)
    n_exp = len(exp_sorted)
    
    # Find maximum difference in empirical CDFs (simplified)
    combined_values = sorted(set(normal_sorted + exp_sorted))
    max_diff = 0
    
    for value in combined_values[:20]:  # Check first 20 values for efficiency
        # Empirical CDF for normal data
        cdf_normal = sum(1 for x in normal_sorted if x <= value) / n_normal
        
        # Empirical CDF for exponential data
        cdf_exp = sum(1 for x in exp_sorted if x <= value) / n_exp
        
        diff = abs(cdf_normal - cdf_exp)
        if diff > max_diff:
            max_diff = diff
    
    print(f"     Normal vs Exponential:")
    print(f"       Max CDF difference: {max_diff:.3f}")
    print(f"       Distributions {'significantly different' if max_diff > 0.2 else 'similar'}")
    
    # Quantile-Quantile comparison
    print(f"\n   📊 Quantile Analysis:")
    
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    for dist_name, data in [('Normal', normal_data), ('Exponential', exponential_data)]:
        print(f"     {dist_name} quantiles:")
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        for q in quantiles:
            index = int(q * (n - 1))
            quantile_value = sorted_data[index]
            print(f"       Q{q*100:.0f}: {quantile_value:.2f}")


def optimization_integration_examples():
    """Demonstrate numerical optimization and integration."""
    print("\n🔢 Numerical Methods")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Numerical methods examples:
from refunc.math_stats import minimize_function, integrate_function

# Function optimization
def objective_function(x):
    return x[0]**2 + x[1]**2 + 2*x[0]*x[1]

result = minimize_function(objective_function, x0=[1.0, 1.0])
print(f"Minimum found at: {result.x}")
print(f"Minimum value: {result.fun}")

# Numerical integration
def integrand(x):
    return x**2 + 2*x + 1

integral_result = integrate_function(integrand, 0, 2)
print(f"Integral value: {integral_result.value}")
print(f"Error estimate: {integral_result.error}")

# Root finding
from refunc.math_stats import find_function_root

def equation(x):
    return x**3 - 2*x - 5

root = find_function_root(equation, x0=2.0)
print(f"Root found at: {root.x}")
        """)
        return
    
    print("🧮 Testing numerical methods:")
    
    # Optimization Examples
    print("   🎯 Function Optimization:")
    
    # Example 1: Simple quadratic function
    def quadratic(x):
        """Simple quadratic function: f(x) = (x-3)² + 5"""
        return (x - 3)**2 + 5
    
    # Golden section search (simplified)
    def golden_section_search(f, a, b, tol=1e-5):
        """Find minimum of unimodal function using golden section search."""
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        
        # Initialize points
        x1 = b - (b - a) / phi
        x2 = a + (b - a) / phi
        f1 = f(x1)
        f2 = f(x2)
        
        iterations = 0
        while abs(b - a) > tol and iterations < 100:
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - (b - a) / phi
                f1 = f(x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + (b - a) / phi
                f2 = f(x2)
            
            iterations += 1
        
        x_min = (a + b) / 2
        return {'x': x_min, 'fun': f(x_min), 'iterations': iterations}
    
    # Test optimization
    result = golden_section_search(quadratic, 0, 6)
    print(f"     Quadratic minimum:")
    print(f"       x* = {result['x']:.6f} (expected: 3.0)")
    print(f"       f(x*) = {result['fun']:.6f} (expected: 5.0)")
    print(f"       Iterations: {result['iterations']}")
    
    # Example 2: Statistical optimization (maximum likelihood)
    datasets = generate_sample_data()
    sample_data = datasets['normal_data'][:50]
    
    def negative_log_likelihood(params):
        """Negative log-likelihood for normal distribution."""
        mu, sigma = params
        if sigma <= 0:
            return float('inf')
        
        n = len(sample_data)
        log_likelihood = -n/2 * math.log(2 * math.pi) - n * math.log(sigma)
        log_likelihood -= sum((x - mu)**2 for x in sample_data) / (2 * sigma**2)
        return -log_likelihood
    
    # Simple grid search for MLE
    best_mu = 0
    best_sigma = 1
    best_nll = float('inf')
    
    # Grid search around sample statistics
    sample_mean = sum(sample_data) / len(sample_data)
    sample_std = math.sqrt(sum((x - sample_mean)**2 for x in sample_data) / (len(sample_data) - 1))
    
    for mu in [sample_mean + i * 0.1 for i in range(-10, 11)]:
        for sigma in [sample_std + i * 0.1 for i in range(-5, 6) if sample_std + i * 0.1 > 0]:
            nll = negative_log_likelihood([mu, sigma])
            if nll < best_nll:
                best_nll = nll
                best_mu = mu
                best_sigma = sigma
    
    print(f"\n     Maximum Likelihood Estimation:")
    print(f"       Sample mean: {sample_mean:.3f}")
    print(f"       Sample std: {sample_std:.3f}")
    print(f"       MLE μ: {best_mu:.3f}")
    print(f"       MLE σ: {best_sigma:.3f}")
    print(f"       Log-likelihood: {-best_nll:.2f}")
    
    # Numerical Integration Examples
    print(f"\n   ∫ Numerical Integration:")
    
    # Example 1: Simple polynomial
    def polynomial(x):
        """f(x) = x² + 2x + 1"""
        return x**2 + 2*x + 1
    
    # Trapezoidal rule
    def trapezoidal_rule(f, a, b, n=1000):
        """Numerical integration using trapezoidal rule."""
        h = (b - a) / n
        result = 0.5 * (f(a) + f(b))
        
        for i in range(1, n):
            result += f(a + i * h)
        
        return result * h
    
    # Test integration
    integral_approx = trapezoidal_rule(polynomial, 0, 2)
    
    # Analytical solution: ∫(x² + 2x + 1)dx from 0 to 2 = [x³/3 + x² + x] from 0 to 2
    integral_exact = (8/3 + 4 + 2) - (0)  # = 8/3 + 6 = 26/3 ≈ 8.667
    
    print(f"     Polynomial integration ∫₀² (x² + 2x + 1) dx:")
    print(f"       Numerical result: {integral_approx:.6f}")
    print(f"       Analytical result: {integral_exact:.6f}")
    print(f"       Error: {abs(integral_approx - integral_exact):.6f}")
    
    # Example 2: Statistical application (probability calculation)
    def standard_normal_pdf(x):
        """Standard normal probability density function."""
        return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x**2)
    
    # Calculate P(-1 < Z < 1) for standard normal
    prob_approx = trapezoidal_rule(standard_normal_pdf, -1, 1, 1000)
    prob_theoretical = 0.6827  # Known value for standard normal
    
    print(f"\n     Standard Normal P(-1 < Z < 1):")
    print(f"       Numerical result: {prob_approx:.4f}")
    print(f"       Theoretical result: {prob_theoretical:.4f}")
    print(f"       Error: {abs(prob_approx - prob_theoretical):.4f}")
    
    # Root Finding Example
    print(f"\n   🔍 Root Finding:")
    
    def cubic_equation(x):
        """f(x) = x³ - 2x - 5"""
        return x**3 - 2*x - 5
    
    # Newton-Raphson method
    def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
        """Find root using Newton-Raphson method."""
        x = x0
        
        for i in range(max_iter):
            fx = f(x)
            if abs(fx) < tol:
                return {'x': x, 'fx': fx, 'iterations': i+1}
            
            dfx = df(x)
            if abs(dfx) < 1e-12:
                break
            
            x = x - fx / dfx
        
        return {'x': x, 'fx': f(x), 'iterations': max_iter}
    
    def cubic_derivative(x):
        """f'(x) = 3x² - 2"""
        return 3*x**2 - 2
    
    # Find root starting from x = 2
    root_result = newton_raphson(cubic_equation, cubic_derivative, 2.0)
    
    print(f"     Cubic equation x³ - 2x - 5 = 0:")
    print(f"       Root: x = {root_result['x']:.6f}")
    print(f"       f(x) = {root_result['fx']:.6f}")
    print(f"       Iterations: {root_result['iterations']}")
    
    # Verify the root
    verification = cubic_equation(root_result['x'])
    print(f"       Verification: f({root_result['x']:.6f}) = {verification:.8f}")


def main():
    """Run all statistical analysis examples."""
    print("🚀 Refunc Statistical Analysis Examples")
    print("=" * 65)
    
    if not REFUNC_AVAILABLE:
        print("ℹ️  Running in demo mode (showing code structure)")
    else:
        print("ℹ️  Running with full Refunc functionality")
    
    print()
    
    # Set random seed for reproducible examples
    random.seed(42)
    
    # Run examples
    descriptive_statistics_examples()
    hypothesis_testing_examples()
    correlation_regression_analysis()
    distribution_analysis_examples()
    optimization_integration_examples()
    
    print("\n✅ Statistical analysis examples completed!")
    print("\n📖 Next steps:")
    print("- Apply statistical methods to your data analysis workflows")
    print("- Combine with data validation for comprehensive analysis")
    print("- Use optimization for hyperparameter tuning")
    print("- Check out model_training.py for ML applications")


if __name__ == "__main__":
    main()