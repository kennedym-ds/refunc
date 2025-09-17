# Math & Statistics Module

The `refunc.math_stats` module provides comprehensive statistical functions, descriptive statistics, distribution analysis, and hypothesis testing capabilities designed specifically for machine learning workflows.

## Overview

The module includes:

- **Descriptive Statistics**: Comprehensive statistical summaries
- **Hypothesis Testing**: Statistical tests for normality, correlation, and group comparisons
- **Statistical Test Results**: Structured results with interpretations
- **Bootstrap Methods**: Confidence intervals and resampling techniques
- **Outlier Detection**: Multiple methods for identifying outliers

## Core Classes

### StatTestType

Enumeration of statistical test types.

```python
class StatTestType(Enum):
    NORMALITY = "normality"
    INDEPENDENCE = "independence"
    HOMOGENEITY = "homogeneity"
    CORRELATION = "correlation"
    PARAMETRIC = "parametric"
    NON_PARAMETRIC = "non_parametric"
```

### StatTestResult

Container for statistical test results with interpretation.

```python
@dataclass
class StatTestResult:
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float] = None
    degrees_freedom: Optional[int] = None
    effect_size: Optional[float] = None
    power: Optional[float] = None
    interpretation: Optional[str] = None
    confidence_interval: Optional[Tuple[float, float]] = None
```

**Properties:**

- `is_significant(alpha=0.05)`: Check if result is statistically significant
- `summary()`: Get formatted summary of results

**Example:**

```python
from refunc.math_stats import test_normality
import numpy as np

data = np.random.normal(0, 1, 1000)
result = test_normality(data, method="shapiro")

print(f"Test: {result.test_name}")
print(f"P-value: {result.p_value:.6f}")
print(f"Significant: {result.is_significant}")
print(result.summary())
```

### DescriptiveStats

Comprehensive descriptive statistics container.

```python
@dataclass
class DescriptiveStats:
    count: int
    mean: float
    median: float
    mode: Optional[float]
    std: float
    var: float
    min: float
    max: float
    range: float
    q1: float
    q3: float
    iqr: float
    skewness: float
    kurtosis: float
    cv: float  # Coefficient of variation
    mad: float  # Median absolute deviation
    percentiles: Dict[int, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
```

**Methods:**

- `summary()`: Get formatted summary of all statistics

**Example:**

```python
from refunc.math_stats import describe
import numpy as np

data = np.random.normal(100, 15, 1000)
stats = describe(data)

print(stats.summary())
print(f"Mean: {stats.mean:.2f} ± {stats.std:.2f}")
print(f"Skewness: {stats.skewness:.3f}")
```

## StatisticsEngine Class

The main computational engine for statistical analysis.

### Initialization

```python
class StatisticsEngine:
    def __init__(self, confidence_level: float = 0.95):
        """Initialize statistics engine."""
```

**Parameters:**

- `confidence_level`: Default confidence level for intervals (0.95 = 95%)

### Methods

#### describe()

Compute comprehensive descriptive statistics.

```python
def describe(
    self,
    data: Union[List, np.ndarray, pd.Series],
    percentiles: Optional[List[int]] = None,
    include_ci: bool = True
) -> DescriptiveStats:
```

**Parameters:**

- `data`: Input data (list, numpy array, or pandas Series)
- `percentiles`: Custom percentile values to compute (default: [5, 10, 25, 50, 75, 90, 95])
- `include_ci`: Whether to include confidence intervals

**Returns:**

- `DescriptiveStats`: Complete statistical summary

**Example:**

```python
from refunc.math_stats import StatisticsEngine
import numpy as np

engine = StatisticsEngine(confidence_level=0.99)
data = np.random.exponential(2, 1000)

stats = engine.describe(
    data, 
    percentiles=[1, 5, 25, 50, 75, 95, 99],
    include_ci=True
)

print(f"Distribution shape:")
print(f"  Skewness: {stats.skewness:.3f}")
print(f"  Kurtosis: {stats.kurtosis:.3f}")
print(f"99% CI for mean: {stats.confidence_intervals['mean']}")
```

#### test_normality()

Test for normality of data distribution.

```python
def test_normality(
    self,
    data: Union[List, np.ndarray, pd.Series],
    method: str = "shapiro"
) -> StatTestResult:
```

**Parameters:**

- `data`: Input data
- `method`: Test method ("shapiro", "anderson", "jarque_bera", "normaltest")

**Available Methods:**

- **Shapiro-Wilk** (`"shapiro"`): Best for small to medium samples (n ≤ 5000)
- **Anderson-Darling** (`"anderson"`): Good for larger samples
- **Jarque-Bera** (`"jarque_bera"`): Based on skewness and kurtosis
- **D'Agostino-Pearson** (`"normaltest"`): Combines skewness and kurtosis tests

**Example:**

```python
engine = StatisticsEngine()

# Normal data
normal_data = np.random.normal(0, 1, 500)
result = engine.test_normality(normal_data, method="shapiro")
print(f"Normal data: {result.interpretation}")

# Non-normal data
exponential_data = np.random.exponential(1, 500)
result = engine.test_normality(exponential_data, method="shapiro")
print(f"Exponential data: {result.interpretation}")
```

#### test_correlation()

Test correlation between two variables.

```python
def test_correlation(
    self,
    x: Union[List, np.ndarray, pd.Series],
    y: Union[List, np.ndarray, pd.Series],
    method: str = "pearson"
) -> StatTestResult:
```

**Parameters:**

- `x`: First variable
- `y`: Second variable
- `method`: Correlation method ("pearson", "spearman", "kendall")

**Correlation Methods:**

- **Pearson** (`"pearson"`): Linear correlation, assumes normality
- **Spearman** (`"spearman"`): Rank correlation, non-parametric
- **Kendall** (`"kendall"`): Tau correlation, robust to outliers

**Example:**

```python
# Linear relationship
x = np.linspace(0, 10, 100)
y = 2 * x + np.random.normal(0, 1, 100)

pearson_result = engine.test_correlation(x, y, method="pearson")
print(f"Pearson correlation: {pearson_result.statistic:.3f}")
print(f"Interpretation: {pearson_result.interpretation}")

# Non-linear relationship
y_nonlinear = x**2 + np.random.normal(0, 5, 100)
spearman_result = engine.test_correlation(x, y_nonlinear, method="spearman")
print(f"Spearman correlation: {spearman_result.statistic:.3f}")
```

#### compare_means()

Compare means between groups using various statistical tests.

```python
def compare_means(
    self,
    group1: Union[List, np.ndarray, pd.Series],
    group2: Optional[Union[List, np.ndarray, pd.Series]] = None,
    groups: Optional[List[Union[List, np.ndarray, pd.Series]]] = None,
    paired: bool = False,
    parametric: bool = True,
    alternative: str = "two-sided"
) -> StatTestResult:
```

**Parameters:**

- `group1`: First group data
- `group2`: Second group data (for two-group comparison)
- `groups`: Multiple groups (for ANOVA/Kruskal-Wallis)
- `paired`: Whether observations are paired
- `parametric`: Whether to use parametric tests
- `alternative`: Alternative hypothesis ("two-sided", "less", "greater")

**Test Selection:**

| Scenario | Parametric | Non-parametric |
|----------|------------|----------------|
| Two independent groups | Independent t-test | Mann-Whitney U |
| Two paired groups | Paired t-test | Wilcoxon signed-rank |
| Multiple groups | One-way ANOVA | Kruskal-Wallis |

**Example:**

```python
# Two independent groups
group_a = np.random.normal(100, 15, 50)
group_b = np.random.normal(105, 15, 50)

result = engine.compare_means(group_a, group_b, parametric=True)
print(f"T-test result: {result.interpretation}")
print(f"Effect size (Cohen's d): {result.effect_size:.3f}")

# Multiple groups comparison
group_c = np.random.normal(110, 15, 50)
anova_result = engine.compare_means(
    group1=group_a, 
    groups=[group_a, group_b, group_c], 
    parametric=True
)
print(f"ANOVA result: {anova_result.interpretation}")

# Paired comparison
before = np.random.normal(100, 15, 30)
after = before + np.random.normal(5, 10, 30)  # Some improvement

paired_result = engine.compare_means(
    before, after, 
    paired=True, 
    parametric=True
)
print(f"Paired t-test: {paired_result.interpretation}")
```

#### bootstrap_confidence_interval()

Calculate bootstrap confidence intervals for any statistic.

```python
def bootstrap_confidence_interval(
    self,
    data: Union[List, np.ndarray, pd.Series],
    statistic_func: Callable = np.mean,
    n_bootstrap: int = 10000,
    confidence_level: Optional[float] = None
) -> Tuple[float, float]:
```

**Parameters:**

- `data`: Input data
- `statistic_func`: Function to compute statistic
- `n_bootstrap`: Number of bootstrap samples
- `confidence_level`: Confidence level (uses instance default if None)

**Example:**

```python
# Bootstrap confidence interval for mean
data = np.random.gamma(2, 2, 1000)
ci_mean = engine.bootstrap_confidence_interval(data, np.mean)
print(f"95% CI for mean: [{ci_mean[0]:.3f}, {ci_mean[1]:.3f}]")

# Bootstrap confidence interval for median
ci_median = engine.bootstrap_confidence_interval(data, np.median)
print(f"95% CI for median: [{ci_median[0]:.3f}, {ci_median[1]:.3f}]")

# Custom statistic - coefficient of variation
def cv(x):
    return np.std(x) / np.mean(x)

ci_cv = engine.bootstrap_confidence_interval(data, cv)
print(f"95% CI for CV: [{ci_cv[0]:.3f}, {ci_cv[1]:.3f}]")
```

#### outlier_detection()

Detect outliers using various methods.

```python
def outlier_detection(
    self,
    data: Union[List, np.ndarray, pd.Series],
    method: str = "iqr",
    threshold: float = 1.5
) -> Tuple[np.ndarray, np.ndarray]:
```

**Parameters:**

- `data`: Input data
- `method`: Detection method ("iqr", "zscore", "modified_zscore")
- `threshold`: Threshold for outlier detection

**Methods:**

- **IQR Method** (`"iqr"`): Outliers beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR
- **Z-Score** (`"zscore"`): Outliers with |z-score| > threshold
- **Modified Z-Score** (`"modified_zscore"`): Robust to outliers themselves

**Returns:**

- Tuple of (outlier_indices, outlier_values)

**Example:**

```python
# Generate data with outliers
np.random.seed(42)
clean_data = np.random.normal(50, 10, 95)
outliers = [100, -20, 150, -10, 90]
data = np.concatenate([clean_data, outliers])

# Detect outliers using different methods
iqr_indices, iqr_values = engine.outlier_detection(data, method="iqr")
zscore_indices, zscore_values = engine.outlier_detection(data, method="zscore", threshold=3)
mod_zscore_indices, mod_zscore_values = engine.outlier_detection(data, method="modified_zscore", threshold=3.5)

print(f"IQR method found {len(iqr_indices)} outliers: {iqr_values}")
print(f"Z-score method found {len(zscore_indices)} outliers: {zscore_values}")
print(f"Modified Z-score found {len(mod_zscore_indices)} outliers: {mod_zscore_values}")
```

## Convenience Functions

For quick analysis without creating an engine instance:

### describe() Function

```python
def describe(data: Union[List, np.ndarray, pd.Series], **kwargs) -> DescriptiveStats:
    """Compute descriptive statistics for data."""
```

### test_normality() Function

```python
def test_normality(data: Union[List, np.ndarray, pd.Series], method: str = "shapiro") -> StatTestResult:
    """Test normality of data."""
```

### test_correlation() Function

```python
def test_correlation(
    x: Union[List, np.ndarray, pd.Series],
    y: Union[List, np.ndarray, pd.Series],
    method: str = "pearson"
) -> StatTestResult:
    """Test correlation between variables."""
```

### compare_groups()

```python
def compare_groups(
    group1: Union[List, np.ndarray, pd.Series],
    group2: Optional[Union[List, np.ndarray, pd.Series]] = None,
    **kwargs
) -> StatTestResult:
    """Compare means between groups."""
```

### bootstrap_ci()

```python
def bootstrap_ci(
    data: Union[List, np.ndarray, pd.Series],
    statistic_func: Callable = np.mean,
    **kwargs
) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval."""
```

### detect_outliers()

```python
def detect_outliers(
    data: Union[List, np.ndarray, pd.Series],
    method: str = "iqr",
    threshold: float = 1.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect outliers in data."""
```

## Complete Examples

### Data Quality Assessment

```python
from refunc.math_stats import describe, test_normality, detect_outliers
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
data = np.concatenate([
    np.random.normal(100, 15, 950),  # Main distribution
    np.random.uniform(150, 200, 50)  # Some outliers
])

# Comprehensive data assessment
print("=== DATA QUALITY ASSESSMENT ===")

# 1. Descriptive statistics
stats = describe(data)
print(f"\nSample size: {stats.count}")
print(f"Mean ± SD: {stats.mean:.2f} ± {stats.std:.2f}")
print(f"Median [IQR]: {stats.median:.2f} [{stats.q1:.2f}, {stats.q3:.2f}]")
print(f"Skewness: {stats.skewness:.3f}")
print(f"Kurtosis: {stats.kurtosis:.3f}")

# 2. Normality assessment
normality = test_normality(data, method="shapiro")
print(f"\nNormality test: {normality.interpretation}")
print(f"Shapiro-Wilk p-value: {normality.p_value:.6f}")

# 3. Outlier detection
outlier_indices, outlier_values = detect_outliers(data, method="iqr")
print(f"\nOutliers detected: {len(outlier_indices)} ({len(outlier_indices)/len(data)*100:.1f}%)")
if len(outlier_values) > 0:
    print(f"Outlier range: [{outlier_values.min():.1f}, {outlier_values.max():.1f}]")

# 4. Distribution assessment
if stats.skewness > 1:
    print("Distribution: Highly right-skewed")
elif stats.skewness > 0.5:
    print("Distribution: Moderately right-skewed")
elif stats.skewness < -1:
    print("Distribution: Highly left-skewed")
elif stats.skewness < -0.5:
    print("Distribution: Moderately left-skewed")
else:
    print("Distribution: Approximately symmetric")
```

### A/B Testing Analysis

```python
from refunc.math_stats import compare_groups, describe, StatisticsEngine
import numpy as np

# Simulate A/B test data
np.random.seed(42)
control_group = np.random.normal(100, 15, 1000)  # Control: mean=100
treatment_group = np.random.normal(103, 15, 1000)  # Treatment: mean=103

print("=== A/B TEST ANALYSIS ===")

# Descriptive statistics for both groups
control_stats = describe(control_group)
treatment_stats = describe(treatment_group)

print(f"\nControl Group (n={control_stats.count}):")
print(f"  Mean ± SD: {control_stats.mean:.2f} ± {control_stats.std:.2f}")
print(f"  95% CI: [{control_stats.confidence_intervals['mean'][0]:.2f}, {control_stats.confidence_intervals['mean'][1]:.2f}]")

print(f"\nTreatment Group (n={treatment_stats.count}):")
print(f"  Mean ± SD: {treatment_stats.mean:.2f} ± {treatment_stats.std:.2f}")
print(f"  95% CI: [{treatment_stats.confidence_intervals['mean'][0]:.2f}, {treatment_stats.confidence_intervals['mean'][1]:.2f}]")

# Statistical comparison
comparison = compare_groups(control_group, treatment_group, parametric=True)
print(f"\nStatistical Test: {comparison.test_name}")
print(f"Result: {comparison.interpretation}")
print(f"P-value: {comparison.p_value:.6f}")
print(f"Effect size (Cohen's d): {comparison.effect_size:.3f}")

# Effect size interpretation
abs_effect = abs(comparison.effect_size)
if abs_effect < 0.2:
    effect_magnitude = "negligible"
elif abs_effect < 0.5:
    effect_magnitude = "small"
elif abs_effect < 0.8:
    effect_magnitude = "medium"
else:
    effect_magnitude = "large"

print(f"Effect magnitude: {effect_magnitude}")

# Business interpretation
improvement = ((treatment_stats.mean - control_stats.mean) / control_stats.mean) * 100
print(f"Relative improvement: {improvement:.2f}%")
```

### Time Series Analysis

```python
from refunc.math_stats import StatisticsEngine, describe
import numpy as np

# Generate time series data
np.random.seed(42)
n_points = 365  # One year of daily data
trend = np.linspace(100, 120, n_points)
seasonal = 10 * np.sin(2 * np.pi * np.arange(n_points) / 365.25 * 4)  # Quarterly pattern
noise = np.random.normal(0, 5, n_points)
time_series = trend + seasonal + noise

print("=== TIME SERIES ANALYSIS ===")

engine = StatisticsEngine()

# Overall statistics
overall_stats = describe(time_series)
print(f"Overall Statistics:")
print(f"  Mean: {overall_stats.mean:.2f}")
print(f"  Volatility (SD): {overall_stats.std:.2f}")
print(f"  Coefficient of Variation: {overall_stats.cv:.3f}")

# Quarterly analysis
quarters = np.array_split(time_series, 4)
quarterly_means = [np.mean(quarter) for quarter in quarters]

print(f"\nQuarterly Analysis:")
for i, quarter_mean in enumerate(quarterly_means, 1):
    print(f"  Q{i}: {quarter_mean:.2f}")

# Test for differences between quarters
anova_result = engine.compare_means(quarters[0], groups=quarters, parametric=True)
print(f"\nQuarterly differences: {anova_result.interpretation}")
print(f"ANOVA p-value: {anova_result.p_value:.6f}")

# Trend analysis (comparing first and last quarters)
trend_test = engine.compare_means(quarters[0], quarters[-1], parametric=True)
print(f"Trend analysis: {trend_test.interpretation}")
print(f"Change from Q1 to Q4: {quarterly_means[-1] - quarterly_means[0]:.2f}")
```

## Best Practices

### 1. Choosing Statistical Tests

```python
# Check assumptions before selecting tests
def choose_test_strategy(data):
    stats = describe(data)
    normality = test_normality(data)
    
    if normality.is_significant:
        print("Data is not normally distributed - consider non-parametric tests")
        return "non_parametric"
    else:
        print("Data appears normally distributed - parametric tests appropriate")
        return "parametric"

# Apply appropriate tests
strategy = choose_test_strategy(your_data)
```

### 2. Sample Size Considerations

```python
def assess_sample_size(data, min_size=30):
    n = len(data)
    if n < min_size:
        print(f"Warning: Small sample size (n={n}). Results may be unreliable.")
    if n < 3:
        raise ValueError("Need at least 3 observations for statistical analysis")
    return n
```

### 3. Multiple Comparisons

```python
# Adjust p-values for multiple comparisons
def adjust_p_values(p_values, method="bonferroni"):
    """Apply multiple comparison correction."""
    if method == "bonferroni":
        return [p * len(p_values) for p in p_values]
    # Add other methods as needed
```

### 4. Effect Size Reporting

```python
# Always report effect sizes alongside p-values
def interpret_effect_size(effect_size, test_type="cohen_d"):
    """Interpret effect size magnitude."""
    abs_effect = abs(effect_size)
    
    if test_type == "cohen_d":
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
    
    return "unknown"
```

## Dependencies

The math_stats module requires:

- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `scipy`: Statistical functions
- `typing`: Type annotations

## See Also

- [Configuration Guide](../guides/configuration.md) - Setting up statistical analysis parameters
- [Performance Guide](../guides/performance.md) - Optimizing statistical computations
- [Examples](../examples/statistics.md) - Comprehensive usage examples
- [API Reference](utils.md) - Related utility functions
