"""
Comprehensive statistics utilities for ML workflows.

This module provides statistical functions, descriptive statistics,
distribution analysis, and hypothesis testing capabilities.
"""

import warnings
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import normaltest, shapiro, jarque_bera, anderson
from scipy.stats import kstest, chi2_contingency, mannwhitneyu, wilcoxon
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel
from scipy.stats import f_oneway, kruskal, friedmanchisquare

from ..exceptions import RefuncError, ValidationError


class StatTestType(Enum):
    """Statistical test types."""
    NORMALITY = "normality"
    INDEPENDENCE = "independence"
    HOMOGENEITY = "homogeneity"
    CORRELATION = "correlation"
    PARAMETRIC = "parametric"
    NON_PARAMETRIC = "non_parametric"


@dataclass
class StatTestResult:
    """Results from statistical tests."""
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float] = None
    degrees_freedom: Optional[int] = None
    effect_size: Optional[float] = None
    power: Optional[float] = None
    interpretation: Optional[str] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    
    @property
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if result is statistically significant."""
        return self.p_value < alpha
    
    def summary(self) -> str:
        """Get formatted summary of results."""
        lines = [
            f"Test: {self.test_name}",
            f"Statistic: {self.statistic:.6f}",
            f"P-value: {self.p_value:.6f}",
            f"Significant (α=0.05): {'Yes' if self.is_significant else 'No'}"
        ]
        
        if self.degrees_freedom is not None:
            lines.append(f"Degrees of freedom: {self.degrees_freedom}")
        if self.effect_size is not None:
            lines.append(f"Effect size: {self.effect_size:.6f}")
        if self.confidence_interval is not None:
            ci_low, ci_high = self.confidence_interval
            lines.append(f"95% CI: [{ci_low:.6f}, {ci_high:.6f}]")
        if self.interpretation:
            lines.append(f"Interpretation: {self.interpretation}")
            
        return "\n".join(lines)


@dataclass
class DescriptiveStats:
    """Comprehensive descriptive statistics."""
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
    
    def summary(self) -> str:
        """Get formatted summary of descriptive statistics."""
        lines = [
            "Descriptive Statistics",
            "=" * 30,
            f"Count: {self.count}",
            f"Mean: {self.mean:.6f}",
            f"Median: {self.median:.6f}",
            f"Mode: {self.mode:.6f}" if self.mode is not None else "Mode: N/A",
            f"Std Dev: {self.std:.6f}",
            f"Variance: {self.var:.6f}",
            f"Min: {self.min:.6f}",
            f"Max: {self.max:.6f}",
            f"Range: {self.range:.6f}",
            f"Q1: {self.q1:.6f}",
            f"Q3: {self.q3:.6f}",
            f"IQR: {self.iqr:.6f}",
            f"Skewness: {self.skewness:.6f}",
            f"Kurtosis: {self.kurtosis:.6f}",
            f"CV: {self.cv:.6f}",
            f"MAD: {self.mad:.6f}"
        ]
        
        # Add percentiles
        lines.append("\nPercentiles:")
        for p, value in sorted(self.percentiles.items()):
            lines.append(f"  {p}th: {value:.6f}")
            
        # Add confidence intervals
        lines.append("\nConfidence Intervals:")
        for level, (ci_low, ci_high) in self.confidence_intervals.items():
            lines.append(f"  {level}: [{ci_low:.6f}, {ci_high:.6f}]")
            
        return "\n".join(lines)


class StatisticsEngine:
    """Comprehensive statistics computation engine."""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistics engine.
        
        Args:
            confidence_level: Default confidence level for intervals
        """
        self.confidence_level = confidence_level
        self._cache = {}
        
    def describe(
        self,
        data: Union[List, np.ndarray, pd.Series],
        percentiles: Optional[List[int]] = None,
        include_ci: bool = True
    ) -> DescriptiveStats:
        """
        Compute comprehensive descriptive statistics.
        
        Args:
            data: Input data
            percentiles: Custom percentile values to compute
            include_ci: Whether to include confidence intervals
            
        Returns:
            DescriptiveStats object with all statistics
        """
        if percentiles is None:
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            
        data = np.array(data)
        if len(data) == 0:
            raise ValidationError("Cannot compute statistics for empty data")
            
        # Basic statistics
        count = len(data)
        mean = np.mean(data)
        median = np.median(data)
        std = np.std(data, ddof=1) if count > 1 else 0.0
        var = np.var(data, ddof=1) if count > 1 else 0.0
        min_val = np.min(data)
        max_val = np.max(data)
        range_val = max_val - min_val
        
        # Quartiles and IQR
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        # Shape statistics
        skewness = stats.skew(data) if count > 2 else 0.0
        kurtosis = stats.kurtosis(data) if count > 3 else 0.0
        
        # Additional statistics
        cv = std / mean if mean != 0 else np.inf
        mad = stats.median_abs_deviation(data)
        
        # Mode (for numerical data, use most frequent value)
        try:
            mode_result = stats.mode(data, keepdims=False)
            mode = float(mode_result.mode) if mode_result.count > 1 else None
        except:
            mode = None
            
        # Percentiles
        percentile_values = {p: np.percentile(data, p) for p in percentiles}
        
        # Confidence intervals
        confidence_intervals = {}
        if include_ci and count > 1:
            alpha = 1 - self.confidence_level
            se = std / np.sqrt(count)
            
            # Mean confidence interval
            t_critical = stats.t.ppf(1 - alpha/2, count - 1)
            margin_error = t_critical * se
            confidence_intervals['mean'] = (mean - margin_error, mean + margin_error)
            
            # Variance confidence interval (using chi-square)
            chi2_low = stats.chi2.ppf(alpha/2, count - 1)
            chi2_high = stats.chi2.ppf(1 - alpha/2, count - 1)
            var_ci_low = (count - 1) * var / chi2_high
            var_ci_high = (count - 1) * var / chi2_low
            confidence_intervals['variance'] = (var_ci_low, var_ci_high)
            
        return DescriptiveStats(
            count=count,
            mean=float(mean),
            median=float(median),
            mode=mode,
            std=float(std),
            var=float(var),
            min=min_val,
            max=max_val,
            range=range_val,
            q1=float(q1),
            q3=float(q3),
            iqr=float(iqr),
            skewness=float(skewness),
            kurtosis=float(kurtosis),
            cv=float(cv),
            mad=float(mad),
            percentiles={k: float(v) for k, v in percentile_values.items()},
            confidence_intervals=confidence_intervals
        )
    
    def test_normality(
        self,
        data: Union[List, np.ndarray, pd.Series],
        method: str = "shapiro"
    ) -> StatTestResult:
        """
        Test for normality of data.
        
        Args:
            data: Input data
            method: Test method (shapiro, anderson, jarque_bera, normaltest)
            
        Returns:
            Statistical test result
        """
        data = np.array(data)
        if len(data) < 3:
            raise ValidationError("Need at least 3 observations for normality test")
            
        if method == "shapiro":
            from scipy.stats import shapiro
            if len(data) > 5000:
                warnings.warn("Shapiro-Wilk test may be unreliable for large samples")
            statistic, p_value = shapiro(data)
            statistic = float(statistic)
            p_value = float(p_value)
            interpretation = "Data appears normally distributed" if p_value > 0.05 else "Data appears non-normal"
            
        elif method == "anderson":
            from scipy.stats import anderson
            result = anderson(data, dist='norm')
            statistic = float(result.statistic)  # type: ignore
            # Use 5% significance level
            critical_value = float(result.critical_values[2])  # 5% level  # type: ignore
            p_value = 0.05 if statistic > critical_value else 0.1  # Approximate
            interpretation = "Data appears normally distributed" if statistic < critical_value else "Data appears non-normal"
            
        elif method == "jarque_bera":
            from scipy.stats import jarque_bera
            result = jarque_bera(data)
            statistic = float(result[0])  # type: ignore
            p_value = float(result[1])  # type: ignore
            interpretation = "Data appears normally distributed" if p_value > 0.05 else "Data appears non-normal"
            
        elif method == "normaltest":
            from scipy.stats import normaltest
            result = normaltest(data)
            statistic = float(result[0])  # type: ignore
            p_value = float(result[1])  # type: ignore
            interpretation = "Data appears normally distributed" if p_value > 0.05 else "Data appears non-normal"
            
        else:
            raise ValidationError(f"Unknown normality test method: {method}")
            
        return StatTestResult(
            test_name=f"{method.title()} Normality Test",
            statistic=statistic,
            p_value=p_value,
            interpretation=interpretation
        )
    
    def test_independence(
        self,
        x: Union[List, np.ndarray, pd.Series],
        y: Optional[Union[List, np.ndarray, pd.Series]] = None,
        method: str = "chi2"
    ) -> StatTestResult:
        """
        Test for independence between variables.
        
        Args:
            x: First variable (or contingency table if y is None)
            y: Second variable (optional)
            method: Test method (chi2, fisher)
            
        Returns:
            Statistical test result
        """
        if method == "chi2":
            if y is not None:
                # Create contingency table
                contingency_table = pd.crosstab(x, y)
            else:
                contingency_table = np.array(x)
                
            result = chi2_contingency(contingency_table)
            
            # Convert to float  
            chi2_stat = float(result[0])  # type: ignore
            p_value = float(result[1])  # type: ignore
            dof = int(result[2])  # type: ignore
            
            interpretation = "Variables appear independent" if p_value > 0.05 else "Variables appear dependent"
            
            return StatTestResult(
                test_name="Chi-square Test of Independence",
                statistic=chi2_stat,
                p_value=p_value,
                degrees_freedom=dof,
                interpretation=interpretation
            )
        else:
            raise ValidationError(f"Unknown independence test method: {method}")
    
    def test_correlation(
        self,
        x: Union[List, np.ndarray, pd.Series],
        y: Union[List, np.ndarray, pd.Series],
        method: str = "pearson"
    ) -> StatTestResult:
        """
        Test correlation between two variables.
        
        Args:
            x: First variable
            y: Second variable  
            method: Correlation method (pearson, spearman, kendall)
            
        Returns:
            Statistical test result
        """
        x = np.array(x)
        y = np.array(y)
        
        if len(x) != len(y):
            raise ValidationError("Variables must have same length")
            
        if method == "pearson":
            result = stats.pearsonr(x, y)
            correlation = float(result[0])  # type: ignore
            p_value = float(result[1])  # type: ignore
            test_name = "Pearson Correlation Test"
        elif method == "spearman":
            result = stats.spearmanr(x, y)
            correlation = float(result[0])  # type: ignore
            p_value = float(result[1])  # type: ignore
            test_name = "Spearman Rank Correlation Test"
        elif method == "kendall":
            result = stats.kendalltau(x, y)
            correlation = float(result[0])  # type: ignore
            p_value = float(result[1])  # type: ignore
            test_name = "Kendall Tau Correlation Test"
        else:
            raise ValidationError(f"Unknown correlation method: {method}")
            
        # Convert to float - already done above
        # correlation = float(correlation)
        # p_value = float(p_value)
        
        # Interpret correlation strength
        abs_corr = abs(correlation)
        if abs_corr < 0.3:
            strength = "weak"
        elif abs_corr < 0.7:
            strength = "moderate"
        else:
            strength = "strong"
            
        direction = "positive" if correlation > 0 else "negative"
        interpretation = f"{strength.title()} {direction} correlation"
        
        return StatTestResult(
            test_name=test_name,
            statistic=correlation,
            p_value=p_value,
            effect_size=correlation,
            interpretation=interpretation
        )
    
    def compare_means(
        self,
        group1: Union[List, np.ndarray, pd.Series],
        group2: Optional[Union[List, np.ndarray, pd.Series]] = None,
        groups: Optional[List[Union[List, np.ndarray, pd.Series]]] = None,
        paired: bool = False,
        parametric: bool = True,
        alternative: str = "two-sided"
    ) -> StatTestResult:
        """
        Compare means between groups.
        
        Args:
            group1: First group data
            group2: Second group data (for two-group comparison)
            groups: Multiple groups (for ANOVA/Kruskal-Wallis)
            paired: Whether observations are paired
            parametric: Whether to use parametric tests
            alternative: Alternative hypothesis ("two-sided", "less", "greater")
            
        Returns:
            Statistical test result
        """
        if groups is not None:
            # Multiple group comparison
            if parametric:
                statistic, p_value = f_oneway(*groups)
                test_name = "One-way ANOVA"
            else:
                statistic, p_value = kruskal(*groups)
                test_name = "Kruskal-Wallis Test"
                
        elif group2 is not None:
            # Two group comparison
            group1 = np.array(group1)
            group2 = np.array(group2)
            
            if paired:
                if parametric:
                    statistic, p_value = ttest_rel(group1, group2, alternative=alternative)
                    test_name = "Paired t-test"
                else:
                    statistic, p_value = wilcoxon(group1, group2, alternative=alternative)
                    test_name = "Wilcoxon Signed-rank Test"
            else:
                if parametric:
                    statistic, p_value = ttest_ind(group1, group2, alternative=alternative)
                    test_name = "Independent t-test"
                else:
                    statistic, p_value = mannwhitneyu(group1, group2, alternative=alternative)
                    test_name = "Mann-Whitney U Test"
        else:
            raise ValidationError("Must provide either group2 or groups for comparison")
            
        # Calculate effect size for two-group comparisons
        effect_size = None
        if group2 is not None and not paired:
            # Cohen's d
            pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                                (len(group2) - 1) * np.var(group2, ddof=1)) / 
                                (len(group1) + len(group2) - 2))
            if pooled_std > 0:
                effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std
                
        interpretation = "Significant difference" if p_value < 0.05 else "No significant difference"
        
        return StatTestResult(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            interpretation=interpretation
        )
    
    def bootstrap_confidence_interval(
        self,
        data: Union[List, np.ndarray, pd.Series],
        statistic_func: Callable = np.mean,
        n_bootstrap: int = 10000,
        confidence_level: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for any statistic.
        
        Args:
            data: Input data
            statistic_func: Function to compute statistic
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (uses instance default if None)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        data = np.array(data)
        n = len(data)
        
        # Generate bootstrap samples
        bootstrap_stats = []
        rng = np.random.RandomState(42)  # For reproducibility
        
        for _ in range(n_bootstrap):
            sample = rng.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic_func(sample))
            
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return float(ci_lower), float(ci_upper)
    
    def outlier_detection(
        self,
        data: Union[List, np.ndarray, pd.Series],
        method: str = "iqr",
        threshold: float = 1.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outliers in data.
        
        Args:
            data: Input data
            method: Detection method (iqr, zscore, modified_zscore)
            threshold: Threshold for outlier detection
            
        Returns:
            Tuple of (outlier_indices, outlier_values)
        """
        data = np.array(data)
        
        if method == "iqr":
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outlier_mask = (data < lower_bound) | (data > upper_bound)
            
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(data))
            outlier_mask = z_scores > threshold
            
        elif method == "modified_zscore":
            median = np.median(data)
            mad = stats.median_abs_deviation(data)
            modified_z_scores = 0.6745 * (data - median) / mad
            outlier_mask = np.abs(modified_z_scores) > threshold
            
        else:
            raise ValidationError(f"Unknown outlier detection method: {method}")
            
        outlier_indices = np.where(outlier_mask)[0]
        outlier_values = data[outlier_mask]
        
        return outlier_indices, outlier_values


# Convenience functions
def describe(data: Union[List, np.ndarray, pd.Series], **kwargs) -> DescriptiveStats:
    """Compute descriptive statistics for data."""
    engine = StatisticsEngine()
    return engine.describe(data, **kwargs)


def test_normality(data: Union[List, np.ndarray, pd.Series], method: str = "shapiro") -> StatTestResult:
    """Test normality of data."""
    engine = StatisticsEngine()
    return engine.test_normality(data, method)


def test_correlation(
    x: Union[List, np.ndarray, pd.Series],
    y: Union[List, np.ndarray, pd.Series],
    method: str = "pearson"
) -> StatTestResult:
    """Test correlation between variables."""
    engine = StatisticsEngine()
    return engine.test_correlation(x, y, method)


def compare_groups(
    group1: Union[List, np.ndarray, pd.Series],
    group2: Optional[Union[List, np.ndarray, pd.Series]] = None,
    **kwargs
) -> StatTestResult:
    """Compare means between groups."""
    engine = StatisticsEngine()
    return engine.compare_means(group1, group2, **kwargs)


def bootstrap_ci(
    data: Union[List, np.ndarray, pd.Series],
    statistic_func: Callable = np.mean,
    **kwargs
) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval."""
    engine = StatisticsEngine()
    return engine.bootstrap_confidence_interval(data, statistic_func, **kwargs)


def detect_outliers(
    data: Union[List, np.ndarray, pd.Series],
    method: str = "iqr",
    threshold: float = 1.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect outliers in data."""
    engine = StatisticsEngine()
    return engine.outlier_detection(data, method, threshold)