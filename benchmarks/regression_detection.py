"""
Performance regression detection utilities for refunc benchmarks.

This module provides utilities for detecting performance regressions by
comparing current benchmark results with historical baselines.
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib


@dataclass
class BenchmarkResult:
    """Container for benchmark result data."""
    name: str
    mean_time: float
    min_time: float
    max_time: float
    stddev: float
    iterations: int
    timestamp: str
    git_commit: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class RegressionAlert:
    """Container for regression alert information."""
    benchmark_name: str
    current_mean: float
    baseline_mean: float
    regression_percent: float
    threshold_percent: float
    severity: str  # 'warning', 'error', 'critical'
    message: str


class PerformanceRegression:
    """Performance regression detection system."""
    
    def __init__(self, threshold_percent: float = 20, baseline_file: Optional[Path] = None):
        """
        Initialize regression detector.
        
        Args:
            threshold_percent: Regression threshold as percentage (e.g., 20 for 20%)
            baseline_file: Path to baseline performance data file
        """
        self.threshold = threshold_percent / 100
        self.baseline_file = baseline_file or Path("benchmarks/.benchmarks/baseline.json")
        self.baseline = self._load_baseline()
        
    def _load_baseline(self) -> Dict[str, BenchmarkResult]:
        """Load baseline performance data from file."""
        if not self.baseline_file.exists():
            return {}
        
        try:
            with open(self.baseline_file, 'r') as f:
                baseline_data = json.load(f)
            
            baseline = {}
            for name, data in baseline_data.items():
                baseline[name] = BenchmarkResult(
                    name=data['name'],
                    mean_time=data['mean_time'],
                    min_time=data['min_time'],
                    max_time=data['max_time'],
                    stddev=data['stddev'],
                    iterations=data['iterations'],
                    timestamp=data['timestamp'],
                    git_commit=data.get('git_commit'),
                    metadata=data.get('metadata', {})
                )
            
            return baseline
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load baseline data: {e}")
            return {}
    
    def save_baseline(self, results: Dict[str, BenchmarkResult]) -> None:
        """Save benchmark results as new baseline."""
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
        
        baseline_data = {}
        for name, result in results.items():
            baseline_data[name] = {
                'name': result.name,
                'mean_time': result.mean_time,
                'min_time': result.min_time,
                'max_time': result.max_time,
                'stddev': result.stddev,
                'iterations': result.iterations,
                'timestamp': result.timestamp,
                'git_commit': result.git_commit,
                'metadata': result.metadata or {}
            }
        
        with open(self.baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        self.baseline = results
    
    def set_baseline(self, benchmark_name: str, baseline_result: BenchmarkResult) -> None:
        """Set baseline for a specific benchmark."""
        self.baseline[benchmark_name] = baseline_result
    
    def check_regression(self, benchmark_name: str, current_result: BenchmarkResult) -> Optional[RegressionAlert]:
        """
        Check if benchmark shows performance regression.
        
        Args:
            benchmark_name: Name of the benchmark
            current_result: Current benchmark result
            
        Returns:
            RegressionAlert if regression detected, None otherwise
        """
        if benchmark_name not in self.baseline:
            return None
        
        baseline_result = self.baseline[benchmark_name]
        current_mean = current_result.mean_time
        baseline_mean = baseline_result.mean_time
        
        if baseline_mean <= 0:
            return None
        
        regression = (current_mean - baseline_mean) / baseline_mean
        
        if regression > self.threshold:
            severity = self._determine_severity(regression)
            message = self._generate_message(
                benchmark_name, current_mean, baseline_mean, regression * 100
            )
            
            return RegressionAlert(
                benchmark_name=benchmark_name,
                current_mean=current_mean,
                baseline_mean=baseline_mean,
                regression_percent=regression * 100,
                threshold_percent=self.threshold * 100,
                severity=severity,
                message=message
            )
        
        return None
    
    def check_all_regressions(self, current_results: Dict[str, BenchmarkResult]) -> List[RegressionAlert]:
        """Check for regressions in all benchmark results."""
        alerts = []
        
        for name, result in current_results.items():
            alert = self.check_regression(name, result)
            if alert:
                alerts.append(alert)
        
        return alerts
    
    def _determine_severity(self, regression: float) -> str:
        """Determine severity level based on regression percentage."""
        regression_percent = regression * 100
        
        if regression_percent >= 50:
            return 'critical'
        elif regression_percent >= 30:
            return 'error'
        else:
            return 'warning'
    
    def _generate_message(self, name: str, current: float, baseline: float, percent: float) -> str:
        """Generate human-readable regression message."""
        return (
            f"Performance regression detected in {name}: "
            f"{current:.4f}s vs baseline {baseline:.4f}s "
            f"({percent:.1f}% slower)"
        )


class BenchmarkComparison:
    """Utility for comparing benchmark results across versions."""
    
    def __init__(self, results_dir: Path = None):
        """Initialize benchmark comparison utility."""
        self.results_dir = results_dir or Path("benchmarks/.benchmarks")
        
    def compare_with_previous(self, current_results: Dict[str, BenchmarkResult]) -> Dict[str, float]:
        """Compare current results with previous run."""
        previous_file = self.results_dir / "previous.json"
        
        if not previous_file.exists():
            return {}
        
        with open(previous_file, 'r') as f:
            previous_data = json.load(f)
        
        comparisons = {}
        
        for name, current in current_results.items():
            if name in previous_data:
                previous_mean = previous_data[name]['mean_time']
                if previous_mean > 0:
                    change_percent = ((current.mean_time - previous_mean) / previous_mean) * 100
                    comparisons[name] = change_percent
        
        return comparisons
    
    def generate_comparison_report(self, 
                                 current_results: Dict[str, BenchmarkResult],
                                 include_history: bool = True) -> Dict[str, Any]:
        """Generate comprehensive comparison report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_benchmarks': len(current_results),
            'comparisons': {},
            'summary': {}
        }
        
        # Compare with previous run
        previous_comparisons = self.compare_with_previous(current_results)
        
        improved = 0
        regressed = 0
        stable = 0
        
        for name, change_percent in previous_comparisons.items():
            status = 'stable'
            if change_percent < -5:  # 5% improvement
                status = 'improved'
                improved += 1
            elif change_percent > 5:  # 5% regression
                status = 'regressed'
                regressed += 1
            else:
                stable += 1
            
            report['comparisons'][name] = {
                'change_percent': change_percent,
                'status': status,
                'current_mean': current_results[name].mean_time,
                'current_stddev': current_results[name].stddev
            }
        
        report['summary'] = {
            'improved': improved,
            'regressed': regressed,
            'stable': stable,
            'new_benchmarks': len(current_results) - len(previous_comparisons)
        }
        
        return report


def has_regressed(benchmark_name: str, baseline_ms: float, tolerance: float = 0.25) -> bool:
    """
    Helper function to check for regression (compatible with existing docs).
    
    Args:
        benchmark_name: Name of the benchmark function
        baseline_ms: Baseline time in milliseconds
        tolerance: Tolerance threshold (0.25 = 25%)
        
    Returns:
        True if regression detected, False otherwise
    """
    try:
        from refunc.decorators import get_timing_stats
        
        stats = get_timing_stats(benchmark_name)
        if not stats:
            return False
        
        current_ms = stats.mean_time * 1000
        return current_ms > baseline_ms * (1 + tolerance)
        
    except ImportError:
        # Fallback if decorators not available
        return False


def benchmark_result_from_pytest(benchmark_data: Dict[str, Any]) -> BenchmarkResult:
    """Convert pytest-benchmark result to BenchmarkResult."""
    stats = benchmark_data.get('stats', {})
    
    return BenchmarkResult(
        name=benchmark_data.get('name', 'unknown'),
        mean_time=stats.get('mean', 0.0),
        min_time=stats.get('min', 0.0),
        max_time=stats.get('max', 0.0),
        stddev=stats.get('stddev', 0.0),
        iterations=stats.get('rounds', 0),
        timestamp=datetime.now().isoformat(),
        metadata={
            'ops': stats.get('ops', 0),
            'outliers': stats.get('outliers', 0)
        }
    )


# Example usage functions
def setup_performance_monitoring():
    """Set up performance monitoring for CI/CD."""
    detector = PerformanceRegression(threshold_percent=25)
    
    # Example baseline establishment
    baseline_results = {
        'test_timing_decorator_overhead': BenchmarkResult(
            name='test_timing_decorator_overhead',
            mean_time=0.000050,  # 50 microseconds
            min_time=0.000045,
            max_time=0.000055,
            stddev=0.000002,
            iterations=1000,
            timestamp=datetime.now().isoformat()
        )
    }
    
    detector.save_baseline(baseline_results)
    return detector


def check_benchmark_regressions(pytest_results_file: Path) -> List[RegressionAlert]:
    """Check for regressions from pytest-benchmark results."""
    if not pytest_results_file.exists():
        return []
    
    with open(pytest_results_file, 'r') as f:
        pytest_data = json.load(f)
    
    # Convert pytest results to BenchmarkResult objects
    current_results = {}
    for benchmark in pytest_data.get('benchmarks', []):
        result = benchmark_result_from_pytest(benchmark)
        current_results[result.name] = result
    
    # Check for regressions
    detector = PerformanceRegression()
    alerts = detector.check_all_regressions(current_results)
    
    return alerts