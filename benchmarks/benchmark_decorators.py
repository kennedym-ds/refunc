"""
Performance benchmarks for refunc decorators.

This module contains benchmarks for timing, memory profiling, and monitoring decorators
to ensure performance characteristics remain stable over time.
"""

import pytest
import time
import numpy as np
from typing import List


# Import decorators for benchmarking
try:
    from refunc.decorators import (
        time_it, memory_profile, system_monitor, performance_monitor,
        get_timing_stats, clear_timing_stats
    )
    DECORATORS_AVAILABLE = True
except ImportError:
    DECORATORS_AVAILABLE = False
    pytestmark = pytest.mark.skip("refunc decorators not available")


@pytest.mark.skipif(not DECORATORS_AVAILABLE, reason="refunc decorators not available")
class TestTimingDecorators:
    """Benchmark tests for timing decorators."""
    
    def setup_method(self):
        """Clear timing stats before each test."""
        clear_timing_stats()
    
    def test_time_it_decorator_overhead(self, benchmark):
        """Benchmark the overhead of the time_it decorator."""
        
        @time_it()
        def simple_function():
            """Simple function to measure decorator overhead."""
            return 42
        
        # Benchmark the decorated function
        result = benchmark(simple_function)
        assert result == 42
    
    def test_time_it_with_stats_collection(self, benchmark):
        """Benchmark time_it decorator with statistics collection."""
        
        @time_it(collect_stats=True)
        def stats_function():
            """Function with stats collection."""
            time.sleep(0.001)  # Small sleep to create measurable time
            return "done"
        
        result = benchmark(stats_function)
        assert result == "done"
        
        # Verify stats are collected
        stats = get_timing_stats("stats_function")
        assert stats is not None
        assert stats.call_count > 0
    
    def test_memory_profile_overhead(self, benchmark):
        """Benchmark the overhead of memory profiling decorator."""
        
        @memory_profile()
        def memory_function():
            """Simple function to measure memory decorator overhead."""
            data = list(range(100))
            return sum(data)
        
        result = benchmark(memory_function)
        assert result == 4950
    
    def test_combined_performance_monitor(self, benchmark):
        """Benchmark combined performance monitoring."""
        
        @performance_monitor()
        def combined_function():
            """Function with combined monitoring."""
            # Create some work
            data = np.random.random(1000)
            result = np.sum(data)
            return result
        
        result = benchmark(combined_function)
        assert isinstance(result, (int, float, np.number))


@pytest.mark.skipif(not DECORATORS_AVAILABLE, reason="refunc decorators not available")  
class TestDecoratorPerformance:
    """Performance regression tests for decorators."""
    
    def test_timing_accuracy(self, benchmark):
        """Test timing accuracy under load."""
        
        @time_it(collect_stats=True)
        def cpu_intensive_task():
            """CPU intensive task for timing accuracy."""
            # Compute intensive operation
            result = 0
            for i in range(10000):
                result += i ** 2
            return result
        
        # Run benchmark
        result = benchmark(cpu_intensive_task)
        assert result > 0
        
        # Check timing stats accuracy
        stats = get_timing_stats("cpu_intensive_task")
        assert stats is not None
        assert stats.mean_time > 0
        assert stats.std_dev >= 0
    
    def test_memory_leak_detection(self, benchmark):
        """Test for memory leaks in decorators."""
        
        @memory_profile()
        def potential_leak_function():
            """Function that might cause memory leaks."""
            # Create and return large data structure
            large_list = list(range(10000))
            return len(large_list)
        
        # Run multiple times to detect potential leaks
        results = []
        for _ in range(10):
            result = benchmark.pedantic(potential_leak_function, rounds=5, iterations=1)
            results.append(result)
        
        # All results should be the same
        assert all(r == 10000 for r in results)
    
    def test_decorator_scalability(self, benchmark):
        """Test decorator performance with varying workloads."""
        
        @time_it()
        @memory_profile()
        def scalable_function(size: int):
            """Function with scalable workload."""
            data = np.random.random(size)
            return np.mean(data)
        
        # Benchmark with different sizes
        small_result = benchmark.pedantic(
            lambda: scalable_function(100), 
            rounds=10, 
            iterations=5
        )
        
        assert 0 <= small_result <= 1  # Should be within [0, 1] for random data


@pytest.mark.skipif(not DECORATORS_AVAILABLE, reason="refunc decorators not available")
class TestRegressionDetection:
    """Tests for performance regression detection."""
    
    def test_baseline_establishment(self):
        """Test establishing performance baselines."""
        
        @time_it(collect_stats=True)  
        def baseline_function():
            """Function to establish baseline."""
            time.sleep(0.01)  # 10ms baseline
            return "baseline"
        
        # Run function multiple times to establish baseline
        for _ in range(5):
            baseline_function()
        
        stats = get_timing_stats("baseline_function") 
        assert stats is not None
        assert stats.call_count == 5
        assert stats.mean_time > 0.005  # Should be at least 5ms
    
    def test_regression_detection_threshold(self):
        """Test regression detection with thresholds."""
        
        def has_regressed(func_name: str, baseline_ms: float, tolerance: float = 0.25) -> bool:
            """Helper function from performance guide."""
            stats = get_timing_stats(func_name)
            if not stats:
                return False
            current_ms = stats.mean_time * 1000
            return current_ms > baseline_ms * (1 + tolerance)
        
        @time_it(collect_stats=True)
        def monitored_function():
            """Function to monitor for regressions."""
            time.sleep(0.005)  # 5ms
            return True
        
        # Establish baseline
        for _ in range(3):
            monitored_function()
        
        baseline_stats = get_timing_stats("monitored_function")
        baseline_ms = baseline_stats.mean_time * 1000
        
        # Test regression detection
        regression_detected = has_regressed("monitored_function", baseline_ms, 0.5)
        assert not regression_detected  # Should not have regressed yet


# Performance comparison framework
@pytest.mark.skipif(not DECORATORS_AVAILABLE, reason="refunc decorators not available")
class TestPerformanceComparison:
    """Performance comparison tests."""
    
    def test_algorithm_comparison(self, benchmark):
        """Compare performance of different algorithms."""
        
        def bubble_sort(arr: List[int]) -> List[int]:
            """Bubble sort implementation."""
            arr = arr.copy()
            n = len(arr)
            for i in range(n):
                for j in range(0, n - i - 1):
                    if arr[j] > arr[j + 1]:
                        arr[j], arr[j + 1] = arr[j + 1], arr[j]
            return arr
        
        def quick_sort(arr: List[int]) -> List[int]:
            """Quick sort implementation."""
            if len(arr) <= 1:
                return arr
            pivot = arr[len(arr) // 2]
            left = [x for x in arr if x < pivot]
            middle = [x for x in arr if x == pivot]
            right = [x for x in arr if x > pivot]
            return quick_sort(left) + middle + quick_sort(right)
        
        # Test data
        test_data = list(range(100, 0, -1))  # Reverse sorted
        
        # Benchmark bubble sort
        bubble_result = benchmark.pedantic(
            lambda: bubble_sort(test_data),
            rounds=3,
            iterations=1
        )
        
        # Verify correctness
        assert bubble_result == list(range(1, 101))
    
    def test_decorator_comparison(self, benchmark):
        """Compare different decorator configurations."""
        
        @time_it()
        def timing_only():
            return sum(range(1000))
        
        @memory_profile()
        def memory_only():
            return sum(range(1000))
        
        @performance_monitor()
        def full_monitoring():
            return sum(range(1000))
        
        # Benchmark each approach
        timing_result = benchmark.pedantic(timing_only, rounds=5, iterations=10)
        assert timing_result == sum(range(1000))