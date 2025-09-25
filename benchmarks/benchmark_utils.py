"""
Performance benchmarks for refunc core utilities.

This module contains benchmarks for core utility functions including caching,
mathematical operations, and other fundamental operations.
"""

import pytest
import numpy as np
import time
from typing import List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor
import hashlib


# Import core utilities for benchmarking
try:
    from refunc.utils.cache import MemoryCache, DiskCache, cache_result
    from refunc.math_stats import StatisticsEngine
    from refunc.exceptions import retry_on_failure, ValidationError
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    pytestmark = pytest.mark.skip("refunc utilities not available")


@pytest.fixture
def large_numeric_array():
    """Create large numeric array for testing."""
    np.random.seed(42)
    return np.random.random(100000)


@pytest.fixture
def sample_data_dict():
    """Create sample dictionary data."""
    return {
        f'key_{i}': {
            'value': i * 2,
            'metadata': f'data_{i}',
            'array': np.random.random(100).tolist()
        }
        for i in range(1000)
    }


@pytest.mark.skipif(not UTILS_AVAILABLE, reason="refunc utilities not available")
class TestCachePerformance:
    """Benchmark tests for caching systems."""
    
    def test_memory_cache_initialization(self, benchmark):
        """Benchmark memory cache initialization."""
        
        def create_cache():
            return MemoryCache(max_size=10000, ttl_seconds=3600)
        
        cache = benchmark(create_cache)
        assert cache.max_size == 10000
    
    def test_memory_cache_single_operations(self, benchmark):
        """Benchmark single cache operations."""
        
        cache = MemoryCache(max_size=10000)
        
        def single_cache_ops():
            # Set operation
            cache.set("test_key", "test_value")
            
            # Get operation  
            value = cache.get("test_key")
            
            # Check operation
            exists = cache.has("test_key")
            
            return value, exists
        
        result = benchmark(single_cache_ops)
        assert result[0] == "test_value"
        assert result[1] is True
    
    def test_memory_cache_bulk_operations(self, benchmark, sample_data_dict):
        """Benchmark bulk cache operations."""
        
        cache = MemoryCache(max_size=50000)
        
        def bulk_cache_ops():
            # Bulk set
            for key, value in sample_data_dict.items():
                cache.set(key, value)
            
            # Bulk get
            results = []
            for key in sample_data_dict.keys():
                result = cache.get(key)
                results.append(result)
            
            return len(results)
        
        result = benchmark(bulk_cache_ops)
        assert result == len(sample_data_dict)
    
    def test_cache_eviction_performance(self, benchmark):
        """Benchmark cache eviction under memory pressure."""
        
        cache = MemoryCache(max_size=100)  # Small cache to trigger eviction
        
        def cache_eviction_test():
            # Fill cache beyond capacity
            for i in range(200):
                cache.set(f"key_{i}", f"value_{i}" * 100)  # Larger values
            
            # Try to retrieve recent items
            retrieved = 0
            for i in range(150, 200):  # Recent items should still be there
                if cache.get(f"key_{i}") is not None:
                    retrieved += 1
            
            return retrieved
        
        result = benchmark(cache_eviction_test)
        assert result > 0  # Should retrieve some recent items
    
    def test_cache_decorator_performance(self, benchmark):
        """Benchmark cache decorator performance."""
        
        call_count = 0
        
        @cache_result(ttl_seconds=300)
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)  # Simulate expensive operation
            return x ** 2
        
        def cached_function_calls():
            results = []
            # First call - cache miss
            results.append(expensive_function(5))
            # Second call - cache hit
            results.append(expensive_function(5))
            # Third call with different arg - cache miss
            results.append(expensive_function(10))
            # Fourth call - cache hit
            results.append(expensive_function(10))
            
            return results
        
        results = benchmark(cached_function_calls)
        assert results == [25, 25, 100, 100]
        assert call_count >= 2  # Should have been called at least twice


@pytest.mark.skipif(not UTILS_AVAILABLE, reason="refunc utilities not available")
class TestMathStatisticsPerformance:
    """Benchmark tests for mathematical and statistical operations."""
    
    def test_statistics_engine_initialization(self, benchmark):
        """Benchmark StatisticsEngine initialization."""
        
        def create_engine():
            return StatisticsEngine()
        
        engine = benchmark(create_engine)
        assert engine is not None
    
    def test_descriptive_statistics_performance(self, benchmark, large_numeric_array):
        """Benchmark descriptive statistics computation."""
        
        engine = StatisticsEngine()
        
        def compute_descriptive_stats():
            return engine.descriptive_statistics(large_numeric_array)
        
        stats = benchmark(compute_descriptive_stats)
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
    
    def test_percentile_calculations(self, benchmark, large_numeric_array):
        """Benchmark percentile calculations."""
        
        engine = StatisticsEngine()
        
        def compute_percentiles():
            percentiles = [25, 50, 75, 90, 95, 99]
            results = {}
            for p in percentiles:
                results[f'p{p}'] = engine.percentile(large_numeric_array, p)
            return results
        
        percentile_results = benchmark(compute_percentiles)
        assert len(percentile_results) == 6
        assert all(isinstance(v, (int, float, np.number)) for v in percentile_results.values())
    
    def test_correlation_computation(self, benchmark):
        """Benchmark correlation computation."""
        
        np.random.seed(42)
        x = np.random.random(10000)
        y = x + np.random.random(10000) * 0.1  # Correlated data
        
        engine = StatisticsEngine()
        
        def compute_correlation():
            return engine.correlation(x, y, method='pearson')
        
        correlation = benchmark(compute_correlation)
        assert isinstance(correlation, (float, np.number))
        assert -1 <= correlation <= 1
    
    def test_hypothesis_testing_performance(self, benchmark, large_numeric_array):
        """Benchmark hypothesis testing."""
        
        engine = StatisticsEngine()
        
        # Split data into two groups
        mid_point = len(large_numeric_array) // 2
        group1 = large_numeric_array[:mid_point]
        group2 = large_numeric_array[mid_point:]
        
        def run_t_test():
            return engine.t_test_independent(group1, group2)
        
        test_result = benchmark(run_t_test)
        assert 'statistic' in test_result
        assert 'p_value' in test_result
    
    def test_bootstrap_confidence_intervals(self, benchmark):
        """Benchmark bootstrap confidence interval computation."""
        
        np.random.seed(42)
        sample_data = np.random.normal(50, 10, 1000)  # Smaller sample for bootstrap
        
        engine = StatisticsEngine()
        
        def bootstrap_ci():
            return engine.bootstrap_confidence_interval(
                sample_data,
                statistic_func=np.mean,
                confidence_level=0.95,
                n_bootstrap=1000
            )
        
        ci_result = benchmark(bootstrap_ci)
        assert 'lower' in ci_result
        assert 'upper' in ci_result
        assert 'confidence_level' in ci_result


@pytest.mark.skipif(not UTILS_AVAILABLE, reason="refunc utilities not available")
class TestExceptionHandlingPerformance:
    """Benchmark tests for exception handling utilities."""
    
    def test_retry_decorator_success(self, benchmark):
        """Benchmark retry decorator when function succeeds."""
        
        call_count = 0
        
        @retry_on_failure(max_attempts=3, delay=0.001)
        def reliable_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        def retry_success():
            nonlocal call_count
            call_count = 0
            return reliable_function()
        
        result = benchmark(retry_success)
        assert result == "success"
        assert call_count == 1  # Should succeed on first try
    
    def test_retry_decorator_with_failures(self, benchmark):
        """Benchmark retry decorator with initial failures."""
        
        call_count = 0
        
        @retry_on_failure(max_attempts=3, delay=0.001)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Simulated failure")
            return "success"
        
        def retry_with_failure():
            nonlocal call_count
            call_count = 0
            return flaky_function()
        
        result = benchmark(retry_with_failure)
        assert result == "success"
        assert call_count == 2  # Should succeed on second try
    
    def test_retry_performance_under_load(self, benchmark):
        """Test retry performance under concurrent load."""
        
        success_count = 0
        
        @retry_on_failure(max_attempts=2, delay=0.001)
        def concurrent_function(thread_id: int):
            nonlocal success_count
            # Simulate occasional failures
            if thread_id % 5 == 0:  # 20% failure rate
                raise ValueError("Simulated failure")
            success_count += 1
            return f"thread_{thread_id}_success"
        
        def concurrent_retry_test():
            results = []
            for i in range(50):  # 50 simulated concurrent calls
                try:
                    result = concurrent_function(i)
                    results.append(result)
                except Exception:
                    results.append("failed")
            return len([r for r in results if "success" in r])
        
        successful_calls = benchmark(concurrent_retry_test)
        assert successful_calls > 0


@pytest.mark.skipif(not UTILS_AVAILABLE, reason="refunc utilities not available")
class TestConcurrencyPerformance:
    """Benchmark tests for concurrent operations."""
    
    def test_concurrent_cache_access(self, benchmark):
        """Benchmark concurrent cache access."""
        
        cache = MemoryCache(max_size=10000)
        
        def concurrent_cache_test():
            def worker(thread_id: int):
                results = []
                for i in range(100):
                    key = f"thread_{thread_id}_key_{i}"
                    value = f"thread_{thread_id}_value_{i}"
                    
                    # Set value
                    cache.set(key, value)
                    
                    # Get value
                    retrieved = cache.get(key)
                    results.append(retrieved == value)
                
                return sum(results)
            
            # Use ThreadPoolExecutor for concurrent access
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(worker, i) for i in range(5)]
                results = [future.result() for future in futures]
            
            return sum(results)
        
        total_success = benchmark(concurrent_cache_test)
        assert total_success > 400  # Should succeed most of the time
    
    def test_concurrent_statistics_computation(self, benchmark):
        """Benchmark concurrent statistics computation."""
        
        engine = StatisticsEngine()
        
        def concurrent_stats_test():
            def compute_stats(seed: int):
                np.random.seed(seed)
                data = np.random.random(5000)
                return engine.descriptive_statistics(data)
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(compute_stats, i) for i in range(10)]
                results = [future.result() for future in futures]
            
            return len(results)
        
        result_count = benchmark(concurrent_stats_test)
        assert result_count == 10


@pytest.mark.skipif(not UTILS_AVAILABLE, reason="refunc utilities not available")
class TestMemoryEfficiency:
    """Benchmark tests for memory efficiency."""
    
    def test_memory_efficient_iteration(self, benchmark):
        """Test memory-efficient iteration patterns."""
        
        def memory_efficient_processing():
            # Process data in chunks to avoid large memory usage
            total_sum = 0
            chunk_size = 10000
            
            for chunk_start in range(0, 100000, chunk_size):
                chunk_end = min(chunk_start + chunk_size, 100000)
                chunk_data = np.random.random(chunk_end - chunk_start)
                total_sum += np.sum(chunk_data)
                
                # Clear chunk to free memory
                del chunk_data
            
            return total_sum
        
        result = benchmark(memory_efficient_processing)
        assert isinstance(result, (int, float, np.number))
        assert result > 0
    
    def test_cache_memory_management(self, benchmark):
        """Test cache memory management under pressure."""
        
        def cache_memory_test():
            cache = MemoryCache(max_size=1000)  # Limited size cache
            
            # Fill cache with increasingly large objects
            for i in range(2000):  # More items than cache can hold
                large_object = {
                    'id': i,
                    'data': list(range(i % 100)),  # Variable size objects
                    'metadata': f'object_{i}'
                }
                cache.set(f'large_obj_{i}', large_object)
            
            # Count how many items are still in cache
            count = 0
            for i in range(1900, 2000):  # Check recent items
                if cache.get(f'large_obj_{i}') is not None:
                    count += 1
            
            return count
        
        cached_items = benchmark(cache_memory_test)
        assert cached_items > 0  # Should retain some recent items
    
    def test_efficient_data_structures(self, benchmark):
        """Benchmark efficient data structure operations."""
        
        def efficient_operations():
            # Use efficient data structures for common operations
            results = {}
            
            # Efficient set operations
            large_set = set(range(50000))
            results['set_lookup'] = 25000 in large_set
            
            # Efficient dictionary operations
            large_dict = {i: i ** 2 for i in range(50000)}
            results['dict_lookup'] = large_dict.get(25000, 0)
            
            # Efficient list comprehension
            squares = [x ** 2 for x in range(1000)]
            results['list_comp'] = len(squares)
            
            return results
        
        results = benchmark(efficient_operations)
        assert results['set_lookup'] is True
        assert results['dict_lookup'] == 25000 ** 2
        assert results['list_comp'] == 1000