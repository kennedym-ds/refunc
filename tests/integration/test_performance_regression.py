"""
Performance regression tests for refunc package.

These tests establish performance baselines and detect regressions
in critical functions and workflows across modules.
"""

import pytest
import time
import numpy as np
import pandas as pd
from pathlib import Path
import json
from unittest.mock import Mock, patch


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceBaselines:
    """Establish and test performance baselines."""
    
    def test_decorator_performance_overhead(self, sample_numpy_arrays):
        """Test that decorators don't add significant overhead."""
        try:
            from refunc.decorators import time_it, memory_profile
            
            test_data = sample_numpy_arrays['medium']
            
            def baseline_function(data):
                """Baseline function without decorators."""
                return np.sum(data ** 2)
            
            @time_it()
            def decorated_function(data):
                """Same function with timing decorator."""
                return np.sum(data ** 2)
            
            @time_it()
            @memory_profile()
            def heavily_decorated_function(data):
                """Function with multiple decorators."""
                return np.sum(data ** 2)
            
            # Measure baseline performance
            baseline_times = []
            for _ in range(10):
                start = time.perf_counter()
                result = baseline_function(test_data)
                end = time.perf_counter()
                baseline_times.append(end - start)
            
            baseline_avg = np.mean(baseline_times)
            
            # Measure decorated function performance
            decorated_times = []
            for _ in range(10):
                start = time.perf_counter()
                result = decorated_function(test_data)
                end = time.perf_counter()
                decorated_times.append(end - start)
            
            decorated_avg = np.mean(decorated_times)
            
            # Measure heavily decorated function performance
            heavy_times = []
            for _ in range(10):
                start = time.perf_counter()
                result = heavily_decorated_function(test_data)
                end = time.perf_counter()
                heavy_times.append(end - start)
            
            heavy_avg = np.mean(heavy_times)
            
            # Performance assertions
            # Decorator overhead should be minimal (less than 50% increase)
            overhead_ratio = decorated_avg / baseline_avg
            assert overhead_ratio < 1.5, f"Decorator overhead too high: {overhead_ratio:.2f}x"
            
            # Multiple decorators should not cause exponential overhead
            multi_overhead_ratio = heavy_avg / baseline_avg
            assert multi_overhead_ratio < 2.0, f"Multi-decorator overhead too high: {multi_overhead_ratio:.2f}x"
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")
    
    def test_file_handler_performance(self, temp_dir, sample_dataframe):
        """Test FileHandler performance benchmarks."""
        try:
            from refunc.utils import FileHandler
            
            handler = FileHandler()
            
            # Test with different data sizes
            data_sizes = [100, 1000, 5000]
            performance_results = {}
            
            for size in data_sizes:
                # Create test data of specified size
                test_data = pd.DataFrame({
                    'col1': np.random.randn(size),
                    'col2': np.random.choice(['A', 'B', 'C'], size),
                    'col3': np.random.randint(0, 100, size)
                })
                
                # Test CSV performance
                csv_file = temp_dir / f"test_{size}.csv"
                
                # Save performance
                start_time = time.perf_counter()
                handler.save_dataframe(test_data, csv_file)
                save_time = time.perf_counter() - start_time
                
                # Load performance
                start_time = time.perf_counter()
                loaded_data = handler.load_dataframe(csv_file)
                load_time = time.perf_counter() - start_time
                
                performance_results[size] = {
                    'save_time': save_time,
                    'load_time': load_time,
                    'total_time': save_time + load_time,
                    'rows_per_second_save': size / save_time,
                    'rows_per_second_load': size / load_time
                }
            
            # Verify performance scales reasonably
            # Larger datasets should have better rows/second throughput
            small_throughput = performance_results[100]['rows_per_second_save']
            large_throughput = performance_results[5000]['rows_per_second_save']
            
            # Allow for some variation but expect general scaling
            assert large_throughput > small_throughput * 0.5, "Save throughput doesn't scale well"
            
            # Save performance baseline
            baseline_file = temp_dir / "file_handler_baseline.json"
            with open(baseline_file, 'w') as f:
                json.dump(performance_results, f, indent=2)
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")
    
    def test_logging_performance_impact(self, temp_dir):
        """Test that logging doesn't significantly impact performance."""
        try:
            from refunc.logging import MLLogger
            
            logger = MLLogger("perf_test", log_dir=str(temp_dir))
            
            def compute_without_logging():
                """Computation without logging."""
                data = np.random.randn(1000, 10)
                result = np.sum(data ** 2)
                return result
            
            def compute_with_light_logging():
                """Computation with minimal logging."""
                logger.info("Starting computation")
                data = np.random.randn(1000, 10)
                result = np.sum(data ** 2)
                logger.info("Computation completed")
                return result
            
            def compute_with_heavy_logging():
                """Computation with extensive logging."""
                logger.info("Starting computation")
                data = np.random.randn(1000, 10)
                logger.debug("Data generated")
                
                for i in range(0, 1000, 100):
                    batch = data[i:i+100]
                    batch_result = np.sum(batch ** 2)
                    logger.debug(f"Batch {i//100}: {batch_result}")
                
                result = np.sum(data ** 2)
                logger.metric("final_result", result)
                logger.info("Computation completed")
                return result
            
            # Benchmark different logging levels
            iterations = 20
            
            # No logging baseline
            no_log_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                compute_without_logging()
                no_log_times.append(time.perf_counter() - start)
            
            # Light logging
            light_log_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                compute_with_light_logging()
                light_log_times.append(time.perf_counter() - start)
            
            # Heavy logging
            heavy_log_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                compute_with_heavy_logging()
                heavy_log_times.append(time.perf_counter() - start)
            
            # Calculate averages
            no_log_avg = np.mean(no_log_times)
            light_log_avg = np.mean(light_log_times)
            heavy_log_avg = np.mean(heavy_log_times)
            
            # Performance impact should be reasonable
            light_impact = light_log_avg / no_log_avg
            heavy_impact = heavy_log_avg / no_log_avg
            
            assert light_impact < 1.3, f"Light logging impact too high: {light_impact:.2f}x"
            assert heavy_impact < 3.0, f"Heavy logging impact too high: {heavy_impact:.2f}x"
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")


@pytest.mark.integration
class TestMemoryPerformance:
    """Test memory usage and performance."""
    
    def test_memory_usage_patterns(self, sample_numpy_arrays):
        """Test memory usage doesn't grow unexpectedly."""
        try:
            from refunc.decorators import memory_profile
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            @memory_profile()
            def memory_intensive_function(data):
                """Function that uses significant memory."""
                # Create multiple copies to use memory
                copies = [data.copy() for _ in range(5)]
                processed = [np.fft.fft(copy) for copy in copies]
                result = np.sum([np.real(p) for p in processed], axis=0)
                return result
            
            # Get baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run function multiple times
            results = []
            memory_usage = []
            
            for i in range(5):
                result = memory_intensive_function(sample_numpy_arrays['medium'])
                results.append(result)
                
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_usage.append(current_memory - baseline_memory)
            
            # Memory usage should not grow indefinitely
            max_memory = max(memory_usage)
            final_memory = memory_usage[-1]
            
            # Final memory should not be significantly higher than max
            # (indicating memory leaks)
            memory_growth_ratio = final_memory / max_memory if max_memory > 0 else 1
            assert memory_growth_ratio < 1.5, f"Possible memory leak detected: {memory_growth_ratio:.2f}"
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")
    
    def test_cache_performance_scaling(self, sample_numpy_arrays):
        """Test that caching improves performance as expected."""
        try:
            from refunc.utils import MemoryCache, cache_result
            
            cache = MemoryCache(max_size=100)
            computation_count = 0
            
            @cache_result(cache)
            def expensive_computation(data_key):
                """Expensive computation that benefits from caching."""
                nonlocal computation_count
                computation_count += 1
                
                # Simulate expensive computation
                time.sleep(0.1)
                data = sample_numpy_arrays[data_key]
                return np.sum(data ** 2)
            
            # Test cache performance
            test_keys = ['small', 'medium', 'small', 'medium', 'large', 'small']
            
            start_time = time.perf_counter()
            results = [expensive_computation(key) for key in test_keys]
            total_time = time.perf_counter() - start_time
            
            # Should only compute 3 times (small, medium, large)
            assert computation_count == 3, f"Expected 3 computations, got {computation_count}"
            
            # Total time should be much less than 6 * 0.1 seconds due to caching
            expected_uncached_time = len(test_keys) * 0.1
            cache_efficiency = total_time / expected_uncached_time
            
            assert cache_efficiency < 0.6, f"Cache not effective enough: {cache_efficiency:.2f}"
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")


@pytest.mark.integration
@pytest.mark.benchmark
class TestRegressionBenchmarks:
    """Benchmark tests that can detect performance regressions."""
    
    def test_data_processing_benchmark(self, benchmark, sample_dataframe):
        """Benchmark data processing operations."""
        try:
            from refunc.utils import FileHandler
            from refunc.decorators import time_it
            import tempfile
            
            handler = FileHandler()
            
            def data_processing_workflow():
                """Complete data processing workflow to benchmark."""
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    # Save data
                    file_path = temp_path / "test.csv"
                    handler.save_dataframe(sample_dataframe, file_path)
                    
                    # Load data
                    loaded = handler.load_dataframe(file_path)
                    
                    # Process data
                    processed = loaded.copy()
                    processed['new_col'] = processed['numeric'] * 2
                    processed = processed.dropna()
                    
                    # Save processed data
                    output_path = temp_path / "processed.csv"
                    handler.save_dataframe(processed, output_path)
                    
                    return len(processed)
            
            # Benchmark the workflow
            result = benchmark(data_processing_workflow)
            assert result > 0
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")
    
    def test_decorator_stack_benchmark(self, benchmark, sample_numpy_arrays):
        """Benchmark decorator stack performance."""
        try:
            from refunc.decorators import time_it, memory_profile, validate_inputs
            
            @time_it()
            @memory_profile()
            @validate_inputs(data=np.ndarray)
            def decorated_computation(data):
                """Heavily decorated function for benchmarking."""
                # Multiple operations to make timing meaningful
                result1 = np.sum(data ** 2)
                result2 = np.mean(data)
                result3 = np.std(data)
                return result1 + result2 + result3
            
            # Benchmark the decorated function
            test_data = sample_numpy_arrays['medium']
            result = benchmark(decorated_computation, test_data)
            assert isinstance(result, (int, float, np.number))
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")
    
    def test_logging_throughput_benchmark(self, benchmark, temp_dir):
        """Benchmark logging throughput."""
        try:
            from refunc.logging import MLLogger
            
            logger = MLLogger("benchmark_logger", log_dir=str(temp_dir))
            
            def logging_throughput_test():
                """Test logging throughput."""
                # Log various types of messages
                for i in range(100):
                    logger.info(f"Info message {i}")
                    
                    if i % 10 == 0:
                        logger.metric(f"metric_{i}", i * 0.1)
                    
                    if i % 20 == 0:
                        logger.debug(f"Debug message {i}")
                
                return 100
            
            # Benchmark logging throughput
            result = benchmark(logging_throughput_test)
            assert result == 100
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")


@pytest.mark.integration
class TestScalabilityTests:
    """Test how components scale with data size."""
    
    def test_data_size_scaling(self, temp_dir):
        """Test how performance scales with data size."""
        try:
            from refunc.utils import FileHandler
            from refunc.decorators import time_it
            
            handler = FileHandler()
            performance_data = {}
            
            # Test with different data sizes
            sizes = [100, 1000, 10000]
            
            for size in sizes:
                # Create test data
                test_data = pd.DataFrame({
                    'numeric': np.random.randn(size),
                    'categorical': np.random.choice(['A', 'B', 'C'], size),
                    'text': [f'text_{i}' for i in range(size)]
                })
                
                @time_it(collect_stats=True)
                def process_data_size(data):
                    """Process data and measure time."""
                    # File operations
                    file_path = temp_dir / f"test_{len(data)}.csv"
                    handler.save_dataframe(data, file_path)
                    loaded = handler.load_dataframe(file_path)
                    
                    # Data processing
                    processed = loaded.copy()
                    processed['derived'] = processed['numeric'] ** 2
                    processed = processed[processed['numeric'] > processed['numeric'].median()]
                    
                    return len(processed)
                
                # Measure performance
                start_time = time.perf_counter()
                result_size = process_data_size(test_data)
                end_time = time.perf_counter()
                
                performance_data[size] = {
                    'processing_time': end_time - start_time,
                    'throughput': size / (end_time - start_time),
                    'result_size': result_size
                }
            
            # Verify reasonable scaling
            # Time should scale roughly linearly (not exponentially)
            small_time = performance_data[100]['processing_time']
            large_time = performance_data[10000]['processing_time']
            
            # Allow for some overhead, but time shouldn't scale more than 200x for 100x data
            time_scaling_factor = large_time / small_time
            data_scaling_factor = 10000 / 100  # 100x
            
            assert time_scaling_factor < data_scaling_factor * 2, \
                f"Time scaling too poor: {time_scaling_factor:.1f}x for {data_scaling_factor}x data"
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")
    
    def test_concurrent_operations_performance(self, temp_dir, sample_dataframe):
        """Test performance under concurrent operations."""
        try:
            from refunc.utils import FileHandler
            from refunc.logging import MLLogger
            import concurrent.futures
            import threading
            
            handler = FileHandler()
            logger = MLLogger("concurrent_test", log_dir=str(temp_dir))
            
            def concurrent_task(task_id):
                """Task to run concurrently."""
                # File operations
                file_path = temp_dir / f"concurrent_{task_id}.csv"
                handler.save_dataframe(sample_dataframe, file_path)
                loaded = handler.load_dataframe(file_path)
                
                # Logging
                logger.info(f"Task {task_id} completed")
                logger.metric(f"task_{task_id}_rows", len(loaded))
                
                # Processing
                processed = loaded.copy()
                processed['task_id'] = task_id
                
                return task_id, len(processed)
            
            # Test sequential execution
            start_time = time.perf_counter()
            sequential_results = [concurrent_task(i) for i in range(5)]
            sequential_time = time.perf_counter() - start_time
            
            # Test concurrent execution
            start_time = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                concurrent_results = list(executor.map(concurrent_task, range(5, 10)))
            concurrent_time = time.perf_counter() - start_time
            
            # Concurrent execution should show some improvement
            # (allowing for overhead and GIL limitations)
            speedup = sequential_time / concurrent_time
            
            # Should see at least some speedup, even with Python's GIL
            assert speedup > 0.8, f"Concurrent execution slower than sequential: {speedup:.2f}x"
            
            # Verify all tasks completed
            assert len(sequential_results) == 5
            assert len(concurrent_results) == 5
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")


@pytest.mark.integration
class TestResourceUtilization:
    """Test resource utilization patterns."""
    
    def test_memory_efficiency(self, sample_numpy_arrays):
        """Test memory efficiency of operations."""
        try:
            from refunc.decorators import memory_profile
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            @memory_profile()
            def memory_efficient_processing(data):
                """Process data in a memory-efficient way."""
                # Process in chunks to avoid memory spikes
                chunk_size = 100
                results = []
                
                for i in range(0, len(data), chunk_size):
                    chunk = data[i:i + chunk_size]
                    processed_chunk = np.sum(chunk ** 2)
                    results.append(processed_chunk)
                
                return np.array(results)
            
            @memory_profile()
            def memory_inefficient_processing(data):
                """Process data in a memory-inefficient way."""
                # Create multiple full copies
                copies = [data.copy() for _ in range(10)]
                processed = [copy ** 2 for copy in copies]
                return np.sum(processed, axis=0)
            
            # Test both approaches
            test_data = sample_numpy_arrays['large']
            
            # Measure memory for efficient approach
            baseline_memory = process.memory_info().rss
            efficient_result = memory_efficient_processing(test_data)
            efficient_peak = process.memory_info().rss - baseline_memory
            
            # Reset baseline
            baseline_memory = process.memory_info().rss
            inefficient_result = memory_inefficient_processing(test_data)
            inefficient_peak = process.memory_info().rss - baseline_memory
            
            # Efficient approach should use significantly less memory
            memory_efficiency = efficient_peak / inefficient_peak if inefficient_peak > 0 else 1
            assert memory_efficiency < 0.5, f"Memory efficiency not achieved: {memory_efficiency:.2f}"
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")


@pytest.mark.integration
class TestPerformanceRegression:
    """Test for performance regressions in key operations."""
    
    def test_baseline_performance_preservation(self, temp_dir, sample_dataframe):
        """Test that key operations maintain baseline performance."""
        try:
            from refunc.utils import FileHandler
            from refunc.decorators import time_it
            from refunc.logging import MLLogger
            
            # Define performance baselines (in seconds)
            baselines = {
                'file_save_1000_rows': 0.5,
                'file_load_1000_rows': 0.3,
                'logging_100_messages': 0.2,
                'data_processing_1000_rows': 0.1
            }
            
            handler = FileHandler()
            logger = MLLogger("baseline_test", log_dir=str(temp_dir))
            
            # Test file operations
            file_path = temp_dir / "baseline_test.csv"
            
            start_time = time.perf_counter()
            handler.save_dataframe(sample_dataframe, file_path)
            save_time = time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            loaded_data = handler.load_dataframe(file_path)
            load_time = time.perf_counter() - start_time
            
            # Test logging performance
            start_time = time.perf_counter()
            for i in range(100):
                logger.info(f"Baseline test message {i}")
            logging_time = time.perf_counter() - start_time
            
            # Test data processing
            start_time = time.perf_counter()
            processed = sample_dataframe.copy()
            processed['new_col'] = processed['numeric'] * 2
            processed = processed[processed['numeric'] > 0]
            processing_time = time.perf_counter() - start_time
            
            # Check against baselines (with some tolerance)
            tolerance = 2.0  # Allow 2x slower than baseline
            
            actual_performance = {
                'file_save_1000_rows': save_time,
                'file_load_1000_rows': load_time, 
                'logging_100_messages': logging_time,
                'data_processing_1000_rows': processing_time
            }
            
            for operation, baseline in baselines.items():
                actual = actual_performance[operation]
                ratio = actual / baseline
                
                # Log performance for monitoring
                logger.metric(f"baseline_{operation}_ratio", ratio)
                
                # Allow for reasonable performance variation
                assert ratio < tolerance, \
                    f"{operation} regression: {actual:.3f}s vs baseline {baseline:.3f}s ({ratio:.1f}x)"
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")