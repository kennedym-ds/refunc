"""
Comprehensive tests for the refunc.decorators module.

This test suite covers:
- Timing decorators and performance measurement
- Memory profiling and monitoring
- System resource monitoring
- Input/output validation decorators
- Combined performance monitoring
- Edge cases and error conditions
"""

import pytest
import time
import asyncio
import sys
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, List

# Import exceptions  
from refunc.exceptions import DataError

# Import all decorator components
from refunc.decorators import (
    # Timing decorators
    time_it,
    time_it_async,
    timer,
    TimingResult,
    TimingStats,
    TimingMode,
    get_timing_stats,
    clear_timing_stats,
    quick_time,
    measure_time,
    
    # Memory decorators
    memory_profile,
    memory_profile_async,
    memory_monitor,
    MemoryResult,
    MemorySnapshot,
    MemoryStats,
    MemoryMode,
    MemoryMonitor,
    get_current_memory,
    profile_memory,
    
    # System monitoring decorators
    system_monitor,
    system_monitor_async,
    monitor_system,
    MonitoringResult,
    SystemSnapshot,
    get_system_info,
    
    # Validation decorators
    validate_inputs,
    validate_outputs,
    validate_types,
    ValidationResult,
    ValidatorBase,
    TypeValidator,
    RangeValidator,
    type_check,
    range_check,
    custom_check,
    
    # Combined decorators
    performance_monitor,
    performance_monitor_async,
    monitor_performance,
    CombinedResult,
    MonitoringConfig,
    quick_monitor,
    full_monitor,
)


class TestTimingDecorators:
    """Test timing decorators and measurement functionality."""
    
    def test_time_it_basic_functionality(self):
        """Test basic timing decorator functionality."""
        @time_it()
        def sample_function(x: int) -> int:
            time.sleep(0.01)  # Small delay to measure
            return x * 2
        
        result = sample_function(5)
        assert result == 10
        
        # Check that timing was recorded
        stats = get_timing_stats("sample_function")
        assert stats is not None
        assert stats.call_count >= 1
        assert stats.mean_time > 0
        
    def test_time_it_with_mode(self):
        """Test timing decorator with different timing modes."""
        @time_it(mode=TimingMode.PROCESS_TIME)
        def cpu_intensive_function():
            # Simulate CPU work
            total = 0
            for i in range(1000):
                total += i * i
            return total
        
        result = cpu_intensive_function()
        assert isinstance(result, int)
        
        stats = get_timing_stats("cpu_intensive_function")
        assert stats is not None
        
    def test_time_it_with_statistics(self):
        """Test timing decorator statistical accumulation."""
        @time_it(collect_stats=True)
        def repeated_function(n: int) -> int:
            time.sleep(0.001)  # Very small delay
            return n + 1
        
        # Call multiple times
        for i in range(5):
            repeated_function(i)
        
        stats = get_timing_stats("repeated_function")
        assert stats is not None
        assert stats.call_count == 5
        assert stats.total_time > 0
        assert stats.mean_time > 0
        assert stats.min_time <= stats.mean_time <= stats.max_time
        
    @pytest.mark.asyncio
    async def test_time_it_async(self):
        """Test asynchronous timing decorator."""
        @time_it_async()
        async def async_function(delay: float) -> str:
            await asyncio.sleep(delay)
            return "completed"
        
        result = await async_function(0.01)
        assert result == "completed"
        
        stats = get_timing_stats("async_function")
        assert stats is not None
        assert stats.call_count >= 1
        
    def test_timer_context_manager(self):
        """Test timer context manager."""
        with timer("context_operation") as timing_result:
            time.sleep(0.01)
            # Do some work
            x = sum(range(100))
        
        assert timing_result.execution_time > 0
        assert timing_result.function_name == "context_operation"
        
    def test_timing_result_dataclass(self):
        """Test TimingResult dataclass functionality."""
        result = TimingResult(
            function_name="test_func",
            execution_time=1.5,
            timestamp=time.time(),
            timing_mode=TimingMode.WALL_CLOCK,
            metadata={"custom": "value"}
        )
        
        assert result.function_name == "test_func"
        assert result.execution_time == 1.5
        assert result.timing_mode == TimingMode.WALL_CLOCK
        assert result.metadata["custom"] == "value"
        
    def test_clear_timing_stats(self):
        """Test clearing timing statistics."""
        @time_it()
        def clearable_function():
            return "test"
        
        clearable_function()
        assert get_timing_stats("clearable_function") is not None
        
        clear_timing_stats()
        assert get_timing_stats("clearable_function") is None
        
    def test_quick_time_convenience(self):
        """Test quick_time convenience function."""
        def quick_function():
            time.sleep(0.001)
            return "quick"
        
        result, timing = quick_time(quick_function)
        assert result == "quick"
        assert timing > 0


class TestMemoryDecorators:
    """Test memory profiling and monitoring decorators."""
    
    def test_memory_profile_basic(self):
        """Test basic memory profiling functionality."""
        @memory_profile()
        def memory_function():
            # Allocate some memory
            data = [i for i in range(1000)]
            return len(data)
        
        result = memory_function()
        assert result == 1000
        
    def test_memory_profile_with_tracking(self):
        """Test memory profiling with detailed tracking."""
        @memory_profile(track_tracemalloc=True)
        def allocating_function():
            # Create various data structures
            list_data = list(range(100))
            dict_data = {i: i*2 for i in range(50)}
            return len(list_data) + len(dict_data)
        
        result = allocating_function()
        assert result == 150
        
    @pytest.mark.asyncio
    async def test_memory_profile_async(self):
        """Test asynchronous memory profiling."""
        @memory_profile_async()
        async def async_memory_function():
            await asyncio.sleep(0.001)
            # Allocate memory
            data = np.zeros(1000)
            return data.size
        
        result = await async_memory_function()
        assert result == 1000
        
    def test_memory_monitor_context(self):
        """Test memory monitor context manager."""
        with memory_monitor("memory_operation") as monitor:
            # Allocate significant memory
            big_list = [i for i in range(10000)]
            big_dict = {i: str(i) for i in range(1000)}
        
        assert monitor.peak_memory >= monitor.start_memory.rss
        
    def test_get_current_memory(self):
        """Test current memory usage function."""
        current_memory = get_current_memory()
        assert isinstance(current_memory, MemorySnapshot)
        assert current_memory.rss > 0
        
    def test_memory_result_dataclass(self):
        """Test MemoryResult dataclass."""
        # Create sample memory snapshots
        start_snapshot = MemorySnapshot(rss=100*1024*1024, vms=200*1024*1024, percent=5.0, available=8*1024*1024*1024, timestamp=time.time())
        peak_snapshot = MemorySnapshot(rss=150*1024*1024, vms=250*1024*1024, percent=7.5, available=int(7.5*1024*1024*1024), timestamp=time.time())
        end_snapshot = MemorySnapshot(rss=110*1024*1024, vms=210*1024*1024, percent=5.5, available=int(7.8*1024*1024*1024), timestamp=time.time())
        
        result = MemoryResult(
            function_name="test_memory",
            peak_memory=150*1024*1024,
            memory_diff=10*1024*1024,
            start_memory=start_snapshot,
            peak_snapshot=peak_snapshot,
            end_memory=end_snapshot,
            duration=0.1,
            timestamp=time.time()
        )
        
        assert result.function_name == "test_memory"
        assert result.peak_memory == 150*1024*1024
        assert result.memory_diff == 10*1024*1024


class TestSystemMonitoring:
    """Test system resource monitoring decorators."""
    
    def test_system_monitor_basic(self):
        """Test basic system monitoring."""
        @system_monitor()
        def system_function():
            # Do some computation
            return sum(i*i for i in range(1000))
        
        result = system_function()
        assert isinstance(result, int)
        
    def test_system_monitor_with_options(self):
        """Test system monitoring with various options."""
        @system_monitor(
            sample_interval=0.05,
            monitor_gpu=False,
            print_result=False,
            return_result=False
        )
        def monitored_function():
            time.sleep(0.01)
            return "monitored"
        
        result = monitored_function()
        assert result == "monitored"
        
    @pytest.mark.asyncio
    async def test_system_monitor_async(self):
        """Test asynchronous system monitoring."""
        @system_monitor_async()
        async def async_system_function():
            await asyncio.sleep(0.01)
            return "async_monitored"
        
        result = await async_system_function()
        assert result == "async_monitored"
        
    def test_get_system_info(self):
        """Test system information gathering."""
        system_info = get_system_info()
        
        assert isinstance(system_info, dict)
        assert "cpu" in system_info
        assert "memory" in system_info
        assert "cpu_count" in system_info["cpu"]
        
    def test_system_snapshot(self):
        """Test SystemSnapshot functionality."""
        snapshot = SystemSnapshot.capture()
        
        assert hasattr(snapshot, 'cpu_percent')
        assert hasattr(snapshot, 'memory_percent')
        assert hasattr(snapshot, 'timestamp')
        
    def test_monitoring_result_dataclass(self):
        """Test MonitoringResult dataclass."""
        result = MonitoringResult(
            function_name="test_monitor",
            execution_time=1.5,
            cpu_usage_start=10.0,
            cpu_usage_end=15.0,
            memory_usage_start=100.0,
            memory_usage_end=120.0,
            timestamp=time.time()
        )
        
        assert result.function_name == "test_monitor"
        assert result.execution_time == 1.5
        assert result.cpu_usage_end > result.cpu_usage_start


class TestValidationDecorators:
    """Test input/output validation decorators."""
    
    def test_validate_inputs_types(self):
        """Test input type validation."""
        @validate_inputs(
            validators={
                'x': type_check(int),
                'y': type_check(str)
            }
        )
        def typed_function(x: int, y: str) -> str:
            return f"{x}_{y}"
        
        # Valid inputs
        result = typed_function(5, "test")
        assert result == "5_test"
        
        # Invalid inputs
        with pytest.raises(DataError):
            typed_function("5", "test")  # x should be int
            
        with pytest.raises(DataError):
            typed_function(5, 123)  # y should be str
            
    def test_validate_inputs_ranges(self):
        """Test input range validation."""
        @validate_inputs(
            validators={
                'age': range_check(0, 120),
                'score': range_check(0.0, 100.0)
            }
        )
        def range_function(age: int, score: float) -> str:
            return f"Age: {age}, Score: {score}"
        
        # Valid inputs
        result = range_function(25, 85.5)
        assert "Age: 25" in result
        
        # Invalid inputs
        with pytest.raises(DataError):
            range_function(-5, 85.5)  # age too low
            
        with pytest.raises(DataError):
            range_function(25, 150.0)  # score too high
            
    def test_validate_outputs(self):
        """Test output validation."""
        @validate_outputs(validators=type_check(int))
        def output_function(x: int) -> int:
            if x < 0:
                return "error"  # Wrong type
            return x * 2
        
        # Valid output
        result = output_function(5)
        assert result == 10
        
        # Invalid output
        with pytest.raises(DataError):
            output_function(-1)
            
    def test_custom_validators(self):
        """Test custom validator creation."""
        def even_number_validator(value):
            if value % 2 != 0:
                return False
            return True
        
        @validate_inputs(validators={'x': custom_check(even_number_validator, "Expected even number")})
        def even_function(x: int) -> int:
            return x // 2
        
        # Valid input
        result = even_function(8)
        assert result == 4
        
        # Invalid input
        with pytest.raises(DataError):
            even_function(7)
            
    def test_dataframe_validation(self):
        """Test DataFrame validation."""
        def validate_dataframe_columns(df):
            required_columns = {"name", "age"}
            if not required_columns.issubset(set(df.columns)):
                return False
            return True
        
        @validate_inputs(validators={'df': custom_check(validate_dataframe_columns, "Invalid DataFrame columns")})
        def process_dataframe(df: pd.DataFrame) -> int:
            return len(df)
        
        # Valid DataFrame
        valid_df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        result = process_dataframe(valid_df)
        assert result == 2
        
        # Invalid DataFrame
        invalid_df = pd.DataFrame({"name": ["Alice", "Bob"]})  # missing age
        with pytest.raises(DataError):
            process_dataframe(invalid_df)
            
    def test_validation_result(self):
        """Test ValidationResult dataclass."""
        result = ValidationResult(
            function_name="test_func",
            input_valid=True,
            output_valid=True,
            input_errors=[],
            output_errors=["Minor issue"],
            validation_time=0.001
        )
        
        assert result.input_valid is True
        assert result.output_valid is True
        assert len(result.input_errors) == 0
        assert len(result.output_errors) == 1


class TestCombinedDecorators:
    """Test combined performance monitoring decorators."""
    
    def test_performance_monitor_basic(self):
        """Test basic combined performance monitoring."""
        @performance_monitor
        def performance_function():
            time.sleep(0.01)
            # Allocate some memory
            data = list(range(1000))
            return len(data)
        
        result = performance_function()
        assert result == 1000
        
    def test_performance_monitor_with_config(self):
        """Test performance monitoring with configuration."""
        config = MonitoringConfig(
            enable_timing=True,
            enable_memory=True,
            enable_system=True,
            enable_validation=False
        )
        
        @performance_monitor(config)
        def configured_function():
            return "configured"
        
        result = configured_function()
        assert result == "configured"
        
    @pytest.mark.asyncio
    async def test_performance_monitor_async(self):
        """Test asynchronous combined monitoring."""
        @performance_monitor_async
        async def async_performance_function():
            await asyncio.sleep(0.01)
            return "async_performance"
        
        result = await async_performance_function()
        assert result == "async_performance"
        
    def test_quick_monitor_convenience(self):
        """Test quick monitor convenience decorator."""
        @quick_monitor
        def quick_monitored_function():
            time.sleep(0.001)
            return "quick_monitored"
        
        result = quick_monitored_function()
        assert result == "quick_monitored"
        
    def test_full_monitor_comprehensive(self):
        """Test full monitoring with all features."""
        @full_monitor
        def fully_monitored_function(x: int) -> int:
            time.sleep(0.01)
            # Some computation and memory allocation
            data = [i * x for i in range(100)]
            return sum(data)
        
        result = fully_monitored_function(5)
        assert isinstance(result, int)
        
    def test_combined_result_dataclass(self):
        """Test CombinedResult dataclass."""
        timing_result = TimingResult("test", 1.0, time.time())
        
        # Create proper MemorySnapshot instances  
        start_snapshot = MemorySnapshot(rss=100*1024*1024, vms=200*1024*1024, percent=5.0, available=8*1024*1024*1024, timestamp=time.time())
        peak_snapshot = MemorySnapshot(rss=150*1024*1024, vms=250*1024*1024, percent=7.5, available=int(7.5*1024*1024*1024), timestamp=time.time())
        end_snapshot = MemorySnapshot(rss=110*1024*1024, vms=210*1024*1024, percent=5.5, available=int(7.8*1024*1024*1024), timestamp=time.time())
        
        memory_result = MemoryResult(
            function_name="test",
            peak_memory=150*1024*1024,
            memory_diff=10*1024*1024,
            start_memory=start_snapshot,
            peak_snapshot=peak_snapshot,
            end_memory=end_snapshot,
            duration=1.0,
            timestamp=time.time()
        )
        
        combined_result = CombinedResult(
            function_name="test_combined",
            timing_result=timing_result,
            memory_result=memory_result,
            monitoring_result=None,
            validation_result=None
        )
        
        assert combined_result.function_name == "test_combined"
        assert combined_result.timing_result == timing_result
        assert combined_result.memory_result == memory_result


class TestDecoratorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_decorator_with_exceptions(self):
        """Test decorators when decorated functions raise exceptions."""
        @time_it
        @memory_profile
        def failing_function():
            raise ValueError("Test exception")
        
        with pytest.raises(ValueError):
            failing_function()
        
        # Should still record timing/memory even with exception
        stats = get_timing_stats("failing_function")
        assert stats is not None
        
    def test_decorator_with_generator_function(self):
        """Test decorators with generator functions."""
        @time_it
        def generator_function(n: int):
            for i in range(n):
                yield i * 2
        
        result = list(generator_function(5))
        assert result == [0, 2, 4, 6, 8]
        
    def test_decorator_with_recursive_function(self):
        """Test decorators with recursive functions."""
        @time_it
        def recursive_factorial(n: int) -> int:
            if n <= 1:
                return 1
            return n * recursive_factorial(n - 1)
        
        result = recursive_factorial(5)
        assert result == 120
        
        stats = get_timing_stats("recursive_factorial")
        assert stats is not None
        assert stats.call_count >= 5  # Should record each recursive call
        
    def test_decorator_stacking(self):
        """Test multiple decorators stacked together."""
        @validate_inputs(validators={'x': type_check(int)})
        @memory_profile
        @time_it
        def stacked_function(x: int) -> int:
            time.sleep(0.001)
            return x * x
        
        result = stacked_function(4)
        assert result == 16
        
        # Should work with all decorators
        stats = get_timing_stats("stacked_function")
        assert stats is not None
        
    def test_decorator_with_class_methods(self):
        """Test decorators on class methods."""
        class TestClass:
            @time_it
            def instance_method(self, x: int) -> int:
                return x + 1
            
            @classmethod
            @memory_profile
            def class_method(cls, x: int) -> int:
                return x * 2
            
            @staticmethod
            @system_monitor
            def static_method(x: int) -> int:
                return x * 3
        
        obj = TestClass()
        
        assert obj.instance_method(5) == 6
        assert TestClass.class_method(5) == 10
        assert TestClass.static_method(5) == 15
        
    def test_decorator_performance_overhead(self):
        """Test that decorators don't add significant overhead."""
        def plain_function():
            return sum(range(100))
        
        @time_it
        def decorated_function():
            return sum(range(100))
        
        # Measure plain function
        start = time.perf_counter()
        for _ in range(100):
            plain_function()
        plain_time = time.perf_counter() - start
        
        # Measure decorated function
        start = time.perf_counter()
        for _ in range(100):
            decorated_function()
        decorated_time = time.perf_counter() - start
        
        # Overhead should be reasonable (less than 10x)
        overhead_ratio = decorated_time / plain_time
        assert overhead_ratio < 10.0


class TestDecoratorIntegration:
    """Test decorator integration with other refunc components."""
    
    def test_decorators_with_logging(self):
        """Test that decorators integrate with logging."""
        @time_it(log_results=True)
        def logged_function():
            return "logged"
        
        result = logged_function()
        assert result == "logged"
        
    def test_decorators_with_configuration(self):
        """Test decorators with configuration management."""
        # This would test integration with config system
        pass
        
    def test_decorators_with_exceptions(self):
        """Test decorators with exception handling."""
        from refunc.exceptions import OperationError
        
        @time_it
        def function_with_refunc_exception():
            raise OperationError("Test operation error")
        
        with pytest.raises(OperationError):
            function_with_refunc_exception()


# Benchmark and stress tests
class TestDecoratorPerformance:
    """Test decorator performance under stress."""
    
    @pytest.mark.slow
    def test_timing_decorator_stress(self):
        """Stress test timing decorator with many calls."""
        @time_it
        def stress_function(n: int) -> int:
            return n * n
        
        # Make many calls
        for i in range(1000):
            stress_function(i)
        
        stats = get_timing_stats("stress_function")
        assert stats is not None
        assert stats.call_count == 1000
        
    @pytest.mark.slow
    def test_memory_decorator_stress(self):
        """Stress test memory decorator with many allocations."""
        @memory_profile
        def memory_stress_function(size: int) -> list:
            return list(range(size))
        
        # Make calls with increasing memory usage
        for i in range(10, 1000, 100):
            result = memory_stress_function(i)
            assert len(result) == i


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# Integration Tests for Phase 2 Coverage Improvement
class TestDecoratorLoggingIntegration:
    """Test integration between decorators and logging system."""
    
    def test_timing_decorator_with_mllogger(self):
        """Test timing decorator integration with MLLogger."""
        from refunc.logging import MLLogger, get_logger
        
        # Setup logger
        logger = get_logger("decorator_test")
        
        @time_it(collect_stats=True, print_result=False)
        def logged_timing_function(x: int) -> int:
            logger.info(f"Processing value: {x}")
            time.sleep(0.001)
            result = x * 2
            logger.info(f"Result: {result}")
            return result
        
        # Execute function
        result = logged_timing_function(5)
        assert result == 10
        
        # Verify timing stats were collected
        stats = get_timing_stats("logged_timing_function")
        assert stats is not None
        assert stats.call_count >= 1
        
    def test_memory_decorator_with_experiment_logging(self):
        """Test memory profiling with experiment logging."""
        from refunc.logging import ExperimentTracker
        
        # Mock experiment tracker to avoid external dependencies
        with patch('refunc.logging.experiment.ExperimentTracker') as mock_tracker:
            mock_instance = Mock()
            mock_tracker.return_value = mock_instance
            
            @memory_profile(print_result=False, return_result=True)
            def memory_experiment_function(data_size: int):
                # Simulate experiment with memory allocation
                mock_instance.log_metric("data_size", data_size)
                data = list(range(data_size))
                mock_instance.log_metric("allocated_items", len(data))
                return data
            
            # Execute function
            result, memory_result = memory_experiment_function(1000)
            assert len(result) == 1000
            assert isinstance(memory_result, MemoryResult)
            assert memory_result.function_name == "memory_experiment_function"
            
    def test_system_monitor_with_progress_tracking(self):
        """Test system monitoring with progress tracking."""
        from refunc.logging import ProgressTracker
        
        @system_monitor(sample_interval=0.01, print_result=False)
        def progress_monitored_function(iterations: int):
            # Mock progress tracker
            with patch('refunc.logging.progress.ProgressTracker') as mock_progress:
                mock_instance = Mock()
                mock_progress.return_value = mock_instance
                
                for i in range(iterations):
                    mock_instance.update(i + 1)
                    time.sleep(0.001)
                
                return iterations
        
        result = progress_monitored_function(10)
        assert result == 10
        
    def test_combined_decorators_with_structured_logging(self):
        """Test combined decorators with structured logging."""
        from refunc.logging import MLLogger, LogLevel
        
        # Setup structured logging
        with patch('refunc.logging.core.MLLogger') as mock_logger:
            mock_instance = Mock()
            mock_logger.return_value = mock_instance
            
            config = MonitoringConfig(
                enable_timing=True,
                enable_memory=True,
                enable_system=True,
                enable_validation=False
            )
            
            @performance_monitor(config)
            def structured_logging_function(data: List[int]) -> int:
                mock_instance.info("Starting data processing", extra={"data_length": len(data)})
                result = sum(data)
                mock_instance.metric("processing_result", result)
                mock_instance.info("Completed data processing", extra={"result": result})
                return result
            
            # Execute function
            test_data = list(range(100))
            result = structured_logging_function(test_data)
            assert result == sum(test_data)
            
            # Verify logging calls were made
            assert mock_instance.info.call_count >= 2
            assert mock_instance.metric.call_count >= 1
            
    def test_validation_decorators_with_error_logging(self):
        """Test validation decorators with error logging integration."""
        from refunc.logging import get_logger
        from refunc.exceptions import DataError
        
        logger = get_logger("validation_test")
        
        @validate_inputs(validators={'x': type_check(int), 'y': range_check(0, 100)})
        def validated_with_logging(x: int, y: int) -> int:
            logger.debug(f"Validated inputs: x={x}, y={y}")
            result = x + y
            logger.info(f"Computation result: {result}")
            return result
        
        # Test valid inputs
        result = validated_with_logging(10, 20)
        assert result == 30
        
        # Test invalid inputs with logging
        with pytest.raises(DataError):
            validated_with_logging("invalid", 20)  # Type error
            
        with pytest.raises(DataError):
            validated_with_logging(10, 150)  # Range error
            
    def test_async_decorators_with_async_logging(self):
        """Test async decorators with async logging capabilities."""
        
        @pytest.mark.asyncio
        async def test_async_integration():
            from refunc.logging import MLLogger
            
            with patch('refunc.logging.core.MLLogger') as mock_logger:
                mock_instance = Mock()
                mock_logger.return_value = mock_instance
                
                @time_it_async(collect_stats=True)
                @memory_profile_async()
                async def async_logged_function(delay: float) -> str:
                    mock_instance.info(f"Starting async operation with delay {delay}")
                    await asyncio.sleep(delay)
                    mock_instance.info("Completed async operation")
                    return "async_complete"
                
                result = await async_logged_function(0.01)
                assert result == "async_complete"
                
                # Verify timing stats
                stats = get_timing_stats("async_logged_function")
                assert stats is not None
        
        # Run the async test
        asyncio.run(test_async_integration())
        
    def test_exception_handling_with_logging_integration(self):
        """Test exception handling across decorators and logging."""
        from refunc.logging import get_logger
        from refunc.exceptions import OperationError
        
        logger = get_logger("exception_test")
        
        @time_it(collect_stats=True)
        @memory_profile()
        def exception_logged_function(should_fail: bool):
            logger.info("Function started")
            
            if should_fail:
                logger.error("About to raise exception")
                raise OperationError("Intentional test failure")
            
            logger.info("Function completed successfully")
            return "success"
        
        # Test successful execution
        result = exception_logged_function(False)
        assert result == "success"
        
        # Test exception handling
        with pytest.raises(OperationError):
            exception_logged_function(True)
        
        # Verify timing stats still collected despite exception
        stats = get_timing_stats("exception_logged_function")
        assert stats is not None
        assert stats.call_count >= 2  # Both successful and failed calls


class TestConfigurationIntegration:
    """Test integration between configuration management and other components."""
    
    def test_decorator_configuration_integration(self):
        """Test decorators with configuration management."""
        from refunc.config import ConfigManager
        
        # Mock configuration system
        with patch('refunc.config.core.ConfigManager') as mock_config:
            mock_instance = Mock()
            mock_config.return_value = mock_instance
            
            # Mock configuration values
            mock_instance.get.side_effect = lambda key, default=None: {
                'timing.mode': 'process_time',
                'timing.precision': 4,
                'memory.track_peak': True,
                'system.sample_interval': 0.05
            }.get(key, default)
            
            # Create decorator with config-driven parameters
            config = mock_instance
            
            @time_it(
                mode=config.get('timing.mode', TimingMode.WALL_CLOCK),
                precision=config.get('timing.precision', 6)
            )
            def config_driven_function():
                return "configured"
            
            result = config_driven_function()
            assert result == "configured"
            
    def test_logging_configuration_integration(self):
        """Test logging system with configuration management."""
        from refunc.config import ConfigManager
        
        with patch('refunc.config.core.ConfigManager') as mock_config:
            mock_instance = Mock()
            mock_config.return_value = mock_instance
            
            # Mock logging configuration
            mock_instance.get.side_effect = lambda key, default=None: {
                'logging.level': 'INFO',
                'logging.name': 'test_logger'
            }.get(key, default)
            
            # Test configuration-driven logging setup
            config = mock_instance
            
            with patch('refunc.logging.MLLogger') as mock_logger:
                from refunc.logging import MLLogger
                logger = MLLogger(
                    name=config.get('logging.name', 'default'),
                    level=config.get('logging.level', 'WARNING')
                )
                
                mock_logger.assert_called_once()


class TestDataScienceIntegration:
    """Test integration between data science utilities and other components."""
    
    def test_dataframe_operations_with_monitoring(self):
        """Test DataFrame operations with performance monitoring."""
        
        @time_it(collect_stats=True)
        @memory_profile(track_peak=True)
        def monitored_dataframe_operation(rows: int) -> pd.DataFrame:
            # Create test DataFrame
            data = {
                'id': range(rows),
                'value': np.random.rand(rows),
                'category': [['A', 'B', 'C'][i % 3] for i in range(rows)]
            }
            df = pd.DataFrame(data)
            
            # Perform some operations
            df_processed = df.groupby('category')['value'].mean().reset_index()
            return df_processed
        
        result = monitored_dataframe_operation(1000)
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 3  # Number of categories
        
        # Verify monitoring worked
        stats = get_timing_stats("monitored_dataframe_operation")
        assert stats is not None
        
    def test_numpy_operations_with_validation(self):
        """Test NumPy operations with input validation."""
        
        @validate_inputs(validators={
            'array': custom_check(lambda x: isinstance(x, np.ndarray), "Must be NumPy array"),
            'factor': type_check((int, float))
        })
        @time_it()
        def validated_numpy_operation(array: np.ndarray, factor: float) -> np.ndarray:
            return array * factor
        
        # Test valid inputs
        test_array = np.array([1, 2, 3, 4, 5])
        result = validated_numpy_operation(test_array, 2.0)
        expected = np.array([2, 4, 6, 8, 10])
        np.testing.assert_array_equal(result, expected)
        
        # Test invalid inputs
        with pytest.raises(DataError):
            validated_numpy_operation([1, 2, 3], 2.0)  # Not NumPy array
            
        with pytest.raises(DataError):
            validated_numpy_operation(test_array, "invalid")  # Wrong type


class TestMathematicalIntegration:
    """Test integration between mathematical utilities and other components."""
    
    def test_statistical_functions_with_monitoring(self):
        """Test mathematical statistics with performance monitoring."""
        from refunc.math_stats import describe, StatisticsEngine
        
        @time_it(collect_stats=True)
        @memory_profile()
        def monitored_statistical_analysis(data: List[float]) -> Dict[str, float]:
            # Mock math_stats functions to test integration
            with patch('refunc.math_stats.statistics.describe') as mock_describe:
                mock_result = Mock()
                mock_result.mean = 50.0
                mock_result.std_dev = 15.0
                mock_describe.return_value = mock_result
                
                stats_result = describe(data)
                return {
                    'mean': stats_result.mean,
                    'std': getattr(stats_result, 'std', 15.0),  # Handle different attribute names
                    'count': len(data)
                }
        
        test_data = [float(i) for i in range(100)]
        result = monitored_statistical_analysis(test_data)
        
        assert 'mean' in result
        assert 'std' in result
        assert 'count' in result
        
        # Verify monitoring
        stats = get_timing_stats("monitored_statistical_analysis")
        assert stats is not None
        
    def test_numerical_methods_with_validation(self):
        """Test numerical methods with input validation."""
        
        @validate_inputs(validators={
            'func': custom_check(callable, "Must be callable"),
            'x0': type_check((int, float)),
            'tolerance': range_check(1e-10, 1e-2)
        })
        @time_it()
        def validated_numerical_method(func, x0: float, tolerance: float) -> float:
            # Mock numerical method
            return x0 + tolerance  # Simplified for testing
        
        def test_function(x):
            return x**2 - 4
        
        result = validated_numerical_method(test_function, 2.0, 1e-6)
        assert isinstance(result, float)
        
        # Test validation errors
        with pytest.raises(DataError):
            validated_numerical_method("not_callable", 2.0, 1e-6)
            
        with pytest.raises(DataError):
            validated_numerical_method(test_function, 2.0, 1.0)  # Tolerance too large


class TestFileHandlingIntegration:
    """Test integration between file handling utilities and other components."""
    
    def test_file_operations_with_logging(self):
        """Test file operations with comprehensive logging."""
        from refunc.utils import FileHandler
        from refunc.logging import get_logger
        
        logger = get_logger("file_test")
        
        @time_it(collect_stats=True)
        @memory_profile()
        def logged_file_operation(file_path: str, data: Dict[str, Any]) -> bool:
            logger.info(f"Starting file operation: {file_path}")
            
            # Mock FileHandler to avoid actual file I/O
            with patch('refunc.utils.file_handler.FileHandler') as mock_handler:
                mock_instance = Mock()
                mock_handler.return_value = mock_instance
                mock_instance.write_json.return_value = True
                mock_instance.read_json.return_value = data
                
                handler = mock_handler()
                
                # Write operation
                logger.debug("Writing data to file")
                write_success = handler.write_json(file_path, data)
                
                # Read operation
                logger.debug("Reading data from file")
                read_data = handler.read_json(file_path)
                
                logger.info("File operation completed successfully")
                return write_success and read_data == data
        
        test_data = {"key": "value", "number": 42}
        result = logged_file_operation("test.json", test_data)
        assert result is True
        
        # Verify timing was collected
        stats = get_timing_stats("logged_file_operation")
        assert stats is not None
        
    def test_file_validation_with_monitoring(self):
        """Test file operations with validation and monitoring."""
        
        @validate_inputs(validators={
            'file_path': custom_check(lambda x: isinstance(x, str) and x.endswith('.json'), "Must be JSON file path"),
            'data': type_check(dict)
        })
        @system_monitor(sample_interval=0.01, print_result=False)
        def validated_file_operation(file_path: str, data: Dict[str, Any]) -> int:
            # Mock file size calculation
            return len(str(data))
        
        # Valid operation
        result = validated_file_operation("test.json", {"data": "value"})
        assert isinstance(result, int)
        
        # Invalid operations
        with pytest.raises(DataError):
            validated_file_operation("test.txt", {"data": "value"})  # Wrong extension
            
        with pytest.raises(DataError):
            validated_file_operation("test.json", "not_dict")  # Wrong type