"""
Timing decorators for performance monitoring.

This module provides decorators and context managers for measuring execution time
of functions, methods, and code blocks with various timing modes and statistics.
"""

import time
import functools
import statistics
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Generic
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio
from contextlib import contextmanager

from ..exceptions import RefuncError


F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class TimingResult:
    """Container for timing measurement results."""
    
    function_name: str
    execution_time: float
    timestamp: float
    args_hash: Optional[str] = None
    timing_mode: str = "wall_clock"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class TimingStats:
    """Statistical summary of multiple timing measurements."""
    
    function_name: str
    call_count: int
    total_time: float
    mean_time: float
    min_time: float
    max_time: float
    std_dev: float
    median_time: float
    p95_time: float
    p99_time: float


class TimingMode:
    """Enumeration of available timing modes."""
    
    WALL_CLOCK = "wall_clock"  # Real-world elapsed time
    PROCESS_TIME = "process_time"  # CPU time for current process
    THREAD_TIME = "thread_time"  # CPU time for current thread (if available)
    PERF_COUNTER = "perf_counter"  # High-resolution monotonic timer


class TimingCollector:
    """Collects and manages timing measurements for statistical analysis."""
    
    def __init__(self):
        self._measurements: Dict[str, List[TimingResult]] = defaultdict(list)
        self._enabled = True
    
    def add_measurement(self, result: TimingResult) -> None:
        """Add a timing measurement."""
        if self._enabled:
            self._measurements[result.function_name].append(result)
    
    def get_stats(self, function_name: str) -> Optional[TimingStats]:
        """Get statistical summary for a function."""
        measurements = self._measurements.get(function_name, [])
        if not measurements:
            return None
        
        times = [m.execution_time for m in measurements]
        
        return TimingStats(
            function_name=function_name,
            call_count=len(times),
            total_time=sum(times),
            mean_time=statistics.mean(times),
            min_time=min(times),
            max_time=max(times),
            std_dev=statistics.stdev(times) if len(times) > 1 else 0.0,
            median_time=statistics.median(times),
            p95_time=self._percentile(times, 0.95),
            p99_time=self._percentile(times, 0.99)
        )
    
    def get_all_stats(self) -> Dict[str, TimingStats]:
        """Get statistics for all measured functions."""
        result = {}
        for name in self._measurements.keys():
            stats = self.get_stats(name)
            if stats is not None:
                result[name] = stats
        return result
    
    def clear(self, function_name: Optional[str] = None) -> None:
        """Clear measurements for a specific function or all functions."""
        if function_name:
            self._measurements.pop(function_name, None)
        else:
            self._measurements.clear()
    
    def enable(self) -> None:
        """Enable timing collection."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable timing collection."""
        self._enabled = False
    
    @staticmethod
    def _percentile(data: List[float], p: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p
        f = int(k)
        c = k - f
        if f == len(sorted_data) - 1:
            return sorted_data[f]
        return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c


# Global timing collector instance
_timing_collector = TimingCollector()


def get_timing_stats(function_name: Optional[str] = None) -> Union[TimingStats, Dict[str, TimingStats], None]:
    """
    Get timing statistics.
    
    Args:
        function_name: Specific function name, or None for all functions
    
    Returns:
        TimingStats for specific function, dict of all stats, or None if not found
    """
    if function_name:
        return _timing_collector.get_stats(function_name)
    return _timing_collector.get_all_stats()


def clear_timing_stats(function_name: Optional[str] = None) -> None:
    """Clear timing statistics."""
    _timing_collector.clear(function_name)


def _get_timer_function(mode: str) -> Callable[[], float]:
    """Get the appropriate timer function for the specified mode."""
    if mode == TimingMode.WALL_CLOCK:
        return time.time
    elif mode == TimingMode.PROCESS_TIME:
        return time.process_time
    elif mode == TimingMode.THREAD_TIME:
        if hasattr(time, 'thread_time'):
            return time.thread_time  # type: ignore
        else:
            # Fallback to process time if thread time not available
            return time.process_time
    elif mode == TimingMode.PERF_COUNTER:
        return time.perf_counter
    else:
        raise RefuncError(f"Unknown timing mode: {mode}")


def time_it(
    func: Optional[F] = None,
    *,
    mode: str = TimingMode.WALL_CLOCK,
    collect_stats: bool = True,
    print_result: bool = False,
    return_result: bool = False,
    precision: int = 6,
    log_results: bool = False
) -> Union[F, Callable[[F], F]]:
    """
    Decorator to measure function execution time.
    
    Can be used with or without parentheses:
    - @time_it
    - @time_it()
    - @time_it(mode=TimingMode.PROCESS_TIME)
    
    Args:
        func: Function to decorate (when used without parentheses)
        mode: Timing mode (wall_clock, process_time, thread_time, perf_counter)
        collect_stats: Whether to collect statistics
        print_result: Whether to print timing result
        return_result: Whether to return TimingResult as part of function result
        precision: Number of decimal places for timing display
        log_results: Whether to log timing results (compatibility parameter)
    
    Returns:
        Decorated function or decorator function
    """
    def decorator(func: F) -> F:
        timer_func = _get_timer_function(mode)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = timer_func()
            success = False
            error = None
            
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                error = str(e)
                raise
            finally:
                end_time = timer_func()
                execution_time = end_time - start_time
                
                # Create timing result
                timing_result = TimingResult(
                    function_name=func.__name__,
                    execution_time=execution_time,
                    timestamp=time.time(),
                    timing_mode=mode,
                    metadata={
                        "success": success,
                        "error": error,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs)
                    }
                )
                
                # Collect statistics
                if collect_stats:
                    _timing_collector.add_measurement(timing_result)
                
                # Print result
                if print_result:
                    print(f"{func.__name__} executed in {execution_time:.{precision}f}s ({mode})")
            
            # Return result with timing info if requested
            if return_result:
                return result, timing_result
            return result
        
        return wrapper  # type: ignore
    
    # Handle dual-mode usage: @time_it vs @time_it()
    if func is None:
        # Called with parentheses: @time_it() or @time_it(mode=...)
        return decorator
    else:
        # Called without parentheses: @time_it
        return decorator(func)


def time_it_async(
    mode: str = TimingMode.WALL_CLOCK,
    collect_stats: bool = True,
    print_result: bool = False,
    return_result: bool = False,
    precision: int = 6
) -> Callable[[F], F]:
    """
    Decorator to measure async function execution time.
    
    Args:
        mode: Timing mode (wall_clock, process_time, thread_time, perf_counter)
        collect_stats: Whether to collect statistics
        print_result: Whether to print timing result
        return_result: Whether to return TimingResult as part of function result
        precision: Number of decimal places for timing display
    
    Returns:
        Decorated async function
    """
    def decorator(func: F) -> F:
        timer_func = _get_timer_function(mode)
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = timer_func()
            success = False
            error = None
            
            try:
                result = await func(*args, **kwargs)
                success = True
            except Exception as e:
                error = str(e)
                raise
            finally:
                end_time = timer_func()
                execution_time = end_time - start_time
                
                # Create timing result
                timing_result = TimingResult(
                    function_name=func.__name__,
                    execution_time=execution_time,
                    timestamp=time.time(),
                    timing_mode=mode,
                    metadata={
                        "success": success,
                        "error": error,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs),
                        "async": True
                    }
                )
                
                # Collect statistics
                if collect_stats:
                    _timing_collector.add_measurement(timing_result)
                
                # Print result
                if print_result:
                    print(f"{func.__name__} (async) executed in {execution_time:.{precision}f}s ({mode})")
            
            # Return result with timing info if requested
            if return_result:
                return result, timing_result
            return result
        
        return wrapper  # type: ignore
    
    return decorator


@contextmanager
def timer(
    name: str = "code_block",
    mode: str = TimingMode.WALL_CLOCK,
    collect_stats: bool = True,
    print_result: bool = True,
    precision: int = 6
):
    """
    Context manager for timing code blocks.
    
    Args:
        name: Name for the timed operation
        mode: Timing mode
        collect_stats: Whether to collect statistics
        print_result: Whether to print timing result
        precision: Number of decimal places for timing display
    
    Yields:
        TimingResult object that gets populated with timing data
    """
    timer_func = _get_timer_function(mode)
    
    # Create result object that will be populated
    result = TimingResult(
        function_name=name,
        execution_time=0.0,
        timestamp=time.time(),
        timing_mode=mode
    )
    
    start_time = timer_func()
    success = True
    error = None
    
    try:
        yield result
    except Exception as e:
        success = False
        error = str(e)
        raise
    finally:
        end_time = timer_func()
        execution_time = end_time - start_time
        
        # Update result object
        result.execution_time = execution_time
        result.metadata = {
            "success": success,
            "error": error
        }
        
        # Collect statistics
        if collect_stats:
            _timing_collector.add_measurement(result)
        
        # Print result
        if print_result:
            print(f"{name} executed in {execution_time:.{precision}f}s ({mode})")


class TimingProfiler:
    """Advanced timing profiler with hierarchical measurement support."""
    
    def __init__(self, mode: str = TimingMode.WALL_CLOCK):
        self.mode = mode
        self.timer_func = _get_timer_function(mode)
        self._stack: List[Dict[str, Any]] = []
        self._results: List[TimingResult] = []
        self._enabled = True
    
    def start(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Start timing a named operation."""
        if not self._enabled:
            return
        
        self._stack.append({
            'name': name,
            'start_time': self.timer_func(),
            'metadata': metadata or {}
        })
    
    def stop(self, name: Optional[str] = None) -> Optional[TimingResult]:
        """
        Stop timing the most recent operation.
        
        Args:
            name: Optional name to verify we're stopping the right operation
        
        Returns:
            TimingResult if operation was found and timed
        """
        if not self._enabled or not self._stack:
            return None
        
        entry = self._stack.pop()
        end_time = self.timer_func()
        
        # Verify name if provided
        if name and entry['name'] != name:
            raise RefuncError(f"Timing mismatch: expected '{name}', got '{entry['name']}'")
        
        # Create result
        result = TimingResult(
            function_name=entry['name'],
            execution_time=end_time - entry['start_time'],
            timestamp=time.time(),
            timing_mode=self.mode,
            metadata=entry['metadata']
        )
        
        self._results.append(result)
        _timing_collector.add_measurement(result)
        
        return result
    
    def reset(self) -> None:
        """Reset the profiler state."""
        self._stack.clear()
        self._results.clear()
    
    def get_results(self) -> List[TimingResult]:
        """Get all timing results."""
        return self._results.copy()
    
    def enable(self) -> None:
        """Enable the profiler."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable the profiler."""
        self._enabled = False
    
    @contextmanager
    def measure(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for measuring code blocks."""
        self.start(name, metadata)
        try:
            yield
        finally:
            self.stop(name)


# Convenience functions for quick timing
def quick_time(func: Callable, *args, **kwargs) -> tuple[Any, float]:
    """
    Quickly time a function call and return result and execution time.
    
    Args:
        func: Function to time
        *args: Function arguments
        **kwargs: Function keyword arguments
    
    Returns:
        Tuple of (function_result, execution_time)
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    
    return result, end_time - start_time


async def quick_time_async(func: Callable, *args, **kwargs) -> tuple[Any, float]:
    """
    Quickly time an async function call and return result and execution time.
    
    Args:
        func: Async function to time
        *args: Function arguments  
        **kwargs: Function keyword arguments
    
    Returns:
        Tuple of (function_result, execution_time)
    """
    start_time = time.perf_counter()
    result = await func(*args, **kwargs)
    end_time = time.perf_counter()
    
    return result, end_time - start_time


# Create a global profiler instance
profiler = TimingProfiler()


# Convenience aliases
measure_time = time_it
measure_time_async = time_it_async
time_block = timer