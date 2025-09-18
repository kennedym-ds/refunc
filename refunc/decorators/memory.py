"""
Memory profiling decorators for monitoring memory usage.

This module provides decorators and utilities for measuring memory consumption
of functions and code blocks, including peak memory usage, memory leaks detection,
and detailed memory profiling.
"""

import gc
import psutil
import functools
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, NamedTuple
from dataclasses import dataclass, field
import threading
import time
from contextlib import contextmanager

from ..exceptions import RefuncError


F = TypeVar('F', bound=Callable[..., Any])


class MemorySnapshot(NamedTuple):
    """Memory usage snapshot."""
    
    rss: int  # Resident Set Size (physical memory)
    vms: int  # Virtual Memory Size
    percent: float  # Memory percentage
    available: int  # Available memory
    timestamp: float


@dataclass
class MemoryResult:
    """Container for memory measurement results."""
    
    function_name: str
    peak_memory: int  # Peak memory usage in bytes
    memory_diff: int  # Memory difference (end - start)
    start_memory: MemorySnapshot
    peak_snapshot: MemorySnapshot
    end_memory: MemorySnapshot
    duration: float
    timestamp: float
    gc_collections: Dict[int, int] = field(default_factory=dict)
    tracemalloc_peak: Optional[int] = None
    tracemalloc_current: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryStats:
    """Statistical summary of memory measurements."""
    
    function_name: str
    call_count: int
    avg_peak_memory: float
    max_peak_memory: int
    min_peak_memory: int
    avg_memory_diff: float
    max_memory_diff: int
    min_memory_diff: int
    total_memory_leaked: int  # Sum of positive memory diffs


class MemoryMode:
    """Memory monitoring modes."""
    
    BASIC = "basic"  # Basic RSS monitoring using psutil
    DETAILED = "detailed"  # Detailed monitoring with tracemalloc
    PEAK_ONLY = "peak_only"  # Only track peak memory usage
    CONTINUOUS = "continuous"  # Continuous monitoring with sampling


class MemoryMonitor:
    """Continuous memory monitoring with background sampling."""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self._monitoring = False
        self._thread = None
        self._process = psutil.Process()
        self._snapshots: List[MemorySnapshot] = []
        self._lock = threading.Lock()
    
    def start(self) -> None:
        """Start continuous memory monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._snapshots.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> List[MemorySnapshot]:
        """Stop monitoring and return collected snapshots."""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=1.0)
        
        with self._lock:
            return self._snapshots.copy()
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring:
            try:
                memory_info = self._process.memory_info()
                memory_percent = self._process.memory_percent()
                virtual_memory = psutil.virtual_memory()
                
                snapshot = MemorySnapshot(
                    rss=memory_info.rss,
                    vms=memory_info.vms,
                    percent=memory_percent,
                    available=virtual_memory.available,
                    timestamp=time.time()
                )
                
                with self._lock:
                    self._snapshots.append(snapshot)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            
            time.sleep(self.interval)
    
    def get_peak_memory(self) -> Optional[MemorySnapshot]:
        """Get the snapshot with peak memory usage."""
        with self._lock:
            if not self._snapshots:
                return None
            return max(self._snapshots, key=lambda s: s.rss)


def _get_memory_snapshot() -> MemorySnapshot:
    """Get current memory snapshot."""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    virtual_memory = psutil.virtual_memory()
    
    return MemorySnapshot(
        rss=memory_info.rss,
        vms=memory_info.vms,
        percent=memory_percent,
        available=virtual_memory.available,
        timestamp=time.time()
    )


def _format_bytes(bytes_value: int) -> str:
    """Format bytes in human-readable format."""
    value = float(bytes_value)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def memory_profile(
    func: Optional[F] = None,
    *,
    mode: str = MemoryMode.BASIC,
    track_peak: bool = True,
    track_tracemalloc: bool = False,
    print_result: bool = False,
    return_result: bool = False,
    gc_before: bool = False,
    gc_after: bool = False
) -> Union[F, Callable[[F], F]]:
    """
    Decorator to monitor memory usage of functions.
    
    Can be used with or without parentheses:
    - @memory_profile
    - @memory_profile()
    - @memory_profile(mode=MemoryMode.DETAILED)
    
    Args:
        func: Function to decorate (when used without parentheses)
        mode: Memory monitoring mode
        track_peak: Whether to track peak memory usage
        track_tracemalloc: Whether to use tracemalloc for detailed tracking
        print_result: Whether to print memory usage
        return_result: Whether to return MemoryResult
        gc_before: Whether to run garbage collection before
        gc_after: Whether to run garbage collection after
    
    Returns:
        Decorated function or decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Run garbage collection before if requested
            gc_before_counts = {}
            if gc_before:
                gc_before_counts = {i: gc.get_count()[i] for i in range(3)}
                gc.collect()
            
            # Start tracemalloc if requested
            tracemalloc_started = False
            if track_tracemalloc and not tracemalloc.is_tracing():
                tracemalloc.start()
                tracemalloc_started = True
            
            # Get initial state
            start_memory = _get_memory_snapshot()
            start_time = time.time()
            
            # Start continuous monitoring if needed
            monitor = None
            if mode == MemoryMode.CONTINUOUS or track_peak:
                monitor = MemoryMonitor()
                monitor.start()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Get end state
                end_time = time.time()
                end_memory = _get_memory_snapshot()
                
                # Stop monitoring
                snapshots = []
                if monitor:
                    snapshots = monitor.stop()
                
                # Get peak memory
                peak_snapshot = end_memory
                if snapshots and monitor:
                    peak_from_monitor = monitor.get_peak_memory()
                    if peak_from_monitor and peak_from_monitor.rss > peak_snapshot.rss:
                        peak_snapshot = peak_from_monitor
                
                # Get tracemalloc info
                tracemalloc_current = None
                tracemalloc_peak = None
                if track_tracemalloc and tracemalloc.is_tracing():
                    tracemalloc_current, _ = tracemalloc.get_traced_memory()
                    tracemalloc_peak = tracemalloc.get_traced_memory()[1]
                
                # Stop tracemalloc if we started it
                if tracemalloc_started:
                    tracemalloc.stop()
                
                # Run garbage collection after if requested
                gc_after_counts = {}
                if gc_after:
                    gc.collect()
                    gc_after_counts = {i: gc.get_count()[i] for i in range(3)}
                
                # Calculate GC collections
                gc_collections = {}
                if gc_before and gc_after:
                    current_counts = gc.get_count()
                    for i in range(3):
                        gc_collections[i] = current_counts[i] - gc_before_counts[i]
                
                # Create result
                memory_result = MemoryResult(
                    function_name=func.__name__,
                    peak_memory=peak_snapshot.rss,
                    memory_diff=end_memory.rss - start_memory.rss,
                    start_memory=start_memory,
                    peak_snapshot=peak_snapshot,
                    end_memory=end_memory,
                    duration=end_time - start_time,
                    timestamp=time.time(),
                    gc_collections=gc_collections,
                    tracemalloc_peak=tracemalloc_peak,
                    tracemalloc_current=tracemalloc_current,
                    metadata={
                        "mode": mode,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs),
                        "snapshots_count": len(snapshots)
                    }
                )
                
                # Print result if requested
                if print_result:
                    peak_mb = peak_snapshot.rss / (1024 * 1024)
                    diff_mb = memory_result.memory_diff / (1024 * 1024)
                    print(f"{func.__name__} - Peak: {peak_mb:.2f} MB, "
                          f"Diff: {diff_mb:+.2f} MB, Duration: {memory_result.duration:.3f}s")
                
                # Return result with memory info if requested
                if return_result:
                    return result, memory_result
                return result
                
            except Exception as e:
                # Clean up monitoring on error
                if monitor:
                    monitor.stop()
                if tracemalloc_started:
                    tracemalloc.stop()
                raise
        
        return wrapper  # type: ignore
    
    # Handle dual-mode usage: @memory_profile vs @memory_profile()
    if func is None:
        # Called with parentheses: @memory_profile() or @memory_profile(mode=...)
        return decorator
    else:
        # Called without parentheses: @memory_profile
        return decorator(func)


def memory_profile_async(
    mode: str = MemoryMode.BASIC,
    track_peak: bool = True,
    track_tracemalloc: bool = False,
    print_result: bool = False,
    return_result: bool = False,
    gc_before: bool = False,
    gc_after: bool = False
) -> Callable[[F], F]:
    """
    Decorator to monitor memory usage of async functions.
    
    Args:
        mode: Memory monitoring mode
        track_peak: Whether to track peak memory usage
        track_tracemalloc: Whether to use tracemalloc for detailed tracking
        print_result: Whether to print memory usage
        return_result: Whether to return MemoryResult
        gc_before: Whether to run garbage collection before
        gc_after: Whether to run garbage collection after
    
    Returns:
        Decorated async function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Run garbage collection before if requested
            gc_before_counts = {}
            if gc_before:
                gc_before_counts = {i: gc.get_count()[i] for i in range(3)}
                gc.collect()
            
            # Start tracemalloc if requested
            tracemalloc_started = False
            if track_tracemalloc and not tracemalloc.is_tracing():
                tracemalloc.start()
                tracemalloc_started = True
            
            # Get initial state
            start_memory = _get_memory_snapshot()
            start_time = time.time()
            
            # Start continuous monitoring if needed
            monitor = None
            if mode == MemoryMode.CONTINUOUS or track_peak:
                monitor = MemoryMonitor()
                monitor.start()
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                
                # Get end state
                end_time = time.time()
                end_memory = _get_memory_snapshot()
                
                # Stop monitoring
                snapshots = []
                if monitor:
                    snapshots = monitor.stop()
                
                # Get peak memory
                peak_snapshot = end_memory
                if snapshots and monitor:
                    peak_from_monitor = monitor.get_peak_memory()
                    if peak_from_monitor and peak_from_monitor.rss > peak_snapshot.rss:
                        peak_snapshot = peak_from_monitor
                
                # Get tracemalloc info
                tracemalloc_current = None
                tracemalloc_peak = None
                if track_tracemalloc and tracemalloc.is_tracing():
                    tracemalloc_current, _ = tracemalloc.get_traced_memory()
                    tracemalloc_peak = tracemalloc.get_traced_memory()[1]
                
                # Stop tracemalloc if we started it
                if tracemalloc_started:
                    tracemalloc.stop()
                
                # Run garbage collection after if requested
                gc_after_counts = {}
                if gc_after:
                    gc.collect()
                    gc_after_counts = {i: gc.get_count()[i] for i in range(3)}
                
                # Calculate GC collections
                gc_collections = {}
                if gc_before and gc_after:
                    current_counts = gc.get_count()
                    for i in range(3):
                        gc_collections[i] = current_counts[i] - gc_before_counts[i]
                
                # Create result
                memory_result = MemoryResult(
                    function_name=func.__name__,
                    peak_memory=peak_snapshot.rss,
                    memory_diff=end_memory.rss - start_memory.rss,
                    start_memory=start_memory,
                    peak_snapshot=peak_snapshot,
                    end_memory=end_memory,
                    duration=end_time - start_time,
                    timestamp=time.time(),
                    gc_collections=gc_collections,
                    tracemalloc_peak=tracemalloc_peak,
                    tracemalloc_current=tracemalloc_current,
                    metadata={
                        "mode": mode,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs),
                        "snapshots_count": len(snapshots),
                        "async": True
                    }
                )
                
                # Print result if requested
                if print_result:
                    peak_mb = peak_snapshot.rss / (1024 * 1024)
                    diff_mb = memory_result.memory_diff / (1024 * 1024)
                    print(f"{func.__name__} (async) - Peak: {peak_mb:.2f} MB, "
                          f"Diff: {diff_mb:+.2f} MB, Duration: {memory_result.duration:.3f}s")
                
                # Return result with memory info if requested
                if return_result:
                    return result, memory_result
                return result
                
            except Exception as e:
                # Clean up monitoring on error
                if monitor:
                    monitor.stop()
                if tracemalloc_started:
                    tracemalloc.stop()
                raise
        
        return wrapper  # type: ignore
    return decorator


@contextmanager
def memory_monitor(
    name: str = "code_block",
    mode: str = MemoryMode.BASIC,
    track_peak: bool = True,
    track_tracemalloc: bool = False,
    print_result: bool = True,
    gc_before: bool = False,
    gc_after: bool = False
):
    """
    Context manager for monitoring memory usage of code blocks.
    
    Args:
        name: Name for the monitored operation
        mode: Memory monitoring mode
        track_peak: Whether to track peak memory usage
        track_tracemalloc: Whether to use tracemalloc for detailed tracking
        print_result: Whether to print memory usage
        gc_before: Whether to run garbage collection before
        gc_after: Whether to run garbage collection after
    
    Yields:
        MemoryResult object that gets populated with memory data
    """
    # Run garbage collection before if requested
    gc_before_counts = {}
    if gc_before:
        gc_before_counts = {i: gc.get_count()[i] for i in range(3)}
        gc.collect()
    
    # Start tracemalloc if requested
    tracemalloc_started = False
    if track_tracemalloc and not tracemalloc.is_tracing():
        tracemalloc.start()
        tracemalloc_started = True
    
    # Get initial state
    start_memory = _get_memory_snapshot()
    start_time = time.time()
    
    # Start continuous monitoring if needed
    monitor = None
    if mode == MemoryMode.CONTINUOUS or track_peak:
        monitor = MemoryMonitor()
        monitor.start()
    
    # Create result object that will be populated
    result = MemoryResult(
        function_name=name,
        peak_memory=0,
        memory_diff=0,
        start_memory=start_memory,
        peak_snapshot=start_memory,
        end_memory=start_memory,
        duration=0.0,
        timestamp=time.time()
    )
    
    try:
        yield result
    finally:
        # Get end state
        end_time = time.time()
        end_memory = _get_memory_snapshot()
        
        # Stop monitoring
        snapshots = []
        if monitor:
            snapshots = monitor.stop()
        
        # Get peak memory
        peak_snapshot = end_memory
        if snapshots and monitor:
            peak_from_monitor = monitor.get_peak_memory()
            if peak_from_monitor and peak_from_monitor.rss > peak_snapshot.rss:
                peak_snapshot = peak_from_monitor
        
        # Get tracemalloc info
        tracemalloc_current = None
        tracemalloc_peak = None
        if track_tracemalloc and tracemalloc.is_tracing():
            tracemalloc_current, _ = tracemalloc.get_traced_memory()
            tracemalloc_peak = tracemalloc.get_traced_memory()[1]
        
        # Stop tracemalloc if we started it
        if tracemalloc_started:
            tracemalloc.stop()
        
        # Run garbage collection after if requested
        gc_after_counts = {}
        if gc_after:
            gc.collect()
            gc_after_counts = {i: gc.get_count()[i] for i in range(3)}
        
        # Calculate GC collections
        gc_collections = {}
        if gc_before and gc_after:
            current_counts = gc.get_count()
            for i in range(3):
                gc_collections[i] = current_counts[i] - gc_before_counts[i]
        
        # Update result object
        result.peak_memory = peak_snapshot.rss
        result.memory_diff = end_memory.rss - start_memory.rss
        result.peak_snapshot = peak_snapshot
        result.end_memory = end_memory
        result.duration = end_time - start_time
        result.gc_collections = gc_collections
        result.tracemalloc_peak = tracemalloc_peak
        result.tracemalloc_current = tracemalloc_current
        result.metadata = {
            "mode": mode,
            "snapshots_count": len(snapshots)
        }
        
        # Print result if requested
        if print_result:
            peak_mb = peak_snapshot.rss / (1024 * 1024)
            diff_mb = result.memory_diff / (1024 * 1024)
            print(f"{name} - Peak: {peak_mb:.2f} MB, "
                  f"Diff: {diff_mb:+.2f} MB, Duration: {result.duration:.3f}s")


def get_current_memory() -> MemorySnapshot:
    """Get current memory usage snapshot."""
    return _get_memory_snapshot()


def get_peak_memory_usage() -> int:
    """Get peak memory usage since process start."""
    process = psutil.Process()
    return getattr(process, 'memory_info_ex', process.memory_info)().peak_wset if hasattr(process.memory_info(), 'peak_wset') else process.memory_info().rss


class MemoryLeakDetector:
    """Utility for detecting memory leaks."""
    
    def __init__(self, threshold_mb: float = 10.0, sample_interval: float = 1.0):
        self.threshold_bytes = int(threshold_mb * 1024 * 1024)
        self.sample_interval = sample_interval
        self._baseline: Optional[MemorySnapshot] = None
        self._samples: List[MemorySnapshot] = []
    
    def set_baseline(self) -> None:
        """Set the baseline memory usage."""
        self._baseline = _get_memory_snapshot()
        self._samples.clear()
    
    def sample(self) -> MemorySnapshot:
        """Take a memory sample."""
        snapshot = _get_memory_snapshot()
        self._samples.append(snapshot)
        return snapshot
    
    def check_leak(self) -> Dict[str, Any]:
        """Check for memory leaks based on samples."""
        if not self._baseline or len(self._samples) < 2:
            return {"leak_detected": False, "reason": "Insufficient data"}
        
        current = self._samples[-1]
        memory_growth = current.rss - self._baseline.rss
        
        # Check if memory has grown significantly
        leak_detected = memory_growth > self.threshold_bytes
        
        # Calculate growth rate
        if len(self._samples) > 1:
            time_diff = current.timestamp - self._samples[0].timestamp
            growth_rate = memory_growth / time_diff if time_diff > 0 else 0
        else:
            growth_rate = 0
        
        return {
            "leak_detected": leak_detected,
            "memory_growth_bytes": memory_growth,
            "memory_growth_mb": memory_growth / (1024 * 1024),
            "growth_rate_mb_per_sec": growth_rate / (1024 * 1024),
            "baseline_mb": self._baseline.rss / (1024 * 1024),
            "current_mb": current.rss / (1024 * 1024),
            "samples_count": len(self._samples),
            "threshold_mb": self.threshold_bytes / (1024 * 1024)
        }


# Convenience aliases
profile_memory = memory_profile
profile_memory_async = memory_profile_async
monitor_memory = memory_monitor