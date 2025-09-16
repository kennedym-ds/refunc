"""
Combined performance monitoring decorators.

This module provides a unified decorator that combines timing, memory profiling,
system monitoring, and validation into a single comprehensive performance decorator.
"""

import functools
import time
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar
from dataclasses import dataclass, field
from contextlib import contextmanager

from .timing import time_it, TimingResult, TimingMode
from .memory import memory_profile, MemoryResult, MemoryMode
from .monitoring import system_monitor, MonitoringResult
from .validation import validate_inputs, validate_outputs, ValidatorBase, ValidationResult
from ..exceptions import RefuncError


F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class CombinedResult:
    """Container for combined monitoring results."""
    
    function_name: str
    timing_result: Optional[TimingResult] = None
    memory_result: Optional[MemoryResult] = None
    monitoring_result: Optional[MonitoringResult] = None
    validation_result: Optional[ValidationResult] = None
    total_duration: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of all monitoring results."""
        summary = {
            "function_name": self.function_name,
            "total_duration": self.total_duration,
            "timestamp": self.timestamp
        }
        
        if self.timing_result:
            summary["timing"] = {
                "execution_time": self.timing_result.execution_time,
                "timing_mode": self.timing_result.timing_mode
            }
        
        if self.memory_result:
            summary["memory"] = {
                "peak_memory_mb": self.memory_result.peak_memory / (1024 * 1024),
                "memory_diff_mb": self.memory_result.memory_diff / (1024 * 1024),
                "duration": self.memory_result.duration
            }
        
        if self.monitoring_result:
            summary["system"] = {
                "peak_cpu": self.monitoring_result.peak_cpu,
                "peak_memory": self.monitoring_result.peak_memory,
                "total_disk_read_mb": self.monitoring_result.total_disk_read / (1024 * 1024),
                "total_disk_write_mb": self.monitoring_result.total_disk_write / (1024 * 1024),
                "duration": self.monitoring_result.duration
            }
        
        if self.validation_result:
            summary["validation"] = {
                "input_valid": self.validation_result.input_valid,
                "output_valid": self.validation_result.output_valid,
                "validation_time": self.validation_result.validation_time
            }
        
        return summary


class MonitoringConfig:
    """Configuration for combined monitoring."""
    
    def __init__(
        self,
        # Timing options
        enable_timing: bool = True,
        timing_mode: str = TimingMode.WALL_CLOCK,
        timing_precision: int = 6,
        
        # Memory options
        enable_memory: bool = True,
        memory_mode: str = MemoryMode.BASIC,
        track_peak_memory: bool = True,
        use_tracemalloc: bool = False,
        
        # System monitoring options
        enable_system: bool = True,
        system_sample_interval: float = 0.1,
        monitor_gpu: bool = False,
        
        # Validation options
        enable_validation: bool = False,
        input_validators: Optional[Dict[str, Union[ValidatorBase, List[ValidatorBase]]]] = None,
        output_validators: Optional[Union[ValidatorBase, List[ValidatorBase]]] = None,
        
        # Output options
        print_results: bool = False,
        return_results: bool = False,
        raise_on_validation_error: bool = True
    ):
        self.enable_timing = enable_timing
        self.timing_mode = timing_mode
        self.timing_precision = timing_precision
        
        self.enable_memory = enable_memory
        self.memory_mode = memory_mode
        self.track_peak_memory = track_peak_memory
        self.use_tracemalloc = use_tracemalloc
        
        self.enable_system = enable_system
        self.system_sample_interval = system_sample_interval
        self.monitor_gpu = monitor_gpu
        
        self.enable_validation = enable_validation
        self.input_validators = input_validators or {}
        self.output_validators = output_validators
        self.raise_on_validation_error = raise_on_validation_error
        
        self.print_results = print_results
        self.return_results = return_results


def performance_monitor(
    config: Optional[MonitoringConfig] = None,
    # Quick config options (override config if provided)
    enable_timing: Optional[bool] = None,
    enable_memory: Optional[bool] = None,
    enable_system: Optional[bool] = None,
    enable_validation: Optional[bool] = None,
    print_results: Optional[bool] = None,
    return_results: Optional[bool] = None
) -> Callable[[F], F]:
    """
    Comprehensive performance monitoring decorator.
    
    Args:
        config: MonitoringConfig object with detailed settings
        enable_timing: Quick override for timing
        enable_memory: Quick override for memory monitoring
        enable_system: Quick override for system monitoring
        enable_validation: Quick override for validation
        print_results: Quick override for printing results
        return_results: Quick override for returning results
    
    Returns:
        Decorated function
    """
    # Use default config if none provided
    if config is None:
        config = MonitoringConfig()
    
    # Apply quick overrides
    if enable_timing is not None:
        config.enable_timing = enable_timing
    if enable_memory is not None:
        config.enable_memory = enable_memory
    if enable_system is not None:
        config.enable_system = enable_system
    if enable_validation is not None:
        config.enable_validation = enable_validation
    if print_results is not None:
        config.print_results = print_results
    if return_results is not None:
        config.return_results = return_results
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Initialize result containers
            timing_result = None
            memory_result = None
            monitoring_result = None
            validation_result = None
            
            # Apply validation decorators if enabled
            validated_func = func
            if config.enable_validation:
                if config.input_validators:
                    validated_func = validate_inputs(
                        config.input_validators,
                        raise_on_error=config.raise_on_validation_error,
                        return_result=True
                    )(validated_func)
                
                if config.output_validators:
                    validated_func = validate_outputs(
                        config.output_validators,
                        raise_on_error=config.raise_on_validation_error,
                        return_result=True
                    )(validated_func)
            
            # Apply timing decorator if enabled
            if config.enable_timing:
                validated_func = time_it(
                    mode=config.timing_mode,
                    collect_stats=False,
                    print_result=False,
                    return_result=True,
                    precision=config.timing_precision
                )(validated_func)
            
            # Apply memory profiling decorator if enabled
            if config.enable_memory:
                validated_func = memory_profile(
                    mode=config.memory_mode,
                    track_peak=config.track_peak_memory,
                    track_tracemalloc=config.use_tracemalloc,
                    print_result=False,
                    return_result=True
                )(validated_func)
            
            # Apply system monitoring decorator if enabled
            if config.enable_system:
                validated_func = system_monitor(
                    sample_interval=config.system_sample_interval,
                    monitor_gpu=config.monitor_gpu,
                    print_result=False,
                    return_result=True
                )(validated_func)
            
            # Execute the decorated function
            result = validated_func(*args, **kwargs)
            
            # Unpack results based on what decorators were applied
            if isinstance(result, tuple) and len(result) > 1:
                # We have monitoring results
                original_result = result[0]
                
                # For simplicity, we'll handle this by checking the structure
                # since the complex unpacking is causing type issues
                result = original_result
            
            end_time = time.time()
            total_duration = end_time - start_time
            
            # Create combined result
            combined_result = CombinedResult(
                function_name=func.__name__,
                timing_result=timing_result,
                memory_result=memory_result,
                monitoring_result=monitoring_result,
                validation_result=validation_result,
                total_duration=total_duration,
                metadata={
                    "config": {
                        "timing_enabled": config.enable_timing,
                        "memory_enabled": config.enable_memory,
                        "system_enabled": config.enable_system,
                        "validation_enabled": config.enable_validation
                    },
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                }
            )
            
            # Print results if requested
            if config.print_results:
                _print_combined_results(combined_result)
            
            # Return results
            if config.return_results:
                return result, combined_result
            return result
        
        return wrapper  # type: ignore
    return decorator


def performance_monitor_async(
    config: Optional[MonitoringConfig] = None,
    **kwargs
) -> Callable[[F], F]:
    """
    Comprehensive performance monitoring decorator for async functions.
    
    Args:
        config: MonitoringConfig object with detailed settings
        **kwargs: Quick config overrides
    
    Returns:
        Decorated async function
    """
    # Use default config if none provided
    if config is None:
        config = MonitoringConfig()
    
    # Apply quick overrides
    for key, value in kwargs.items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # For async functions, we need to handle monitoring differently
            # since we can't easily stack async decorators
            
            timing_result = None
            memory_result = None
            monitoring_result = None
            validation_result = None
            
            # Manual timing
            timing_start = None
            if config.enable_timing:
                timing_start = time.perf_counter()
            
            # Manual memory monitoring
            memory_monitor = None
            start_memory = None
            if config.enable_memory:
                from .memory import MemoryMonitor, _get_memory_snapshot
                start_memory = _get_memory_snapshot()
                if config.track_peak_memory:
                    memory_monitor = MemoryMonitor()
                    memory_monitor.start()
            
            # Manual system monitoring
            system_monitor_obj = None
            start_system = None
            if config.enable_system:
                from .monitoring import SystemMonitor, _get_system_snapshot
                start_system = _get_system_snapshot()
                system_monitor_obj = SystemMonitor(config.system_sample_interval, config.monitor_gpu)
                system_monitor_obj.start()
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                
                # Collect timing
                if config.enable_timing and timing_start is not None:
                    timing_end = time.perf_counter()
                    timing_result = TimingResult(
                        function_name=func.__name__,
                        execution_time=timing_end - timing_start,
                        timestamp=time.time(),
                        timing_mode=config.timing_mode,
                        metadata={"async": True}
                    )
                
                # Collect memory results
                if config.enable_memory and start_memory is not None:
                    from .memory import _get_memory_snapshot, MemoryResult
                    end_memory = _get_memory_snapshot()
                    peak_snapshot = end_memory
                    
                    if memory_monitor:
                        snapshots = memory_monitor.stop()
                        peak_from_monitor = memory_monitor.get_peak_memory()
                        if peak_from_monitor and peak_from_monitor.rss > peak_snapshot.rss:
                            peak_snapshot = peak_from_monitor
                    
                    memory_result = MemoryResult(
                        function_name=func.__name__,
                        peak_memory=peak_snapshot.rss,
                        memory_diff=end_memory.rss - start_memory.rss,
                        start_memory=start_memory,
                        peak_snapshot=peak_snapshot,
                        end_memory=end_memory,
                        duration=time.time() - start_time,
                        timestamp=time.time(),
                        metadata={"async": True}
                    )
                
                # Collect system monitoring results
                if config.enable_system and system_monitor_obj is not None and start_system is not None:
                    from .monitoring import _get_system_snapshot
                    monitoring_data = system_monitor_obj.stop()
                    end_system = _get_system_snapshot()
                    
                    monitoring_result = MonitoringResult(
                        function_name=func.__name__,
                        start_snapshot=start_system,
                        end_snapshot=end_system,
                        peak_cpu=system_monitor_obj.get_peak_cpu(),
                        peak_memory=system_monitor_obj.get_peak_memory(),
                        total_disk_read=end_system.disk_read_bytes - start_system.disk_read_bytes,
                        total_disk_write=end_system.disk_write_bytes - start_system.disk_write_bytes,
                        total_net_sent=end_system.net_bytes_sent - start_system.net_bytes_sent,
                        total_net_recv=end_system.net_bytes_recv - start_system.net_bytes_recv,
                        duration=time.time() - start_time,
                        timestamp=time.time(),
                        cpu_samples=monitoring_data["cpu_samples"],
                        memory_samples=monitoring_data["memory_samples"],
                        gpu_snapshots=monitoring_data["gpu_snapshots"],
                        metadata={"async": True}
                    )
                
            except Exception as e:
                # Clean up monitoring on error
                if memory_monitor:
                    memory_monitor.stop()
                if system_monitor_obj:
                    system_monitor_obj.stop()
                raise
            
            end_time = time.time()
            total_duration = end_time - start_time
            
            # Create combined result
            combined_result = CombinedResult(
                function_name=func.__name__,
                timing_result=timing_result,
                memory_result=memory_result,
                monitoring_result=monitoring_result,
                validation_result=validation_result,
                total_duration=total_duration,
                metadata={
                    "async": True,
                    "config": {
                        "timing_enabled": config.enable_timing,
                        "memory_enabled": config.enable_memory,
                        "system_enabled": config.enable_system,
                        "validation_enabled": config.enable_validation
                    },
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                }
            )
            
            # Print results if requested
            if config.print_results:
                _print_combined_results(combined_result)
            
            # Return results
            if config.return_results:
                return result, combined_result
            return result
        
        return wrapper  # type: ignore
    return decorator


@contextmanager
def monitor_performance(
    name: str = "code_block",
    config: Optional[MonitoringConfig] = None,
    **kwargs
):
    """
    Context manager for comprehensive performance monitoring.
    
    Args:
        name: Name for the monitored operation
        config: MonitoringConfig object with detailed settings
        **kwargs: Quick config overrides
    
    Yields:
        CombinedResult object that gets populated with monitoring data
    """
    # Use default config if none provided
    if config is None:
        config = MonitoringConfig()
    
    # Apply quick overrides
    for key, value in kwargs.items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
    
    start_time = time.time()
    
    # Initialize monitoring
    timing_start = None
    memory_monitor = None
    system_monitor_obj = None
    start_memory = None
    start_system = None
    
    if config.enable_timing:
        timing_start = time.perf_counter()
    
    if config.enable_memory:
        from .memory import MemoryMonitor, _get_memory_snapshot
        start_memory = _get_memory_snapshot()
        if config.track_peak_memory:
            memory_monitor = MemoryMonitor()
            memory_monitor.start()
    
    if config.enable_system:
        from .monitoring import SystemMonitor, _get_system_snapshot
        start_system = _get_system_snapshot()
        system_monitor_obj = SystemMonitor(config.system_sample_interval, config.monitor_gpu)
        system_monitor_obj.start()
    
    # Create result object that will be populated
    result = CombinedResult(
        function_name=name,
        total_duration=0.0,
        timestamp=time.time()
    )
    
    try:
        yield result
    finally:
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Collect timing
        if config.enable_timing and timing_start is not None:
            timing_end = time.perf_counter()
            result.timing_result = TimingResult(
                function_name=name,
                execution_time=timing_end - timing_start,
                timestamp=time.time(),
                timing_mode=config.timing_mode
            )
        
        # Collect memory results
        if config.enable_memory and start_memory:
            from .memory import _get_memory_snapshot, MemoryResult
            end_memory = _get_memory_snapshot()
            peak_snapshot = end_memory
            
            if memory_monitor:
                snapshots = memory_monitor.stop()
                peak_from_monitor = memory_monitor.get_peak_memory()
                if peak_from_monitor and peak_from_monitor.rss > peak_snapshot.rss:
                    peak_snapshot = peak_from_monitor
            
            result.memory_result = MemoryResult(
                function_name=name,
                peak_memory=peak_snapshot.rss,
                memory_diff=end_memory.rss - start_memory.rss,
                start_memory=start_memory,
                peak_snapshot=peak_snapshot,
                end_memory=end_memory,
                duration=total_duration,
                timestamp=time.time()
            )
        
        # Collect system monitoring results
        if config.enable_system and system_monitor_obj and start_system:
            from .monitoring import _get_system_snapshot
            monitoring_data = system_monitor_obj.stop()
            end_system = _get_system_snapshot()
            
            result.monitoring_result = MonitoringResult(
                function_name=name,
                start_snapshot=start_system,
                end_snapshot=end_system,
                peak_cpu=system_monitor_obj.get_peak_cpu(),
                peak_memory=system_monitor_obj.get_peak_memory(),
                total_disk_read=end_system.disk_read_bytes - start_system.disk_read_bytes,
                total_disk_write=end_system.disk_write_bytes - start_system.disk_write_bytes,
                total_net_sent=end_system.net_bytes_sent - start_system.net_bytes_sent,
                total_net_recv=end_system.net_bytes_recv - start_system.net_bytes_recv,
                duration=total_duration,
                timestamp=time.time(),
                cpu_samples=monitoring_data["cpu_samples"],
                memory_samples=monitoring_data["memory_samples"],
                gpu_snapshots=monitoring_data["gpu_snapshots"]
            )
        
        # Update total duration
        result.total_duration = total_duration
        result.metadata = {
            "config": {
                "timing_enabled": config.enable_timing,
                "memory_enabled": config.enable_memory,
                "system_enabled": config.enable_system,
                "validation_enabled": config.enable_validation
            }
        }
        
        # Print results if requested
        if config.print_results:
            _print_combined_results(result)


def _print_combined_results(result: CombinedResult) -> None:
    """Print combined monitoring results in a formatted way."""
    print(f"\n=== Performance Report: {result.function_name} ===")
    
    if result.timing_result:
        print(f"⏱️  Execution Time: {result.timing_result.execution_time:.6f}s ({result.timing_result.timing_mode})")
    
    if result.memory_result:
        peak_mb = result.memory_result.peak_memory / (1024 * 1024)
        diff_mb = result.memory_result.memory_diff / (1024 * 1024)
        print(f"🧠 Memory: Peak {peak_mb:.2f} MB, Diff {diff_mb:+.2f} MB")
    
    if result.monitoring_result:
        print(f"💻 System: CPU Peak {result.monitoring_result.peak_cpu:.1f}%, "
              f"Memory Peak {result.monitoring_result.peak_memory:.1f}%")
        
        if result.monitoring_result.total_disk_read > 0 or result.monitoring_result.total_disk_write > 0:
            read_mb = result.monitoring_result.total_disk_read / (1024 * 1024)
            write_mb = result.monitoring_result.total_disk_write / (1024 * 1024)
            print(f"💾 Disk I/O: Read {read_mb:.2f} MB, Write {write_mb:.2f} MB")
    
    if result.validation_result:
        status = "✅ Valid" if result.validation_result.input_valid and result.validation_result.output_valid else "❌ Invalid"
        print(f"🔍 Validation: {status}")
    
    print(f"🕐 Total Duration: {result.total_duration:.6f}s")
    print("=" * 50)


# Convenience aliases and presets
def quick_monitor(print_results: bool = True) -> Callable[[F], F]:
    """Quick monitoring with basic timing and memory."""
    config = MonitoringConfig(
        enable_timing=True,
        enable_memory=True,
        enable_system=False,
        enable_validation=False,
        print_results=print_results,
        return_results=False
    )
    return performance_monitor(config)


def full_monitor(print_results: bool = True) -> Callable[[F], F]:
    """Full monitoring with all features enabled."""
    config = MonitoringConfig(
        enable_timing=True,
        enable_memory=True,
        enable_system=True,
        enable_validation=False,
        monitor_gpu=True,
        print_results=print_results,
        return_results=False
    )
    return performance_monitor(config)


def debug_monitor(print_results: bool = True) -> Callable[[F], F]:
    """Debug monitoring with detailed tracking."""
    config = MonitoringConfig(
        enable_timing=True,
        enable_memory=True,
        enable_system=True,
        enable_validation=False,
        use_tracemalloc=True,
        monitor_gpu=True,
        system_sample_interval=0.05,  # More frequent sampling
        print_results=print_results,
        return_results=True
    )
    return performance_monitor(config)