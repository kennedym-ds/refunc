"""
Performance decorators for monitoring execution metrics.

This package provides comprehensive decorators for timing, memory profiling,
system monitoring, validation, and combined performance tracking.
"""

# Timing decorators
from .timing import (
    time_it,
    time_it_async,
    timer,
    TimingProfiler,
    TimingResult,
    TimingStats,
    TimingMode,
    get_timing_stats,
    clear_timing_stats,
    quick_time,
    quick_time_async,
    profiler,
    # Aliases
    measure_time,
    measure_time_async,
    time_block
)

# Memory decorators
from .memory import (
    memory_profile,
    memory_profile_async,
    memory_monitor,
    MemoryResult,
    MemoryStats,
    MemoryMode,
    MemoryMonitor,
    MemorySnapshot,
    MemoryLeakDetector,
    get_current_memory,
    get_peak_memory_usage,
    # Aliases
    profile_memory,
    profile_memory_async,
    monitor_memory
)

# System monitoring decorators
from .monitoring import (
    system_monitor,
    system_monitor_async,
    monitor_system,
    MonitoringResult,
    SystemSnapshot,
    GPUSnapshot,
    SystemMonitor,
    get_system_info,
    get_gpu_info,
    # Aliases
    monitor_system_resources,
    monitor_system_resources_async
)

# Validation decorators
from .validation import (
    validate_inputs,
    validate_outputs,
    validate_types,
    validate_data_schema,
    ValidationResult,
    ValidatorBase,
    TypeValidator,
    RangeValidator,
    ShapeValidator,
    DataFrameValidator,
    CustomValidator,
    # Factory functions
    type_check,
    range_check,
    shape_check,
    dataframe_check,
    custom_check
)

# Combined decorators
from .combined import (
    performance_monitor,
    performance_monitor_async,
    monitor_performance,
    CombinedResult,
    MonitoringConfig,
    # Convenience presets
    quick_monitor,
    full_monitor,
    debug_monitor
)

# Convenience imports for common use cases
__all__ = [
    # Core timing
    "time_it",
    "time_it_async", 
    "timer",
    "TimingResult",
    "measure_time",
    "measure_time_async",
    
    # Core memory
    "memory_profile",
    "memory_profile_async",
    "memory_monitor",
    "MemoryResult",
    "profile_memory",
    
    # Core system monitoring
    "system_monitor",
    "system_monitor_async", 
    "monitor_system",
    "MonitoringResult",
    "get_system_info",
    
    # Core validation
    "validate_inputs",
    "validate_outputs",
    "validate_types",
    "ValidationResult",
    "type_check",
    "range_check",
    "dataframe_check",
    
    # Combined monitoring
    "performance_monitor",
    "performance_monitor_async",
    "monitor_performance",
    "CombinedResult",
    "MonitoringConfig",
    "quick_monitor",
    "full_monitor",
    "debug_monitor",
    
    # Results classes
    "TimingResult",
    "MemoryResult", 
    "MonitoringResult",
    "ValidationResult",
    "CombinedResult",
    
    # Utility classes
    "TimingProfiler",
    "MemoryMonitor",
    "SystemMonitor",
    "MemoryLeakDetector",
    
    # Validators
    "ValidatorBase",
    "TypeValidator",
    "RangeValidator", 
    "ShapeValidator",
    "DataFrameValidator",
    "CustomValidator"
]