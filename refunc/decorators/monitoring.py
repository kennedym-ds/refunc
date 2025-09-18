"""
System monitoring decorators for comprehensive performance tracking.

This module provides decorators for monitoring system resources including CPU usage,
disk I/O, network activity, GPU usage, and other system metrics during function execution.
"""

import os
import psutil
import functools
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, NamedTuple
from dataclasses import dataclass, field
from contextlib import contextmanager

from ..exceptions import RefuncError


F = TypeVar('F', bound=Callable[..., Any])


class SystemSnapshot(NamedTuple):
    """System resource snapshot."""
    
    cpu_percent: float
    memory_percent: float
    memory_rss: int
    memory_vms: int
    disk_read_bytes: int
    disk_write_bytes: int
    net_bytes_sent: int
    net_bytes_recv: int
    open_files: int
    num_threads: int
    timestamp: float
    
    @classmethod
    def capture(cls) -> 'SystemSnapshot':
        """Capture current system resource snapshot."""
        return _get_system_snapshot()


class GPUSnapshot(NamedTuple):
    """GPU resource snapshot."""
    
    gpu_utilization: Optional[float]
    gpu_memory_used: Optional[int]
    gpu_memory_total: Optional[int]
    gpu_temperature: Optional[float]
    gpu_power_draw: Optional[float]
    timestamp: float


@dataclass
class MonitoringResult:
    """Container for system monitoring results."""
    
    function_name: str
    execution_time: float
    cpu_usage_start: float
    cpu_usage_end: float
    memory_usage_start: float
    memory_usage_end: float
    timestamp: float
    
    # Legacy compatibility fields
    start_snapshot: Optional[SystemSnapshot] = None
    end_snapshot: Optional[SystemSnapshot] = None
    peak_cpu: Optional[float] = None
    peak_memory: Optional[float] = None
    total_disk_read: Optional[int] = None
    total_disk_write: Optional[int] = None
    total_net_sent: Optional[int] = None
    total_net_recv: Optional[int] = None
    duration: Optional[float] = None  # Legacy alias for execution_time
    cpu_samples: List[float] = field(default_factory=list)
    memory_samples: List[float] = field(default_factory=list)
    gpu_snapshots: List[GPUSnapshot] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set legacy duration field to match execution_time."""
        if self.duration is None:
            self.duration = self.execution_time


class SystemMonitor:
    """Continuous system resource monitoring."""
    
    def __init__(self, interval: float = 0.1, monitor_gpu: bool = False):
        self.interval = interval
        self.monitor_gpu = monitor_gpu
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process()
        self._cpu_samples: List[float] = []
        self._memory_samples: List[float] = []
        self._gpu_snapshots: List[GPUSnapshot] = []
        self._lock = threading.Lock()
        
        # Check GPU availability
        self._gpu_available = False
        self._gputil = None
        if monitor_gpu:
            try:
                import GPUtil  # type: ignore
                self._gpu_available = True
                self._gputil = GPUtil
            except ImportError:
                self._gpu_available = False
    
    def start(self) -> None:
        """Start system monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._cpu_samples.clear()
        self._memory_samples.clear()
        self._gpu_snapshots.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> Dict[str, List]:
        """Stop monitoring and return collected data."""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=1.0)
        
        with self._lock:
            return {
                "cpu_samples": self._cpu_samples.copy(),
                "memory_samples": self._memory_samples.copy(),
                "gpu_snapshots": self._gpu_snapshots.copy()
            }
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring:
            try:
                # Sample CPU and memory
                cpu_percent = self._process.cpu_percent()
                memory_percent = self._process.memory_percent()
                
                with self._lock:
                    self._cpu_samples.append(cpu_percent)
                    self._memory_samples.append(memory_percent)
                
                # Sample GPU if available
                if self._gpu_available and self._gputil:
                    try:
                        gpus = self._gputil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # Use first GPU
                            gpu_snapshot = GPUSnapshot(
                                gpu_utilization=gpu.load * 100,
                                gpu_memory_used=gpu.memoryUsed,
                                gpu_memory_total=gpu.memoryTotal,
                                gpu_temperature=gpu.temperature,
                                gpu_power_draw=getattr(gpu, 'powerDraw', None),
                                timestamp=time.time()
                            )
                            with self._lock:
                                self._gpu_snapshots.append(gpu_snapshot)
                    except Exception:
                        pass  # Ignore GPU monitoring errors
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            
            time.sleep(self.interval)
    
    def get_peak_cpu(self) -> float:
        """Get peak CPU usage."""
        with self._lock:
            return max(self._cpu_samples) if self._cpu_samples else 0.0
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage."""
        with self._lock:
            return max(self._memory_samples) if self._memory_samples else 0.0


def _get_system_snapshot() -> SystemSnapshot:
    """Get current system resource snapshot."""
    process = psutil.Process()
    
    # Get memory info
    memory_info = process.memory_info()
    
    # Get I/O info
    try:
        io_info = process.io_counters()
        disk_read = io_info.read_bytes
        disk_write = io_info.write_bytes
    except (psutil.AccessDenied, AttributeError):
        disk_read = disk_write = 0
    
    # Get network info (system-wide)
    try:
        net_info = psutil.net_io_counters()
        net_sent = net_info.bytes_sent if net_info else 0
        net_recv = net_info.bytes_recv if net_info else 0
    except (psutil.AccessDenied, AttributeError):
        net_sent = net_recv = 0
    
    # Get file handles
    try:
        open_files = len(process.open_files())
    except (psutil.AccessDenied, AttributeError):
        open_files = 0
    
    return SystemSnapshot(
        cpu_percent=process.cpu_percent(),
        memory_percent=process.memory_percent(),
        memory_rss=memory_info.rss,
        memory_vms=memory_info.vms,
        disk_read_bytes=disk_read,
        disk_write_bytes=disk_write,
        net_bytes_sent=net_sent,
        net_bytes_recv=net_recv,
        open_files=open_files,
        num_threads=process.num_threads(),
        timestamp=time.time()
    )


def system_monitor(
    func: Optional[F] = None,
    *,
    sample_interval: float = 0.1,
    monitor_gpu: bool = False,
    print_result: bool = False,
    return_result: bool = False
) -> Union[F, Callable[[F], F]]:
    """
    Decorator to monitor system resources during function execution.
    
    Can be used with or without parentheses:
    - @system_monitor
    - @system_monitor()
    - @system_monitor(sample_interval=0.05)
    
    Args:
        func: Function to decorate (when used without parentheses)
        sample_interval: Interval between resource samples (seconds)
        monitor_gpu: Whether to monitor GPU resources
        print_result: Whether to print monitoring results
        return_result: Whether to return MonitoringResult
    
    Returns:
        Decorated function or decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get initial snapshot
            start_snapshot = _get_system_snapshot()
            start_time = time.time()
            
            # Start continuous monitoring
            monitor = SystemMonitor(sample_interval, monitor_gpu)
            monitor.start()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Stop monitoring and get data
                monitoring_data = monitor.stop()
                end_time = time.time()
                end_snapshot = _get_system_snapshot()
                
                # Calculate differences
                total_disk_read = end_snapshot.disk_read_bytes - start_snapshot.disk_read_bytes
                total_disk_write = end_snapshot.disk_write_bytes - start_snapshot.disk_write_bytes
                total_net_sent = end_snapshot.net_bytes_sent - start_snapshot.net_bytes_sent
                total_net_recv = end_snapshot.net_bytes_recv - start_snapshot.net_bytes_recv
                
                # Create monitoring result
                monitoring_result = MonitoringResult(
                    function_name=func.__name__,
                    execution_time=end_time - start_time,
                    cpu_usage_start=start_snapshot.cpu_percent,
                    cpu_usage_end=end_snapshot.cpu_percent,
                    memory_usage_start=start_snapshot.memory_percent,
                    memory_usage_end=end_snapshot.memory_percent,
                    timestamp=time.time(),
                    # Legacy/advanced fields
                    start_snapshot=start_snapshot,
                    end_snapshot=end_snapshot,
                    peak_cpu=monitor.get_peak_cpu(),
                    peak_memory=monitor.get_peak_memory(),
                    total_disk_read=total_disk_read,
                    total_disk_write=total_disk_write,
                    total_net_sent=total_net_sent,
                    total_net_recv=total_net_recv,
                    duration=end_time - start_time,
                    cpu_samples=monitoring_data["cpu_samples"],
                    memory_samples=monitoring_data["memory_samples"],
                    gpu_snapshots=monitoring_data["gpu_snapshots"],
                    metadata={
                        "sample_interval": sample_interval,
                        "monitor_gpu": monitor_gpu,
                        "samples_count": len(monitoring_data["cpu_samples"]),
                        "args_count": len(args),
                        "kwargs_count": len(kwargs)
                    }
                )
                
                # Print result if requested
                if print_result:
                    print(f"{func.__name__} - CPU Peak: {monitoring_result.peak_cpu:.1f}%, "
                          f"Memory Peak: {monitoring_result.peak_memory:.1f}%, "
                          f"Duration: {monitoring_result.duration:.3f}s")
                
                # Return result with monitoring info if requested
                if return_result:
                    return result, monitoring_result
                return result
                
            except Exception as e:
                # Clean up monitoring on error
                monitor.stop()
                raise
        
        return wrapper  # type: ignore
    
    # Handle dual-mode usage: @system_monitor vs @system_monitor()
    if func is None:
        # Called with parentheses: @system_monitor() or @system_monitor(sample_interval=...)
        return decorator
    else:
        # Called without parentheses: @system_monitor
        return decorator(func)


def system_monitor_async(
    sample_interval: float = 0.1,
    monitor_gpu: bool = False,
    print_result: bool = False,
    return_result: bool = False
) -> Callable[[F], F]:
    """
    Decorator to monitor system resources during async function execution.
    
    Args:
        sample_interval: Interval between resource samples (seconds)
        monitor_gpu: Whether to monitor GPU resources
        print_result: Whether to print monitoring results
        return_result: Whether to return MonitoringResult
    
    Returns:
        Decorated async function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get initial snapshot
            start_snapshot = _get_system_snapshot()
            start_time = time.time()
            
            # Start continuous monitoring
            monitor = SystemMonitor(sample_interval, monitor_gpu)
            monitor.start()
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                
                # Stop monitoring and get data
                monitoring_data = monitor.stop()
                end_time = time.time()
                end_snapshot = _get_system_snapshot()
                
                # Calculate differences
                total_disk_read = end_snapshot.disk_read_bytes - start_snapshot.disk_read_bytes
                total_disk_write = end_snapshot.disk_write_bytes - start_snapshot.disk_write_bytes
                total_net_sent = end_snapshot.net_bytes_sent - start_snapshot.net_bytes_sent
                total_net_recv = end_snapshot.net_bytes_recv - start_snapshot.net_bytes_recv
                
                # Create monitoring result
                monitoring_result = MonitoringResult(
                    function_name=func.__name__,
                    execution_time=end_time - start_time,
                    cpu_usage_start=start_snapshot.cpu_percent,
                    cpu_usage_end=end_snapshot.cpu_percent,
                    memory_usage_start=start_snapshot.memory_percent,
                    memory_usage_end=end_snapshot.memory_percent,
                    timestamp=time.time(),
                    # Legacy/advanced fields
                    start_snapshot=start_snapshot,
                    end_snapshot=end_snapshot,
                    peak_cpu=monitor.get_peak_cpu(),
                    peak_memory=monitor.get_peak_memory(),
                    total_disk_read=total_disk_read,
                    total_disk_write=total_disk_write,
                    total_net_sent=total_net_sent,
                    total_net_recv=total_net_recv,
                    duration=end_time - start_time,
                    cpu_samples=monitoring_data["cpu_samples"],
                    memory_samples=monitoring_data["memory_samples"],
                    gpu_snapshots=monitoring_data["gpu_snapshots"],
                    metadata={
                        "sample_interval": sample_interval,
                        "monitor_gpu": monitor_gpu,
                        "samples_count": len(monitoring_data["cpu_samples"]),
                        "args_count": len(args),
                        "kwargs_count": len(kwargs),
                        "async": True
                    }
                )
                
                # Print result if requested
                if print_result:
                    print(f"{func.__name__} (async) - CPU Peak: {monitoring_result.peak_cpu:.1f}%, "
                          f"Memory Peak: {monitoring_result.peak_memory:.1f}%, "
                          f"Duration: {monitoring_result.duration:.3f}s")
                
                # Return result with monitoring info if requested
                if return_result:
                    return result, monitoring_result
                return result
                
            except Exception as e:
                # Clean up monitoring on error
                monitor.stop()
                raise
        
        return wrapper  # type: ignore
    return decorator


@contextmanager
def monitor_system(
    name: str = "code_block",
    sample_interval: float = 0.1,
    monitor_gpu: bool = False,
    print_result: bool = True
):
    """
    Context manager for monitoring system resources of code blocks.
    
    Args:
        name: Name for the monitored operation
        sample_interval: Interval between resource samples (seconds)
        monitor_gpu: Whether to monitor GPU resources
        print_result: Whether to print monitoring results
    
    Yields:
        MonitoringResult object that gets populated with monitoring data
    """
    # Get initial snapshot
    start_snapshot = _get_system_snapshot()
    start_time = time.time()
    
    # Start continuous monitoring
    monitor = SystemMonitor(sample_interval, monitor_gpu)
    monitor.start()
    
    # Create result object that will be populated
    result = MonitoringResult(
        function_name=name,
        execution_time=0.0,
        cpu_usage_start=start_snapshot.cpu_percent,
        cpu_usage_end=start_snapshot.cpu_percent,
        memory_usage_start=start_snapshot.memory_percent,
        memory_usage_end=start_snapshot.memory_percent,
        timestamp=time.time(),
        # Legacy/advanced fields
        start_snapshot=start_snapshot,
        end_snapshot=start_snapshot,
        peak_cpu=0.0,
        peak_memory=0.0,
        total_disk_read=0,
        total_disk_write=0,
        total_net_sent=0,
        total_net_recv=0,
        duration=0.0
    )
    
    try:
        yield result
    finally:
        # Stop monitoring and get data
        monitoring_data = monitor.stop()
        end_time = time.time()
        end_snapshot = _get_system_snapshot()
        
        # Calculate differences
        total_disk_read = end_snapshot.disk_read_bytes - start_snapshot.disk_read_bytes
        total_disk_write = end_snapshot.disk_write_bytes - start_snapshot.disk_write_bytes
        total_net_sent = end_snapshot.net_bytes_sent - start_snapshot.net_bytes_sent
        total_net_recv = end_snapshot.net_bytes_recv - start_snapshot.net_bytes_recv
        
        # Update result object
        result.end_snapshot = end_snapshot
        result.peak_cpu = monitor.get_peak_cpu()
        result.peak_memory = monitor.get_peak_memory()
        result.total_disk_read = total_disk_read
        result.total_disk_write = total_disk_write
        result.total_net_sent = total_net_sent
        result.total_net_recv = total_net_recv
        result.duration = end_time - start_time
        result.cpu_samples = monitoring_data["cpu_samples"]
        result.memory_samples = monitoring_data["memory_samples"]
        result.gpu_snapshots = monitoring_data["gpu_snapshots"]
        result.metadata = {
            "sample_interval": sample_interval,
            "monitor_gpu": monitor_gpu,
            "samples_count": len(monitoring_data["cpu_samples"])
        }
        
        # Print result if requested
        if print_result:
            print(f"{name} - CPU Peak: {result.peak_cpu:.1f}%, "
                  f"Memory Peak: {result.peak_memory:.1f}%, "
                  f"Duration: {result.duration:.3f}s")


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    try:
        # CPU info
        cpu_info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "cpu_percent": psutil.cpu_percent(interval=1),
        }
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_info = {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used,
            "free": memory.free
        }
        
        # Disk info
        disk_info = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_info.append({
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": (usage.used / usage.total) * 100
                })
            except PermissionError:
                continue
        
        # Network info
        network_info = psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
        
        # Process info
        process = psutil.Process()
        process_info = {
            "pid": process.pid,
            "name": process.name(),
            "memory_percent": process.memory_percent(),
            "cpu_percent": process.cpu_percent(),
            "num_threads": process.num_threads(),
            "create_time": process.create_time()
        }
        
        return {
            "cpu": cpu_info,
            "memory": memory_info,
            "disk": disk_info,
            "network": network_info,
            "process": process_info,
            "platform": {
                "system": os.name,
                "platform": "unknown"
            }
        }
        
    except Exception as e:
        return {"error": str(e)}


def get_gpu_info() -> List[Dict[str, Any]]:
    """Get GPU information if available."""
    try:
        import GPUtil  # type: ignore
        gpus = GPUtil.getGPUs()
        return [
            {
                "id": gpu.id,
                "name": gpu.name,
                "driver": gpu.driver,
                "memory_total": gpu.memoryTotal,
                "memory_used": gpu.memoryUsed,
                "memory_free": gpu.memoryFree,
                "temperature": gpu.temperature,
                "load": gpu.load,
                "uuid": gpu.uuid
            }
            for gpu in gpus
        ]
    except ImportError:
        return [{"error": "GPUtil not available"}]
    except Exception as e:
        return [{"error": str(e)}]


# Convenience aliases
monitor_system_resources = system_monitor
monitor_system_resources_async = system_monitor_async