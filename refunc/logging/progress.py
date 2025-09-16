"""
Progress tracking and visualization for ML workflows.

This module provides progress bars, step tracking, and visual
feedback for long-running ML operations and training loops.
"""

import time
import sys
from typing import Optional, Dict, Any, Union, Iterator, Callable, List
from dataclasses import dataclass
from contextlib import contextmanager

try:
    from tqdm import tqdm as tqdm_class
    TQDM_AVAILABLE = True
except ImportError:
    tqdm_class = None
    TQDM_AVAILABLE = False

try:
    import colorama
    colorama.init(autoreset=True)
    COLORAMA_AVAILABLE = True
    # Get colors for use
    GREEN = colorama.Fore.GREEN
    YELLOW = colorama.Fore.YELLOW
    RED = colorama.Fore.RED
    BLUE = colorama.Fore.BLUE
    CYAN = colorama.Fore.CYAN
    MAGENTA = colorama.Fore.MAGENTA
    RESET = colorama.Style.RESET_ALL
    BRIGHT = colorama.Style.BRIGHT
except ImportError:
    COLORAMA_AVAILABLE = False
    GREEN = YELLOW = RED = BLUE = CYAN = MAGENTA = RESET = BRIGHT = ""


@dataclass
class ProgressState:
    """State information for progress tracking."""
    
    current: int = 0
    total: Optional[int] = None
    start_time: float = 0.0
    last_update: float = 0.0
    rate: float = 0.0
    eta: Optional[float] = None
    completed: bool = False
    description: str = ""
    metrics: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.start_time == 0.0:
            self.start_time = time.time()
            self.last_update = self.start_time


class ProgressTracker:
    """
    Advanced progress tracker with metrics and ML-specific features.
    
    Provides progress tracking with rate calculation, ETA estimation,
    metrics display, and integration with logging systems.
    """
    
    def __init__(
        self,
        total: Optional[int] = None,
        description: str = "",
        unit: str = "it",
        unit_scale: bool = False,
        disable: bool = False,
        logger = None,
        log_interval: float = 1.0,
        show_metrics: bool = True,
        show_rate: bool = True,
        show_eta: bool = True,
        width: int = 50
    ):
        self.total = total
        self.description = description
        self.unit = unit
        self.unit_scale = unit_scale
        self.disable = disable
        self.logger = logger
        self.log_interval = log_interval
        self.show_metrics = show_metrics
        self.show_rate = show_rate
        self.show_eta = show_eta
        self.width = width
        
        # State
        self.state = ProgressState(total=total, description=description)
        self._last_log_time = 0.0
        self._metrics_history: List[Dict[str, Any]] = []
        
    def update(self, n: int = 1, **kwargs) -> None:
        """Update progress by n steps."""
        if self.disable:
            return
        
        current_time = time.time()
        
        # Update state
        self.state.current += n
        self.state.last_update = current_time
        
        # Calculate rate
        if self.state.current > 0:
            elapsed = current_time - self.state.start_time
            self.state.rate = self.state.current / elapsed if elapsed > 0 else 0
        
        # Calculate ETA
        if self.total and self.state.rate > 0:
            remaining = self.total - self.state.current
            self.state.eta = remaining / self.state.rate
        
        # Update metrics
        if kwargs:
            if self.state.metrics is None:
                self.state.metrics = {}
            self.state.metrics.update(kwargs)
            self._metrics_history.append({
                'time': current_time,
                'step': self.state.current,
                **kwargs
            })
        
        # Check if completed
        if self.total and self.state.current >= self.total:
            self.state.completed = True
        
        # Display progress
        self._display_progress()
        
        # Log progress if needed
        if self.logger and (current_time - self._last_log_time) >= self.log_interval:
            self._log_progress()
            self._last_log_time = current_time
    
    def set_description(self, description: str) -> None:
        """Set the progress description."""
        self.state.description = description
        self._display_progress()
    
    def set_postfix(self, **kwargs) -> None:
        """Set postfix metrics."""
        if self.state.metrics is None:
            self.state.metrics = {}
        self.state.metrics.update(kwargs)
        self._display_progress()
    
    def _display_progress(self) -> None:
        """Display the progress bar."""
        if self.disable:
            return
        
        # Build progress bar string
        progress_str = self._build_progress_string()
        
        # Write to stderr (like tqdm)
        sys.stderr.write(f'\r{progress_str}')
        sys.stderr.flush()
        
        if self.state.completed:
            sys.stderr.write('\n')
    
    def _build_progress_string(self) -> str:
        """Build the progress bar string."""
        parts = []
        
        # Description
        if self.state.description:
            parts.append(f"{GREEN}{self.state.description}{RESET}")
        
        # Progress percentage
        if self.total:
            percent = (self.state.current / self.total) * 100
            parts.append(f"{percent:3.0f}%")
        
        # Progress bar
        if self.total:
            bar = self._create_progress_bar()
            parts.append(bar)
        
        # Current/total
        if self.total:
            parts.append(f"{self.state.current}/{self.total}")
        else:
            parts.append(str(self.state.current))
        
        # Rate
        if self.show_rate and self.state.rate > 0:
            if self.unit_scale and self.state.rate >= 1000:
                rate_str = f"{self.state.rate/1000:.1f}k{self.unit}/s"
            else:
                rate_str = f"{self.state.rate:.1f}{self.unit}/s"
            parts.append(f"[{CYAN}{rate_str}{RESET}]")
        
        # ETA
        if self.show_eta and self.state.eta:
            eta_str = self._format_time(self.state.eta)
            parts.append(f"<{YELLOW}{eta_str}{RESET}>")
        
        # Metrics
        if self.show_metrics and self.state.metrics:
            metrics_str = self._format_metrics(self.state.metrics)
            if metrics_str:
                parts.append(f"({metrics_str})")
        
        return " ".join(parts)
    
    def _create_progress_bar(self) -> str:
        """Create the visual progress bar."""
        if not self.total:
            return ""
        
        filled_width = int((self.state.current / self.total) * self.width)
        bar = "█" * filled_width + "░" * (self.width - filled_width)
        
        if COLORAMA_AVAILABLE:
            # Color the filled portion
            filled_colored = f"{GREEN}{'█' * filled_width}{RESET}"
            remaining = "░" * (self.width - filled_width)
            bar = f"|{filled_colored}{remaining}|"
        else:
            bar = f"|{bar}|"
        
        return bar
    
    def _format_time(self, seconds: float) -> str:
        """Format time duration."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for display."""
        formatted = []
        for key, value in metrics.items():
            if isinstance(value, float):
                if abs(value) < 0.01 or abs(value) > 1000:
                    formatted.append(f"{key}={value:.2e}")
                else:
                    formatted.append(f"{key}={value:.3f}")
            else:
                formatted.append(f"{key}={value}")
        return ", ".join(formatted)
    
    def _log_progress(self) -> None:
        """Log progress to the logger."""
        if not self.logger:
            return
        
        message = f"Progress: {self.state.current}"
        if self.total:
            percent = (self.state.current / self.total) * 100
            message += f"/{self.total} ({percent:.1f}%)"
        
        if self.state.rate > 0:
            message += f" [{self.state.rate:.1f}{self.unit}/s]"
        
        if self.state.eta:
            eta_str = self._format_time(self.state.eta)
            message += f" ETA: {eta_str}"
        
        self.logger.progress(message, metrics=self.state.metrics)
    
    def __iter__(self):
        """Make ProgressTracker iterable for use with yield from."""
        # This should not be used directly but helps with type checking
        raise RuntimeError("ProgressTracker is not meant to be iterated directly. Use progress_context() or track_progress().")
    
    def close(self) -> None:
        """Close the progress tracker."""
        if not self.disable and not self.state.completed:
            sys.stderr.write('\n')
    
    def reset(self) -> None:
        """Reset the progress tracker."""
        self.state = ProgressState(total=self.total, description=self.description)
        self._last_log_time = 0.0
        self._metrics_history.clear()
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get the history of metrics."""
        return self._metrics_history.copy()


class TqdmProgressTracker(ProgressTracker):
    """
    Progress tracker using tqdm backend.
    
    Falls back to basic tracker if tqdm is not available.
    """
    
    def __init__(self, *args, **kwargs):
        if not TQDM_AVAILABLE:
            super().__init__(*args, **kwargs)
            self._use_tqdm = False
            return
        
        # Extract our custom parameters
        logger = kwargs.pop('logger', None)
        log_interval = kwargs.pop('log_interval', 1.0)
        show_metrics = kwargs.pop('show_metrics', True)
        
        # Create tqdm instance
        if TQDM_AVAILABLE and tqdm_class:
            self._tqdm = tqdm_class(*args, **kwargs)
        else:
            raise ImportError("tqdm is not available")
        self._use_tqdm = True
        
        # Setup our tracking
        self.logger = logger
        self.log_interval = log_interval
        self.show_metrics = show_metrics
        self._last_log_time = 0.0
        self._metrics_history: List[Dict[str, Any]] = []
        self._current_metrics: Dict[str, Any] = {}
    
    def update(self, n: int = 1, **kwargs) -> None:
        """Update progress."""
        if not self._use_tqdm:
            super().update(n, **kwargs)
            return
        
        current_time = time.time()
        
        # Update tqdm
        self._tqdm.update(n)
        
        # Track metrics
        if kwargs:
            self._current_metrics.update(kwargs)
            self._metrics_history.append({
                'time': current_time,
                'step': self._tqdm.n,
                **kwargs
            })
            
            # Update tqdm postfix
            if self.show_metrics:
                self._tqdm.set_postfix(**self._current_metrics)
        
        # Log if needed
        if self.logger and (current_time - self._last_log_time) >= self.log_interval:
            self._log_progress()
            self._last_log_time = current_time
    
    def set_description(self, description: str) -> None:
        """Set description."""
        if self._use_tqdm:
            self._tqdm.set_description(description)
        else:
            super().set_description(description)
    
    def set_postfix(self, **kwargs) -> None:
        """Set postfix."""
        if self._use_tqdm:
            self._current_metrics.update(kwargs)
            if self.show_metrics:
                self._tqdm.set_postfix(**self._current_metrics)
        else:
            super().set_postfix(**kwargs)
    
    def _log_progress(self) -> None:
        """Log progress."""
        if not self.logger:
            return
        
        if self._use_tqdm:
            message = f"Progress: {self._tqdm.n}"
            if self._tqdm.total:
                percent = (self._tqdm.n / self._tqdm.total) * 100
                message += f"/{self._tqdm.total} ({percent:.1f}%)"
            
            if hasattr(self._tqdm, 'avg') and getattr(self._tqdm, 'avg', None):
                message += f" [{getattr(self._tqdm, 'avg'):.1f}it/s]"
            
            self.logger.progress(message, metrics=self._current_metrics)
        else:
            super()._log_progress()
    
    def close(self) -> None:
        """Close the tracker."""
        if self._use_tqdm:
            self._tqdm.close()
        else:
            super().close()
    
    def reset(self) -> None:
        """Reset the tracker."""
        if self._use_tqdm:
            self._tqdm.reset()
            self._current_metrics.clear()
            self._metrics_history.clear()
        else:
            super().reset()
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get metrics history."""
        return self._metrics_history.copy()


def create_progress_tracker(
    total: Optional[int] = None,
    description: str = "",
    use_tqdm: bool = True,
    **kwargs
) -> ProgressTracker:
    """
    Create a progress tracker.
    
    Args:
        total: Total number of iterations
        description: Description text
        use_tqdm: Whether to use tqdm if available
        **kwargs: Additional arguments
    
    Returns:
        Progress tracker instance
    """
    if use_tqdm and TQDM_AVAILABLE:
        return TqdmProgressTracker(total=total, desc=description, **kwargs)
    else:
        return ProgressTracker(total=total, description=description, **kwargs)


@contextmanager
def progress_context(
    iterable = None,
    total: Optional[int] = None,
    description: str = "",
    **kwargs
):
    """
    Context manager for progress tracking.
    
    Args:
        iterable: Iterable to track
        total: Total iterations (if iterable is None)
        description: Progress description
        **kwargs: Additional tracker arguments
    
    Yields:
        Progress tracker or wrapped iterable
    """
    if iterable is not None:
        # Wrap iterable
        if hasattr(iterable, '__len__'):
            total = len(iterable)
        
        tracker = create_progress_tracker(
            total=total,
            description=description,
            **kwargs
        )
        
        try:
            def wrapped_iterable():
                for item in iterable:
                    yield item
                    tracker.update(1)
            
            yield wrapped_iterable()
        finally:
            tracker.close()
    else:
        # Return tracker for manual updates
        tracker = create_progress_tracker(
            total=total,
            description=description,
            **kwargs
        )
        
        try:
            yield tracker
        finally:
            tracker.close()


class EpochTracker:
    """
    Specialized tracker for ML training epochs.
    
    Tracks both epoch-level and step-level progress with
    comprehensive metrics logging.
    """
    
    def __init__(
        self,
        total_epochs: int,
        steps_per_epoch: Optional[int] = None,
        logger = None,
        epoch_description: str = "Epoch",
        step_description: str = "Step"
    ):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.logger = logger
        self.epoch_description = epoch_description
        self.step_description = step_description
        
        # Current state
        self.current_epoch = 0
        self.current_step = 0
        self.epoch_start_time = 0.0
        self.step_start_time = 0.0
        
        # Metrics tracking
        self.epoch_metrics: Dict[str, Any] = {}
        self.step_metrics: Dict[str, Any] = {}
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Progress trackers
        self.epoch_tracker: Optional[ProgressTracker] = None
        self.step_tracker: Optional[ProgressTracker] = None
    
    def start_epoch(self, epoch: int) -> None:
        """Start a new epoch."""
        self.current_epoch = epoch
        self.current_step = 0
        self.epoch_start_time = time.time()
        self.epoch_metrics.clear()
        
        # Create epoch tracker
        if self.epoch_tracker:
            self.epoch_tracker.close()
        
        self.epoch_tracker = create_progress_tracker(
            total=self.total_epochs,
            description=f"{self.epoch_description}",
            logger=self.logger,
            position=0,
            leave=True
        )
        self.epoch_tracker.update(epoch)
        
        # Create step tracker if we know steps per epoch
        if self.steps_per_epoch:
            if self.step_tracker:
                self.step_tracker.close()
            
            self.step_tracker = create_progress_tracker(
                total=self.steps_per_epoch,
                description=f"  {self.step_description} ",
                logger=None,  # Don't double-log
                position=1,
                leave=False
            )
        
        if self.logger:
            self.logger.info(f"Starting epoch {epoch}/{self.total_epochs}")
    
    def update_step(self, step_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Update step progress."""
        self.current_step += 1
        
        if step_metrics:
            self.step_metrics.update(step_metrics)
        
        # Update step tracker
        if self.step_tracker:
            self.step_tracker.update(1, **step_metrics or {})
        
        # Log step metrics
        if self.logger and step_metrics:
            self.logger.metric(
                f"Step {self.current_step}",
                metrics=step_metrics,
                step=self.current_step,
                epoch=self.current_epoch
            )
    
    def end_epoch(self, epoch_metrics: Optional[Dict[str, Any]] = None) -> None:
        """End the current epoch."""
        epoch_duration = time.time() - self.epoch_start_time
        
        if epoch_metrics:
            self.epoch_metrics.update(epoch_metrics)
        
        # Add timing metrics
        self.epoch_metrics['epoch_duration'] = epoch_duration
        if self.current_step > 0:
            self.epoch_metrics['avg_step_time'] = epoch_duration / self.current_step
        
        # Store in history
        history_entry = {
            'epoch': self.current_epoch,
            'steps': self.current_step,
            'duration': epoch_duration,
            **self.epoch_metrics
        }
        self.metrics_history.append(history_entry)
        
        # Update epoch tracker
        if self.epoch_tracker:
            self.epoch_tracker.set_postfix(**self.epoch_metrics)
        
        # Close step tracker
        if self.step_tracker:
            self.step_tracker.close()
            self.step_tracker = None
        
        # Log epoch completion
        if self.logger:
            self.logger.info(
                f"Completed epoch {self.current_epoch}/{self.total_epochs}",
                metrics=self.epoch_metrics
            )
    
    def close(self) -> None:
        """Close all trackers."""
        if self.epoch_tracker:
            self.epoch_tracker.close()
        if self.step_tracker:
            self.step_tracker.close()
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get the full metrics history."""
        return self.metrics_history.copy()


@contextmanager
def epoch_context(
    total_epochs: int,
    steps_per_epoch: Optional[int] = None,
    logger = None,
    **kwargs
):
    """
    Context manager for epoch tracking.
    
    Args:
        total_epochs: Total number of epochs
        steps_per_epoch: Steps per epoch (optional)
        logger: Logger instance
        **kwargs: Additional arguments
    
    Yields:
        EpochTracker instance
    """
    tracker = EpochTracker(
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        logger=logger,
        **kwargs
    )
    
    try:
        yield tracker
    finally:
        tracker.close()


# Convenience functions
def track_progress(
    iterable,
    description: str = "",
    **kwargs
) -> Iterator:
    """
    Track progress of an iterable.
    
    Args:
        iterable: Iterable to track
        description: Progress description
        **kwargs: Additional arguments
    
    Yields:
        Items from iterable
    """
    if iterable is None:
        raise ValueError("track_progress requires an iterable. Use progress_context() directly for manual tracking.")
    
    with progress_context(iterable, description=description, **kwargs) as wrapped:
        yield from wrapped


def track_epochs(
    epochs: Union[int, range],
    steps_per_epoch: Optional[int] = None,
    logger = None,
    **kwargs
) -> Iterator[int]:
    """
    Track training epochs.
    
    Args:
        epochs: Number of epochs or range
        steps_per_epoch: Steps per epoch
        logger: Logger instance
        **kwargs: Additional arguments
    
    Yields:
        Epoch numbers
    """
    if isinstance(epochs, int):
        epochs = range(1, epochs + 1)
    
    total_epochs = len(epochs) if hasattr(epochs, '__len__') else max(epochs)
    
    with epoch_context(total_epochs, steps_per_epoch, logger, **kwargs) as tracker:
        for epoch in epochs:
            tracker.start_epoch(epoch)
            yield epoch
            # Note: User should call tracker.end_epoch() when done


# Progress utilities
def format_bytes(num_bytes: float) -> str:
    """Format bytes in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}PB"


def format_time(seconds: float) -> str:
    """Format time duration in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def estimate_remaining_time(current: int, total: int, elapsed: float) -> Optional[float]:
    """Estimate remaining time based on progress."""
    if current <= 0 or total <= 0 or current >= total:
        return None
    
    rate = current / elapsed
    remaining_items = total - current
    
    return remaining_items / rate if rate > 0 else None