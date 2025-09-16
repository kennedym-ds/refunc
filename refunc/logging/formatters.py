"""
Log formatters for different output formats and styles.

This module provides various formatters for ML logging including
colored console output, JSON formatting, and ML-specific formatting.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional, Literal

try:
    import colorama
    colorama.init(autoreset=True)
    COLORAMA_AVAILABLE = True
    # Import constants to use them
    FORE_BLACK = colorama.Fore.BLACK
    FORE_RED = colorama.Fore.RED
    FORE_GREEN = colorama.Fore.GREEN
    FORE_YELLOW = colorama.Fore.YELLOW
    FORE_BLUE = colorama.Fore.BLUE
    FORE_MAGENTA = colorama.Fore.MAGENTA
    FORE_CYAN = colorama.Fore.CYAN
    FORE_WHITE = colorama.Fore.WHITE
    FORE_LIGHTBLACK_EX = colorama.Fore.LIGHTBLACK_EX
    FORE_LIGHTRED_EX = colorama.Fore.LIGHTRED_EX
    FORE_LIGHTGREEN_EX = colorama.Fore.LIGHTGREEN_EX
    FORE_LIGHTYELLOW_EX = colorama.Fore.LIGHTYELLOW_EX
    FORE_LIGHTBLUE_EX = colorama.Fore.LIGHTBLUE_EX
    FORE_LIGHTMAGENTA_EX = colorama.Fore.LIGHTMAGENTA_EX
    FORE_LIGHTCYAN_EX = colorama.Fore.LIGHTCYAN_EX
    FORE_LIGHTWHITE_EX = colorama.Fore.LIGHTWHITE_EX
    STYLE_DIM = colorama.Style.DIM
    STYLE_NORMAL = colorama.Style.NORMAL
    STYLE_BRIGHT = colorama.Style.BRIGHT
    STYLE_RESET_ALL = colorama.Style.RESET_ALL
except ImportError:
    COLORAMA_AVAILABLE = False
    # Fallback constants
    FORE_BLACK = FORE_RED = FORE_GREEN = FORE_YELLOW = FORE_BLUE = ""
    FORE_MAGENTA = FORE_CYAN = FORE_WHITE = FORE_LIGHTBLACK_EX = ""
    FORE_LIGHTRED_EX = FORE_LIGHTGREEN_EX = FORE_LIGHTYELLOW_EX = ""
    FORE_LIGHTBLUE_EX = FORE_LIGHTMAGENTA_EX = FORE_LIGHTCYAN_EX = ""
    FORE_LIGHTWHITE_EX = STYLE_DIM = STYLE_NORMAL = STYLE_BRIGHT = ""
    STYLE_RESET_ALL = ""


class BaseFormatter(logging.Formatter):
    """Base formatter with common functionality."""
    
    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: Literal['%', '{', '$'] = '%',
        validate: bool = True
    ):
        super().__init__(fmt, datefmt, style, validate)
    
    def format_timestamp(self, timestamp: float) -> str:
        """Format timestamp consistently."""
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Include milliseconds
    
    def extract_metrics(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Extract metrics from log record."""
        metrics = {}
        
        # Check for log_entry in extra
        if hasattr(record, 'log_entry') and getattr(record, 'log_entry'):
            log_entry = getattr(record, 'log_entry')
            if hasattr(log_entry, 'metrics'):
                metrics.update(log_entry.metrics)
        
        # Check for direct metrics in extra
        if hasattr(record, 'metrics') and getattr(record, 'metrics'):
            metrics.update(getattr(record, 'metrics'))
        
        return metrics
    
    def extract_tags(self, record: logging.LogRecord) -> Dict[str, str]:
        """Extract tags from log record."""
        tags = {}
        
        # Check for log_entry in extra
        if hasattr(record, 'log_entry') and getattr(record, 'log_entry'):
            log_entry = getattr(record, 'log_entry')
            if hasattr(log_entry, 'tags'):
                tags.update(log_entry.tags)
        
        # Check for direct tags in extra
        if hasattr(record, 'tags') and getattr(record, 'tags'):
            tags.update(getattr(record, 'tags'))
        
        return tags


class ColoredFormatter(BaseFormatter):
    """Colored console formatter for better readability."""
    
    # Color mapping for log levels
    LEVEL_COLORS = {
        'TRACE': FORE_LIGHTBLACK_EX,
        'DEBUG': FORE_CYAN,
        'INFO': FORE_GREEN,
        'METRIC': FORE_LIGHTBLUE_EX,
        'EXPERIMENT': FORE_MAGENTA,
        'PROGRESS': FORE_YELLOW,
        'RESULT': FORE_LIGHTGREEN_EX,
        'WARNING': FORE_YELLOW,
        'ERROR': FORE_RED,
        'CRITICAL': FORE_LIGHTRED_EX + STYLE_BRIGHT,
    }
    
    # Icons for different log types
    LEVEL_ICONS = {
        'TRACE': '🔍',
        'DEBUG': '🐛',
        'INFO': 'ℹ️ ',
        'METRIC': '📊',
        'EXPERIMENT': '🧪',
        'PROGRESS': '⏳',
        'RESULT': '✅',
        'WARNING': '⚠️ ',
        'ERROR': '❌',
        'CRITICAL': '💥',
    }
    
    def __init__(
        self,
        use_colors: bool = True,
        use_icons: bool = True,
        show_timestamp: bool = True,
        show_logger: bool = True,
        show_level: bool = True,
        compact: bool = False
    ):
        self.use_colors = use_colors and COLORAMA_AVAILABLE
        self.use_icons = use_icons
        self.show_timestamp = show_timestamp
        self.show_logger = show_logger
        self.show_level = show_level
        self.compact = compact
        
        # Build format string based on options
        fmt_parts = []
        
        if self.show_timestamp and not self.compact:
            fmt_parts.append('%(asctime)s')
        elif self.show_timestamp and self.compact:
            fmt_parts.append('%(asctime)s')
        
        if self.show_level:
            fmt_parts.append('%(levelname)s')
        
        if self.show_logger and not self.compact:
            fmt_parts.append('[%(name)s]')
        
        fmt_parts.append('%(message)s')
        
        fmt_string = ' | ' if not self.compact else ' '
        fmt_string = fmt_string.join(fmt_parts)
        
        super().__init__(fmt_string, datefmt='%H:%M:%S' if self.compact else '%Y-%m-%d %H:%M:%S')
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors and icons."""
        # Get base formatted message
        formatted = super().format(record)
        
        if not self.use_colors:
            return formatted
        
        level_name = record.levelname
        color = self.LEVEL_COLORS.get(level_name, FORE_WHITE)
        icon = self.LEVEL_ICONS.get(level_name, '') if self.use_icons else ''
        
        # Apply color to level name
        if self.show_level:
            formatted = formatted.replace(
                level_name,
                f"{color}{icon}{level_name}{STYLE_RESET_ALL}"
            )
        
        # Extract and format metrics
        metrics = self.extract_metrics(record)
        if metrics:
            metrics_str = self._format_metrics(metrics)
            formatted += f" {FORE_LIGHTBLUE_EX}[{metrics_str}]{STYLE_RESET_ALL}"
        
        # Extract and format tags
        tags = self.extract_tags(record)
        if tags:
            tags_str = self._format_tags(tags)
            formatted += f" {FORE_LIGHTMAGENTA_EX}[{tags_str}]{STYLE_RESET_ALL}"
        
        return formatted
    
    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for display."""
        formatted_metrics = []
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_metrics.append(f"{key}={value:.4f}")
            else:
                formatted_metrics.append(f"{key}={value}")
        return " ".join(formatted_metrics)
    
    def _format_tags(self, tags: Dict[str, str]) -> str:
        """Format tags for display."""
        return " ".join(f"{key}:{value}" for key, value in tags.items())


class JSONFormatter(BaseFormatter):
    """JSON formatter for structured logging."""
    
    def __init__(
        self,
        include_extra: bool = True,
        sort_keys: bool = True,
        ensure_ascii: bool = False
    ):
        self.include_extra = include_extra
        self.sort_keys = sort_keys
        self.ensure_ascii = ensure_ascii
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        # Base log data
        log_data = {
            'timestamp': record.created,
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add thread and process info if available
        if hasattr(record, 'thread') and record.thread:
            log_data['thread'] = record.thread
        if hasattr(record, 'process') and record.process:
            log_data['process'] = record.process
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add log entry data if available
        if hasattr(record, 'log_entry') and getattr(record, 'log_entry'):
            entry = getattr(record, 'log_entry')
            log_data.update({
                'experiment_id': getattr(entry, 'experiment_id', None),
                'run_id': getattr(entry, 'run_id', None),
                'step': getattr(entry, 'step', None),
                'epoch': getattr(entry, 'epoch', None),
                'metrics': getattr(entry, 'metrics', {}),
                'tags': getattr(entry, 'tags', {}),
                'artifacts': getattr(entry, 'artifacts', []),
                'extra': getattr(entry, 'extra', {})
            })
        
        # Add extra fields if requested
        if self.include_extra:
            extra_fields = {
                key: value for key, value in record.__dict__.items()
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'lineno', 'funcName', 'created',
                    'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'getMessage', 'exc_info',
                    'exc_text', 'stack_info', 'log_entry'
                }
            }
            if extra_fields:
                log_data['extra_fields'] = extra_fields
        
        # Convert to JSON
        try:
            return json.dumps(
                log_data,
                sort_keys=self.sort_keys,
                ensure_ascii=self.ensure_ascii,
                default=self._json_default
            )
        except (TypeError, ValueError) as e:
            # Fallback for non-serializable objects
            log_data['serialization_error'] = str(e)
            return json.dumps(log_data, default=str)
    
    def _json_default(self, obj: Any) -> Any:
        """Default JSON serializer for non-standard types."""
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)


class MLFormatter(BaseFormatter):
    """ML-specific formatter with experiment context."""
    
    def __init__(
        self,
        show_experiment: bool = True,
        show_step: bool = True,
        show_metrics: bool = True,
        compact: bool = False
    ):
        self.show_experiment = show_experiment
        self.show_step = show_step
        self.show_metrics = show_metrics
        self.compact = compact
        
        if compact:
            fmt = '%(asctime)s | %(levelname)-8s | %(message)s'
            datefmt = '%H:%M:%S'
        else:
            fmt = '%(asctime)s | %(levelname)-8s | [%(name)s] | %(message)s'
            datefmt = '%Y-%m-%d %H:%M:%S'
        
        super().__init__(fmt, datefmt)
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with ML context."""
        # Get base formatted message
        formatted = super().format(record)
        
        # Add experiment context if available
        if hasattr(record, 'log_entry') and getattr(record, 'log_entry'):
            entry = getattr(record, 'log_entry')
            context_parts = []
            
            experiment_id = getattr(entry, 'experiment_id', None)
            run_id = getattr(entry, 'run_id', None)
            step = getattr(entry, 'step', None)
            epoch = getattr(entry, 'epoch', None)
            metrics = getattr(entry, 'metrics', {})
            tags = getattr(entry, 'tags', {})
            
            if self.show_experiment and experiment_id:
                exp_short = experiment_id[:8]
                if run_id:
                    run_short = run_id[:8]
                    context_parts.append(f"exp:{exp_short}/run:{run_short}")
                else:
                    context_parts.append(f"exp:{exp_short}")
            
            if self.show_step and (step is not None or epoch is not None):
                step_parts = []
                if epoch is not None:
                    step_parts.append(f"epoch:{epoch}")
                if step is not None:
                    step_parts.append(f"step:{step}")
                context_parts.append("/".join(step_parts))
            
            if context_parts:
                formatted += f" [{'/'.join(context_parts)}]"
            
            # Add metrics if available and requested
            if self.show_metrics and metrics:
                metrics_str = self._format_metrics(metrics)
                formatted += f" metrics:({metrics_str})"
            
            # Add tags if available
            if tags:
                tags_str = self._format_tags(tags)
                formatted += f" tags:({tags_str})"
        
        return formatted
    
    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for display."""
        formatted_metrics = []
        for key, value in metrics.items():
            if isinstance(value, float):
                if abs(value) < 0.01 or abs(value) > 1000:
                    formatted_metrics.append(f"{key}={value:.2e}")
                else:
                    formatted_metrics.append(f"{key}={value:.4f}")
            else:
                formatted_metrics.append(f"{key}={value}")
        return ", ".join(formatted_metrics)
    
    def _format_tags(self, tags: Dict[str, str]) -> str:
        """Format tags for display."""
        return ", ".join(f"{key}:{value}" for key, value in tags.items())


class CompactFormatter(BaseFormatter):
    """Compact formatter for minimal output."""
    
    def __init__(self, show_level: bool = True, show_time: bool = False):
        self.show_level = show_level
        self.show_time = show_time
        
        fmt_parts = []
        if self.show_time:
            fmt_parts.append('%(asctime)s')
        if self.show_level:
            fmt_parts.append('%(levelname)s')
        fmt_parts.append('%(message)s')
        
        fmt_string = ' ' if not self.show_time else ' | '
        fmt_string = fmt_string.join(fmt_parts)
        
        super().__init__(fmt_string, datefmt='%H:%M:%S')
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record compactly."""
        formatted = super().format(record)
        
        # Add key metrics only
        if hasattr(record, 'log_entry') and getattr(record, 'log_entry'):
            entry = getattr(record, 'log_entry')
            metrics = getattr(entry, 'metrics', {})
            
            if metrics:
                # Show only the first few important metrics
                important_metrics = ['loss', 'accuracy', 'error', 'score', 'mse', 'mae', 'f1']
                key_metrics = {k: v for k, v in metrics.items() 
                              if any(im in k.lower() for im in important_metrics)}
                
                if key_metrics:
                    metrics_str = " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" 
                                         for k, v in list(key_metrics.items())[:3])
                    formatted += f" ({metrics_str})"
        
        return formatted


class ProgressFormatter(BaseFormatter):
    """Formatter optimized for progress logging."""
    
    def __init__(self):
        super().__init__('%(message)s')
    
    def format(self, record: logging.LogRecord) -> str:
        """Format progress messages."""
        message = super().format(record)
        
        # Add progress indicators
        if hasattr(record, 'log_entry') and getattr(record, 'log_entry'):
            entry = getattr(record, 'log_entry')
            
            step = getattr(entry, 'step', None)
            epoch = getattr(entry, 'epoch', None)
            metrics = getattr(entry, 'metrics', {})
            
            if step is not None or epoch is not None:
                progress_parts = []
                if epoch is not None:
                    progress_parts.append(f"Epoch {epoch}")
                if step is not None:
                    progress_parts.append(f"Step {step}")
                
                if progress_parts:
                    message = f"[{'/'.join(progress_parts)}] {message}"
            
            # Add key metrics inline
            if metrics:
                important_metrics = ['loss', 'accuracy', 'lr', 'learning_rate']
                key_metrics = {k: v for k, v in metrics.items() 
                              if any(im in k.lower() for im in important_metrics)}
                
                if key_metrics:
                    metrics_str = " | ".join(
                        f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                        for k, v in key_metrics.items()
                    )
                    message += f" - {metrics_str}"
        
        return message