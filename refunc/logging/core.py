"""
Core logging functionality for ML workflows.

This module provides the MLLogger class and core logging infrastructure
for experiment tracking, structured logging, and ML-specific logging patterns.
"""

import logging
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TextIO
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from datetime import datetime

from ..exceptions import RefuncError


@dataclass
class LogEntry:
    """Structured log entry for ML workflows."""
    
    timestamp: float
    level: str
    message: str
    logger_name: str
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    step: Optional[int] = None
    epoch: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class ExperimentContext:
    """Context for ML experiments."""
    
    experiment_id: str
    experiment_name: str
    run_id: str
    run_name: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "running"
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LogLevel:
    """Extended log levels for ML workflows."""
    
    TRACE = 5
    DEBUG = 10
    INFO = 20
    METRIC = 25  # Custom level for metrics
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    
    # ML-specific levels
    EXPERIMENT = 22
    PROGRESS = 23
    RESULT = 24


class MLLogger:
    """
    Advanced logger for ML workflows with experiment tracking.
    
    Provides structured logging, colored output, progress tracking,
    and integration with ML experiment tracking platforms.
    """
    
    def __init__(
        self,
        name: str = "refunc",
        level: Union[int, str] = LogLevel.INFO,
        log_dir: Optional[Union[str, Path]] = None,
        experiment_tracking: bool = True,
        colored_output: bool = True,
        json_logging: bool = False,
        max_log_files: int = 10,
        max_file_size: str = "100MB"
    ):
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set experiment tracking first
        self.experiment_tracking = experiment_tracking
        
        # Create base logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        
        # Clear any existing handlers
        self._logger.handlers.clear()
        
        # Setup handlers
        self._setup_handlers(colored_output, json_logging, max_log_files, max_file_size)
        
        # Experiment tracking state
        self._current_experiment: Optional[ExperimentContext] = None
        self._current_step = 0
        self._current_epoch = 0
        
        # Log entries for structured access
        self._log_entries: List[LogEntry] = []
        
        # Register custom levels
        self._register_custom_levels()
    
    def _setup_handlers(
        self,
        colored_output: bool,
        json_logging: bool,
        max_log_files: int,
        max_file_size: str
    ) -> None:
        """Setup logging handlers."""
        
        # Console handler with basic formatting
        console_handler = logging.StreamHandler()
        
        # Use basic formatter to avoid circular imports
        if colored_output:
            # Try to use colored formatter if available
            try:
                from .formatters import ColoredFormatter
                console_handler.setFormatter(ColoredFormatter())
            except (ImportError, AttributeError):
                # Fall back to basic formatter
                console_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
                ))
        else:
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
            ))
        
        self._logger.addHandler(console_handler)
        
        # File handler with basic formatting
        log_file = self.log_dir / f"{self.name}.log"
        
        try:
            from .handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                filename=str(log_file),
                max_files=max_log_files,
                max_size=max_file_size
            )
        except (ImportError, AttributeError):
            # Fall back to basic file handler
            file_handler = logging.FileHandler(str(log_file))
        
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
        ))
        self._logger.addHandler(file_handler)
        
        # JSON handler if requested
        if json_logging:
            try:
                from .handlers import RotatingFileHandler
                from .formatters import JSONFormatter
                
                json_file = self.log_dir / f"{self.name}.json"
                json_handler = RotatingFileHandler(
                    filename=str(json_file),
                    max_files=max_log_files,
                    max_size=max_file_size
                )
                json_handler.setFormatter(JSONFormatter())
                self._logger.addHandler(json_handler)
            except (ImportError, AttributeError):
                pass
        
        # Experiment handler if tracking enabled
        if self.experiment_tracking:
            try:
                from .handlers import ExperimentHandler
                exp_handler = ExperimentHandler(self.log_dir / "experiments")
                self._logger.addHandler(exp_handler)
            except (ImportError, AttributeError):
                pass
    
    def _register_custom_levels(self) -> None:
        """Register custom logging levels."""
        logging.addLevelName(LogLevel.TRACE, "TRACE")
        logging.addLevelName(LogLevel.METRIC, "METRIC")
        logging.addLevelName(LogLevel.EXPERIMENT, "EXPERIMENT")
        logging.addLevelName(LogLevel.PROGRESS, "PROGRESS")
        logging.addLevelName(LogLevel.RESULT, "RESULT")
    
    def _create_log_entry(
        self,
        level: str,
        message: str,
        metrics: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        artifacts: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> LogEntry:
        """Create a structured log entry."""
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            logger_name=self.name,
            experiment_id=self._current_experiment.experiment_id if self._current_experiment else None,
            run_id=self._current_experiment.run_id if self._current_experiment else None,
            step=self._current_step,
            epoch=self._current_epoch,
            metrics=metrics or {},
            tags=tags or {},
            artifacts=artifacts or [],
            extra=extra or {}
        )
        
        self._log_entries.append(entry)
        return entry
    
    # Basic logging methods
    def trace(self, message: str, **kwargs) -> None:
        """Log trace message."""
        entry = self._create_log_entry("TRACE", message, **kwargs)
        self._logger.log(LogLevel.TRACE, message, extra={"log_entry": entry})
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        entry = self._create_log_entry("DEBUG", message, **kwargs)
        self._logger.debug(message, extra={"log_entry": entry})
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        entry = self._create_log_entry("INFO", message, **kwargs)
        self._logger.info(message, extra={"log_entry": entry})
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        entry = self._create_log_entry("WARNING", message, **kwargs)
        self._logger.warning(message, extra={"log_entry": entry})
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        entry = self._create_log_entry("ERROR", message, **kwargs)
        self._logger.error(message, extra={"log_entry": entry})
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        entry = self._create_log_entry("CRITICAL", message, **kwargs)
        self._logger.critical(message, extra={"log_entry": entry})
    
    # ML-specific logging methods
    def metric(
        self, 
        message: str, 
        metrics: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None,
        **kwargs
    ) -> None:
        """Log metrics."""
        if step is not None:
            self._current_step = step
        
        entry = self._create_log_entry("METRIC", message, metrics=metrics, **kwargs)
        self._logger.log(LogLevel.METRIC, message, extra={"log_entry": entry})
        
        # Update experiment metrics
        if self._current_experiment and metrics:
            self._current_experiment.metrics.update(metrics)
    
    def experiment(self, message: str, **kwargs) -> None:
        """Log experiment information."""
        entry = self._create_log_entry("EXPERIMENT", message, **kwargs)
        self._logger.log(LogLevel.EXPERIMENT, message, extra={"log_entry": entry})
    
    def progress(self, message: str, **kwargs) -> None:
        """Log progress information."""
        entry = self._create_log_entry("PROGRESS", message, **kwargs)
        self._logger.log(LogLevel.PROGRESS, message, extra={"log_entry": entry})
    
    def result(self, message: str, **kwargs) -> None:
        """Log results."""
        entry = self._create_log_entry("RESULT", message, **kwargs)
        self._logger.log(LogLevel.RESULT, message, extra={"log_entry": entry})
    
    # Experiment management
    def start_experiment(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Start a new experiment."""
        experiment_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        
        self._current_experiment = ExperimentContext(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            run_id=run_id,
            run_name=run_name,
            parameters=parameters or {},
            tags=tags or {}
        )
        
        self._current_step = 0
        self._current_epoch = 0
        
        self.experiment(
            f"Started experiment: {experiment_name}",
            extra={
                "experiment_id": experiment_id,
                "run_id": run_id,
                "parameters": parameters,
                "tags": tags
            }
        )
        
        return experiment_id
    
    def end_experiment(self, status: str = "completed") -> None:
        """End the current experiment."""
        if not self._current_experiment:
            self.warning("No active experiment to end")
            return
        
        self._current_experiment.end_time = time.time()
        self._current_experiment.status = status
        
        duration = self._current_experiment.end_time - self._current_experiment.start_time
        
        self.experiment(
            f"Ended experiment: {self._current_experiment.experiment_name}",
            extra={
                "experiment_id": self._current_experiment.experiment_id,
                "run_id": self._current_experiment.run_id,
                "status": status,
                "duration": duration
            }
        )
        
        self._current_experiment = None
    
    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """Log experiment parameters."""
        if self._current_experiment:
            self._current_experiment.parameters.update(parameters)
        
        self.info("Parameters logged", extra={"parameters": parameters})
    
    def log_metrics(
        self, 
        metrics: Dict[str, Any], 
        step: Optional[int] = None,
        commit: bool = True
    ) -> None:
        """Log metrics for the current step."""
        if step is not None:
            self._current_step = step
        
        self.metric(
            f"Metrics at step {self._current_step}",
            metrics=metrics,
            step=self._current_step
        )
    
    def log_artifact(self, artifact_path: Union[str, Path]) -> None:
        """Log an artifact."""
        artifact_str = str(artifact_path)
        
        if self._current_experiment:
            self._current_experiment.artifacts.append(artifact_str)
        
        self.info(f"Artifact logged: {artifact_str}", artifacts=[artifact_str])
    
    def set_step(self, step: int) -> None:
        """Set the current step."""
        self._current_step = step
    
    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch."""
        self._current_epoch = epoch
    
    def increment_step(self) -> int:
        """Increment and return the current step."""
        self._current_step += 1
        return self._current_step
    
    def increment_epoch(self) -> int:
        """Increment and return the current epoch."""
        self._current_epoch += 1
        return self._current_epoch
    
    # Context managers
    @contextmanager
    def experiment_context(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """Context manager for experiments."""
        experiment_id = self.start_experiment(experiment_name, run_name, parameters, tags)
        try:
            yield experiment_id
        except Exception as e:
            self.error(f"Experiment failed: {str(e)}")
            self.end_experiment("failed")
            raise
        else:
            self.end_experiment("completed")
    
    @contextmanager
    def timing_context(self, operation_name: str):
        """Context manager for timing operations."""
        start_time = time.time()
        self.debug(f"Starting: {operation_name}")
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.info(f"Completed: {operation_name}", metrics={"duration": duration})
    
    # Utility methods
    def get_experiment_info(self) -> Optional[Dict[str, Any]]:
        """Get current experiment information."""
        if not self._current_experiment:
            return None
        
        return {
            "experiment_id": self._current_experiment.experiment_id,
            "experiment_name": self._current_experiment.experiment_name,
            "run_id": self._current_experiment.run_id,
            "run_name": self._current_experiment.run_name,
            "start_time": self._current_experiment.start_time,
            "status": self._current_experiment.status,
            "current_step": self._current_step,
            "current_epoch": self._current_epoch
        }
    
    def get_log_entries(
        self,
        level: Optional[str] = None,
        experiment_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[LogEntry]:
        """Get log entries with optional filtering."""
        entries = self._log_entries
        
        if level:
            entries = [e for e in entries if e.level == level]
        
        if experiment_id:
            entries = [e for e in entries if e.experiment_id == experiment_id]
        
        if limit:
            entries = entries[-limit:]
        
        return entries
    
    def export_logs(
        self,
        output_path: Union[str, Path],
        format: str = "json",
        experiment_id: Optional[str] = None
    ) -> None:
        """Export logs to file."""
        output_path = Path(output_path)
        entries = self.get_log_entries(experiment_id=experiment_id)
        
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump([entry.to_dict() for entry in entries], f, indent=2, default=str)
        elif format.lower() == "csv":
            import pandas as pd
            df = pd.DataFrame([entry.to_dict() for entry in entries])
            df.to_csv(output_path, index=False)
        else:
            raise RefuncError(f"Unsupported export format: {format}")
        
        self.info(f"Logs exported to {output_path}")
    
    def clear_logs(self) -> None:
        """Clear stored log entries."""
        self._log_entries.clear()
        self.debug("Log entries cleared")
    
    def set_level(self, level: Union[int, str]) -> None:
        """Set the logging level."""
        self._logger.setLevel(level)
    
    def add_handler(self, handler: logging.Handler) -> None:
        """Add a custom handler."""
        self._logger.addHandler(handler)
    
    def remove_handler(self, handler: logging.Handler) -> None:
        """Remove a handler."""
        self._logger.removeHandler(handler)


# Global logger instance
_default_logger: Optional[MLLogger] = None


def get_logger(
    name: str = "refunc",
    **kwargs
) -> MLLogger:
    """Get or create a logger instance."""
    global _default_logger
    
    if _default_logger is None or _default_logger.name != name:
        _default_logger = MLLogger(name=name, **kwargs)
    
    return _default_logger


def setup_logging(
    name: str = "refunc",
    level: Union[int, str] = LogLevel.INFO,
    log_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> MLLogger:
    """Setup global logging configuration."""
    global _default_logger
    
    _default_logger = MLLogger(
        name=name,
        level=level,
        log_dir=log_dir,
        **kwargs
    )
    
    return _default_logger


# Convenience functions
def info(message: str, **kwargs) -> None:
    """Log info message using default logger."""
    get_logger().info(message, **kwargs)


def debug(message: str, **kwargs) -> None:
    """Log debug message using default logger."""
    get_logger().debug(message, **kwargs)


def warning(message: str, **kwargs) -> None:
    """Log warning message using default logger."""
    get_logger().warning(message, **kwargs)


def error(message: str, **kwargs) -> None:
    """Log error message using default logger."""
    get_logger().error(message, **kwargs)


def metric(message: str, metrics: Optional[Dict[str, Any]] = None, **kwargs) -> None:
    """Log metrics using default logger."""
    get_logger().metric(message, metrics=metrics, **kwargs)