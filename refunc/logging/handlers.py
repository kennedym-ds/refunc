"""
Custom log handlers for ML workflows.

This module provides specialized handlers for rotating files,
experiment tracking, and integration with external systems.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, TextIO
from datetime import datetime
import gzip
import shutil

from ..exceptions import RefuncError


class RotatingFileHandler(logging.Handler):
    """
    Enhanced rotating file handler with size-based rotation.
    
    Supports automatic compression of old log files and
    configurable retention policies.
    """
    
    def __init__(
        self,
        filename: Union[str, Path],
        max_size: str = "100MB",
        max_files: int = 10,
        compress_old: bool = True,
        encoding: str = "utf-8"
    ):
        super().__init__()
        
        self.filename = Path(filename)
        self.max_size = self._parse_size(max_size)
        self.max_files = max_files
        self.compress_old = compress_old
        self.encoding = encoding
        
        # Ensure directory exists
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        
        # Current file handle
        self._file: Optional[TextIO] = None
        self._open_file()
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '100MB' to bytes."""
        size_str = size_str.upper().strip()
        
        multipliers = {
            'B': 1,
            'KB': 1024,
            'MB': 1024 ** 2,
            'GB': 1024 ** 3
        }
        
        for suffix, multiplier in multipliers.items():
            if size_str.endswith(suffix):
                try:
                    value = float(size_str[:-len(suffix)])
                    return int(value * multiplier)
                except ValueError:
                    break
        
        # Default to bytes if no suffix or parsing fails
        try:
            return int(size_str)
        except ValueError:
            return 100 * 1024 * 1024  # Default 100MB
    
    def _open_file(self) -> None:
        """Open the log file for writing."""
        if self._file:
            self._file.close()
        
        self._file = open(self.filename, 'a', encoding=self.encoding)
    
    def _should_rotate(self) -> bool:
        """Check if file should be rotated."""
        if not self.filename.exists():
            return False
        
        return self.filename.stat().st_size >= self.max_size
    
    def _rotate_files(self) -> None:
        """Rotate log files."""
        if self._file:
            self._file.close()
            self._file = None
        
        # Find existing backup files
        backup_files = []
        for i in range(1, self.max_files + 1):
            backup_path = self.filename.with_suffix(f"{self.filename.suffix}.{i}")
            if backup_path.exists():
                backup_files.append((i, backup_path))
        
        # Sort by index (highest first)
        backup_files.sort(key=lambda x: x[0], reverse=True)
        
        # Remove oldest files if we're at the limit
        while len(backup_files) >= self.max_files:
            _, oldest_file = backup_files.pop()
            oldest_file.unlink()
        
        # Shift existing backup files
        for index, backup_path in backup_files:
            new_path = self.filename.with_suffix(f"{self.filename.suffix}.{index + 1}")
            backup_path.rename(new_path)
        
        # Move current file to .1
        if self.filename.exists():
            backup_path = self.filename.with_suffix(f"{self.filename.suffix}.1")
            self.filename.rename(backup_path)
            
            # Compress if requested
            if self.compress_old:
                self._compress_file(backup_path)
        
        # Reopen new file
        self._open_file()
    
    def _compress_file(self, file_path: Path) -> None:
        """Compress a log file with gzip."""
        compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
        
        try:
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove original file
            file_path.unlink()
        except Exception as e:
            # If compression fails, just keep the original file
            if compressed_path.exists():
                compressed_path.unlink()
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record."""
        try:
            if self._should_rotate():
                self._rotate_files()
            
            if self._file:
                msg = self.format(record)
                self._file.write(msg + '\n')
                self._file.flush()
                
        except Exception:
            self.handleError(record)
    
    def close(self) -> None:
        """Close the handler."""
        if self._file:
            self._file.close()
            self._file = None
        super().close()


class ExperimentHandler(logging.Handler):
    """
    Handler for experiment-specific logging.
    
    Creates separate log files for each experiment and
    maintains experiment metadata.
    """
    
    def __init__(
        self,
        base_dir: Union[str, Path],
        auto_create_dirs: bool = True
    ):
        super().__init__()
        
        self.base_dir = Path(base_dir)
        self.auto_create_dirs = auto_create_dirs
        
        if self.auto_create_dirs:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Track experiment files
        self._experiment_files: Dict[str, TextIO] = {}
        self._experiment_metadata: Dict[str, Dict[str, Any]] = {}
    
    def _get_experiment_file(self, experiment_id: str) -> TextIO:
        """Get or create file handle for experiment."""
        if experiment_id not in self._experiment_files:
            exp_dir = self.base_dir / experiment_id
            if self.auto_create_dirs:
                exp_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = exp_dir / "experiment.log"
            self._experiment_files[experiment_id] = open(log_file, 'a', encoding='utf-8')
            
            # Initialize metadata
            self._experiment_metadata[experiment_id] = {
                'experiment_id': experiment_id,
                'start_time': time.time(),
                'log_file': str(log_file),
                'record_count': 0
            }
        
        return self._experiment_files[experiment_id]
    
    def _update_metadata(self, experiment_id: str, record: logging.LogRecord) -> None:
        """Update experiment metadata."""
        if experiment_id in self._experiment_metadata:
            metadata = self._experiment_metadata[experiment_id]
            metadata['record_count'] += 1
            metadata['last_log_time'] = time.time()
            
            # Extract additional info from log entry
            if hasattr(record, 'log_entry') and getattr(record, 'log_entry'):
                entry = getattr(record, 'log_entry')
                
                # Update with latest experiment info
                if hasattr(entry, 'experiment_id'):
                    metadata['experiment_name'] = getattr(entry, 'experiment_name', None)
                if hasattr(entry, 'run_id'):
                    metadata['run_id'] = getattr(entry, 'run_id', None)
                if hasattr(entry, 'step'):
                    metadata['latest_step'] = getattr(entry, 'step', None)
                if hasattr(entry, 'epoch'):
                    metadata['latest_epoch'] = getattr(entry, 'epoch', None)
    
    def _save_metadata(self, experiment_id: str) -> None:
        """Save experiment metadata to file."""
        if experiment_id in self._experiment_metadata:
            metadata = self._experiment_metadata[experiment_id]
            exp_dir = self.base_dir / experiment_id
            metadata_file = exp_dir / "metadata.json"
            
            try:
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            except Exception:
                # Ignore metadata save errors
                pass
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to appropriate experiment file."""
        try:
            # Extract experiment ID
            experiment_id = None
            
            if hasattr(record, 'log_entry') and getattr(record, 'log_entry'):
                entry = getattr(record, 'log_entry')
                experiment_id = getattr(entry, 'experiment_id', None)
            
            if not experiment_id:
                # Check for experiment_id in record attributes
                experiment_id = getattr(record, 'experiment_id', None)
            
            if experiment_id:
                exp_file = self._get_experiment_file(experiment_id)
                msg = self.format(record)
                exp_file.write(msg + '\n')
                exp_file.flush()
                
                # Update metadata
                self._update_metadata(experiment_id, record)
                
        except Exception:
            self.handleError(record)
    
    def close(self) -> None:
        """Close all experiment files and save metadata."""
        for experiment_id, file_handle in self._experiment_files.items():
            try:
                file_handle.close()
                self._save_metadata(experiment_id)
            except Exception:
                pass
        
        self._experiment_files.clear()
        super().close()


class BufferedHandler(logging.Handler):
    """
    Buffered handler that flushes logs in batches.
    
    Useful for high-throughput logging scenarios where
    immediate writes would impact performance.
    """
    
    def __init__(
        self,
        target_handler: logging.Handler,
        buffer_size: int = 100,
        flush_interval: float = 5.0
    ):
        super().__init__()
        
        self.target_handler = target_handler
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        self._buffer: List[logging.LogRecord] = []
        self._last_flush = time.time()
    
    def _should_flush(self) -> bool:
        """Check if buffer should be flushed."""
        return (
            len(self._buffer) >= self.buffer_size or
            (time.time() - self._last_flush) >= self.flush_interval
        )
    
    def _flush_buffer(self) -> None:
        """Flush all buffered records."""
        for record in self._buffer:
            try:
                self.target_handler.emit(record)
            except Exception:
                self.handleError(record)
        
        self._buffer.clear()
        self._last_flush = time.time()
    
    def emit(self, record: logging.LogRecord) -> None:
        """Add record to buffer and flush if needed."""
        self._buffer.append(record)
        
        if self._should_flush():
            self._flush_buffer()
    
    def flush(self) -> None:
        """Force flush of buffer."""
        self._flush_buffer()
        if hasattr(self.target_handler, 'flush'):
            self.target_handler.flush()
    
    def close(self) -> None:
        """Flush and close handler."""
        self._flush_buffer()
        if hasattr(self.target_handler, 'close'):
            self.target_handler.close()
        super().close()


class MetricsHandler(logging.Handler):
    """
    Handler specifically for metrics logging.
    
    Extracts metrics from log records and writes them
    to structured files for analysis.
    """
    
    def __init__(
        self,
        metrics_dir: Union[str, Path],
        output_format: str = "json"  # json, csv, or parquet
    ):
        super().__init__()
        
        self.metrics_dir = Path(metrics_dir)
        self.output_format = output_format.lower()
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Track metrics files by experiment
        self._metrics_files: Dict[str, TextIO] = {}
        self._metrics_data: Dict[str, List[Dict[str, Any]]] = {}
    
    def _extract_metrics(self, record: logging.LogRecord) -> Optional[Dict[str, Any]]:
        """Extract metrics from log record."""
        metrics_data = {
            'timestamp': record.created,
            'level': record.levelname,
            'message': record.getMessage()
        }
        
        # Extract from log entry
        if hasattr(record, 'log_entry') and getattr(record, 'log_entry'):
            entry = getattr(record, 'log_entry')
            
            # Add experiment context
            metrics_data.update({
                'experiment_id': getattr(entry, 'experiment_id', None),
                'run_id': getattr(entry, 'run_id', None),
                'step': getattr(entry, 'step', None),
                'epoch': getattr(entry, 'epoch', None)
            })
            
            # Add metrics
            if hasattr(entry, 'metrics') and getattr(entry, 'metrics'):
                metrics_data['metrics'] = getattr(entry, 'metrics')
                return metrics_data
        
        # Check for direct metrics
        if hasattr(record, 'metrics') and getattr(record, 'metrics'):
            metrics_data['metrics'] = getattr(record, 'metrics')
            return metrics_data
        
        return None
    
    def _write_json_metrics(self, experiment_id: str, metrics_data: Dict[str, Any]) -> None:
        """Write metrics in JSON format."""
        metrics_file = self.metrics_dir / f"{experiment_id}_metrics.jsonl"
        
        with open(metrics_file, 'a') as f:
            json.dump(metrics_data, f, default=str)
            f.write('\n')
    
    def _write_csv_metrics(self, experiment_id: str, metrics_data: Dict[str, Any]) -> None:
        """Write metrics in CSV format."""
        if experiment_id not in self._metrics_data:
            self._metrics_data[experiment_id] = []
        
        # Flatten metrics for CSV
        flattened = {
            'timestamp': metrics_data['timestamp'],
            'level': metrics_data['level'],
            'experiment_id': metrics_data.get('experiment_id'),
            'run_id': metrics_data.get('run_id'),
            'step': metrics_data.get('step'),
            'epoch': metrics_data.get('epoch'),
        }
        
        # Add individual metrics as columns
        if 'metrics' in metrics_data:
            for key, value in metrics_data['metrics'].items():
                flattened[f'metric_{key}'] = value
        
        self._metrics_data[experiment_id].append(flattened)
        
        # Write periodically (every 100 records)
        if len(self._metrics_data[experiment_id]) % 100 == 0:
            self._flush_csv_metrics(experiment_id)
    
    def _flush_csv_metrics(self, experiment_id: str) -> None:
        """Flush CSV metrics to file."""
        if experiment_id not in self._metrics_data or not self._metrics_data[experiment_id]:
            return
        
        try:
            import pandas as pd
            
            df = pd.DataFrame(self._metrics_data[experiment_id])
            csv_file = self.metrics_dir / f"{experiment_id}_metrics.csv"
            
            # Append to existing file or create new
            if csv_file.exists():
                df.to_csv(csv_file, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_file, index=False)
            
            # Clear buffer
            self._metrics_data[experiment_id].clear()
            
        except ImportError:
            # Fallback to basic CSV writing
            self._write_basic_csv(experiment_id)
    
    def _write_basic_csv(self, experiment_id: str) -> None:
        """Write CSV without pandas."""
        import csv
        
        if experiment_id not in self._metrics_data or not self._metrics_data[experiment_id]:
            return
        
        csv_file = self.metrics_dir / f"{experiment_id}_metrics.csv"
        data = self._metrics_data[experiment_id]
        
        # Get all possible fields
        all_fields = set()
        for row in data:
            all_fields.update(row.keys())
        
        fieldnames = sorted(all_fields)
        
        file_exists = csv_file.exists()
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerows(data)
        
        self._metrics_data[experiment_id].clear()
    
    def emit(self, record: logging.LogRecord) -> None:
        """Extract and write metrics from log record."""
        try:
            metrics_data = self._extract_metrics(record)
            if not metrics_data:
                return
            
            experiment_id = metrics_data.get('experiment_id', 'default')
            
            if self.output_format == 'json':
                self._write_json_metrics(experiment_id, metrics_data)
            elif self.output_format == 'csv':
                self._write_csv_metrics(experiment_id, metrics_data)
            # TODO: Add parquet support
            
        except Exception:
            self.handleError(record)
    
    def close(self) -> None:
        """Flush any remaining metrics and close."""
        # Flush any remaining CSV data
        for experiment_id in list(self._metrics_data.keys()):
            if self._metrics_data[experiment_id]:
                self._flush_csv_metrics(experiment_id)
        
        super().close()


class AsyncHandler(logging.Handler):
    """
    Asynchronous handler that processes logs in a background thread.
    
    Useful for expensive log operations that shouldn't block
    the main application thread.
    """
    
    def __init__(
        self,
        target_handler: logging.Handler,
        queue_size: int = 1000
    ):
        super().__init__()
        
        self.target_handler = target_handler
        self.queue_size = queue_size
        
        # Import queue and threading
        import queue
        import threading
        
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Start worker thread
        self._start_worker()
    
    def _start_worker(self) -> None:
        """Start the background worker thread."""
        import threading
        
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
    
    def _worker(self) -> None:
        """Background worker that processes queued log records."""
        import queue
        
        while not self._stop_event.is_set():
            try:
                # Get record with timeout
                record = self._queue.get(timeout=1.0)
                
                if record is None:  # Sentinel value to stop
                    break
                
                # Process the record
                try:
                    self.target_handler.emit(record)
                except Exception:
                    self.handleError(record)
                finally:
                    self._queue.task_done()
                    
            except queue.Empty:
                continue
    
    def emit(self, record: logging.LogRecord) -> None:
        """Add record to processing queue."""
        try:
            self._queue.put_nowait(record)
        except:
            # Queue is full, drop the record or handle error
            self.handleError(record)
    
    def close(self) -> None:
        """Stop worker and close handler."""
        if self._worker_thread and self._worker_thread.is_alive():
            # Signal stop and wait for queue to empty
            self._stop_event.set()
            
            # Add sentinel value
            try:
                self._queue.put_nowait(None)
            except:
                pass
            
            # Wait for worker to finish
            self._worker_thread.join(timeout=5.0)
        
        if hasattr(self.target_handler, 'close'):
            self.target_handler.close()
        
        super().close()


class FilterHandler(logging.Handler):
    """
    Handler wrapper that applies filters before forwarding.
    
    Useful for creating conditional logging pipelines.
    """
    
    def __init__(
        self,
        target_handler: logging.Handler,
        level_filter: Optional[Union[int, str]] = None,
        experiment_filter: Optional[List[str]] = None,
        metric_filter: Optional[List[str]] = None
    ):
        super().__init__()
        
        self.target_handler = target_handler
        self.level_filter = level_filter
        self.experiment_filter = experiment_filter
        self.metric_filter = metric_filter
    
    def _passes_filters(self, record: logging.LogRecord) -> bool:
        """Check if record passes all filters."""
        
        # Level filter
        if self.level_filter is not None:
            if isinstance(self.level_filter, str):
                level_num = getattr(logging, self.level_filter.upper(), None)
            else:
                level_num = self.level_filter
            
            if level_num is not None and record.levelno < level_num:
                return False
        
        # Experiment filter
        if self.experiment_filter is not None:
            experiment_id = None
            
            if hasattr(record, 'log_entry') and getattr(record, 'log_entry'):
                entry = getattr(record, 'log_entry')
                experiment_id = getattr(entry, 'experiment_id', None)
            
            if experiment_id not in self.experiment_filter:
                return False
        
        # Metric filter
        if self.metric_filter is not None:
            metrics = {}
            
            if hasattr(record, 'log_entry') and getattr(record, 'log_entry'):
                entry = getattr(record, 'log_entry')
                metrics = getattr(entry, 'metrics', {})
            
            if not any(metric in metrics for metric in self.metric_filter):
                return False
        
        return True
    
    def emit(self, record: logging.LogRecord) -> None:
        """Forward record if it passes filters."""
        if self._passes_filters(record):
            self.target_handler.emit(record)
    
    def close(self) -> None:
        """Close target handler."""
        if hasattr(self.target_handler, 'close'):
            self.target_handler.close()
        super().close()