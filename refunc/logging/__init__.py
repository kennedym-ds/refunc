"""
Advanced logging system for ML workflows.

This package provides comprehensive logging capabilities including:
- Structured logging with ML-specific features
- Multiple output formatters (colored, JSON, ML-specific)
- Specialized handlers for experiments and metrics
- Progress tracking and visualization
- Experiment tracking and metadata management
- External integrations (MLflow, W&B, Prometheus, etc.)
"""

from .core import (
    MLLogger,
    LogEntry,
    ExperimentContext,
    LogLevel,
    get_logger,
    setup_logging,
    info,
    debug,
    warning,
    error,
    metric
)

from .formatters import (
    ColoredFormatter,
    JSONFormatter,
    MLFormatter,
    CompactFormatter,
    ProgressFormatter
)

from .handlers import (
    RotatingFileHandler,
    ExperimentHandler,
    MetricsHandler,
    AsyncHandler,
    BufferedHandler
)

from .progress import (
    ProgressTracker,
    TqdmProgressTracker,
    EpochTracker,
    ProgressState,
    progress_context,
    epoch_context
)

from .experiment import (
    ExperimentTracker,
    ExperimentMetadata,
    MetricEntry,
    MLflowIntegration,
    WandBIntegration,
    MultiTracker,
    experiment_context
)

from .integrations import (
    PrometheusIntegration,
    ElasticsearchIntegration,
    RedisIntegration,
    SlackIntegration,
    DiscordIntegration,
    IntegrationsManager,
    integration_context,
    auto_configure_integrations
)

__all__ = [
    # Core logging
    'MLLogger',
    'LogEntry',
    'ExperimentContext',
    'LogLevel',
    'get_logger',
    'setup_logging',
    'info',
    'debug',
    'warning',
    'error',
    'metric',
    
    # Formatters
    'ColoredFormatter',
    'JSONFormatter',
    'MLFormatter',
    'CompactFormatter',
    'ProgressFormatter',
    
    # Handlers
    'RotatingFileHandler',
    'ExperimentHandler',
    'MetricsHandler',
    'AsyncHandler',
    'BufferedHandler',
    
    # Progress tracking
    'ProgressTracker',
    'TqdmProgressTracker',
    'EpochTracker',
    'ProgressState',
    'progress_context',
    'epoch_context',
    
    # Experiment tracking
    'ExperimentTracker',
    'ExperimentMetadata',
    'MetricEntry',
    'MLflowIntegration',
    'WandBIntegration',
    'MultiTracker',
    'experiment_context',
    
    # External integrations
    'PrometheusIntegration',
    'ElasticsearchIntegration',
    'RedisIntegration',
    'SlackIntegration',
    'DiscordIntegration',
    'IntegrationsManager',
    'integration_context',
    'auto_configure_integrations'
]

# Version info
__version__ = "0.1.0"
__author__ = "kennedym-ds"
__description__ = "Advanced logging system for ML workflows"