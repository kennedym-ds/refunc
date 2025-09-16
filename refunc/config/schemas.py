"""
Configuration schemas for ML workflows.

This module provides pre-defined configuration schemas for common
ML use cases, including training, data processing, and deployment.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from enum import Enum


class LogLevel(Enum):
    """Logging levels."""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class OptimizationMode(Enum):
    """Model optimization modes."""
    SPEED = "speed"
    MEMORY = "memory"
    ACCURACY = "accuracy"
    BALANCED = "balanced"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    
    host: str = "localhost"
    port: int = 5432
    name: str = "refunc_db"
    username: str = "user"
    password: str = ""
    pool_size: int = 10
    timeout: float = 30.0
    ssl_enabled: bool = False


@dataclass
class CacheConfig:
    """Cache configuration."""
    
    enabled: bool = True
    backend: str = "memory"  # memory, redis, filesystem
    ttl: int = 3600  # seconds
    max_size: int = 1000
    
    # Redis-specific
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Filesystem-specific
    cache_dir: str = ".cache"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    level: LogLevel = LogLevel.INFO
    colored_output: bool = True
    json_logging: bool = False
    log_dir: str = "logs"
    max_log_files: int = 10
    max_file_size: str = "100MB"
    
    # External integrations
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str = "http://localhost:5000"
    
    wandb_enabled: bool = False
    wandb_project: str = "refunc-experiments"
    wandb_entity: Optional[str] = None
    
    # Notifications
    slack_webhook: Optional[str] = None
    discord_webhook: Optional[str] = None


@dataclass
class DataConfig:
    """Data processing configuration."""
    
    # Data sources
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    cache_dir: str = "data/cache"
    
    # Data loading
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True
    
    # Data validation
    validate_data: bool = True
    schema_file: Optional[str] = None
    
    # Data formats
    default_format: str = "parquet"  # csv, json, parquet, pickle
    compression: Optional[str] = None
    
    # Preprocessing
    normalize: bool = True
    standardize: bool = False
    handle_missing: str = "drop"  # drop, fill, interpolate
    outlier_detection: bool = False


@dataclass
class ModelConfig:
    """Model configuration."""
    
    # Model architecture
    model_type: str = "linear"
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping: bool = True
    patience: int = 10
    
    # Optimization
    optimizer: str = "adam"
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    
    # Regularization
    dropout: float = 0.0
    weight_decay: float = 0.0
    l1_regularization: float = 0.0
    l2_regularization: float = 0.0
    
    # Validation
    validation_split: float = 0.2
    cross_validation: bool = False
    cv_folds: int = 5
    
    # Model persistence
    model_dir: str = "models"
    save_best_only: bool = True
    save_frequency: int = 10  # epochs


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Training setup
    seed: int = 42
    deterministic: bool = True
    mixed_precision: bool = False
    gradient_clipping: float = 1.0
    
    # Hardware
    device: str = "auto"  # auto, cpu, cuda, mps
    num_gpus: int = 1
    distributed: bool = False
    
    # Monitoring
    log_frequency: int = 10  # steps
    eval_frequency: int = 100  # steps
    save_frequency: int = 1000  # steps
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    resume_from_checkpoint: bool = True
    max_checkpoints: int = 5
    
    # Profiling
    profile_training: bool = False
    profile_memory: bool = False
    profile_gpu: bool = False


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration."""
    
    # Experiment metadata
    experiment_name: str = "default"
    experiment_description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Tracking
    track_parameters: bool = True
    track_metrics: bool = True
    track_artifacts: bool = True
    track_code: bool = True
    
    # Storage
    experiment_dir: str = "experiments"
    auto_save: bool = True
    save_interval: int = 300  # seconds
    
    # External tracking
    mlflow_experiment: Optional[str] = None
    wandb_run_name: Optional[str] = None


@dataclass
class PerformanceConfig:
    """Performance monitoring configuration."""
    
    # Monitoring
    monitor_memory: bool = True
    monitor_cpu: bool = True
    monitor_gpu: bool = True
    monitor_io: bool = False
    
    # Profiling
    enable_profiling: bool = False
    profile_functions: bool = True
    profile_lines: bool = False
    
    # Optimization
    optimization_mode: OptimizationMode = OptimizationMode.BALANCED
    cache_size: int = 1000
    parallel_workers: int = 4


@dataclass
class SecurityConfig:
    """Security configuration."""
    
    # Authentication
    auth_enabled: bool = False
    auth_method: str = "basic"  # basic, oauth, jwt
    
    # API Keys
    api_keys: Dict[str, str] = field(default_factory=dict)
    
    # Encryption
    encrypt_data: bool = False
    encryption_key: Optional[str] = None
    
    # Access control
    allowed_hosts: List[str] = field(default_factory=list)
    rate_limiting: bool = False
    max_requests_per_minute: int = 100


@dataclass
class RefuncConfig:
    """Main refunc configuration schema."""
    
    # Core configuration sections
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Global settings
    debug: bool = False
    verbose: bool = False
    project_name: str = "refunc-project"
    version: str = "1.0.0"
    
    # Environment
    environment: str = "development"  # development, staging, production
    timezone: str = "UTC"
    
    # Paths
    project_root: str = "."
    data_root: str = "data"
    output_root: str = "output"
    temp_dir: str = "tmp"


# Specialized configurations for specific use cases

@dataclass
class MLTrainingConfig:
    """Simplified ML training configuration."""
    
    # Model
    model_type: str = "linear"
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    
    # Data
    data_path: str = "data/train.csv"
    target_column: str = "target"
    
    # Output
    model_output_dir: str = "models"
    experiment_name: str = "ml_training"


@dataclass
class DataProcessingConfig:
    """Data processing pipeline configuration."""
    
    # Input/Output
    input_path: str = "data/raw"
    output_path: str = "data/processed"
    
    # Processing
    batch_size: int = 1000
    parallel_workers: int = 4
    
    # Validation
    validate_schema: bool = True
    schema_path: Optional[str] = None
    
    # Transformations
    transformations: List[str] = field(default_factory=list)
    normalize: bool = True
    handle_missing: str = "drop"


@dataclass
class ModelInferenceConfig:
    """Model inference configuration."""
    
    # Model
    model_path: str = "models/best_model.pkl"
    model_type: str = "pickle"
    
    # Input/Output
    input_format: str = "json"
    output_format: str = "json"
    
    # Performance
    batch_size: int = 1
    cache_predictions: bool = True
    
    # Monitoring
    log_predictions: bool = True
    monitor_latency: bool = True


# Configuration templates for common scenarios

DEVELOPMENT_CONFIG = {
    "debug": True,
    "verbose": True,
    "logging": {
        "level": "DEBUG",
        "colored_output": True
    },
    "performance": {
        "enable_profiling": True
    }
}

PRODUCTION_CONFIG = {
    "debug": False,
    "verbose": False,
    "logging": {
        "level": "INFO",
        "colored_output": False,
        "json_logging": True
    },
    "security": {
        "auth_enabled": True,
        "rate_limiting": True
    }
}

TRAINING_CONFIG = {
    "training": {
        "deterministic": True,
        "mixed_precision": True,
        "gradient_clipping": 1.0
    },
    "experiment": {
        "track_parameters": True,
        "track_metrics": True,
        "track_artifacts": True
    }
}

INFERENCE_CONFIG = {
    "performance": {
        "optimization_mode": "speed",
        "cache_size": 10000
    },
    "logging": {
        "level": "WARNING"
    }
}