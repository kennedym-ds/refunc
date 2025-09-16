"""
Configuration management for refunc.

This package provides flexible configuration management with support for
multiple file formats (YAML, JSON, TOML), environment variables, and
schema validation.

Main components:
- ConfigManager: Core configuration loading and management
- Configuration schemas for common ML workflows
- Utilities for template generation and validation
- CLI interface for configuration management

Example:
    Basic usage:
    
    >>> from refunc.config import ConfigManager, RefuncConfig
    >>> config = ConfigManager()
    >>> config.add_file_source('config.yaml')
    >>> settings = config.get('model.parameters', {})
    
    Auto-configuration:
    
    >>> from refunc.config import auto_configure
    >>> config = auto_configure()
    >>> model_config = config.get('model')
    
    Schema validation:
    
    >>> from refunc.config import ConfigManager, RefuncConfig
    >>> config = ConfigManager(schema=RefuncConfig)
    >>> config.add_file_source('config.yaml')
    >>> validated_config = config.get_validated()
"""

from .core import (
    ConfigManager,
    ConfigSource,
    ConfigError,
    ValidationError
)

from .schemas import (
    RefuncConfig,
    DatabaseConfig,
    CacheConfig,
    LoggingConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    ExperimentConfig,
    PerformanceConfig,
    SecurityConfig
)

from .utils import (
    auto_configure,
    create_config_template,
    validate_config_file,
    merge_config_files,
    export_config,
    get_config_summary
)

# CLI module is available but not imported by default
# Use: from refunc.config.cli import main

__all__ = [
    # Core classes
    'ConfigManager',
    'ConfigSource',
    'ConfigError',
    'ValidationError',
    
    # Schema classes
    'RefuncConfig',
    'DatabaseConfig',
    'CacheConfig',
    'LoggingConfig',
    'DataConfig',
    'ModelConfig',
    'TrainingConfig',
    'ExperimentConfig',
    'PerformanceConfig',
    'SecurityConfig',
    
    # Utility functions
    'auto_configure',
    'create_config_template',
    'validate_config_file',
    'merge_config_files',
    'export_config',
    'get_config_summary'
]