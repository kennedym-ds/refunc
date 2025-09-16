"""
Configuration utilities and helpers.

This module provides utility functions for configuration management,
including file discovery, template generation, and validation helpers.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type
import json

from .core import ConfigManager, get_config
from .schemas import RefuncConfig, DEVELOPMENT_CONFIG, PRODUCTION_CONFIG, TRAINING_CONFIG, INFERENCE_CONFIG

# Optional dependencies for template generation
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def find_config_files(
    search_paths: Optional[List[Union[str, Path]]] = None,
    config_names: Optional[List[str]] = None
) -> List[Path]:
    """
    Find configuration files in search paths.
    
    Args:
        search_paths: Directories to search in
        config_names: Configuration file names to look for
    
    Returns:
        List of found configuration files
    """
    if search_paths is None:
        search_paths = [
            Path.cwd(),
            Path.cwd() / "config",
            Path.home() / ".config" / "refunc",
            Path("/etc/refunc") if os.name == "posix" else Path("C:\\ProgramData\\refunc")
        ]
    
    if config_names is None:
        config_names = [
            "refunc.yaml", "refunc.yml",
            "refunc.json",
            "refunc.toml",
            "config.yaml", "config.yml",
            "config.json",
            "config.toml"
        ]
    
    found_files = []
    
    for search_path in search_paths:
        search_path = Path(search_path)
        if not search_path.exists():
            continue
        
        for config_name in config_names:
            config_file = search_path / config_name
            if config_file.exists():
                found_files.append(config_file)
    
    return found_files


def auto_configure(
    config_files: Optional[List[Union[str, Path]]] = None,
    env_prefix: str = "REFUNC_",
    schema: Optional[Type] = None,
    auto_find: bool = True
) -> ConfigManager:
    """
    Automatically configure with discovered files.
    
    Args:
        config_files: Specific config files to use
        env_prefix: Environment variable prefix
        schema: Configuration schema class
        auto_find: Whether to auto-discover config files
    
    Returns:
        Configured ConfigManager instance
    """
    if config_files is None and auto_find:
        found_files = find_config_files()
        config_files = [str(f) for f in found_files]  # Convert to string paths
    
    config = ConfigManager(env_prefix=env_prefix)
    
    # Add file sources
    if config_files:
        for config_file in config_files:
            try:
                config.add_file_source(config_file, required=False)
            except Exception as e:
                print(f"Warning: Failed to load config file {config_file}: {e}")
    
    # Set schema
    if schema:
        config.set_schema(schema)
    # Remove automatic schema setting for default case
    
    return config


def create_config_template(
    output_path: Union[str, Path],
    template_type: str = "full",
    format: str = "yaml",
    include_comments: bool = True
) -> None:
    """
    Create a configuration template file.
    
    Args:
        output_path: Path to write the template
        template_type: Type of template (full, development, production, training, inference)
        format: Output format (yaml, json, toml)
        include_comments: Whether to include helpful comments
    """
    output_path = Path(output_path)
    
    # Get template data
    if template_type == "development":
        template_data = DEVELOPMENT_CONFIG
    elif template_type == "production":
        template_data = PRODUCTION_CONFIG
    elif template_type == "training":
        template_data = TRAINING_CONFIG
    elif template_type == "inference":
        template_data = INFERENCE_CONFIG
    else:
        # Full template with all sections
        template_data = _get_full_template()
    
    # Write template
    if format.lower() == "yaml" and YAML_AVAILABLE:
        _write_yaml_template(output_path, template_data, include_comments)
    elif format.lower() == "json":
        _write_json_template(output_path, template_data, include_comments)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Configuration template created: {output_path}")


def validate_config_file(
    config_path: Union[str, Path],
    schema: Optional[Type] = None
) -> bool:
    """
    Validate a configuration file against schema.
    
    Args:
        config_path: Path to configuration file
        schema: Schema class to validate against
    
    Returns:
        True if valid, False otherwise
    """
    try:
        config = ConfigManager()
        config.add_file_source(config_path)
        
        if schema:
            config.set_schema(schema)
        else:
            config.set_schema(RefuncConfig)
        
        # Try to convert to object (this validates the schema)
        config.to_object()
        return True
    
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


def merge_config_files(
    input_files: List[Union[str, Path]],
    output_file: Union[str, Path],
    format: str = "yaml"
) -> None:
    """
    Merge multiple configuration files into one.
    
    Args:
        input_files: List of input configuration files
        output_file: Output file path
        format: Output format
    """
    config = ConfigManager()
    
    # Add all input files
    for i, input_file in enumerate(input_files):
        config.add_file_source(input_file, priority=i * 100)
    
    # Get merged configuration
    merged_data = config.to_dict()
    
    # Write merged configuration
    output_file = Path(output_file)
    
    if format.lower() == "yaml" and YAML_AVAILABLE:
        import yaml
        with open(output_file, 'w') as f:
            yaml.dump(merged_data, f, default_flow_style=False, indent=2)
    elif format.lower() == "json":
        with open(output_file, 'w') as f:
            json.dump(merged_data, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Merged configuration written to: {output_file}")


def get_config_summary(config: Optional[ConfigManager] = None) -> Dict[str, Any]:
    """
    Get a summary of current configuration.
    
    Args:
        config: ConfigManager instance (uses global if None)
    
    Returns:
        Configuration summary
    """
    if config is None:
        config = get_config()
    
    summary = {
        "sources": [
            {
                "name": source.name,
                "priority": source.priority,
                "format": source.format,
                "path": str(source.path) if source.path else None
            }
            for source in config._sources
        ],
        "total_settings": _count_settings(config.to_dict()),
        "schema": config._schema.__name__ if config._schema else None,
        "validation_enabled": config.validation_enabled,
        "auto_reload": config.auto_reload
    }
    
    return summary


def export_config(
    output_path: Union[str, Path],
    config: Optional[ConfigManager] = None,
    format: str = "yaml",
    include_metadata: bool = True
) -> None:
    """
    Export current configuration to file.
    
    Args:
        output_path: Output file path
        config: ConfigManager instance (uses global if None)
        format: Output format
        include_metadata: Whether to include metadata comments
    """
    if config is None:
        config = get_config()
    
    output_path = Path(output_path)
    config_data = config.to_dict()
    
    if include_metadata:
        metadata = {
            "_metadata": {
                "exported_by": "refunc",
                "sources": [s.name for s in config._sources],
                "schema": config._schema.__name__ if config._schema else None
            }
        }
        config_data.update(metadata)
    
    if format.lower() == "yaml" and YAML_AVAILABLE:
        import yaml
        with open(output_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    elif format.lower() == "json":
        with open(output_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Configuration exported to: {output_path}")


def _get_full_template() -> Dict[str, Any]:
    """Get full configuration template."""
    return {
        # Global settings
        "debug": False,
        "verbose": False,
        "project_name": "my-refunc-project",
        "version": "1.0.0",
        "environment": "development",
        
        # Database configuration
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "refunc_db",
            "username": "user",
            "password": "${DB_PASSWORD}",
            "pool_size": 10,
            "timeout": 30.0
        },
        
        # Cache configuration
        "cache": {
            "enabled": True,
            "backend": "memory",
            "ttl": 3600,
            "max_size": 1000
        },
        
        # Logging configuration
        "logging": {
            "level": "INFO",
            "colored_output": True,
            "json_logging": False,
            "log_dir": "logs",
            "max_log_files": 10,
            "max_file_size": "100MB"
        },
        
        # Data configuration
        "data": {
            "data_dir": "data",
            "batch_size": 32,
            "num_workers": 4,
            "default_format": "parquet",
            "validate_data": True
        },
        
        # Model configuration
        "model": {
            "model_type": "linear",
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "validation_split": 0.2
        },
        
        # Training configuration
        "training": {
            "seed": 42,
            "deterministic": True,
            "device": "auto",
            "checkpoint_dir": "checkpoints"
        },
        
        # Experiment configuration
        "experiment": {
            "experiment_name": "default",
            "track_parameters": True,
            "track_metrics": True,
            "auto_save": True
        }
    }


def _write_yaml_template(
    output_path: Path,
    template_data: Dict[str, Any],
    include_comments: bool
) -> None:
    """Write YAML template with optional comments."""
    if not YAML_AVAILABLE:
        raise ValueError("PyYAML is required for YAML template generation")
    
    import yaml
    
    lines = []
    
    if include_comments:
        lines.extend([
            "# Refunc Configuration File",
            "# Generated template - customize as needed",
            "",
        ])
    
    yaml_content = yaml.dump(template_data, default_flow_style=False, indent=2)
    lines.append(yaml_content)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def _write_json_template(
    output_path: Path,
    template_data: Dict[str, Any],
    include_comments: bool
) -> None:
    """Write JSON template."""
    if include_comments:
        template_data["_comment"] = "Refunc configuration file - customize as needed"
    
    with open(output_path, 'w') as f:
        json.dump(template_data, f, indent=2)


def _count_settings(data: Dict[str, Any], prefix: str = "") -> int:
    """Count total number of settings in configuration."""
    count = 0
    for key, value in data.items():
        if isinstance(value, dict):
            count += _count_settings(value, f"{prefix}{key}.")
        else:
            count += 1
    return count


# Environment-specific utilities

def setup_development_environment() -> ConfigManager:
    """Setup configuration for development environment."""
    config = ConfigManager()
    config.add_dict_source(DEVELOPMENT_CONFIG, "development", priority=300)
    return config


def setup_production_environment() -> ConfigManager:
    """Setup configuration for production environment."""
    config = ConfigManager()
    config.add_dict_source(PRODUCTION_CONFIG, "production", priority=300)
    return config


def setup_training_environment() -> ConfigManager:
    """Setup configuration for ML training."""
    config = ConfigManager()
    config.add_dict_source(TRAINING_CONFIG, "training", priority=300)
    return config


def setup_inference_environment() -> ConfigManager:
    """Setup configuration for model inference."""
    config = ConfigManager()
    config.add_dict_source(INFERENCE_CONFIG, "inference", priority=300)
    return config