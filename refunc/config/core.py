"""
Core configuration management for refunc.

This module provides comprehensive configuration management with:
- Environment variable integration
- Multi-format config file support (YAML, JSON, TOML)
- Schema validation and type conversion
- Hierarchical configuration merging
- Dynamic config reloading
"""

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, TypeVar, Generic
from dataclasses import dataclass, field, fields, is_dataclass, MISSING
from abc import ABC, abstractmethod
import logging

from ..exceptions import RefuncError

# Optional dependencies
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


try:
    import tomllib
    tomli = tomllib
    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False

T = TypeVar('T')


class ConfigError(RefuncError):
    """Configuration-related errors."""
    pass


class ValidationError(ConfigError):
    """Configuration validation errors."""
    pass


@dataclass
@dataclass
class ConfigSource:
    """Configuration source metadata."""
    
    name: str
    path: Optional[Path] = None
    priority: int = 0  # Higher numbers take precedence
    format: Optional[str] = None
    last_modified: Optional[float] = None
    _data: Optional[Dict[str, Any]] = None  # For in-memory dict sources
    

class ConfigLoader(ABC):
    """Abstract base class for configuration loaders."""
    
    @abstractmethod
    def can_load(self, source: Union[str, Path]) -> bool:
        """Check if this loader can handle the given source."""
        pass
    
    @abstractmethod
    def load(self, source: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from source."""
        pass


class JSONLoader(ConfigLoader):
    """JSON configuration loader."""
    
    def can_load(self, source: Union[str, Path]) -> bool:
        """Check if source is JSON format."""
        if isinstance(source, str):
            return source.lower().endswith('.json')
        return source.suffix.lower() == '.json'
    
    def load(self, source: Union[str, Path]) -> Dict[str, Any]:
        """Load JSON configuration."""
        try:
            with open(source, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ConfigError(f"Failed to load JSON config from {source}: {e}")


class YAMLLoader(ConfigLoader):
    """YAML configuration loader."""
    
    def can_load(self, source: Union[str, Path]) -> bool:
        """Check if source is YAML format."""
        if not YAML_AVAILABLE:
            return False
        
        if isinstance(source, str):
            return source.lower().endswith(('.yaml', '.yml'))
        return source.suffix.lower() in {'.yaml', '.yml'}
    
    def load(self, source: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML configuration."""
        if not YAML_AVAILABLE:
            raise ConfigError("PyYAML not available for YAML config loading")
        
        import yaml  # Import here to ensure it's available
        try:
            with open(source, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigError(f"Failed to load YAML config from {source}: {e}")


class TOMLLoader(ConfigLoader):
    """TOML configuration loader."""
    
    def can_load(self, source: Union[str, Path]) -> bool:
        """Check if source is TOML format."""
        if not TOML_AVAILABLE:
            return False
        
        if isinstance(source, str):
            return source.lower().endswith('.toml')
        return source.suffix.lower() == '.toml'
    
    def load(self, source: Union[str, Path]) -> Dict[str, Any]:
        """Load TOML configuration."""
        if not TOML_AVAILABLE:
            raise ConfigError("tomli/tomllib not available for TOML config loading")
        
        try:
            with open(source, 'rb') as f:
                return tomli.load(f)
        except Exception as e:
            raise ConfigError(f"Failed to load TOML config from {source}: {e}")


class EnvironmentLoader(ConfigLoader):
    """Environment variable loader."""
    
    def __init__(self, prefix: str = "REFUNC_"):
        self.prefix = prefix.upper()
    
    def can_load(self, source: Union[str, Path]) -> bool:
        """Always can load environment variables."""
        return True
    
    def load(self, source: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                # Remove prefix and convert to nested dict
                config_key = key[len(self.prefix):].lower()
                
                # Handle nested keys (e.g., REFUNC_DB_HOST -> db.host)
                keys = config_key.split('_')
                current = config
                
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                
                # Convert value to appropriate type
                current[keys[-1]] = self._convert_value(value)
        
        return config
    
    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        # Try boolean
        if value.lower() in {'true', 'false'}:
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try JSON (for lists, dicts)
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Return as string
        return value


class ConfigManager:
    """
    Central configuration management system.
    
    Manages multiple configuration sources with priority-based merging,
    schema validation, and dynamic reloading.
    """
    
    def __init__(
        self,
        env_prefix: str = "REFUNC_",
        auto_reload: bool = False,
        validation_enabled: bool = True
    ):
        self.env_prefix = env_prefix
        self.auto_reload = auto_reload
        self.validation_enabled = validation_enabled
        
        # Configuration data
        self._config: Dict[str, Any] = {}
        self._sources: List[ConfigSource] = []
        self._schema: Optional[Type] = None
        
        # Loaders
        self._loaders: List[ConfigLoader] = [
            JSONLoader(),
            YAMLLoader(),
            TOMLLoader(),
            EnvironmentLoader(env_prefix)
        ]
        
        # Load environment variables by default
        self.add_env_source()
    
    def add_file_source(
        self,
        path: Union[str, Path],
        priority: int = 100,
        required: bool = True
    ) -> None:
        """Add a configuration file source."""
        path = Path(path)
        
        if required and not path.exists():
            raise ConfigError(f"Required config file not found: {path}")
        
        if not path.exists():
            return
        
        # Find appropriate loader
        loader = None
        for l in self._loaders:
            if l.can_load(path):
                loader = l
                break
        
        if not loader:
            raise ConfigError(f"No loader available for config file: {path}")
        
        source = ConfigSource(
            name=str(path),
            path=path,
            priority=priority,
            format=path.suffix.lower(),
            last_modified=path.stat().st_mtime
        )
        
        self._sources.append(source)
        self._reload_config()
    
    def add_env_source(self, priority: int = 200) -> None:
        """Add environment variables as a configuration source."""
        source = ConfigSource(
            name="environment",
            priority=priority,
            format="env"
        )
        
        self._sources.append(source)
        self._reload_config()
    
    def add_dict_source(
        self,
        data: Dict[str, Any],
        name: str = "dict",
        priority: int = 50
    ) -> None:
        """Add a dictionary as a configuration source."""
        source = ConfigSource(
            name=name,
            priority=priority,
            format="dict"
        )
        
        # Store the data in source object
        source._data = data
        self._sources.append(source)
        self._reload_config()
    
    def set_schema(self, schema_class: Type[T]) -> None:
        """Set a schema class for validation."""
        if not is_dataclass(schema_class):
            raise ConfigError("Schema must be a dataclass")
        
        self._schema = schema_class
        self._validate_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        if self.auto_reload:
            self._check_reload()
        
        # Support nested keys with dot notation
        keys = key.split('.')
        current = self._config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def get_typed(self, key: str, type_: Type[T], default: Optional[T] = None) -> Optional[T]:
        """Get configuration value with type conversion."""
        value = self.get(key, default)
        
        if value is None:
            return default
        
        if isinstance(value, type_):
            return value
        
        # Try to convert
        try:
            if type_ == bool and isinstance(value, str):
                return value.lower() in {'true', '1', 'yes', 'on'}  # type: ignore
            
            return type_(value)  # type: ignore
        except (ValueError, TypeError) as e:
            raise ConfigError(f"Cannot convert {key}={value} to {type_.__name__}: {e}")
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        keys = key.split('.')
        current = self._config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
        self._validate_config()
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        if self.auto_reload:
            self._check_reload()
        
        return self._config.copy()
    
    def to_object(self, schema_class: Optional[Type[T]] = None) -> T:
        """Convert configuration to typed object."""
        schema = schema_class or self._schema
        
        if not schema:
            raise ConfigError("No schema class provided or set")
        
        if not is_dataclass(schema):
            raise ConfigError("Schema must be a dataclass")
        
        try:
            return self._dict_to_dataclass(self._config, schema)  # type: ignore
        except Exception as e:
            raise ConfigError(f"Failed to convert config to {schema.__name__}: {e}")
    
    def reload(self) -> None:
        """Force reload configuration from all sources."""
        self._reload_config()
    
    def _reload_config(self) -> None:
        """Reload configuration from all sources."""
        # Sort sources by priority (lower numbers first)
        sorted_sources = sorted(self._sources, key=lambda s: s.priority)
        
        # Merge configurations
        merged_config = {}
        
        for source in sorted_sources:
            try:
                if source.format == "env":
                    # Load from environment
                    loader = EnvironmentLoader(self.env_prefix)
                    config_data = loader.load()
                elif source.format == "dict":
                    # Load from stored dict
                    config_data = getattr(source, '_data', {})
                else:
                    # Load from file
                    if source.path is None:
                        continue
                        
                    loader = None
                    for l in self._loaders:
                        if l.can_load(source.path):
                            loader = l
                            break
                    
                    if loader:
                        config_data = loader.load(source.path)
                    else:
                        continue
                
                # Deep merge configuration
                self._deep_merge(merged_config, config_data)
                
            except Exception as e:
                logging.warning(f"Failed to load config from {source.name}: {e}")
                continue
        
        self._config = merged_config
        self._validate_config()
    
    def _check_reload(self) -> None:
        """Check if any file sources have been modified."""
        reload_needed = False
        
        for source in self._sources:
            if source.path and source.path.exists():
                current_mtime = source.path.stat().st_mtime
                if source.last_modified != current_mtime:
                    source.last_modified = current_mtime
                    reload_needed = True
        
        if reload_needed:
            self._reload_config()
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge source dictionary into target."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def _validate_config(self) -> None:
        """Validate configuration against schema."""
        if not self.validation_enabled or not self._schema:
            return
        
        try:
            self.to_object(self._schema)
        except Exception as e:
            if isinstance(e, ConfigError):
                raise ValidationError(f"Configuration validation failed: {e}")
            raise ValidationError(f"Configuration validation failed: {e}")
    
    def _dict_to_dataclass(self, data: Dict[str, Any], dataclass_type: Type[T]) -> T:
        """Convert dictionary to dataclass instance."""
        if not is_dataclass(dataclass_type):
            raise ValueError(f"{dataclass_type} is not a dataclass")
        
        field_values = {}
        
        for field_obj in fields(dataclass_type):
            field_name = field_obj.name
            field_type = field_obj.type
            
            if field_name in data:
                value = data[field_name]
                
                # Handle nested dataclasses
                if is_dataclass(field_type):
                    if isinstance(value, dict):
                        # Ensure field_type is a proper type, not an instance
                        if isinstance(field_type, type):
                            value = self._dict_to_dataclass(value, field_type)
                        else:
                            # Skip conversion if field_type is not a proper type
                            pass
                
                field_values[field_name] = value
            elif field_obj.default is not MISSING:
                # Field has a default value, skip setting it
                pass
            elif field_obj.default_factory is not MISSING:
                # Field has a default factory, skip setting it
                pass
            else:
                # Field is required but missing
                raise ValueError(f"Required field '{field_name}' missing from configuration")
        
        return dataclass_type(**field_values)


# Global configuration instance
_global_config: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get or create global configuration manager."""
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager()
    return _global_config


def configure(
    config_files: Optional[List[Union[str, Path]]] = None,
    env_prefix: str = "REFUNC_",
    schema: Optional[Type] = None,
    auto_reload: bool = False
) -> ConfigManager:
    """Configure the global configuration manager."""
    global _global_config
    
    _global_config = ConfigManager(
        env_prefix=env_prefix,
        auto_reload=auto_reload
    )
    
    # Add file sources
    if config_files:
        for config_file in config_files:
            _global_config.add_file_source(config_file, required=False)
    
    # Set schema
    if schema:
        _global_config.set_schema(schema)
    
    return _global_config


# Convenience functions
def get_setting(key: str, default: Any = None) -> Any:
    """Get configuration setting."""
    return get_config().get(key, default)


def get_typed_setting(key: str, type_: Type[T], default: Optional[T] = None) -> Optional[T]:
    """Get typed configuration setting."""
    return get_config().get_typed(key, type_, default)


def set_setting(key: str, value: Any) -> None:
    """Set configuration setting."""
    get_config().set(key, value)


def reload_config() -> None:
    """Reload global configuration."""
    get_config().reload()