# ⚙️ Configuration API Reference

> **Flexible configuration management with multi-format support, schema validation, and environment variable integration.**

## Overview

The configuration module provides comprehensive configuration management designed for ML workflows. It supports multiple file formats, environment variables, schema validation, and hierarchical configuration merging.

### Key Features

- **🗂️ Multi-Format Support**: YAML, JSON, TOML, and environment variables
- **📋 Schema Validation**: Type-safe configuration with dataclass schemas
- **🔗 Hierarchical Merging**: Multiple configuration sources with priority handling
- **🔄 Dynamic Reloading**: Hot-reload configuration changes during runtime
- **🌍 Environment Integration**: Seamless environment variable integration
- **📝 Template Generation**: Auto-generate configuration templates

## Quick Start

```python
from refunc.config import ConfigManager, auto_configure

# Auto-configuration with sensible defaults
config = auto_configure()

# Manual configuration setup
manager = ConfigManager()
manager.add_file_source('config.yaml')
manager.add_env_source('MYAPP_')

# Access configuration values
db_host = config.get('database.host', 'localhost')
model_params = config.get('model.parameters', {})

# Schema validation
from refunc.config import RefuncConfig
validated_config = manager.get_validated(RefuncConfig)
```

---

## Core Classes

### ConfigManager

The main configuration management class that handles multiple sources and provides unified access.

```python
class ConfigManager:
    def __init__(
        self,
        schema: Optional[Type] = None,
        auto_reload: bool = False,
        case_sensitive: bool = False
    )
```

**Parameters:**

- `schema`: Optional dataclass schema for validation
- `auto_reload`: Enable automatic reloading when files change
- `case_sensitive`: Whether configuration keys are case-sensitive

**Example:**

```python
from refunc.config import ConfigManager, RefuncConfig

# Basic setup
config = ConfigManager()

# With schema validation
config = ConfigManager(schema=RefuncConfig, auto_reload=True)

# Add configuration sources
config.add_file_source('app.yaml', priority=10)
config.add_file_source('local.yaml', priority=20)  # Higher priority
config.add_env_source('APP_')
```

#### Source Management

```python
# Add file sources
config.add_file_source(
    path='config.yaml',
    priority=10,
    format='yaml',  # Auto-detected if None
    required=True
)

# Add environment variable source
config.add_env_source(
    prefix='MYAPP_',
    priority=5
)

# Add dictionary source (for programmatic config)
config.add_dict_source({
    'debug': True,
    'database': {'host': 'localhost'}
}, priority=15)

# List all sources
sources = config.get_sources()
for source in sources:
    print(f"{source.name}: priority={source.priority}")
```

#### Value Access

```python
# Get configuration values
value = config.get('key.nested', default_value)
all_config = config.get_all()

# Check if key exists
if config.has('database.host'):
    host = config.get('database.host')

# Get with type conversion
port = config.get_int('database.port', 5432)
enabled = config.get_bool('features.caching', False)
urls = config.get_list('api.endpoints', [])

# Get nested sections
db_config = config.get_section('database')
```

#### Schema Validation

```python
from refunc.config import RefuncConfig, ValidationError

try:
    # Validate entire configuration
    validated = config.get_validated(RefuncConfig)
    
    # Validate specific section
    db_config = config.get_validated_section('database', DatabaseConfig)
    
except ValidationError as e:
    print(f"Configuration validation failed: {e}")
```

#### Configuration Updates

```python
# Update configuration
config.set('database.host', 'new_host')
config.update({'logging.level': 'DEBUG'})

# Merge configurations
config.merge_from_file('override.yaml')
config.merge_from_dict({'new_key': 'value'})

# Reload from sources
config.reload()

# Save current configuration
config.export('current_config.yaml', format='yaml')
```

### ConfigSource

Represents a configuration source with metadata.

```python
@dataclass
class ConfigSource:
    name: str
    path: Optional[Path] = None
    priority: int = 0
    format: Optional[str] = None
    last_modified: Optional[float] = None
```

**Example:**

```python
# Create custom source
source = ConfigSource(
    name="production_config",
    path=Path("prod.yaml"),
    priority=100,
    format="yaml"
)
```

---

## Configuration Schemas

### RefuncConfig

Main configuration schema for refunc applications.

```python
@dataclass
class RefuncConfig:
    # Application settings
    app_name: str = "refunc_app"
    version: str = "1.0.0"
    debug: bool = False
    
    # Database configuration
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # Logging configuration
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Performance settings
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Model configuration
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Training configuration
    training: TrainingConfig = field(default_factory=TrainingConfig)
```

### DatabaseConfig

Database connection and configuration settings.

```python
@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "refunc"
    username: str = "user"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    connection_timeout: int = 30
    ssl_mode: str = "prefer"
    
    # Advanced settings
    echo_sql: bool = False
    auto_migrate: bool = False
    backup_enabled: bool = True
```

### LoggingConfig

Logging system configuration.

```python
@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File logging
    file_enabled: bool = True
    file_path: str = "logs/app.log"
    max_file_size: str = "100MB"
    backup_count: int = 5
    
    # Console logging
    console_enabled: bool = True
    colored_output: bool = True
    
    # Structured logging
    json_logging: bool = False
    include_extra: bool = True
    
    # External integrations
    sentry_dsn: Optional[str] = None
    elk_enabled: bool = False
```

### ModelConfig

ML model configuration settings.

```python
@dataclass
class ModelConfig:
    name: str = "default_model"
    type: str = "sklearn"  # sklearn, tensorflow, pytorch, etc.
    version: str = "1.0.0"
    
    # Model parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Storage settings
    model_dir: str = "models"
    checkpoint_dir: str = "checkpoints"
    auto_save: bool = True
    save_format: str = "pickle"  # pickle, joblib, onnx, etc.
    
    # Optimization
    enable_optimization: bool = True
    optimization_level: str = "O2"
```

### TrainingConfig

Training pipeline configuration.

```python
@dataclass
class TrainingConfig:
    # Basic training settings
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    
    # Validation settings
    validation_split: float = 0.2
    validation_frequency: int = 1
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_frequency: int = 5
    keep_best_only: bool = True
    
    # Data settings
    shuffle: bool = True
    seed: int = 42
    num_workers: int = 4
```

### PerformanceConfig

Performance optimization settings.

```python
@dataclass
class PerformanceConfig:
    # Caching
    cache_enabled: bool = True
    cache_backend: str = "memory"  # memory, redis, file
    cache_ttl: int = 3600
    
    # Parallelization
    num_workers: int = 4
    multiprocessing: bool = True
    thread_pool_size: int = 8
    
    # Memory management
    memory_limit: str = "8GB"
    gc_threshold: int = 1000
    
    # Profiling
    profiling_enabled: bool = False
    profile_memory: bool = False
    profile_cpu: bool = False
```

---

## Utility Functions

### Auto-Configuration

```python
from refunc.config import auto_configure

# Auto-detect and load configuration files
config = auto_configure(
    config_dirs=['./config', '~/.refunc', '/etc/refunc'],
    env_prefix='REFUNC_',
    schema=RefuncConfig
)
```

**Auto-detection order:**

1. `refunc.yaml` / `refunc.yml`
2. `config.yaml` / `config.yml`  
3. `app.yaml` / `app.yml`
4. `refunc.json`
5. `config.json`
6. `refunc.toml`
7. Environment variables

### Template Generation

```python
from refunc.config import create_config_template

# Generate configuration template
template = create_config_template(
    schema=RefuncConfig,
    format='yaml',
    include_docs=True,
    include_examples=True
)

# Save template to file
with open('config_template.yaml', 'w') as f:
    f.write(template)
```

**Generated template example:**

```yaml
# Refunc Configuration Template
# Generated with documentation and examples

# Application settings
app_name: "my_app"          # Application name
version: "1.0.0"            # Application version
debug: false                # Enable debug mode

# Database configuration
database:
  host: "localhost"         # Database host
  port: 5432               # Database port
  database: "refunc"       # Database name
  username: "user"         # Database username
  # ... more fields with documentation
```

### Configuration Validation

```python
from refunc.config import validate_config_file, ValidationError

try:
    # Validate configuration file
    result = validate_config_file(
        'config.yaml',
        schema=RefuncConfig,
        strict=True
    )
    
    if result.is_valid:
        print("Configuration is valid!")
    else:
        for error in result.errors:
            print(f"Error: {error}")
            
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### Configuration Merging

```python
from refunc.config import merge_config_files

# Merge multiple configuration files
merged = merge_config_files([
    'base_config.yaml',
    'environment_config.yaml',
    'local_overrides.yaml'
], output_format='yaml')

# Save merged configuration
with open('final_config.yaml', 'w') as f:
    f.write(merged)
```

### Configuration Export

```python
from refunc.config import export_config

# Export current configuration
config_manager = ConfigManager()
config_manager.add_file_source('config.yaml')

# Export to different formats
export_config(config_manager, 'output.yaml', format='yaml')
export_config(config_manager, 'output.json', format='json')
export_config(config_manager, 'output.toml', format='toml')
```

### Configuration Summary

```python
from refunc.config import get_config_summary

# Get configuration summary
summary = get_config_summary(config_manager)
print(summary)
```

**Output:**

```text
Configuration Summary
====================
Sources: 3
- app.yaml (priority: 10, last modified: 2024-01-15 10:30:00)
- local.yaml (priority: 20, last modified: 2024-01-15 11:00:00) 
- environment (priority: 5)

Total keys: 45
Validation: Passed (RefuncConfig)
```

---

## Environment Variables

### Variable Naming Convention

Environment variables use the prefix `REFUNC_` by default, with nested keys separated by underscores:

```bash
# Simple values
export REFUNC_DEBUG=true
export REFUNC_APP_NAME="my_app"

# Nested configuration
export REFUNC_DATABASE_HOST="db.example.com"
export REFUNC_DATABASE_PORT=5432
export REFUNC_LOGGING_LEVEL="DEBUG"

# Complex values (JSON)
export REFUNC_MODEL_PARAMETERS='{"lr": 0.001, "batch_size": 32}'
export REFUNC_API_ENDPOINTS='["http://api1.com", "http://api2.com"]'
```

### Type Conversion

Environment variables are automatically converted to appropriate types:

```python
# Boolean conversion
REFUNC_DEBUG=true          # -> True
REFUNC_ENABLED=false       # -> False

# Numeric conversion  
REFUNC_PORT=8080           # -> 8080 (int)
REFUNC_RATE=0.001          # -> 0.001 (float)

# JSON conversion
REFUNC_CONFIG='{"key": "value"}'  # -> {"key": "value"} (dict)
REFUNC_LIST='[1, 2, 3]'          # -> [1, 2, 3] (list)

# String (default)
REFUNC_NAME=my_app         # -> "my_app" (str)
```

### Custom Prefix

```python
# Use custom environment prefix
config = ConfigManager()
config.add_env_source('MYAPP_')

# Now variables like MYAPP_DEBUG=true will be loaded
```

---

## CLI Interface

The configuration module includes a command-line interface for common tasks:

### CLI Template Generation

```bash
# Generate configuration template
python -m refunc.config.cli template --output config.yaml --format yaml

# Include documentation and examples
python -m refunc.config.cli template --output config.yaml --docs --examples
```

### CLI Configuration Validation

```bash
# Validate configuration file
python -m refunc.config.cli validate config.yaml

# Validate with specific schema
python -m refunc.config.cli validate config.yaml --schema RefuncConfig

# Strict validation
python -m refunc.config.cli validate config.yaml --strict
```

### CLI Configuration Merging

```bash
# Merge multiple files
python -m refunc.config.cli merge base.yaml prod.yaml local.yaml --output final.yaml

# Merge with environment variables
python -m refunc.config.cli merge config.yaml --env-prefix MYAPP_ --output final.yaml
```

### Format Conversion

```bash
# Convert between formats
python -m refunc.config.cli convert config.yaml config.json
python -m refunc.config.cli convert config.json config.toml
```

### CLI Configuration Summary

```bash
# Get configuration summary
python -m refunc.config.cli summary config.yaml

# Include environment variables
python -m refunc.config.cli summary config.yaml --include-env REFUNC_
```

---

## Advanced Usage

### Custom Loaders

```python
from refunc.config import ConfigLoader, ConfigManager

class CustomLoader(ConfigLoader):
    def can_load(self, source):
        return str(source).endswith('.custom')
    
    def load(self, source):
        # Custom loading logic
        return {"loaded_from": "custom_format"}

# Register custom loader
config = ConfigManager()
config.register_loader(CustomLoader())
```

### Dynamic Configuration

```python
# Watch for configuration changes
config = ConfigManager(auto_reload=True)
config.add_file_source('config.yaml')

# Configuration will automatically reload when file changes
# Use callbacks to handle changes
def on_config_change(old_config, new_config):
    print("Configuration changed!")

config.add_change_callback(on_config_change)
```

### Configuration Encryption

```python
from refunc.config import ConfigManager
from refunc.config.encryption import EncryptedLoader

# Load encrypted configuration
config = ConfigManager()
config.register_loader(EncryptedLoader(key='your-encryption-key'))
config.add_file_source('encrypted_config.yaml.enc')
```

### Configuration Profiles

```python
# Profile-based configuration
config = ConfigManager()
config.add_file_source('base.yaml', priority=10)

# Load profile-specific configuration
profile = os.getenv('ENVIRONMENT', 'development')
config.add_file_source(f'config_{profile}.yaml', priority=20)

# Override with local settings
config.add_file_source('local.yaml', priority=30, required=False)
```

---

## Error Handling

### Configuration Errors

```python
from refunc.config import ConfigError, ValidationError

try:
    config = ConfigManager()
    config.add_file_source('nonexistent.yaml')
    
except ConfigError as e:
    print(f"Configuration error: {e}")

try:
    validated = config.get_validated(RefuncConfig)
    
except ValidationError as e:
    print(f"Validation error: {e}")
    print(f"Field errors: {e.field_errors}")
```

### Graceful Degradation

```python
# Provide fallbacks for missing configuration
config = ConfigManager()

try:
    config.add_file_source('config.yaml')
except ConfigError:
    # Fall back to defaults
    config.add_dict_source({
        'database': {'host': 'localhost', 'port': 5432},
        'logging': {'level': 'INFO'}
    })
```

---

## Best Practices

### 1. Use Schema Validation

```python
# Always define schemas for type safety
@dataclass
class MyAppConfig:
    api_url: str
    timeout: int = 30
    retry_attempts: int = 3

config = ConfigManager(schema=MyAppConfig)
```

### 2. Environment-Specific Configuration

```python
# Use profiles for different environments
def create_config():
    config = ConfigManager()
    
    # Base configuration
    config.add_file_source('base.yaml', priority=10)
    
    # Environment-specific
    env = os.getenv('ENVIRONMENT', 'development')
    config.add_file_source(f'{env}.yaml', priority=20, required=False)
    
    # Local overrides
    config.add_file_source('local.yaml', priority=30, required=False)
    
    # Environment variables (highest priority)
    config.add_env_source('MYAPP_', priority=40)
    
    return config
```

### 3. Secure Sensitive Data

```python
# Don't put secrets in configuration files
# Use environment variables instead
@dataclass
class SecureConfig:
    api_key: str = ""  # Set via MYAPP_API_KEY
    database_password: str = ""  # Set via MYAPP_DATABASE_PASSWORD
    
    def __post_init__(self):
        if not self.api_key:
            raise ValueError("API key must be provided via environment variable")
```

### 4. Configuration Validation

```python
# Validate configuration at startup
def validate_startup_config(config: ConfigManager):
    try:
        validated = config.get_validated(MyAppConfig)
        
        # Additional business logic validation
        if validated.timeout <= 0:
            raise ValueError("Timeout must be positive")
            
        return validated
        
    except ValidationError as e:
        print(f"Configuration validation failed: {e}")
        sys.exit(1)
```

---

## Examples

### Complete Application Configuration

```python
from refunc.config import ConfigManager, auto_configure
from dataclasses import dataclass
import os

@dataclass
class AppConfig:
    # Application
    name: str = "ml_pipeline"
    version: str = "1.0.0"
    debug: bool = False
    
    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "ml_data"
    
    # Model
    model_path: str = "models/current"
    batch_size: int = 32
    learning_rate: float = 0.001

def setup_configuration():
    """Setup application configuration."""
    
    # Auto-configure with schema validation
    config = auto_configure(schema=AppConfig)
    
    # Validate configuration
    try:
        app_config = config.get_validated(AppConfig)
        print(f"Starting {app_config.name} v{app_config.version}")
        
        if app_config.debug:
            print("Debug mode enabled")
            
        return app_config
        
    except Exception as e:
        print(f"Configuration error: {e}")
        return None

# Usage
if __name__ == "__main__":
    config = setup_configuration()
    if config:
        # Application startup with validated configuration
        main(config)
```

### Configuration with Multiple Environments

```yaml
# base.yaml
app:
  name: "ml_pipeline"
  version: "1.0.0"

database:
  pool_size: 5

logging:
  level: "INFO"
  file_enabled: true
```

```yaml
# development.yaml
app:
  debug: true

database:
  host: "localhost"
  port: 5432

logging:
  level: "DEBUG"
```

```yaml
# production.yaml
app:
  debug: false

database:
  host: "prod-db.example.com"
  port: 5432
  ssl_mode: "require"

logging:
  level: "WARNING"
  sentry_dsn: "https://..."
```

```python
# Application code
def create_app_config():
    config = ConfigManager()
    
    # Load base configuration
    config.add_file_source('config/base.yaml', priority=10)
    
    # Load environment-specific configuration
    env = os.getenv('ENVIRONMENT', 'development')
    env_file = f'config/{env}.yaml'
    
    if os.path.exists(env_file):
        config.add_file_source(env_file, priority=20)
    
    # Environment variables override everything
    config.add_env_source('ML_PIPELINE_', priority=30)
    
    return config.get_validated(AppConfig)
```

---

## See Also

- **[⚠️ Exceptions](exceptions.md)** - Error handling for configuration errors
- **[📝 Logging](logging.md)** - Logging configuration integration
- **[🚀 Quick Start Guide](../guides/quickstart.md)** - Getting started with configuration
- **[💡 Examples](../examples/)** - More configuration examples
