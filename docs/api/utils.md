# Utilities Module API Reference

The `refunc.utils` module provides comprehensive file handling, caching, and format detection utilities for efficient data processing in ML workflows.

---

## Overview

The utils module offers three core components:

- **FileHandler**: Smart file detection, multi-format loading/saving with intelligent caching
- **Caching System**: Memory and disk-based caching with TTL, compression, and LRU eviction
- **Format Detection**: Automatic file format detection and validation

**Key Features:**

- Support for 10+ data formats (CSV, JSON, Parquet, Excel, HDF5, etc.)
- Intelligent caching with memory and disk storage options
- Automatic format detection from file extensions
- Dependency checking and error handling
- Thread-safe operations with proper locking

---

## Quick Start

```python
from refunc.utils import FileHandler, MemoryCache, cache_result

# Smart file handling with auto-detection
handler = FileHandler(cache_enabled=True)
data = handler.load_auto('data.csv')  # Auto-detects CSV format
handler.save_auto(data, 'output.parquet')  # Auto-saves as Parquet

# Caching function results
@cache_result(ttl_seconds=3600, use_disk=True)
def expensive_computation(x, y):
    return x ** y + complex_calculation()

# Direct caching usage
cache = MemoryCache(max_size=1000, ttl_seconds=1800)
cache.put("key", expensive_data)
result = cache.get("key")
```

---

## Core Classes

### FileHandler

Smart file handler with automatic format detection and caching.

```python
from refunc.utils import FileHandler

# Initialize with caching options
handler = FileHandler(
    cache_enabled=True,
    cache_ttl_seconds=3600,  # 1 hour
    use_disk_cache=False,
    cache_dir=None,
    default_compression=True
)
```

**Parameters:**

- `cache_enabled` (bool): Enable intelligent caching for file operations
- `cache_ttl_seconds` (float, optional): Time-to-live for cache entries in seconds
- `use_disk_cache` (bool): Use disk-based caching instead of memory
- `cache_dir` (Path, optional): Directory for disk cache files
- `default_compression` (bool): Enable compression for disk cache

**Core Methods:**

#### load_auto()

Automatically detect format and load any supported file.

```python
# Auto-detection based on file extension
data = handler.load_auto('dataset.csv')
df = handler.load_auto('results.parquet')
config = handler.load_auto('settings.yaml')

# With format-specific parameters
df = handler.load_auto('data.csv', encoding='utf-8', sep=';')
data = handler.load_auto('store.h5', key='experiments')
```

#### save_auto()

Automatically detect format and save data.

```python
# Saves in appropriate format based on extension
handler.save_auto(dataframe, 'output.csv')
handler.save_auto(results_dict, 'results.json')
handler.save_auto(model_data, 'model.pkl')

# With format-specific options
handler.save_auto(df, 'export.xlsx', sheet_name='Results')
```

#### Format-Specific Loaders

For when you need explicit format control:

```python
# CSV/TSV
df = handler.load_csv('data.csv', sep=',', encoding='utf-8')

# JSON (returns DataFrame or raw data)
data = handler.load_json('config.json')

# Parquet (high-performance columnar format)
df = handler.load_parquet('large_dataset.parquet')

# Excel with sheet selection
df = handler.load_excel('report.xlsx', sheet_name='Summary')

# Pickle (Python objects)
model = handler.load_pickle('trained_model.pkl')

# HDF5 (requires key parameter)
df = handler.load_hdf5('experiment_data.h5', key='results')

# Feather (fast binary format)
df = handler.load_feather('processed_data.feather')

# YAML configuration files
config = handler.load_yaml('config.yaml')
```

#### File Discovery

Find and search for files:

```python
# Search with glob patterns
csv_files = handler.search_pattern('/data', '*.csv', recursive=True)
log_files = handler.search_pattern('/logs', 'error_*.log', recursive=False)

# Find files by format
parquet_files = handler.find_files_by_format('/datasets', FileFormat.PARQUET)
config_files = handler.find_files_by_format('/config', FileFormat.YAML)
```

#### Format Detection and Validation

Inspect file properties:

```python
# Detect format from file path
format_type = handler.detect_format('data.parquet')  # Returns FileFormat.PARQUET

# Check if format is supported
is_supported = handler.is_supported('data.exotic')  # Returns False

# Get comprehensive file information
info = handler.get_file_info('dataset.csv')
# Returns: {format, extension, supported, dependencies_available, 
#          missing_dependencies, file_exists, file_size}
```

#### Cache Management

Control caching behavior:

```python
# Clear cache manually
handler.clear_cache()

# Get cache statistics
stats = handler.cache_stats()
# Returns: {entry_count, total_size_mb, max_size, ttl_seconds}
```

---

## Caching System

### MemoryCache

High-performance in-memory cache with LRU eviction and TTL support.

```python
from refunc.utils import MemoryCache

cache = MemoryCache(
    max_size=1000,           # Maximum number of entries
    ttl_seconds=3600,        # Time-to-live in seconds
    max_memory_mb=100        # Maximum memory usage in MB
)

# Basic operations
cache.put("user_123", user_data)
user = cache.get("user_123")
cache.remove("old_key")
cache.clear()

# Get cache statistics
stats = cache.stats()
print(f"Entries: {stats['entry_count']}, Size: {stats['total_size_mb']} MB")
```

**Features:**

- **LRU Eviction**: Automatically removes least recently used entries
- **TTL Support**: Entries expire after specified time
- **Memory Limiting**: Prevents excessive memory usage
- **Thread Safety**: Safe for concurrent access
- **Access Tracking**: Monitors usage patterns

### DiskCache

Persistent disk-based cache with compression support.

```python
from refunc.utils import DiskCache

cache = DiskCache(
    cache_dir="./cache",     # Cache directory
    compress=True,           # Enable gzip compression
    max_size_mb=500         # Maximum cache size in MB
)

# Same interface as MemoryCache
cache.put("model_weights", large_model_data)
weights = cache.get("model_weights")

# Automatically manages disk space and file compression
```

**Features:**

- **Compression**: Optional gzip compression for space efficiency
- **Size Management**: Automatic cleanup when size limits exceeded
- **Persistence**: Survives application restarts
- **Corruption Recovery**: Handles corrupted cache files gracefully

### CacheEntry

Metadata wrapper for cached values with access tracking.

```python
from refunc.utils import CacheEntry
import time

# Create cache entry with metadata
entry = CacheEntry(
    value=expensive_data,
    created_at=time.time(),
    access_count=0,
    last_accessed=None,
    size_bytes=1024
)

# Access value and update metadata
data = entry.access()  # Increments access_count, updates last_accessed

# Check entry metadata
age = entry.age()                    # Age in seconds
inactive = entry.time_since_access() # Time since last access
```

### cache_result Decorator

Decorator for automatic function result caching.

```python
from refunc.utils import cache_result

# Memory caching with TTL
@cache_result(ttl_seconds=1800)
def compute_statistics(dataset_path):
    df = pd.read_csv(dataset_path)
    return df.describe()

# Disk caching for large results
@cache_result(
    use_disk=True,
    cache_dir="./computation_cache",
    compress=True
)
def train_model(data, hyperparams):
    model = expensive_training(data, hyperparams)
    return model

# Custom cache key generation
def custom_key_fn(*args, **kwargs):
    return f"custom_{hash(str(args))}"

@cache_result(cache_key_fn=custom_key_fn)
def process_with_custom_key(data):
    return expensive_processing(data)

# Access underlying cache
result = compute_statistics.cache.stats()
compute_statistics.cache.clear()
```

**Parameters:**

- `cache_key_fn` (Callable, optional): Custom function to generate cache keys
- `ttl_seconds` (float, optional): Time-to-live for cache entries
- `use_disk` (bool): Use disk-based caching instead of memory
- `cache_dir` (Path, optional): Directory for disk cache
- `compress` (bool): Enable compression for disk cache

---

## Format Detection

### FileFormat Enum

Enumeration of supported file formats.

```python
from refunc.utils import FileFormat

# Available formats
FileFormat.CSV       # Comma-separated values
FileFormat.TSV       # Tab-separated values  
FileFormat.JSON      # JSON data
FileFormat.PARQUET   # Apache Parquet
FileFormat.PICKLE    # Python pickle
FileFormat.EXCEL     # Excel spreadsheets
FileFormat.HDF5      # Hierarchical Data Format
FileFormat.FEATHER   # Apache Arrow Feather
FileFormat.TXT       # Plain text
FileFormat.YAML      # YAML configuration
FileFormat.UNKNOWN   # Unsupported format
```

### FormatRegistry

Registry for format detection and dependency management.

```python
from refunc.utils import FormatRegistry

# Detect format from file path
format_type = FormatRegistry.detect_format('data.parquet')

# Detect from MIME type
format_type = FormatRegistry.detect_format_from_mime('text/csv')

# Check if format is supported
is_supported = FormatRegistry.is_supported('file.unknown')

# Get required packages for a format
packages = FormatRegistry.get_required_packages(FileFormat.PARQUET)
# Returns: ['pandas', 'pyarrow']

# Check dependency availability
available, missing = FormatRegistry.check_dependencies(FileFormat.EXCEL)
# Returns: (False, ['openpyxl']) if openpyxl not installed

# Get all supported extensions
extensions = FormatRegistry.get_supported_extensions()
# Returns: {'.csv', '.json', '.parquet', ...}

# Group formats by category
categories = FormatRegistry.get_formats_by_category()
# Returns: {'tabular': [...], 'structured': [...], 'binary': [...]}
```

### Format Validation Functions

Utility functions for format validation and information.

```python
from refunc.utils import validate_file_format, get_format_info

# Validate file format
is_valid = validate_file_format('data.csv', FileFormat.CSV)  # True
is_supported = validate_file_format('data.csv')  # True (any supported format)

# Get comprehensive format information
info = get_format_info('dataset.parquet')
print(info)
# {
#     'format': FileFormat.PARQUET,
#     'extension': '.parquet',
#     'supported': True,
#     'dependencies_available': True,
#     'missing_dependencies': [],
#     'required_packages': ['pandas', 'pyarrow'],
#     'file_exists': True,
#     'file_size': 1048576
# }
```

---

## Advanced Usage

### Custom Cache Implementations

```python
# Combining memory and disk caching
class HybridCache:
    def __init__(self):
        self.memory = MemoryCache(max_size=100, ttl_seconds=300)  # 5 min
        self.disk = DiskCache("./hybrid_cache", compress=True)
    
    def get(self, key):
        # Try memory first, fallback to disk
        result = self.memory.get(key)
        if result is None:
            result = self.disk.get(key)
            if result is not None:
                self.memory.put(key, result)  # Promote to memory
        return result
    
    def put(self, key, value):
        self.memory.put(key, value)
        self.disk.put(key, value)  # Persist to disk
```

### Batch File Processing

```python
# Process multiple files efficiently
def process_dataset_directory(directory_path):
    handler = FileHandler(cache_enabled=True)
    
    # Find all data files
    data_files = []
    for format_type in [FileFormat.CSV, FileFormat.PARQUET, FileFormat.JSON]:
        files = handler.find_files_by_format(directory_path, format_type)
        data_files.extend(files)
    
    results = []
    for file_path in data_files:
        try:
            # Auto-load and process
            data = handler.load_auto(file_path)
            processed = process_data(data)
            results.append(processed)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return results
```

### Configuration-Driven File Processing

```python
# YAML configuration for file processing pipeline
processing_config = {
    'input_formats': ['csv', 'parquet', 'json'],
    'output_format': 'parquet',
    'cache_settings': {
        'enabled': True,
        'ttl_seconds': 7200,
        'use_disk': True,
        'compress': True
    },
    'processing_steps': [
        'validate_schema',
        'clean_data',
        'feature_engineering'
    ]
}

def configure_file_processing(config):
    # Create file handler with config settings
    cache_config = config['cache_settings']
    handler = FileHandler(
        cache_enabled=cache_config['enabled'],
        cache_ttl_seconds=cache_config['ttl_seconds'],
        use_disk_cache=cache_config['use_disk'],
        default_compression=cache_config['compress']
    )
    
    return handler
```

---

## Error Handling

The utils module provides comprehensive error handling with descriptive messages and recovery suggestions.

### Common Exceptions

```python
from refunc.exceptions import (
    DataError,
    FileNotFoundError,
    UnsupportedFormatError,
    CorruptedDataError
)

try:
    data = handler.load_auto('missing_file.csv')
except FileNotFoundError as e:
    print(f"File not found: {e}")

try:
    data = handler.load_auto('data.exotic_format')
except UnsupportedFormatError as e:
    print(f"Unsupported format: {e}")
    print(f"Supported extensions: {e.context['supported_extensions']}")

try:
    data = handler.load_parquet('corrupted.parquet')
except CorruptedDataError as e:
    print(f"Corrupted data: {e}")
    print(f"File: {e.context['file_path']}")

try:
    data = handler.load_hdf5('data.h5')  # Missing 'key' parameter
except DataError as e:
    print(f"Data error: {e}")
    print(f"Suggestion: {e.suggestion}")
```

### Dependency Checking

```python
# Check dependencies before using specific formats
format_type = FileFormat.PARQUET
available, missing = FormatRegistry.check_dependencies(format_type)

if not available:
    print(f"Missing dependencies: {missing}")
    print(f"Install with: pip install {' '.join(missing)}")
    # Fallback to alternative format
    handler.save_auto(data, 'output.csv')  # Use CSV instead
else:
    handler.save_auto(data, 'output.parquet')
```

---

## Best Practices

### 1. Cache Strategy

```python
# Use appropriate cache based on data characteristics
handler = FileHandler(
    cache_enabled=True,
    cache_ttl_seconds=3600,  # 1 hour for temporary data
    use_disk_cache=True,     # For large datasets
    default_compression=True  # Save disk space
)

# Clear cache periodically to prevent excessive growth
if handler.cache_stats()['total_size_mb'] > 500:  # 500 MB limit
    handler.clear_cache()
```

### 2. Format Selection

```python
# Choose appropriate formats for your use case
performance_formats = [FileFormat.PARQUET, FileFormat.FEATHER]  # Fast I/O
portable_formats = [FileFormat.CSV, FileFormat.JSON]           # Portable
compressed_formats = [FileFormat.PARQUET, FileFormat.HDF5]     # Space-efficient

# Use format detection to handle mixed datasets
def smart_save(data, base_path):
    if isinstance(data, pd.DataFrame) and len(data) > 100000:
        # Large DataFrames: use Parquet for performance
        handler.save_auto(data, f"{base_path}.parquet")
    elif isinstance(data, dict):
        # Configuration data: use YAML for readability
        handler.save_auto(data, f"{base_path}.yaml")
    else:
        # Default: use JSON for compatibility
        handler.save_auto(data, f"{base_path}.json")
```

### 3. Error Recovery

```python
def robust_load(file_paths, fallback_format=FileFormat.CSV):
    """Load data with automatic fallback strategies."""
    handler = FileHandler()
    
    for file_path in file_paths:
        try:
            return handler.load_auto(file_path)
        except UnsupportedFormatError:
            # Try fallback format
            try:
                fallback_path = Path(file_path).with_suffix(f'.{fallback_format.value}')
                return handler.load_auto(fallback_path)
            except Exception:
                continue
        except CorruptedDataError:
            print(f"Skipping corrupted file: {file_path}")
            continue
    
    raise DataError("No valid data files found")
```

### 4. Performance Optimization

```python
# Pre-warm cache for frequently accessed files
critical_files = ['config.yaml', 'model_weights.pkl', 'feature_names.json']
for file_path in critical_files:
    handler.load_auto(file_path)  # Loads into cache

# Use disk cache for large, infrequently changing datasets
large_data_handler = FileHandler(
    use_disk_cache=True,
    cache_dir="./large_data_cache",
    default_compression=True
)

# Batch operations to minimize I/O
@cache_result(ttl_seconds=86400)  # Cache for 24 hours
def load_all_config_files(config_dir):
    config_files = handler.find_files_by_format(config_dir, FileFormat.YAML)
    return {f.name: handler.load_auto(f) for f in config_files}
```

---

## Examples

### Complete Data Pipeline

```python
from refunc.utils import FileHandler, cache_result
from pathlib import Path

class DataPipeline:
    def __init__(self, cache_dir="./pipeline_cache"):
        self.handler = FileHandler(
            cache_enabled=True,
            use_disk_cache=True,
            cache_dir=cache_dir,
            default_compression=True
        )
    
    @cache_result(ttl_seconds=3600, use_disk=True)
    def load_raw_data(self, data_dir):
        """Load and combine all raw data files."""
        raw_files = self.handler.find_files_by_format(data_dir, FileFormat.CSV)
        
        combined_data = []
        for file_path in raw_files:
            try:
                df = self.handler.load_auto(file_path)
                df['source_file'] = file_path.name
                combined_data.append(df)
            except Exception as e:
                print(f"Skipping {file_path}: {e}")
        
        return pd.concat(combined_data, ignore_index=True)
    
    @cache_result(ttl_seconds=7200, use_disk=True)
    def preprocess_data(self, raw_data):
        """Preprocess the combined dataset."""
        # Data cleaning and feature engineering
        processed = raw_data.copy()
        processed = processed.dropna()
        processed['feature_x'] = processed['col_a'] * processed['col_b']
        return processed
    
    def save_results(self, data, output_dir):
        """Save results in multiple formats."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save in different formats for different use cases
        self.handler.save_auto(data, output_path / "results.parquet")  # Performance
        self.handler.save_auto(data, output_path / "results.csv")      # Compatibility
        
        # Save metadata
        metadata = {
            'rows': len(data),
            'columns': list(data.columns),
            'created_at': time.time()
        }
        self.handler.save_auto(metadata, output_path / "metadata.yaml")

# Usage
pipeline = DataPipeline()
raw_data = pipeline.load_raw_data("./data/raw")
processed_data = pipeline.preprocess_data(raw_data)
pipeline.save_results(processed_data, "./data/processed")
```

### Configuration Management System

```python
from refunc.utils import FileHandler, MemoryCache
from typing import Dict, Any

class ConfigManager:
    def __init__(self):
        self.handler = FileHandler(cache_enabled=True)
        self.config_cache = MemoryCache(max_size=50, ttl_seconds=300)
    
    def load_config_hierarchy(self, config_dirs: List[str]) -> Dict[str, Any]:
        """Load configuration from multiple directories with priority."""
        config = {}
        
        for config_dir in config_dirs:
            config_files = self.handler.find_files_by_format(
                config_dir, FileFormat.YAML
            )
            
            for config_file in sorted(config_files):
                try:
                    file_config = self.handler.load_auto(config_file)
                    config.update(file_config)  # Later files override earlier
                except Exception as e:
                    print(f"Warning: Could not load {config_file}: {e}")
        
        return config
    
    def get_config(self, key: str, config_dirs: List[str]) -> Any:
        """Get configuration value with caching."""
        cache_key = f"config_{hash(tuple(config_dirs))}_{key}"
        
        # Check cache first
        cached_value = self.config_cache.get(cache_key)
        if cached_value is not None:
            return cached_value
        
        # Load configuration hierarchy
        config = self.load_config_hierarchy(config_dirs)
        value = config.get(key)
        
        # Cache the result
        self.config_cache.put(cache_key, value)
        return value
    
    def validate_config_files(self, config_dir: str) -> Dict[str, bool]:
        """Validate all configuration files in a directory."""
        config_files = self.handler.find_files_by_format(
            config_dir, FileFormat.YAML
        )
        
        results = {}
        for config_file in config_files:
            try:
                self.handler.load_auto(config_file)
                results[str(config_file)] = True
            except Exception:
                results[str(config_file)] = False
        
        return results

# Usage
config_manager = ConfigManager()

# Load configuration from multiple sources
database_url = config_manager.get_config(
    'database_url', 
    ['./config', '~/.myapp', '/etc/myapp']
)

# Validate all configuration files
validation_results = config_manager.validate_config_files('./config')
print(f"Valid configs: {sum(validation_results.values())}/{len(validation_results)}")
```

---

## See Also

- **[⚠️ Exceptions](exceptions.md)** - Error handling for file and cache operations
- **[📝 Logging](logging.md)** - Logging integration for file operations
- **[⚙️ Config](config.md)** - Configuration management utilities
- **[🚀 Quick Start Guide](../guides/quickstart.md)** - Getting started with file handling
- **[💡 Examples](../examples/)** - More file processing examples


