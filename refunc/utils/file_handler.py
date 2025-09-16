"""
Comprehensive file handling utilities for ML workflows.

This module provides the FileHandler class for smart file detection,
multi-format loading, directory scanning, and intelligent caching.
"""

import os
import glob
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Pattern, Callable
import pandas as pd
import numpy as np

from .formats import FileFormat, FormatRegistry, get_format_info
from .cache import MemoryCache, DiskCache, cache_result
from ..exceptions import (
    DataError,
    FileNotFoundError as RefuncFileNotFoundError,
    UnsupportedFormatError,
    CorruptedDataError,
)


class FileHandler:
    """
    Comprehensive file handler with smart detection and caching.
    
    Provides unified interface for loading/saving data across multiple formats
    with automatic format detection, validation, and intelligent caching.
    """
    
    def __init__(
        self,
        cache_enabled: bool = True,
        cache_ttl_seconds: Optional[float] = 3600,  # 1 hour default
        use_disk_cache: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        default_compression: bool = True
    ):
        self.cache_enabled = cache_enabled
        self.cache_ttl_seconds = cache_ttl_seconds
        self.use_disk_cache = use_disk_cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".refunc_cache"
        self.default_compression = default_compression
        
        # Initialize cache
        if cache_enabled:
            if use_disk_cache:
                self._cache = DiskCache(
                    cache_dir=self.cache_dir / "file_handler",
                    compress=default_compression
                )
            else:
                self._cache = MemoryCache(ttl_seconds=cache_ttl_seconds)
        else:
            self._cache = None
    
    def _generate_cache_key(self, file_path: Path, **kwargs) -> str:
        """Generate cache key for file operations."""
        # Include file path, modification time, and any additional parameters
        try:
            mtime = file_path.stat().st_mtime
            params_hash = hash(tuple(sorted(kwargs.items())))
            return f"{file_path}_{mtime}_{params_hash}"
        except OSError:
            # File doesn't exist or can't access
            params_hash = hash(tuple(sorted(kwargs.items())))
            return f"{file_path}_{params_hash}"
    
    def _validate_file_exists(self, file_path: Union[str, Path]) -> Path:
        """Validate that file exists and return Path object."""
        path = Path(file_path)
        if not path.exists():
            raise RefuncFileNotFoundError(str(path))
        return path
    
    def detect_format(self, file_path: Union[str, Path]) -> FileFormat:
        """Detect file format from path."""
        return FormatRegistry.detect_format(file_path)
    
    def is_supported(self, file_path: Union[str, Path]) -> bool:
        """Check if file format is supported."""
        return FormatRegistry.is_supported(file_path)
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get comprehensive file information."""
        return get_format_info(file_path)
    
    def load_csv(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load CSV file with pandas."""
        path = self._validate_file_exists(file_path)
        
        try:
            return pd.read_csv(path, **kwargs)
        except Exception as e:
            raise CorruptedDataError(str(path), str(e))
    
    def load_json(self, file_path: Union[str, Path], **kwargs) -> Union[pd.DataFrame, Dict, List]:
        """Load JSON file."""
        path = self._validate_file_exists(file_path)
        
        try:
            # Try loading as DataFrame first, fall back to raw JSON
            try:
                return pd.read_json(path, **kwargs)
            except ValueError:
                import json
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            raise CorruptedDataError(str(path), str(e))
    
    def load_parquet(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load Parquet file with pandas."""
        path = self._validate_file_exists(file_path)
        
        try:
            return pd.read_parquet(path, **kwargs)
        except Exception as e:
            raise CorruptedDataError(str(path), str(e))
    
    def load_excel(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load Excel file with pandas."""
        path = self._validate_file_exists(file_path)
        
        try:
            return pd.read_excel(path, **kwargs)
        except Exception as e:
            raise CorruptedDataError(str(path), str(e))
    
    def load_pickle(self, file_path: Union[str, Path]) -> Any:
        """Load pickle file."""
        path = self._validate_file_exists(file_path)
        
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise CorruptedDataError(str(path), str(e))
    
    def load_hdf5(self, file_path: Union[str, Path], key: str, **kwargs) -> pd.DataFrame:
        """Load HDF5 file with pandas."""
        path = self._validate_file_exists(file_path)
        
        try:
            return pd.read_hdf(path, key=key, **kwargs)
        except Exception as e:
            raise CorruptedDataError(str(path), str(e))
    
    def load_feather(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load Feather file with pandas."""
        path = self._validate_file_exists(file_path)
        
        try:
            return pd.read_feather(path, **kwargs)
        except Exception as e:
            raise CorruptedDataError(str(path), str(e))
    
    def load_yaml(self, file_path: Union[str, Path]) -> Union[Dict, List]:
        """Load YAML file."""
        path = self._validate_file_exists(file_path)
        
        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise CorruptedDataError(str(path), str(e))
    
    def load_auto(self, file_path: Union[str, Path], **kwargs) -> Any:
        """
        Automatically detect format and load file.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional arguments passed to the specific loader
        
        Returns:
            Loaded data in appropriate format
        """
        path = self._validate_file_exists(file_path)
        
        # Check cache first
        cache_key = None
        if self._cache:
            cache_key = self._generate_cache_key(path, **kwargs)
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Detect format
        format_type = self.detect_format(path)
        
        if format_type == FileFormat.UNKNOWN:
            raise UnsupportedFormatError(str(path), list(FormatRegistry.get_supported_extensions()))
        
        # Check dependencies
        deps_available, missing_deps = FormatRegistry.check_dependencies(format_type)
        if not deps_available:
            raise DataError(
                f"Missing dependencies for {format_type.value} format: {missing_deps}",
                context={"format": format_type.value, "missing_packages": missing_deps},
                suggestion=f"Install missing packages: pip install {' '.join(missing_deps)}"
            )
        
        # Load using appropriate loader
        loader_map = {
            FileFormat.CSV: self.load_csv,
            FileFormat.TSV: lambda p, **kw: self.load_csv(p, sep='\t', **kw),
            FileFormat.JSON: self.load_json,
            FileFormat.PARQUET: self.load_parquet,
            FileFormat.EXCEL: self.load_excel,
            FileFormat.PICKLE: self.load_pickle,
            FileFormat.FEATHER: self.load_feather,
            FileFormat.YAML: self.load_yaml,
        }
        
        # Special handling for HDF5 (requires key parameter)
        if format_type == FileFormat.HDF5:
            if 'key' not in kwargs:
                raise DataError(
                    "HDF5 files require a 'key' parameter",
                    suggestion="Specify the dataset key: load_auto(file_path, key='dataset_name')"
                )
            result = self.load_hdf5(path, **kwargs)
        else:
            loader = loader_map.get(format_type)
            if not loader:
                raise UnsupportedFormatError(str(path), [fmt.value for fmt in loader_map.keys()])
            
            result = loader(path, **kwargs)
        
        # Cache result
        if self._cache and cache_key:
            self._cache.put(cache_key, result)
        
        return result
    
    def save_auto(self, data: Any, file_path: Union[str, Path], **kwargs) -> None:
        """
        Automatically detect format and save data.
        
        Args:
            data: Data to save
            file_path: Path where to save the file
            **kwargs: Additional arguments passed to the specific saver
        """
        path = Path(file_path)
        format_type = self.detect_format(path)
        
        if format_type == FileFormat.UNKNOWN:
            raise UnsupportedFormatError(str(path), list(FormatRegistry.get_supported_extensions()))
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save using appropriate method
        if format_type == FileFormat.CSV:
            if isinstance(data, pd.DataFrame):
                data.to_csv(path, index=False, **kwargs)
            else:
                raise DataError("CSV format requires pandas DataFrame")
        
        elif format_type == FileFormat.TSV:
            if isinstance(data, pd.DataFrame):
                data.to_csv(path, sep='\t', index=False, **kwargs)
            else:
                raise DataError("TSV format requires pandas DataFrame")
        
        elif format_type == FileFormat.JSON:
            if isinstance(data, pd.DataFrame):
                data.to_json(path, **kwargs)
            else:
                import json
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, **kwargs)
        
        elif format_type == FileFormat.PARQUET:
            if isinstance(data, pd.DataFrame):
                data.to_parquet(path, **kwargs)
            else:
                raise DataError("Parquet format requires pandas DataFrame")
        
        elif format_type == FileFormat.EXCEL:
            if isinstance(data, pd.DataFrame):
                data.to_excel(path, index=False, **kwargs)
            else:
                raise DataError("Excel format requires pandas DataFrame")
        
        elif format_type == FileFormat.PICKLE:
            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        elif format_type == FileFormat.FEATHER:
            if isinstance(data, pd.DataFrame):
                data.to_feather(path, **kwargs)
            else:
                raise DataError("Feather format requires pandas DataFrame")
        
        elif format_type == FileFormat.HDF5:
            if isinstance(data, pd.DataFrame):
                if 'key' not in kwargs:
                    kwargs['key'] = 'data'
                data.to_hdf(path, **kwargs)
            else:
                raise DataError("HDF5 format requires pandas DataFrame")
        
        elif format_type == FileFormat.YAML:
            import yaml
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, **kwargs)
        
        else:
            raise UnsupportedFormatError(str(path), list(FormatRegistry.get_supported_extensions()))
    
    def search_pattern(self, directory: Union[str, Path], pattern: str, recursive: bool = True) -> List[Path]:
        """
        Search for files matching a pattern.
        
        Args:
            directory: Directory to search in
            pattern: Glob pattern to match
            recursive: Whether to search recursively
        
        Returns:
            List of matching file paths
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise RefuncFileNotFoundError(str(dir_path))
        
        if recursive:
            search_pattern = f"**/{pattern}"
            matches = list(dir_path.glob(search_pattern))
        else:
            matches = list(dir_path.glob(pattern))
        
        # Filter to only return files (not directories)
        return [p for p in matches if p.is_file()]
    
    def find_files_by_format(self, directory: Union[str, Path], format_type: FileFormat, recursive: bool = True) -> List[Path]:
        """
        Find all files of a specific format in a directory.
        
        Args:
            directory: Directory to search in
            format_type: File format to search for
            recursive: Whether to search recursively
        
        Returns:
            List of matching file paths
        """
        # Get extensions for this format
        extensions = [ext for ext, fmt in FormatRegistry.EXTENSION_MAP.items() if fmt == format_type]
        
        all_matches = []
        for ext in extensions:
            pattern = f"*{ext}"
            matches = self.search_pattern(directory, pattern, recursive)
            all_matches.extend(matches)
        
        return sorted(set(all_matches))  # Remove duplicates and sort
    
    def clear_cache(self) -> None:
        """Clear the file handler cache."""
        if self._cache:
            self._cache.clear()
    
    def cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if self._cache and hasattr(self._cache, 'stats'):
            return self._cache.stats()  # type: ignore
        return None