"""
File format detection and handling utilities.

This module provides utilities for detecting file formats, validating file types,
and determining appropriate loading/saving strategies.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from enum import Enum


class FileFormat(Enum):
    """Supported file formats."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    PICKLE = "pickle"
    EXCEL = "excel"
    HDF5 = "hdf5"
    FEATHER = "feather"
    TSV = "tsv"
    TXT = "txt"
    YAML = "yaml"
    UNKNOWN = "unknown"


class FormatRegistry:
    """Registry for file format detection and handling."""
    
    # File extension mappings
    EXTENSION_MAP: Dict[str, FileFormat] = {
        # CSV formats
        ".csv": FileFormat.CSV,
        ".tsv": FileFormat.TSV,
        ".tab": FileFormat.TSV,
        
        # JSON formats
        ".json": FileFormat.JSON,
        ".jsonl": FileFormat.JSON,
        
        # Parquet
        ".parquet": FileFormat.PARQUET,
        ".pq": FileFormat.PARQUET,
        
        # Pickle
        ".pkl": FileFormat.PICKLE,
        ".pickle": FileFormat.PICKLE,
        
        # Excel
        ".xlsx": FileFormat.EXCEL,
        ".xls": FileFormat.EXCEL,
        
        # HDF5
        ".h5": FileFormat.HDF5,
        ".hdf5": FileFormat.HDF5,
        ".hdf": FileFormat.HDF5,
        
        # Feather
        ".feather": FileFormat.FEATHER,
        
        # Text
        ".txt": FileFormat.TXT,
        
        # YAML
        ".yaml": FileFormat.YAML,
        ".yml": FileFormat.YAML,
    }
    
    # MIME type mappings
    MIME_MAP: Dict[str, FileFormat] = {
        "text/csv": FileFormat.CSV,
        "application/json": FileFormat.JSON,
        "application/vnd.apache.parquet": FileFormat.PARQUET,
        "application/octet-stream": FileFormat.PICKLE,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": FileFormat.EXCEL,
        "application/vnd.ms-excel": FileFormat.EXCEL,
        "application/x-hdf": FileFormat.HDF5,
        "text/plain": FileFormat.TXT,
        "application/x-yaml": FileFormat.YAML,
    }
    
    # Required packages for each format
    PACKAGE_REQUIREMENTS: Dict[FileFormat, List[str]] = {
        FileFormat.CSV: ["pandas"],
        FileFormat.JSON: ["pandas"],
        FileFormat.PARQUET: ["pandas", "pyarrow"],
        FileFormat.PICKLE: [],  # Built-in
        FileFormat.EXCEL: ["pandas", "openpyxl"],
        FileFormat.HDF5: ["pandas", "h5py"],
        FileFormat.FEATHER: ["pandas", "pyarrow"],
        FileFormat.TSV: ["pandas"],
        FileFormat.TXT: [],  # Built-in
        FileFormat.YAML: ["pyyaml"],
    }
    
    @classmethod
    def detect_format(cls, file_path: Union[str, Path]) -> FileFormat:
        """Detect file format from file path."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        return cls.EXTENSION_MAP.get(extension, FileFormat.UNKNOWN)
    
    @classmethod
    def detect_format_from_mime(cls, mime_type: str) -> FileFormat:
        """Detect file format from MIME type."""
        return cls.MIME_MAP.get(mime_type.lower(), FileFormat.UNKNOWN)
    
    @classmethod
    def is_supported(cls, file_path: Union[str, Path]) -> bool:
        """Check if file format is supported."""
        return cls.detect_format(file_path) != FileFormat.UNKNOWN
    
    @classmethod
    def get_required_packages(cls, format_type: FileFormat) -> List[str]:
        """Get required packages for a file format."""
        return cls.PACKAGE_REQUIREMENTS.get(format_type, [])
    
    @classmethod
    def check_dependencies(cls, format_type: FileFormat) -> Tuple[bool, List[str]]:
        """Check if required dependencies are available."""
        required = cls.get_required_packages(format_type)
        missing = []
        
        for package in required:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        return len(missing) == 0, missing
    
    @classmethod
    def get_supported_extensions(cls) -> Set[str]:
        """Get all supported file extensions."""
        return set(cls.EXTENSION_MAP.keys())
    
    @classmethod
    def get_formats_by_category(cls) -> Dict[str, List[FileFormat]]:
        """Group formats by category."""
        return {
            "tabular": [
                FileFormat.CSV, FileFormat.TSV, FileFormat.PARQUET, 
                FileFormat.EXCEL, FileFormat.FEATHER
            ],
            "structured": [FileFormat.JSON, FileFormat.YAML],
            "binary": [FileFormat.PICKLE, FileFormat.HDF5],
            "text": [FileFormat.TXT],
        }


def validate_file_format(file_path: Union[str, Path], expected_format: Optional[FileFormat] = None) -> bool:
    """
    Validate that a file matches the expected format.
    
    Args:
        file_path: Path to the file
        expected_format: Expected file format (if None, just check if supported)
    
    Returns:
        True if file format is valid, False otherwise
    """
    detected_format = FormatRegistry.detect_format(file_path)
    
    if expected_format is None:
        return detected_format != FileFormat.UNKNOWN
    
    return detected_format == expected_format


def get_format_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive format information for a file.
    
    Args:
        file_path: Path to the file
    
    Returns:
        Dictionary with format information
    """
    format_type = FormatRegistry.detect_format(file_path)
    deps_available, missing_deps = FormatRegistry.check_dependencies(format_type)
    
    return {
        "format": format_type,
        "extension": Path(file_path).suffix.lower(),
        "supported": format_type != FileFormat.UNKNOWN,
        "dependencies_available": deps_available,
        "missing_dependencies": missing_deps,
        "required_packages": FormatRegistry.get_required_packages(format_type),
        "file_exists": Path(file_path).exists(),
        "file_size": Path(file_path).stat().st_size if Path(file_path).exists() else None,
    }