"""
Performance benchmarks for refunc file handling utilities.

This module contains benchmarks for file I/O operations, format detection,
caching performance, and data loading/saving operations.
"""

import pytest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import json
import os


# Import file handling utilities for benchmarking  
try:
    from refunc.utils import FileHandler
    from refunc.utils.formats import FileFormat, get_format_info
    from refunc.utils.cache import MemoryCache, DiskCache
    FILE_UTILS_AVAILABLE = True
except ImportError:
    FILE_UTILS_AVAILABLE = False
    pytestmark = pytest.mark.skip("refunc file utilities not available")


@pytest.fixture
def temp_dir():
    """Create temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'id': range(1000),
        'value': np.random.random(1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1H')
    })


@pytest.fixture
def sample_json_data():
    """Create sample JSON data for testing."""
    return {
        'metadata': {
            'version': '1.0',
            'created': '2023-01-01T00:00:00',
            'author': 'benchmark'
        },
        'data': [
            {'id': i, 'value': i * 2, 'active': i % 2 == 0}
            for i in range(1000)
        ]
    }


@pytest.mark.skipif(not FILE_UTILS_AVAILABLE, reason="refunc file utilities not available")
class TestFileHandlerPerformance:
    """Benchmark tests for FileHandler class."""
    
    def test_file_handler_initialization(self, benchmark):
        """Benchmark FileHandler initialization."""
        
        def create_file_handler():
            return FileHandler(
                cache_enabled=True,
                cache_ttl_seconds=3600,
                use_disk_cache=False
            )
        
        handler = benchmark(create_file_handler)
        assert handler is not None
    
    def test_csv_loading_performance(self, benchmark, temp_dir, sample_dataframe):
        """Benchmark CSV file loading performance."""
        
        # Create test CSV file
        csv_file = temp_dir / "test_data.csv"
        sample_dataframe.to_csv(csv_file, index=False)
        
        handler = FileHandler(cache_enabled=False)
        
        def load_csv():
            return handler.load_file(csv_file)
        
        loaded_df = benchmark(load_csv)
        assert len(loaded_df) == 1000
        assert list(loaded_df.columns) == ['id', 'value', 'category', 'timestamp']
    
    def test_csv_saving_performance(self, benchmark, temp_dir, sample_dataframe):
        """Benchmark CSV file saving performance."""
        
        handler = FileHandler()
        csv_file = temp_dir / "output.csv"
        
        def save_csv():
            return handler.save_file(sample_dataframe, csv_file)
        
        result = benchmark(save_csv)
        assert csv_file.exists()
        assert csv_file.stat().st_size > 0
    
    def test_json_loading_performance(self, benchmark, temp_dir, sample_json_data):
        """Benchmark JSON file loading performance."""
        
        # Create test JSON file
        json_file = temp_dir / "test_data.json"
        with open(json_file, 'w') as f:
            json.dump(sample_json_data, f)
        
        handler = FileHandler(cache_enabled=False)
        
        def load_json():
            return handler.load_file(json_file)
        
        loaded_data = benchmark(load_json)
        assert 'metadata' in loaded_data
        assert len(loaded_data['data']) == 1000
    
    def test_parquet_performance(self, benchmark, temp_dir, sample_dataframe):
        """Benchmark Parquet file operations."""
        
        handler = FileHandler()
        parquet_file = temp_dir / "test_data.parquet"
        
        # Save to Parquet
        def save_parquet():
            return handler.save_file(sample_dataframe, parquet_file)
        
        save_result = benchmark.pedantic(save_parquet, rounds=3, iterations=1)
        assert parquet_file.exists()
        
        # Load from Parquet
        def load_parquet():
            return handler.load_file(parquet_file)
        
        loaded_df = benchmark.pedantic(load_parquet, rounds=3, iterations=1)
        assert len(loaded_df) == 1000


@pytest.mark.skipif(not FILE_UTILS_AVAILABLE, reason="refunc file utilities not available")
class TestCachingPerformance:
    """Benchmark tests for caching systems."""
    
    def test_memory_cache_performance(self, benchmark):
        """Benchmark memory cache operations."""
        
        cache = MemoryCache(max_size=1000)
        
        def cache_operations():
            # Store items
            for i in range(100):
                cache.set(f"key_{i}", f"value_{i}")
            
            # Retrieve items
            results = []
            for i in range(100):
                result = cache.get(f"key_{i}")
                results.append(result)
            
            return len(results)
        
        result = benchmark(cache_operations)
        assert result == 100
    
    def test_disk_cache_performance(self, benchmark, temp_dir):
        """Benchmark disk cache operations."""
        
        cache = DiskCache(cache_dir=temp_dir, max_size_mb=100)
        
        def disk_cache_operations():
            # Store items
            for i in range(50):  # Fewer items for disk cache
                cache.set(f"disk_key_{i}", f"disk_value_{i}")
            
            # Retrieve items  
            results = []
            for i in range(50):
                result = cache.get(f"disk_key_{i}")
                results.append(result)
            
            return len([r for r in results if r is not None])
        
        result = benchmark(disk_cache_operations)
        assert result == 50
    
    def test_cache_with_file_handler(self, benchmark, temp_dir, sample_dataframe):
        """Benchmark FileHandler with caching enabled."""
        
        # Create test file
        csv_file = temp_dir / "cached_data.csv"
        sample_dataframe.to_csv(csv_file, index=False)
        
        handler = FileHandler(cache_enabled=True, cache_ttl_seconds=3600)
        
        def cached_file_loading():
            # First load (cache miss)
            df1 = handler.load_file(csv_file)
            
            # Second load (cache hit)
            df2 = handler.load_file(csv_file)
            
            return len(df1) + len(df2)
        
        result = benchmark(cached_file_loading)
        assert result == 2000  # 1000 + 1000


@pytest.mark.skipif(not FILE_UTILS_AVAILABLE, reason="refunc file utilities not available") 
class TestFormatDetection:
    """Benchmark tests for format detection."""
    
    def test_format_detection_speed(self, benchmark, temp_dir):
        """Benchmark file format detection."""
        
        # Create files with different formats
        files = {}
        files['csv'] = temp_dir / "test.csv"
        files['json'] = temp_dir / "test.json"
        files['txt'] = temp_dir / "test.txt"
        
        # Create sample files
        files['csv'].write_text("id,value\n1,100\n2,200")
        files['json'].write_text('{"key": "value"}')
        files['txt'].write_text("Sample text content")
        
        def detect_formats():
            results = {}
            for format_name, file_path in files.items():
                format_info = get_format_info(file_path)
                results[format_name] = format_info
            return results
        
        detected_formats = benchmark(detect_formats)
        assert len(detected_formats) == 3
        assert all(info is not None for info in detected_formats.values())
    
    def test_batch_file_processing(self, benchmark, temp_dir, sample_dataframe):
        """Benchmark batch file processing."""
        
        # Create multiple CSV files
        files = []
        for i in range(10):
            file_path = temp_dir / f"batch_{i}.csv"
            # Create smaller datasets for faster processing
            subset_df = sample_dataframe.iloc[:100]  # First 100 rows
            subset_df.to_csv(file_path, index=False)
            files.append(file_path)
        
        handler = FileHandler(cache_enabled=True)
        
        def batch_processing():
            results = []
            for file_path in files:
                df = handler.load_file(file_path)
                results.append(len(df))
            return sum(results)
        
        total_rows = benchmark(batch_processing)
        assert total_rows == 1000  # 10 files * 100 rows each


@pytest.mark.skipif(not FILE_UTILS_AVAILABLE, reason="refunc file utilities not available")
class TestFileIORegression:
    """Regression tests for file I/O performance."""
    
    def test_large_file_handling(self, benchmark, temp_dir):
        """Test performance with larger files."""
        
        # Create larger dataset
        np.random.seed(42)
        large_df = pd.DataFrame({
            'id': range(10000),
            'value': np.random.random(10000),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 10000),
            'data': np.random.random(10000) * 1000
        })
        
        handler = FileHandler()
        large_file = temp_dir / "large_data.csv"
        
        def large_file_operations():
            # Save large file
            handler.save_file(large_df, large_file)
            
            # Load large file
            loaded_df = handler.load_file(large_file)
            
            return len(loaded_df)
        
        result = benchmark.pedantic(large_file_operations, rounds=1, iterations=1)
        assert result == 10000
    
    def test_memory_efficient_processing(self, benchmark, temp_dir):
        """Test memory-efficient file processing."""
        
        # Create file with many small chunks
        handler = FileHandler()
        
        def memory_efficient_processing():
            results = []
            
            # Process multiple small files
            for i in range(20):
                small_df = pd.DataFrame({
                    'batch': [i] * 50,
                    'value': np.random.random(50)
                })
                
                file_path = temp_dir / f"chunk_{i}.csv"
                handler.save_file(small_df, file_path)
                
                loaded_df = handler.load_file(file_path)
                results.append(loaded_df['value'].sum())
            
            return len(results)
        
        result = benchmark(memory_efficient_processing)
        assert result == 20
    
    def test_concurrent_file_access(self, benchmark, temp_dir, sample_dataframe):
        """Test file access patterns that might occur in concurrent scenarios."""
        
        # Create multiple files
        files = []
        for i in range(5):
            file_path = temp_dir / f"concurrent_{i}.csv"
            sample_dataframe.to_csv(file_path, index=False)
            files.append(file_path)
        
        handler = FileHandler(cache_enabled=True)
        
        def concurrent_access_pattern():
            results = []
            
            # Simulate concurrent-like access pattern
            for _ in range(3):  # Multiple rounds
                for file_path in files:
                    df = handler.load_file(file_path)
                    results.append(len(df))
            
            return sum(results)
        
        result = benchmark(concurrent_access_pattern)
        assert result == 15000  # 5 files * 1000 rows * 3 rounds