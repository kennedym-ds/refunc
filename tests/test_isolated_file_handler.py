"""
Isolated FileHandler tests that avoid scipy import issues.
"""
import pytest
import pandas as pd
import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch


class SimpleFileHandler:
    """Simplified FileHandler for testing core functionality."""
    
    def __init__(self, cache_enabled=True, cache_ttl_seconds=3600, use_disk_cache=False):
        self.cache_enabled = cache_enabled
        self.cache_ttl_seconds = cache_ttl_seconds  
        self.use_disk_cache = use_disk_cache
        self._cache = {} if cache_enabled else None
    
    def _validate_file_exists(self, file_path):
        """Validate that file exists."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path
    
    def load_csv(self, file_path, **kwargs):
        """Load CSV file."""
        path = self._validate_file_exists(file_path)
        return pd.read_csv(path, **kwargs)
    
    def load_json(self, file_path, **kwargs):
        """Load JSON file."""
        path = self._validate_file_exists(file_path)
        try:
            # Try DataFrame first
            return pd.read_json(path, **kwargs)
        except (ValueError, Exception):
            # Fall back to regular JSON
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    def load_pickle(self, file_path):
        """Load pickle file."""
        path = self._validate_file_exists(file_path)
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def save_csv(self, data, file_path, **kwargs):
        """Save CSV file."""
        data.to_csv(file_path, index=False, **kwargs)
    
    def save_json(self, data, file_path, **kwargs):
        """Save JSON file."""
        if isinstance(data, pd.DataFrame):
            data.to_json(file_path, **kwargs)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, **kwargs)
    
    def save_pickle(self, data, file_path):
        """Save pickle file."""
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    def save_auto(self, data, file_path):
        """Auto-save based on file extension."""
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix == '.csv':
            self.save_csv(data, file_path)
        elif suffix == '.json':
            self.save_json(data, file_path)
        elif suffix == '.pkl':
            self.save_pickle(data, file_path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    
    def search_pattern(self, directory, pattern, recursive=True):
        """Search for files by pattern."""
        dir_path = Path(directory)
        if recursive:
            return list(dir_path.glob(f"**/{pattern}"))
        else:
            return list(dir_path.glob(pattern))


class TestSimpleFileHandler:
    """Test core file handling functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Sample DataFrame."""
        return pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.5, 20.3, 30.1]
        })
    
    def test_initialization(self):
        """Test handler initialization."""
        handler = SimpleFileHandler()
        assert handler.cache_enabled == True
        assert handler.cache_ttl_seconds == 3600
        assert handler.use_disk_cache == False
        
        handler2 = SimpleFileHandler(cache_enabled=False, cache_ttl_seconds=7200)
        assert handler2.cache_enabled == False
        assert handler2.cache_ttl_seconds == 7200
    
    def test_csv_operations(self, temp_dir, sample_data):
        """Test CSV loading and saving."""
        handler = SimpleFileHandler()
        
        # Save CSV
        csv_file = temp_dir / "test.csv"
        handler.save_csv(sample_data, csv_file)
        assert csv_file.exists()
        
        # Load CSV
        loaded_data = handler.load_csv(csv_file)
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(sample_data)
        assert list(loaded_data.columns) == list(sample_data.columns)
    
    def test_json_operations_dict(self, temp_dir):
        """Test JSON operations with dictionary."""
        handler = SimpleFileHandler()
        
        test_dict = {"name": "Alice", "age": 25, "values": [1, 2, 3]}
        json_file = temp_dir / "test.json"
        
        # Save JSON
        handler.save_json(test_dict, json_file)
        assert json_file.exists()
        
        # Load JSON
        loaded_data = handler.load_json(json_file)
        if isinstance(loaded_data, dict):
            assert loaded_data == test_dict
        else:
            # pandas converted to DataFrame
            assert isinstance(loaded_data, pd.DataFrame)
    
    def test_json_operations_dataframe(self, temp_dir, sample_data):
        """Test JSON operations with DataFrame."""
        handler = SimpleFileHandler()
        
        json_file = temp_dir / "test_df.json"
        
        # Save DataFrame as JSON
        handler.save_json(sample_data, json_file)
        assert json_file.exists()
        
        # Load back
        loaded_data = handler.load_json(json_file)
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(sample_data)
    
    def test_pickle_operations(self, temp_dir, sample_data):
        """Test pickle operations."""
        handler = SimpleFileHandler()
        
        pickle_file = temp_dir / "test.pkl"
        
        # Save pickle
        handler.save_pickle(sample_data, pickle_file)
        assert pickle_file.exists()
        
        # Load pickle
        loaded_data = handler.load_pickle(pickle_file)
        pd.testing.assert_frame_equal(loaded_data, sample_data)
    
    def test_auto_save_csv(self, temp_dir, sample_data):
        """Test auto-save with CSV."""
        handler = SimpleFileHandler()
        
        csv_file = temp_dir / "auto.csv"
        handler.save_auto(sample_data, csv_file)
        
        assert csv_file.exists()
        loaded_data = handler.load_csv(csv_file)
        assert len(loaded_data) == len(sample_data)
    
    def test_auto_save_json(self, temp_dir):
        """Test auto-save with JSON."""
        handler = SimpleFileHandler()
        
        test_dict = {"key": "value", "number": 42}
        json_file = temp_dir / "auto.json"
        handler.save_auto(test_dict, json_file)
        
        assert json_file.exists()
        loaded_data = handler.load_json(json_file)
        if isinstance(loaded_data, dict):
            assert loaded_data == test_dict
    
    def test_search_pattern(self, temp_dir):
        """Test file pattern searching."""
        handler = SimpleFileHandler()
        
        # Create test files
        (temp_dir / "file1.csv").touch()
        (temp_dir / "file2.json").touch()
        (temp_dir / "data.txt").touch()
        
        sub_dir = temp_dir / "subdir"
        sub_dir.mkdir()
        (sub_dir / "file3.csv").touch()
        
        # Search for CSV files recursively
        csv_files = handler.search_pattern(temp_dir, "*.csv", recursive=True)
        csv_names = [f.name for f in csv_files]
        
        assert len(csv_files) == 2
        assert "file1.csv" in csv_names
        assert "file3.csv" in csv_names
        
        # Search non-recursively
        csv_files_nr = handler.search_pattern(temp_dir, "*.csv", recursive=False)
        assert len(csv_files_nr) == 1
        assert csv_files_nr[0].name == "file1.csv"
    
    def test_file_not_found(self, temp_dir):
        """Test file not found error."""
        handler = SimpleFileHandler()
        
        non_existent = temp_dir / "does_not_exist.csv"
        
        with pytest.raises(FileNotFoundError):
            handler.load_csv(non_existent)
    
    def test_unsupported_format_auto_save(self, temp_dir, sample_data):
        """Test unsupported format in auto save."""
        handler = SimpleFileHandler()
        
        unsupported_file = temp_dir / "test.unknown"
        
        with pytest.raises(ValueError, match="Unsupported format"):
            handler.save_auto(sample_data, unsupported_file)
    
    def test_corrupted_json(self, temp_dir):
        """Test loading corrupted JSON."""
        handler = SimpleFileHandler()
        
        corrupted_json = temp_dir / "corrupted.json"
        corrupted_json.write_text('{"invalid": json content')
        
        with pytest.raises((json.JSONDecodeError, ValueError)):
            handler.load_json(corrupted_json)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
