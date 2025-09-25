"""
Shared pytest fixtures and configuration for the refunc test suite.
"""

import tempfile
import shutil
import os
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_file(temp_dir: Path) -> Path:
    """Create a temporary file for testing."""
    file_path = temp_dir / "test_file.txt"
    file_path.write_text("test content")
    return file_path


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'numeric': np.random.randn(100),
        'categorical': np.random.choice(['A', 'B', 'C'], 100),
        'missing': [1, 2, None, 4, 5] * 20,
        'text': ['text_' + str(i) for i in range(100)]
    })


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Create a sample configuration dictionary for testing."""
    return {
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'handlers': ['console', 'file']
        },
        'data': {
            'input_path': '/path/to/input',
            'output_path': '/path/to/output',
            'batch_size': 32
        },
        'model': {
            'type': 'random_forest',
            'parameters': {
                'n_estimators': 100,
                'max_depth': 10
            }
        }
    }


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return Mock()


@pytest.fixture
def mock_file_system(temp_dir: Path):
    """Create a mock file system structure for testing."""
    # Create directories
    (temp_dir / "data").mkdir()
    (temp_dir / "logs").mkdir()
    (temp_dir / "models").mkdir()
    
    # Create files
    (temp_dir / "data" / "train.csv").write_text("col1,col2\n1,2\n3,4")
    (temp_dir / "data" / "test.csv").write_text("col1,col2\n5,6\n7,8")
    (temp_dir / "config.yaml").write_text("key: value\nnested:\n  key: value")
    
    return temp_dir


@pytest.fixture
def sample_numpy_arrays():
    """Create sample numpy arrays for testing."""
    np.random.seed(42)
    return {
        'small': np.random.randn(10),
        'medium': np.random.randn(100, 5),
        'large': np.random.randn(1000, 10),
        'sparse': np.zeros((100, 100)),
        'categorical': np.random.choice([0, 1, 2], 100)
    }


@pytest.fixture
def mock_experiment_data():
    """Create mock experiment data for testing."""
    return {
        'experiment_id': 'test_exp_001',
        'metrics': {
            'accuracy': 0.95,
            'precision': 0.92,
            'recall': 0.88,
            'f1_score': 0.90
        },
        'parameters': {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100
        },
        'artifacts': {
            'model_path': '/path/to/model.pkl',
            'plot_path': '/path/to/plot.png'
        }
    }


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables and global state before each test."""
    # Store original environment
    original_env = os.environ.copy()
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def capture_logs():
    """Capture log output for testing."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger('refunc')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    yield log_capture
    
    logger.removeHandler(handler)


# Additional fixtures for integration tests
@pytest.fixture
def integration_config():
    """Provide configuration for integration tests."""
    return {
        'test_mode': True,
        'integration_test': True,
        'logging': {
            'level': 'DEBUG',
            'console_enabled': True,
            'file_enabled': True
        },
        'performance': {
            'enable_monitoring': True,
            'collect_metrics': True
        },
        'data': {
            'batch_size': 10,  # Small batch size for testing
            'cache_enabled': True
        }
    }


@pytest.fixture
def large_sample_dataframe():
    """Create a larger DataFrame for performance testing."""
    np.random.seed(42)
    size = 10000
    return pd.DataFrame({
        'numeric': np.random.randn(size),
        'categorical': np.random.choice(['A', 'B', 'C', 'D', 'E'], size),
        'missing': np.random.choice([1, 2, 3, None], size),
        'text': [f'text_{i%1000}' for i in range(size)],
        'binary': np.random.choice([0, 1], size),
        'float_col': np.random.uniform(0, 100, size),
        'id': range(size)
    })


@pytest.fixture
def performance_baseline():
    """Provide performance baseline expectations for tests."""
    return {
        'file_operations': {
            'save_1000_rows_max_seconds': 1.0,
            'load_1000_rows_max_seconds': 0.5
        },
        'memory_usage': {
            'max_growth_mb': 100,
            'leak_threshold_mb': 10
        },
        'decorator_overhead': {
            'max_overhead_ratio': 2.0,
            'acceptable_overhead_ratio': 1.5
        }
    }


@pytest.fixture
def mock_ml_model():
    """Provide a mock ML model for testing."""
    class MockMLModel:
        def __init__(self):
            self.is_trained = False
            self.predictions_made = 0
            
        def train(self, X, y):
            self.is_trained = True
            return {'accuracy': 0.85, 'loss': 0.15}
            
        def predict(self, X):
            if not self.is_trained:
                raise ValueError("Model not trained")
            self.predictions_made += len(X)
            return np.random.choice([0, 1], size=len(X))
            
        def predict_proba(self, X):
            if not self.is_trained:
                raise ValueError("Model not trained")
            probas = np.random.uniform(0.1, 0.9, size=(len(X), 2))
            # Normalize to sum to 1
            probas = probas / probas.sum(axis=1, keepdims=True)
            return probas
    
    return MockMLModel()


# Markers for test categorization
pytest_plugins = []

# Configuration for different test environments
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests for performance benchmarking"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add integration marker to tests in integration directory
        if 'integration' in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add unit marker to all other tests by default
        elif not any(mark.name in ['integration', 'slow', 'gpu', 'benchmark'] for mark in item.iter_markers()):
            item.add_marker(pytest.mark.unit)