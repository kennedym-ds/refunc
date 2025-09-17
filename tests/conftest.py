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


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add unit marker to all tests by default
        if not any(mark.name in ['integration', 'slow', 'gpu'] for mark in item.iter_markers()):
            item.add_marker(pytest.mark.unit)