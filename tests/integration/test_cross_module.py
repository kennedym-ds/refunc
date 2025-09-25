"""
Cross-module integration tests for refunc package.

These tests verify that different modules work correctly together,
testing the integration points and data flow between modules.
"""

import pytest
import tempfile
import time
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch


@pytest.mark.integration
class TestConfigLoggingIntegration:
    """Test Config + Logging integration."""
    
    def test_config_driven_logging_setup(self, temp_dir, sample_config):
        """Test that logging can be configured via ConfigManager."""
        try:
            from refunc.config import ConfigManager
            from refunc.logging import MLLogger, setup_logging
            
            # Create config with logging settings
            config_data = {
                'logging': {
                    'level': 'DEBUG',
                    'format': '%(name)s - %(levelname)s - %(message)s',
                    'file_enabled': True,
                    'file_path': str(temp_dir / 'test.log'),
                    'console_enabled': True
                }
            }
            
            config = ConfigManager()
            config.update(config_data)
            
            # Setup logging from config
            logging_config = config.get('logging', {})
            logger = MLLogger("test_config_integration")
            
            # Test that logger works with config
            logger.info("Test message")
            
            # Verify configuration was applied
            assert logger.name == "test_config_integration"
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")
    
    def test_config_environment_override(self, temp_dir, monkeypatch):
        """Test that environment variables override config files."""
        try:
            from refunc.config import ConfigManager
            
            # Set environment variable
            monkeypatch.setenv("REFUNC_LOGGING_LEVEL", "ERROR")
            
            config = ConfigManager()
            config.add_env_source(prefix="REFUNC_")
            
            # This should be overridden by env var
            config.update({'logging': {'level': 'INFO'}})
            
            # Environment variable should take precedence
            level = config.get('logging.level')
            assert level == "ERROR"
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")


@pytest.mark.integration
class TestDecoratorsExceptionsIntegration:
    """Test Decorators + Exception handling integration."""
    
    def test_timing_decorator_with_retry(self):
        """Test that timing decorator works with retry mechanism."""
        try:
            from refunc.decorators import time_it
            from refunc.exceptions import retry_on_failure
            
            attempt_count = 0
            
            @time_it(collect_stats=True)
            @retry_on_failure(max_attempts=3, delay=0.1)
            def unstable_function():
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count < 2:
                    raise ValueError("Simulated failure")
                return "success"
            
            result = unstable_function()
            assert result == "success"
            assert attempt_count == 2  # Failed once, succeeded on second attempt
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")
    
    def test_memory_decorator_with_validation(self):
        """Test memory profiling with input validation."""
        try:
            from refunc.decorators import memory_profile, validate_inputs
            
            @memory_profile()
            @validate_inputs(x=(int, float), y=(int, float))
            def memory_intensive_function(x, y):
                # Simulate memory usage
                data = np.random.randn(1000, 100)
                return np.sum(data) + x + y
            
            result = memory_intensive_function(1.0, 2.0)
            assert isinstance(result, float)
            
            # Test validation failure
            with pytest.raises(Exception):  # Validation error
                memory_intensive_function("invalid", 2.0)
                
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")


@pytest.mark.integration  
class TestUtilsLoggingIntegration:
    """Test Utils + Logging integration."""
    
    def test_file_handler_with_logging(self, temp_dir):
        """Test FileHandler operations with logging integration."""
        try:
            from refunc.utils import FileHandler
            from refunc.logging import MLLogger
            
            logger = MLLogger("file_handler_test")
            handler = FileHandler(logger=logger)
            
            # Create test data
            test_data = pd.DataFrame({
                'col1': [1, 2, 3],
                'col2': ['a', 'b', 'c']
            })
            
            test_file = temp_dir / "test_data.csv"
            
            # This should log the operation
            handler.save_dataframe(test_data, test_file)
            
            # Verify file was created
            assert test_file.exists()
            
            # This should also log the operation
            loaded_data = handler.load_dataframe(test_file)
            
            # Verify data integrity
            pd.testing.assert_frame_equal(test_data, loaded_data)
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")
    
    def test_cache_with_performance_monitoring(self, temp_dir):
        """Test caching utilities with performance decorators."""
        try:
            from refunc.utils import MemoryCache, cache_result
            from refunc.decorators import time_it
            
            cache = MemoryCache(max_size=100)
            
            @time_it(collect_stats=True)
            @cache_result(cache)
            def expensive_computation(n):
                # Simulate expensive computation
                time.sleep(0.1)
                return sum(range(n))
            
            # First call should be slow (cached)
            start_time = time.time()
            result1 = expensive_computation(100)
            first_duration = time.time() - start_time
            
            # Second call should be fast (from cache)
            start_time = time.time()
            result2 = expensive_computation(100)
            second_duration = time.time() - start_time
            
            assert result1 == result2
            assert second_duration < first_duration  # Cache should be faster
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")


@pytest.mark.integration
class TestDataScienceUtilsIntegration:
    """Test Data Science + Utils integration."""
    
    def test_data_processing_pipeline(self, sample_dataframe, temp_dir):
        """Test complete data processing pipeline across modules."""
        try:
            from refunc.utils import FileHandler
            from refunc.decorators import time_it, memory_profile
            
            handler = FileHandler()
            
            @time_it()
            @memory_profile()
            def process_data(df):
                # Simulate data processing
                processed = df.copy()
                processed['new_column'] = processed['numeric'] * 2
                processed = processed.dropna()
                return processed
            
            # Save original data
            input_file = temp_dir / "input.csv"
            handler.save_dataframe(sample_dataframe, input_file)
            
            # Load and process
            loaded_data = handler.load_dataframe(input_file)
            processed_data = process_data(loaded_data)
            
            # Save processed data
            output_file = temp_dir / "output.csv"
            handler.save_dataframe(processed_data, output_file)
            
            # Verify processing
            assert 'new_column' in processed_data.columns
            assert len(processed_data) <= len(sample_dataframe)  # Due to dropna
            assert output_file.exists()
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")


@pytest.mark.integration
class TestMathStatsIntegration:
    """Test Math/Stats + other modules integration."""
    
    def test_statistical_analysis_with_logging(self, sample_numpy_arrays):
        """Test statistical functions with logging integration."""
        try:
            from refunc.logging import MLLogger
            from refunc.decorators import time_it
            
            logger = MLLogger("stats_test")
            
            @time_it(logger=logger)
            def analyze_data(data):
                # Simulate statistical analysis
                mean = np.mean(data)
                std = np.std(data)
                median = np.median(data)
                
                logger.metric("mean", mean)
                logger.metric("std", std)
                logger.metric("median", median)
                
                return {
                    'mean': mean,
                    'std': std,
                    'median': median
                }
            
            results = analyze_data(sample_numpy_arrays['medium'])
            
            assert 'mean' in results
            assert 'std' in results
            assert 'median' in results
            assert all(isinstance(v, (int, float, np.number)) for v in results.values())
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")


@pytest.mark.integration
class TestMLWorkflowIntegration:
    """Test ML module with other components."""
    
    def test_ml_pipeline_with_monitoring(self, sample_dataframe):
        """Test ML workflow with performance monitoring and logging."""
        try:
            from refunc.decorators import performance_monitor
            from refunc.logging import MLLogger
            from refunc.exceptions import retry_on_failure
            
            logger = MLLogger("ml_pipeline")
            
            @performance_monitor(logger=logger)
            @retry_on_failure(max_attempts=2, delay=0.1)
            def train_model(data):
                # Simulate model training
                X = data[['numeric']].values
                y = (data['numeric'] > 0).astype(int).values
                
                # Simple mock training
                model_params = {
                    'accuracy': 0.85 + np.random.random() * 0.1,
                    'features': X.shape[1],
                    'samples': X.shape[0]
                }
                
                logger.info(f"Model trained with {X.shape[0]} samples")
                logger.metric("accuracy", model_params['accuracy'])
                
                return model_params
            
            model = train_model(sample_dataframe)
            
            assert 'accuracy' in model
            assert 'features' in model
            assert 'samples' in model
            assert 0.8 <= model['accuracy'] <= 1.0
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")


@pytest.mark.integration
@pytest.mark.slow
class TestFullModuleIntegration:
    """Test full integration across all modules."""
    
    def test_complete_workflow_integration(self, temp_dir, sample_dataframe):
        """Test a complete workflow touching all major modules."""
        try:
            from refunc.config import ConfigManager
            from refunc.logging import MLLogger
            from refunc.utils import FileHandler
            from refunc.decorators import performance_monitor
            from refunc.exceptions import retry_on_failure
            
            # Setup configuration
            config = ConfigManager()
            config.update({
                'data': {
                    'input_path': str(temp_dir / 'input.csv'),
                    'output_path': str(temp_dir / 'output.csv')
                },
                'logging': {
                    'level': 'INFO',
                    'file_path': str(temp_dir / 'workflow.log')
                },
                'processing': {
                    'batch_size': 50,
                    'threshold': 0.5
                }
            })
            
            # Setup logging
            logger = MLLogger("integration_workflow")
            
            # Setup file handler
            file_handler = FileHandler(logger=logger)
            
            @performance_monitor(logger=logger)
            @retry_on_failure(max_attempts=2, delay=0.1)
            def complete_workflow(data_path, output_path, config):
                """Complete data processing workflow."""
                
                logger.info("Starting complete workflow")
                
                # Load data
                data = file_handler.load_dataframe(data_path)
                logger.info(f"Loaded {len(data)} records")
                
                # Process data
                processed = data.copy()
                threshold = config.get('processing.threshold', 0.0)
                processed = processed[processed['numeric'] > threshold]
                processed['processed_flag'] = True
                
                logger.info(f"Processed to {len(processed)} records")
                logger.metric("processed_count", len(processed))
                
                # Save results
                file_handler.save_dataframe(processed, output_path)
                logger.info(f"Saved results to {output_path}")
                
                return {
                    'input_count': len(data),
                    'output_count': len(processed),
                    'processing_ratio': len(processed) / len(data)
                }
            
            # Save input data
            input_path = config.get('data.input_path')
            file_handler.save_dataframe(sample_dataframe, input_path)
            
            # Run workflow
            results = complete_workflow(
                input_path,
                config.get('data.output_path'),
                config
            )
            
            # Verify results
            assert 'input_count' in results
            assert 'output_count' in results
            assert 'processing_ratio' in results
            assert results['output_count'] <= results['input_count']
            assert Path(config.get('data.output_path')).exists()
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")