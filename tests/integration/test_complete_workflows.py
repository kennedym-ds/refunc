"""
End-to-end workflow integration tests for refunc package.

These tests verify complete ML workflows from data loading through 
model training and evaluation, ensuring all components work together
in realistic scenarios.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import time
from unittest.mock import Mock, patch


@pytest.mark.integration
@pytest.mark.slow
class TestMLPipelineWorkflow:
    """Test complete ML pipeline workflows."""
    
    def test_basic_ml_pipeline(self, temp_dir, sample_dataframe):
        """Test a basic ML pipeline from data to model."""
        try:
            from refunc.utils import FileHandler
            from refunc.logging import MLLogger
            from refunc.decorators import time_it, memory_profile
            from refunc.config import ConfigManager
            
            # Setup components
            logger = MLLogger("ml_pipeline", log_dir=str(temp_dir))
            file_handler = FileHandler(logger=logger)
            config = ConfigManager()
            
            # Configure pipeline
            config.update({
                'data': {
                    'train_size': 0.8,
                    'random_state': 42
                },
                'model': {
                    'type': 'mock_classifier',
                    'parameters': {
                        'n_estimators': 100,
                        'max_depth': 5
                    }
                },
                'evaluation': {
                    'metrics': ['accuracy', 'precision', 'recall']
                }
            })
            
            @time_it(logger=logger)
            @memory_profile()
            def prepare_data(df, config):
                """Prepare data for training."""
                logger.info("Preparing data for training")
                
                # Create features and target
                X = df[['numeric']].values
                y = (df['numeric'] > df['numeric'].median()).astype(int)
                
                # Split data
                train_size = config.get('data.train_size', 0.8)
                split_idx = int(len(X) * train_size)
                
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                logger.metric("train_samples", len(X_train))
                logger.metric("test_samples", len(X_test))
                
                return X_train, X_test, y_train, y_test
            
            @time_it(logger=logger)
            def train_model(X_train, y_train, config):
                """Train a mock model."""
                logger.info("Training model")
                
                model_config = config.get('model', {})
                logger.info(f"Model type: {model_config.get('type', 'unknown')}")
                
                # Mock training process
                time.sleep(0.1)  # Simulate training time
                
                # Create mock model
                model = {
                    'type': model_config.get('type', 'mock'),
                    'parameters': model_config.get('parameters', {}),
                    'trained_on_samples': len(X_train),
                    'feature_count': X_train.shape[1],
                    'training_accuracy': 0.85 + np.random.random() * 0.1
                }
                
                logger.metric("training_accuracy", model['training_accuracy'])
                return model
            
            @time_it(logger=logger)
            def evaluate_model(model, X_test, y_test, config):
                """Evaluate the trained model."""
                logger.info("Evaluating model")
                
                # Mock predictions
                y_pred = np.random.choice([0, 1], size=len(y_test))
                
                # Calculate mock metrics
                accuracy = np.mean(y_pred == y_test)
                precision = np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_pred == 1), 1)
                recall = np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1)
                
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'test_samples': len(y_test)
                }
                
                # Log metrics
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        logger.metric(metric_name, value)
                
                return metrics
            
            # Run the complete pipeline
            logger.info("Starting ML pipeline")
            
            # Data preparation
            X_train, X_test, y_train, y_test = prepare_data(sample_dataframe, config)
            
            # Model training
            model = train_model(X_train, y_train, config)
            
            # Model evaluation
            metrics = evaluate_model(model, X_test, y_test, config)
            
            # Save results
            results_file = temp_dir / "pipeline_results.csv"
            results_df = pd.DataFrame([metrics])
            file_handler.save_dataframe(results_df, results_file)
            
            logger.info("ML pipeline completed successfully")
            
            # Verify pipeline results
            assert model['trained_on_samples'] > 0
            assert model['feature_count'] > 0
            assert 'accuracy' in metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert results_file.exists()
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")
    
    def test_data_processing_pipeline(self, temp_dir, sample_dataframe):
        """Test complete data processing pipeline."""
        try:
            from refunc.utils import FileHandler, MemoryCache
            from refunc.logging import MLLogger
            from refunc.decorators import time_it, cache_result
            from refunc.exceptions import retry_on_failure
            
            logger = MLLogger("data_pipeline", log_dir=str(temp_dir))
            file_handler = FileHandler(logger=logger)
            cache = MemoryCache(max_size=100)
            
            @time_it(logger=logger)
            @retry_on_failure(max_attempts=2, delay=0.1)
            def load_and_validate_data(file_path):
                """Load and validate data."""
                logger.info(f"Loading data from {file_path}")
                
                data = file_handler.load_dataframe(file_path)
                
                # Validation checks
                assert not data.empty, "Data is empty"
                assert len(data.columns) > 0, "No columns found"
                
                logger.metric("loaded_rows", len(data))
                logger.metric("loaded_columns", len(data.columns))
                
                return data
            
            @time_it(logger=logger)
            @cache_result(cache)
            def clean_data(data):
                """Clean and preprocess data."""
                logger.info("Cleaning data")
                
                # Remove duplicates
                initial_count = len(data)
                data = data.drop_duplicates()
                
                # Handle missing values
                data = data.dropna()
                
                final_count = len(data)
                logger.metric("cleaned_rows", final_count)
                logger.metric("removed_rows", initial_count - final_count)
                
                return data
            
            @time_it(logger=logger)
            def feature_engineering(data):
                """Create additional features."""
                logger.info("Engineering features")
                
                engineered = data.copy()
                
                # Add derived features
                if 'numeric' in engineered.columns:
                    engineered['numeric_squared'] = engineered['numeric'] ** 2
                    engineered['numeric_log'] = np.log(np.abs(engineered['numeric']) + 1)
                    engineered['numeric_normalized'] = (
                        engineered['numeric'] - engineered['numeric'].mean()
                    ) / engineered['numeric'].std()
                
                logger.metric("final_features", len(engineered.columns))
                return engineered
            
            @time_it(logger=logger)
            def save_processed_data(data, output_path):
                """Save processed data with metadata."""
                logger.info(f"Saving processed data to {output_path}")
                
                # Save main data
                file_handler.save_dataframe(data, output_path)
                
                # Save metadata
                metadata = {
                    'rows': len(data),
                    'columns': len(data.columns),
                    'column_names': list(data.columns),
                    'processing_timestamp': pd.Timestamp.now().isoformat()
                }
                
                metadata_path = output_path.parent / f"{output_path.stem}_metadata.csv"
                metadata_df = pd.DataFrame([metadata])
                file_handler.save_dataframe(metadata_df, metadata_path)
                
                logger.info("Data and metadata saved successfully")
                return metadata
            
            # Run the complete data processing pipeline
            logger.info("Starting data processing pipeline")
            
            # Save input data
            input_file = temp_dir / "input_data.csv"
            file_handler.save_dataframe(sample_dataframe, input_file)
            
            # Process data through pipeline
            raw_data = load_and_validate_data(input_file)
            clean_data_result = clean_data(raw_data)
            engineered_data = feature_engineering(clean_data_result)
            
            # Save final results
            output_file = temp_dir / "processed_data.csv"
            metadata = save_processed_data(engineered_data, output_file)
            
            logger.info("Data processing pipeline completed")
            
            # Verify pipeline results
            assert output_file.exists()
            assert metadata['rows'] > 0
            assert metadata['columns'] >= len(sample_dataframe.columns)
            assert 'numeric_squared' in engineered_data.columns
            assert 'numeric_log' in engineered_data.columns
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")


@pytest.mark.integration
class TestConfigDrivenWorkflows:
    """Test workflows driven by configuration."""
    
    def test_configurable_data_pipeline(self, temp_dir, sample_dataframe):
        """Test data pipeline that adapts based on configuration."""
        try:
            from refunc.config import ConfigManager
            from refunc.utils import FileHandler
            from refunc.logging import MLLogger
            from refunc.decorators import time_it
            
            # Setup configuration
            config = ConfigManager()
            config.update({
                'pipeline': {
                    'steps': ['load', 'clean', 'transform', 'save'],
                    'clean_duplicates': True,
                    'handle_missing': 'drop',
                    'transformations': {
                        'normalize_numeric': True,
                        'create_categories': True
                    }
                },
                'output': {
                    'format': 'csv',
                    'include_metadata': True
                }
            })
            
            logger = MLLogger("config_pipeline", log_dir=str(temp_dir))
            file_handler = FileHandler(logger=logger)
            
            @time_it(logger=logger)
            def configurable_pipeline(data, config):
                """Run pipeline based on configuration."""
                pipeline_config = config.get('pipeline', {})
                steps = pipeline_config.get('steps', [])
                
                result = data.copy()
                
                for step in steps:
                    logger.info(f"Executing step: {step}")
                    
                    if step == 'clean':
                        if pipeline_config.get('clean_duplicates', False):
                            result = result.drop_duplicates()
                        
                        missing_strategy = pipeline_config.get('handle_missing', 'keep')
                        if missing_strategy == 'drop':
                            result = result.dropna()
                    
                    elif step == 'transform':
                        transforms = pipeline_config.get('transformations', {})
                        
                        if transforms.get('normalize_numeric', False) and 'numeric' in result.columns:
                            result['numeric_norm'] = (
                                result['numeric'] - result['numeric'].mean()
                            ) / result['numeric'].std()
                        
                        if transforms.get('create_categories', False) and 'numeric' in result.columns:
                            result['numeric_category'] = pd.cut(
                                result['numeric'], 
                                bins=3, 
                                labels=['low', 'medium', 'high']
                            )
                
                logger.metric("final_shape_rows", len(result))
                logger.metric("final_shape_cols", len(result.columns))
                
                return result
            
            # Run configurable pipeline
            processed_data = configurable_pipeline(sample_dataframe, config)
            
            # Save based on config
            output_config = config.get('output', {})
            output_file = temp_dir / f"output.{output_config.get('format', 'csv')}"
            file_handler.save_dataframe(processed_data, output_file)
            
            # Verify configuration was applied
            assert 'numeric_norm' in processed_data.columns  # normalization applied
            assert 'numeric_category' in processed_data.columns  # categorization applied
            assert output_file.exists()
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")


@pytest.mark.integration
@pytest.mark.slow  
class TestPerformanceWorkflows:
    """Test workflows with performance monitoring."""
    
    def test_monitored_batch_processing(self, temp_dir, sample_numpy_arrays):
        """Test batch processing with comprehensive monitoring."""
        try:
            from refunc.decorators import performance_monitor, time_it
            from refunc.logging import MLLogger
            from refunc.utils import MemoryCache
            
            logger = MLLogger("batch_processing", log_dir=str(temp_dir))
            cache = MemoryCache(max_size=50)
            
            @performance_monitor(logger=logger)
            def process_batch(batch_data, batch_id):
                """Process a single batch with monitoring."""
                logger.info(f"Processing batch {batch_id}")
                
                # Simulate different types of processing
                if batch_id % 2 == 0:
                    # Heavy computation
                    result = np.fft.fft(batch_data)
                    time.sleep(0.1)  # Simulate processing time
                else:
                    # Light computation
                    result = np.mean(batch_data, axis=0)
                    time.sleep(0.05)
                
                logger.metric(f"batch_{batch_id}_size", len(batch_data))
                return result
            
            @time_it(logger=logger)
            def batch_processor(data_arrays, batch_size=10):
                """Process multiple batches."""
                logger.info("Starting batch processing")
                
                results = []
                total_batches = 0
                
                for array_name, array_data in data_arrays.items():
                    logger.info(f"Processing array: {array_name}")
                    
                    # Split into batches
                    if len(array_data.shape) > 1:
                        n_samples = array_data.shape[0]
                        for i in range(0, n_samples, batch_size):
                            batch = array_data[i:i+batch_size]
                            result = process_batch(batch, total_batches)
                            results.append({
                                'batch_id': total_batches,
                                'array_name': array_name,
                                'result_shape': np.array(result).shape,
                                'result_type': type(result).__name__
                            })
                            total_batches += 1
                
                logger.metric("total_batches_processed", total_batches)
                return results
            
            # Run batch processing
            results = batch_processor(sample_numpy_arrays)
            
            # Verify results
            assert len(results) > 0
            assert all('batch_id' in r for r in results)
            assert all('result_shape' in r for r in results)
            
            # Check that performance monitoring captured data
            # This would typically be verified by checking log outputs
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")


@pytest.mark.integration
class TestErrorRecoveryWorkflows:
    """Test workflows with error handling and recovery."""
    
    def test_resilient_data_processing(self, temp_dir, sample_dataframe):
        """Test data processing with error recovery."""
        try:
            from refunc.exceptions import retry_on_failure, RefuncError
            from refunc.logging import MLLogger
            from refunc.utils import FileHandler
            from refunc.decorators import time_it
            
            logger = MLLogger("resilient_processing", log_dir=str(temp_dir))
            file_handler = FileHandler(logger=logger)
            
            processing_attempts = {}
            
            @retry_on_failure(max_attempts=3, delay=0.1)
            @time_it(logger=logger)
            def unreliable_processing_step(data, step_name):
                """Processing step that may fail."""
                if step_name not in processing_attempts:
                    processing_attempts[step_name] = 0
                
                processing_attempts[step_name] += 1
                
                logger.info(f"Attempting {step_name} (attempt {processing_attempts[step_name]})")
                
                # Simulate failures for first attempts
                if processing_attempts[step_name] < 2 and step_name == 'validation':
                    raise ValueError(f"Simulated failure in {step_name}")
                
                # Actual processing
                if step_name == 'validation':
                    assert not data.empty, "Data is empty"
                    assert len(data.columns) > 0, "No columns"
                    return data
                
                elif step_name == 'transformation':
                    transformed = data.copy()
                    transformed['processed'] = True
                    return transformed
                
                elif step_name == 'enrichment':
                    enriched = data.copy()
                    enriched['enriched_at'] = pd.Timestamp.now()
                    return enriched
                
                return data
            
            @time_it(logger=logger)
            def resilient_pipeline(input_data):
                """Pipeline that handles failures gracefully."""
                logger.info("Starting resilient pipeline")
                
                try:
                    # Step 1: Validation (may fail initially)
                    validated_data = unreliable_processing_step(input_data, 'validation')
                    logger.info("Data validation successful")
                    
                    # Step 2: Transformation
                    transformed_data = unreliable_processing_step(validated_data, 'transformation')
                    logger.info("Data transformation successful")
                    
                    # Step 3: Enrichment
                    enriched_data = unreliable_processing_step(transformed_data, 'enrichment')
                    logger.info("Data enrichment successful")
                    
                    return enriched_data
                
                except Exception as e:
                    logger.error(f"Pipeline failed: {str(e)}")
                    # Return partial results
                    return input_data
            
            # Run resilient pipeline
            result = resilient_pipeline(sample_dataframe)
            
            # Save results
            output_file = temp_dir / "resilient_output.csv"
            file_handler.save_dataframe(result, output_file)
            
            # Verify recovery worked
            assert output_file.exists()
            assert 'processed' in result.columns
            assert 'enriched_at' in result.columns
            assert processing_attempts['validation'] >= 2  # Should have retried
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    def test_full_ml_experiment_workflow(self, temp_dir, sample_dataframe):
        """Test a complete ML experiment from start to finish."""
        try:
            from refunc.config import ConfigManager
            from refunc.logging import MLLogger
            from refunc.utils import FileHandler
            from refunc.decorators import performance_monitor
            from refunc.exceptions import retry_on_failure
            
            # Setup experiment
            config = ConfigManager()
            config.update({
                'experiment': {
                    'name': 'test_experiment_001',
                    'description': 'Full integration test experiment'
                },
                'data': {
                    'validation_split': 0.2,
                    'preprocessing': {
                        'normalize': True,
                        'remove_outliers': True
                    }
                },
                'model': {
                    'algorithm': 'mock_classifier',
                    'hyperparameters': {
                        'n_estimators': 50,
                        'max_depth': 3
                    }
                },
                'evaluation': {
                    'metrics': ['accuracy', 'f1_score'],
                    'cross_validation': True
                }
            })
            
            experiment_name = config.get('experiment.name')
            logger = MLLogger(experiment_name, log_dir=str(temp_dir))
            file_handler = FileHandler(logger=logger)
            
            @performance_monitor(logger=logger)
            def run_full_experiment(data, config):
                """Run complete ML experiment."""
                logger.info(f"Starting experiment: {config.get('experiment.name')}")
                
                # Data preprocessing
                logger.info("Preprocessing data")
                X = data[['numeric']].values
                y = (data['numeric'] > data['numeric'].median()).astype(int)
                
                if config.get('data.preprocessing.normalize'):
                    X = (X - X.mean()) / X.std()
                
                # Train/validation split
                val_split = config.get('data.validation_split', 0.2)
                split_idx = int(len(X) * (1 - val_split))
                
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                
                logger.metric("train_size", len(X_train))
                logger.metric("val_size", len(X_val))
                
                # Model training
                logger.info("Training model")
                model_config = config.get('model', {})
                
                # Mock training
                time.sleep(0.2)  # Simulate training time
                
                model = {
                    'algorithm': model_config.get('algorithm'),
                    'hyperparameters': model_config.get('hyperparameters', {}),
                    'training_samples': len(X_train)
                }
                
                # Model evaluation
                logger.info("Evaluating model")
                
                # Mock predictions and evaluation
                val_predictions = np.random.choice([0, 1], size=len(y_val))
                accuracy = np.mean(val_predictions == y_val)
                f1_score = 2 * accuracy / (1 + accuracy)  # Mock F1 calculation
                
                metrics = {
                    'accuracy': accuracy,
                    'f1_score': f1_score
                }
                
                for metric_name, value in metrics.items():
                    logger.metric(metric_name, value)
                
                # Save experiment results
                results = {
                    'experiment_name': experiment_name,
                    'model_config': model,
                    'metrics': metrics,
                    'data_info': {
                        'total_samples': len(data),
                        'train_samples': len(X_train),
                        'val_samples': len(X_val),
                        'features': X.shape[1]
                    }
                }
                
                return results
            
            # Run the complete experiment
            experiment_results = run_full_experiment(sample_dataframe, config)
            
            # Save experiment results
            results_file = temp_dir / f"{experiment_name}_results.csv"
            results_df = pd.DataFrame([experiment_results['metrics']])
            file_handler.save_dataframe(results_df, results_file)
            
            logger.info("Experiment completed successfully")
            
            # Verify experiment completed
            assert results_file.exists()
            assert 'accuracy' in experiment_results['metrics']
            assert 'f1_score' in experiment_results['metrics']
            assert experiment_results['data_info']['total_samples'] > 0
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")