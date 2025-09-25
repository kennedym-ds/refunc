"""
Real-world scenario integration tests for refunc package.

These tests simulate common usage patterns and realistic scenarios
that users would encounter when using refunc in production environments.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock


@pytest.mark.integration
class TestDataScienceWorkflows:
    """Test common data science workflows."""
    
    def test_exploratory_data_analysis_workflow(self, temp_dir, sample_dataframe):
        """Test typical EDA workflow with logging and performance monitoring."""
        try:
            from refunc.logging import MLLogger
            from refunc.utils import FileHandler
            from refunc.decorators import time_it, memory_profile
            
            logger = MLLogger("eda_workflow", log_dir=str(temp_dir))
            file_handler = FileHandler(logger=logger)
            
            @time_it(logger=logger)
            @memory_profile()
            def data_quality_check(df):
                """Comprehensive data quality assessment."""
                logger.info("Starting data quality assessment")
                
                quality_report = {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'missing_values': df.isnull().sum().to_dict(),
                    'duplicate_rows': df.duplicated().sum(),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
                }
                
                # Log key metrics
                logger.metric("total_rows", quality_report['total_rows'])
                logger.metric("missing_percentage", df.isnull().sum().sum() / df.size)
                logger.metric("duplicate_percentage", quality_report['duplicate_rows'] / len(df))
                
                return quality_report
            
            @time_it(logger=logger)
            def statistical_summary(df):
                """Generate statistical summaries with logging."""
                logger.info("Generating statistical summaries")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(exclude=[np.number]).columns
                
                summary = {
                    'numeric_summary': df[numeric_cols].describe().to_dict(),
                    'categorical_summary': {}
                }
                
                for col in categorical_cols:
                    summary['categorical_summary'][col] = {
                        'unique_values': df[col].nunique(),
                        'most_frequent': df[col].mode().iloc[0] if not df[col].empty else None,
                        'frequency_distribution': df[col].value_counts().head().to_dict()
                    }
                
                logger.metric("numeric_columns", len(numeric_cols))
                logger.metric("categorical_columns", len(categorical_cols))
                
                return summary
            
            @time_it(logger=logger)
            def correlation_analysis(df):
                """Perform correlation analysis."""
                logger.info("Performing correlation analysis")
                
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    correlation_matrix = numeric_df.corr()
                    
                    # Find high correlations
                    high_corr_pairs = []
                    for i in range(len(correlation_matrix.columns)):
                        for j in range(i+1, len(correlation_matrix.columns)):
                            corr_value = correlation_matrix.iloc[i, j]
                            if abs(corr_value) > 0.7:  # High correlation threshold
                                high_corr_pairs.append({
                                    'var1': correlation_matrix.columns[i],
                                    'var2': correlation_matrix.columns[j],
                                    'correlation': corr_value
                                })
                    
                    logger.metric("high_correlations_found", len(high_corr_pairs))
                    return {'correlation_matrix': correlation_matrix.to_dict(), 'high_correlations': high_corr_pairs}
                
                return {'correlation_matrix': {}, 'high_correlations': []}
            
            # Run EDA workflow
            logger.info("Starting comprehensive EDA workflow")
            
            # Data quality assessment
            quality_report = data_quality_check(sample_dataframe)
            
            # Statistical summaries
            stats_summary = statistical_summary(sample_dataframe)
            
            # Correlation analysis
            correlation_results = correlation_analysis(sample_dataframe)
            
            # Compile comprehensive report
            eda_report = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'data_quality': quality_report,
                'statistical_summary': stats_summary,
                'correlation_analysis': correlation_results
            }
            
            # Save EDA report
            report_file = temp_dir / "eda_report.json"
            with open(report_file, 'w') as f:
                json.dump(eda_report, f, indent=2, default=str)
            
            logger.info("EDA workflow completed successfully")
            
            # Verify EDA results
            assert report_file.exists()
            assert quality_report['total_rows'] > 0
            assert quality_report['total_columns'] > 0
            assert 'numeric_summary' in stats_summary
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")
    
    def test_model_training_pipeline(self, temp_dir, sample_dataframe):
        """Test realistic model training pipeline with monitoring."""
        try:
            from refunc.logging import MLLogger
            from refunc.decorators import performance_monitor, retry_on_failure
            from refunc.utils import FileHandler
            from refunc.config import ConfigManager
            
            # Setup experiment configuration
            config = ConfigManager()
            config.update({
                'experiment': {
                    'name': 'real_world_model_training',
                    'random_seed': 42
                },
                'data': {
                    'train_ratio': 0.7,
                    'val_ratio': 0.2,
                    'test_ratio': 0.1
                },
                'model': {
                    'algorithm': 'gradient_boosting',
                    'hyperparameters': {
                        'n_estimators': 100,
                        'learning_rate': 0.1,
                        'max_depth': 6
                    }
                },
                'training': {
                    'early_stopping': True,
                    'patience': 10,
                    'monitor_metric': 'val_accuracy'
                }
            })
            
            logger = MLLogger("model_training", log_dir=str(temp_dir))
            file_handler = FileHandler(logger=logger)
            
            @performance_monitor(logger=logger)
            def prepare_training_data(df, config):
                """Prepare data for model training."""
                logger.info("Preparing training data")
                
                # Create features and target
                X = df[['numeric']].values
                y = (df['numeric'] > df['numeric'].median()).astype(int)
                
                # Data splits
                data_config = config.get('data', {})
                train_ratio = data_config.get('train_ratio', 0.7)
                val_ratio = data_config.get('val_ratio', 0.2)
                
                n_samples = len(X)
                train_end = int(n_samples * train_ratio)
                val_end = int(n_samples * (train_ratio + val_ratio))
                
                X_train = X[:train_end]
                y_train = y[:train_end]
                X_val = X[train_end:val_end]
                y_val = y[train_end:val_end]
                X_test = X[val_end:]
                y_test = y[val_end:]
                
                # Log data split info
                logger.metric("train_samples", len(X_train))
                logger.metric("val_samples", len(X_val))
                logger.metric("test_samples", len(X_test))
                logger.metric("feature_count", X.shape[1])
                
                return (X_train, y_train), (X_val, y_val), (X_test, y_test)
            
            @retry_on_failure(max_attempts=3, delay=0.1)
            @performance_monitor(logger=logger)
            def train_model_with_validation(train_data, val_data, config):
                """Train model with validation monitoring."""
                X_train, y_train = train_data
                X_val, y_val = val_data
                
                logger.info("Starting model training")
                
                model_config = config.get('model', {})
                training_config = config.get('training', {})
                
                # Mock training process with validation
                training_history = {
                    'train_accuracy': [],
                    'val_accuracy': [],
                    'train_loss': [],
                    'val_loss': []
                }
                
                best_val_accuracy = 0
                patience_counter = 0
                max_patience = training_config.get('patience', 10)
                
                # Simulate training epochs
                for epoch in range(50):  # Max epochs
                    # Mock training metrics (would be real training in practice)
                    train_acc = 0.6 + 0.3 * (1 - np.exp(-epoch / 10)) + np.random.normal(0, 0.02)
                    val_acc = 0.55 + 0.25 * (1 - np.exp(-epoch / 10)) + np.random.normal(0, 0.03)
                    train_loss = 0.7 * np.exp(-epoch / 15) + np.random.normal(0, 0.01)
                    val_loss = 0.8 * np.exp(-epoch / 12) + np.random.normal(0, 0.02)
                    
                    training_history['train_accuracy'].append(train_acc)
                    training_history['val_accuracy'].append(val_acc)
                    training_history['train_loss'].append(train_loss)
                    training_history['val_loss'].append(val_loss)
                    
                    # Log metrics every 5 epochs
                    if epoch % 5 == 0:
                        logger.metric(f"epoch_{epoch}_train_acc", train_acc)
                        logger.metric(f"epoch_{epoch}_val_acc", val_acc)
                    
                    # Early stopping check
                    if val_acc > best_val_accuracy:
                        best_val_accuracy = val_acc
                        patience_counter = 0
                        logger.info(f"New best validation accuracy: {val_acc:.4f}")
                    else:
                        patience_counter += 1
                    
                    if training_config.get('early_stopping', False) and patience_counter >= max_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                
                # Create final model object
                final_model = {
                    'algorithm': model_config.get('algorithm'),
                    'hyperparameters': model_config.get('hyperparameters', {}),
                    'training_history': training_history,
                    'best_val_accuracy': best_val_accuracy,
                    'epochs_trained': len(training_history['train_accuracy']),
                    'early_stopped': patience_counter >= max_patience
                }
                
                logger.metric("final_val_accuracy", best_val_accuracy)
                logger.metric("epochs_trained", final_model['epochs_trained'])
                
                return final_model
            
            @performance_monitor(logger=logger)
            def evaluate_final_model(model, test_data):
                """Evaluate final model on test set."""
                X_test, y_test = test_data
                
                logger.info("Evaluating final model")
                
                # Mock evaluation (would use real model predictions)
                # Simulate test accuracy based on validation performance
                base_accuracy = model['best_val_accuracy']
                test_accuracy = base_accuracy + np.random.normal(0, 0.02)
                test_accuracy = np.clip(test_accuracy, 0, 1)
                
                # Mock additional metrics
                test_precision = test_accuracy * (0.9 + np.random.normal(0, 0.05))
                test_recall = test_accuracy * (0.95 + np.random.normal(0, 0.03))
                test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
                
                evaluation_results = {
                    'test_accuracy': test_accuracy,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'test_f1_score': test_f1,
                    'test_samples': len(X_test)
                }
                
                # Log evaluation metrics
                for metric, value in evaluation_results.items():
                    if isinstance(value, (int, float)):
                        logger.metric(metric, value)
                
                return evaluation_results
            
            # Run complete training pipeline
            logger.info("Starting real-world model training pipeline")
            
            # Data preparation
            train_data, val_data, test_data = prepare_training_data(sample_dataframe, config)
            
            # Model training
            trained_model = train_model_with_validation(train_data, val_data, config)
            
            # Final evaluation
            evaluation_results = evaluate_final_model(trained_model, test_data)
            
            # Save training artifacts
            artifacts = {
                'model_config': trained_model,
                'evaluation_results': evaluation_results,
                'experiment_config': dict(config._data)  # Save configuration
            }
            
            artifacts_file = temp_dir / "training_artifacts.json"
            with open(artifacts_file, 'w') as f:
                json.dump(artifacts, f, indent=2, default=str)
            
            logger.info("Model training pipeline completed successfully")
            
            # Verify training results
            assert artifacts_file.exists()
            assert trained_model['best_val_accuracy'] > 0.5
            assert evaluation_results['test_accuracy'] > 0.5
            assert trained_model['epochs_trained'] > 0
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")


@pytest.mark.integration
class TestProductionWorkflows:
    """Test production-ready workflows."""
    
    def test_batch_processing_with_error_handling(self, temp_dir, sample_numpy_arrays):
        """Test batch processing with comprehensive error handling."""
        try:
            from refunc.logging import MLLogger
            from refunc.exceptions import retry_on_failure, RefuncError
            from refunc.decorators import performance_monitor
            from refunc.utils import FileHandler
            
            logger = MLLogger("batch_processing", log_dir=str(temp_dir))
            file_handler = FileHandler(logger=logger)
            
            # Simulate batch processing configuration
            processing_config = {
                'batch_size': 50,
                'max_retries': 3,
                'error_threshold': 0.1,  # 10% error rate threshold
                'output_format': 'csv'
            }
            
            batch_results = []
            error_count = 0
            
            @retry_on_failure(max_attempts=processing_config['max_retries'], delay=0.1)
            @performance_monitor(logger=logger)  
            def process_single_batch(batch_data, batch_id):
                """Process a single batch with error simulation."""
                logger.info(f"Processing batch {batch_id}")
                
                # Simulate occasional processing errors
                if np.random.random() < 0.05:  # 5% error rate
                    raise ValueError(f"Simulated processing error in batch {batch_id}")
                
                # Actual processing
                if len(batch_data.shape) == 1:
                    result = {
                        'batch_id': batch_id,
                        'mean': np.mean(batch_data),
                        'std': np.std(batch_data),
                        'min': np.min(batch_data),
                        'max': np.max(batch_data),
                        'samples': len(batch_data)
                    }
                else:
                    result = {
                        'batch_id': batch_id,
                        'mean': np.mean(batch_data),
                        'std': np.std(batch_data),
                        'shape': batch_data.shape,
                        'samples': batch_data.shape[0]
                    }
                
                # Simulate processing time
                time.sleep(0.01)
                
                logger.metric(f"batch_{batch_id}_samples", result['samples'])
                logger.metric(f"batch_{batch_id}_mean", result['mean'])
                
                return result
            
            def batch_processor_with_monitoring():
                """Main batch processing function with error monitoring."""
                nonlocal error_count
                
                logger.info("Starting batch processing workflow")
                total_batches = 0
                successful_batches = 0
                
                for array_name, array_data in sample_numpy_arrays.items():
                    logger.info(f"Processing array: {array_name}")
                    
                    # Convert to appropriate format for batching
                    if len(array_data.shape) > 1:
                        data_for_batching = array_data
                    else:
                        data_for_batching = array_data.reshape(-1, 1) if len(array_data) > 1 else array_data
                    
                    batch_size = processing_config['batch_size']
                    n_samples = len(data_for_batching)
                    
                    for i in range(0, n_samples, batch_size):
                        batch = data_for_batching[i:i+batch_size]
                        total_batches += 1
                        
                        try:
                            result = process_single_batch(batch, total_batches)
                            batch_results.append(result)
                            successful_batches += 1
                            
                        except Exception as e:
                            error_count += 1
                            logger.error(f"Failed to process batch {total_batches}: {str(e)}")
                            
                            # Check error threshold
                            error_rate = error_count / total_batches
                            if error_rate > processing_config['error_threshold']:
                                logger.error(f"Error rate {error_rate:.2%} exceeds threshold {processing_config['error_threshold']:.2%}")
                                raise RefuncError(f"Batch processing failed: error rate too high")
                
                logger.metric("total_batches", total_batches)
                logger.metric("successful_batches", successful_batches)
                logger.metric("error_rate", error_count / total_batches if total_batches > 0 else 0)
                
                return {
                    'total_batches': total_batches,
                    'successful_batches': successful_batches,
                    'error_count': error_count,
                    'results': batch_results
                }
            
            # Run batch processing
            processing_summary = batch_processor_with_monitoring()
            
            # Save results
            if batch_results:
                results_df = pd.DataFrame(batch_results)
                results_file = temp_dir / f"batch_results.{processing_config['output_format']}"
                file_handler.save_dataframe(results_df, results_file)
                
                # Save processing summary
                summary_file = temp_dir / "processing_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(processing_summary, f, indent=2, default=str)
            
            logger.info("Batch processing workflow completed")
            
            # Verify batch processing results
            assert processing_summary['total_batches'] > 0
            assert processing_summary['successful_batches'] > 0
            assert len(batch_results) == processing_summary['successful_batches']
            
            if batch_results:
                assert results_file.exists()
                assert summary_file.exists()
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")
    
    def test_configuration_driven_pipeline(self, temp_dir, sample_dataframe):
        """Test pipeline that adapts based on runtime configuration."""
        try:
            from refunc.config import ConfigManager
            from refunc.logging import MLLogger
            from refunc.utils import FileHandler
            from refunc.decorators import time_it
            
            # Setup dynamic configuration
            config = ConfigManager()
            
            # Load configuration from multiple sources
            config.update({
                'pipeline': {
                    'name': 'adaptive_processing',
                    'version': '1.0',
                    'steps': [
                        {
                            'name': 'data_validation',
                            'enabled': True,
                            'parameters': {
                                'check_missing': True,
                                'check_duplicates': True,
                                'min_rows': 10
                            }
                        },
                        {
                            'name': 'data_cleaning',
                            'enabled': True,
                            'parameters': {
                                'drop_duplicates': True,
                                'fill_missing': 'median',
                                'outlier_method': 'iqr'
                            }
                        },
                        {
                            'name': 'feature_engineering',
                            'enabled': True,
                            'parameters': {
                                'create_derived': True,
                                'normalize': True,
                                'polynomial_features': False
                            }
                        },
                        {
                            'name': 'data_export',
                            'enabled': True,
                            'parameters': {
                                'format': 'csv',
                                'include_metadata': True
                            }
                        }
                    ]
                }
            })
            
            logger = MLLogger("adaptive_pipeline", log_dir=str(temp_dir))
            file_handler = FileHandler(logger=logger)
            
            @time_it(logger=logger)
            def execute_pipeline_step(data, step_config):
                """Execute a single pipeline step based on configuration."""
                step_name = step_config['name']
                step_params = step_config.get('parameters', {})
                
                logger.info(f"Executing step: {step_name}")
                
                if not step_config.get('enabled', True):
                    logger.info(f"Step {step_name} is disabled, skipping")
                    return data
                
                result = data.copy()
                
                if step_name == 'data_validation':
                    # Validation logic
                    if step_params.get('check_missing', False):
                        missing_count = result.isnull().sum().sum()
                        logger.metric(f"{step_name}_missing_values", missing_count)
                        if missing_count > len(result) * 0.5:
                            logger.warning("High missing value rate detected")
                    
                    if step_params.get('check_duplicates', False):
                        duplicate_count = result.duplicated().sum()
                        logger.metric(f"{step_name}_duplicates", duplicate_count)
                    
                    min_rows = step_params.get('min_rows', 0)
                    if len(result) < min_rows:
                        raise ValueError(f"Insufficient data: {len(result)} < {min_rows}")
                
                elif step_name == 'data_cleaning':
                    initial_rows = len(result)
                    
                    if step_params.get('drop_duplicates', False):
                        result = result.drop_duplicates()
                    
                    fill_method = step_params.get('fill_missing', None)
                    if fill_method and result.isnull().any().any():
                        if fill_method == 'median':
                            numeric_cols = result.select_dtypes(include=[np.number]).columns
                            result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].median())
                    
                    final_rows = len(result)
                    logger.metric(f"{step_name}_rows_removed", initial_rows - final_rows)
                
                elif step_name == 'feature_engineering':
                    if step_params.get('create_derived', False):
                        if 'numeric' in result.columns:
                            result['numeric_squared'] = result['numeric'] ** 2
                            result['numeric_sqrt'] = np.sqrt(np.abs(result['numeric']))
                    
                    if step_params.get('normalize', False):
                        numeric_cols = result.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            if col != 'numeric':  # Don't normalize the derived features twice
                                col_mean = result[col].mean()
                                col_std = result[col].std()
                                if col_std > 0:
                                    result[f'{col}_normalized'] = (result[col] - col_mean) / col_std
                    
                    logger.metric(f"{step_name}_final_features", len(result.columns))
                
                return result
            
            def run_adaptive_pipeline(input_data, config):
                """Run the complete adaptive pipeline."""
                pipeline_config = config.get('pipeline', {})
                steps = pipeline_config.get('steps', [])
                
                logger.info(f"Starting adaptive pipeline: {pipeline_config.get('name', 'unknown')}")
                
                current_data = input_data.copy()
                step_outputs = {}
                
                for step in steps:
                    step_name = step['name']
                    
                    try:
                        current_data = execute_pipeline_step(current_data, step)
                        step_outputs[step_name] = {
                            'success': True,
                            'output_shape': current_data.shape,
                            'output_columns': list(current_data.columns)
                        }
                        logger.info(f"Step {step_name} completed successfully")
                        
                    except Exception as e:
                        logger.error(f"Step {step_name} failed: {str(e)}")
                        step_outputs[step_name] = {
                            'success': False,
                            'error': str(e)
                        }
                        # Continue with next step or fail based on configuration
                        continue
                
                return current_data, step_outputs
            
            # Run the adaptive pipeline
            processed_data, step_results = run_adaptive_pipeline(sample_dataframe, config)
            
            # Save outputs based on configuration
            export_step = next((s for s in config.get('pipeline.steps', []) if s['name'] == 'data_export'), None)
            if export_step and export_step.get('enabled', True):
                export_params = export_step.get('parameters', {})
                output_format = export_params.get('format', 'csv')
                
                output_file = temp_dir / f"processed_data.{output_format}"
                file_handler.save_dataframe(processed_data, output_file)
                
                if export_params.get('include_metadata', False):
                    metadata = {
                        'pipeline_config': dict(config._data),
                        'step_results': step_results,
                        'final_shape': processed_data.shape,
                        'processing_timestamp': pd.Timestamp.now().isoformat()
                    }
                    
                    metadata_file = temp_dir / "pipeline_metadata.json"
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2, default=str)
            
            logger.info("Adaptive pipeline completed successfully")
            
            # Verify pipeline results
            assert len(processed_data) > 0
            assert len(processed_data.columns) >= len(sample_dataframe.columns)
            assert all(result['success'] for result in step_results.values())
            assert output_file.exists()
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")


@pytest.mark.integration
@pytest.mark.slow
class TestMLOpsWorkflows:
    """Test MLOps and production ML workflows."""
    
    def test_model_deployment_simulation(self, temp_dir, sample_dataframe):
        """Test model deployment workflow with monitoring."""
        try:
            from refunc.logging import MLLogger
            from refunc.decorators import performance_monitor, time_it
            from refunc.utils import FileHandler, MemoryCache
            from refunc.config import ConfigManager
            
            # Setup deployment configuration
            config = ConfigManager()
            config.update({
                'deployment': {
                    'model_name': 'production_model_v1',
                    'version': '1.0.0',
                    'environment': 'staging',
                    'monitoring': {
                        'log_predictions': True,
                        'log_performance': True,
                        'alert_thresholds': {
                            'prediction_time_ms': 100,
                            'memory_usage_mb': 500
                        }
                    }
                },
                'inference': {
                    'batch_size': 32,
                    'cache_predictions': True,
                    'cache_ttl_seconds': 3600
                }
            })
            
            logger = MLLogger("model_deployment", log_dir=str(temp_dir))
            file_handler = FileHandler(logger=logger)
            prediction_cache = MemoryCache(max_size=1000)
            
            # Mock model class
            class ProductionModel:
                def __init__(self, model_config):
                    self.config = model_config
                    self.version = model_config.get('version', '1.0.0')
                    self.prediction_count = 0
                
                @performance_monitor()
                def predict(self, X):
                    """Make predictions with monitoring."""
                    self.prediction_count += 1
                    
                    # Simulate model inference
                    time.sleep(0.01)  # Simulate inference time
                    
                    # Mock predictions
                    predictions = np.random.choice([0, 1], size=len(X))
                    confidence_scores = np.random.uniform(0.6, 0.95, size=len(X))
                    
                    return {
                        'predictions': predictions,
                        'confidence': confidence_scores,
                        'model_version': self.version
                    }
            
            @time_it(logger=logger)
            def deploy_model(config):
                """Deploy model with configuration."""
                deployment_config = config.get('deployment', {})
                
                logger.info(f"Deploying model: {deployment_config.get('model_name')}")
                logger.info(f"Version: {deployment_config.get('version')}")
                logger.info(f"Environment: {deployment_config.get('environment')}")
                
                model = ProductionModel(deployment_config)
                
                logger.metric("model_deployed", 1)
                return model
            
            @performance_monitor(logger=logger)
            def serve_predictions(model, input_data, config):
                """Serve predictions with monitoring and caching."""
                inference_config = config.get('inference', {})
                monitoring_config = config.get('deployment.monitoring', {})
                
                batch_size = inference_config.get('batch_size', 32)
                results = []
                
                # Process in batches
                for i in range(0, len(input_data), batch_size):
                    batch = input_data[i:i+batch_size]
                    batch_key = f"batch_{i}_{hash(str(batch.tolist()))}"
                    
                    # Check cache if enabled
                    cached_result = None
                    if inference_config.get('cache_predictions', False):
                        cached_result = prediction_cache.get(batch_key)
                    
                    if cached_result is not None:
                        logger.debug(f"Using cached predictions for batch {i//batch_size}")
                        batch_result = cached_result
                    else:
                        # Make predictions
                        start_time = time.perf_counter()
                        batch_result = model.predict(batch)
                        prediction_time_ms = (time.perf_counter() - start_time) * 1000
                        
                        # Cache results if enabled
                        if inference_config.get('cache_predictions', False):
                            prediction_cache.set(batch_key, batch_result)
                        
                        # Monitor performance
                        logger.metric(f"batch_{i//batch_size}_prediction_time_ms", prediction_time_ms)
                        
                        # Check performance thresholds
                        alert_thresholds = monitoring_config.get('alert_thresholds', {})
                        max_prediction_time = alert_thresholds.get('prediction_time_ms', 1000)
                        
                        if prediction_time_ms > max_prediction_time:
                            logger.warning(f"Prediction time {prediction_time_ms:.1f}ms exceeds threshold {max_prediction_time}ms")
                    
                    results.append(batch_result)
                
                # Combine batch results
                all_predictions = np.concatenate([r['predictions'] for r in results])
                all_confidence = np.concatenate([r['confidence'] for r in results])
                
                final_result = {
                    'predictions': all_predictions,
                    'confidence': all_confidence,
                    'model_version': results[0]['model_version'],
                    'total_samples': len(all_predictions),
                    'batches_processed': len(results)
                }
                
                # Log prediction statistics
                if monitoring_config.get('log_predictions', False):
                    logger.metric("total_predictions", len(all_predictions))
                    logger.metric("avg_confidence", np.mean(all_confidence))
                    logger.metric("positive_predictions", np.sum(all_predictions))
                
                return final_result
            
            @time_it(logger=logger)
            def monitor_model_health(model, predictions_result, config):
                """Monitor model health and performance."""
                monitoring_config = config.get('deployment.monitoring', {})
                
                logger.info("Performing model health check")
                
                # Model statistics
                health_metrics = {
                    'total_predictions_served': model.prediction_count,
                    'avg_confidence': np.mean(predictions_result['confidence']),
                    'prediction_distribution': {
                        'positive': int(np.sum(predictions_result['predictions'])),
                        'negative': int(len(predictions_result['predictions']) - np.sum(predictions_result['predictions']))
                    },
                    'model_version': predictions_result['model_version'],
                    'health_status': 'healthy'
                }
                
                # Check health thresholds
                avg_confidence = health_metrics['avg_confidence']
                if avg_confidence < 0.7:
                    health_metrics['health_status'] = 'warning'
                    logger.warning(f"Low average confidence: {avg_confidence:.3f}")
                
                # Log health metrics
                for metric, value in health_metrics.items():
                    if isinstance(value, (int, float)):
                        logger.metric(f"health_{metric}", value)
                
                return health_metrics
            
            # Run deployment workflow
            logger.info("Starting model deployment workflow")
            
            # Deploy model
            production_model = deploy_model(config)
            
            # Prepare input data (simulate real inference requests)
            X = sample_dataframe[['numeric']].values
            
            # Serve predictions
            prediction_results = serve_predictions(production_model, X, config)
            
            # Monitor model health
            health_status = monitor_model_health(production_model, prediction_results, config)
            
            # Save deployment artifacts
            deployment_summary = {
                'deployment_config': dict(config._data),
                'prediction_results': {
                    'total_samples': int(prediction_results['total_samples']),
                    'avg_confidence': float(prediction_results['confidence'].mean()),
                    'prediction_distribution': {
                        'positive': int(np.sum(prediction_results['predictions'])),
                        'negative': int(len(prediction_results['predictions']) - np.sum(prediction_results['predictions']))
                    }
                },
                'health_status': health_status,
                'deployment_timestamp': pd.Timestamp.now().isoformat()
            }
            
            summary_file = temp_dir / "deployment_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(deployment_summary, f, indent=2, default=str)
            
            # Save predictions log
            predictions_df = pd.DataFrame({
                'prediction': prediction_results['predictions'],
                'confidence': prediction_results['confidence'],
                'model_version': prediction_results['model_version']
            })
            predictions_file = temp_dir / "predictions_log.csv"
            file_handler.save_dataframe(predictions_df, predictions_file)
            
            logger.info("Model deployment workflow completed successfully")
            
            # Verify deployment results
            assert summary_file.exists()
            assert predictions_file.exists()
            assert prediction_results['total_samples'] > 0
            assert health_status['health_status'] in ['healthy', 'warning']
            assert production_model.prediction_count > 0
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")


@pytest.mark.integration
class TestDataPipelineScenarios:
    """Test realistic data pipeline scenarios."""
    
    def test_multi_source_data_integration(self, temp_dir, sample_dataframe):
        """Test integrating data from multiple sources with different formats."""
        try:
            from refunc.utils import FileHandler
            from refunc.logging import MLLogger
            from refunc.decorators import time_it
            from refunc.exceptions import retry_on_failure
            
            logger = MLLogger("data_integration", log_dir=str(temp_dir))  
            file_handler = FileHandler(logger=logger)
            
            # Create multiple data sources
            data_sources = {
                'source_a': sample_dataframe.copy(),
                'source_b': sample_dataframe.copy().assign(source='B'),
                'source_c': sample_dataframe.copy().assign(source='C', extra_col=np.random.randn(len(sample_dataframe)))
            }
            
            # Modify each source to simulate different schemas
            data_sources['source_b'] = data_sources['source_b'].drop(columns=['text'], errors='ignore')
            data_sources['source_c']['categorical'] = data_sources['source_c']['categorical'].map({'A': 'X', 'B': 'Y', 'C': 'Z'})
            
            @retry_on_failure(max_attempts=3, delay=0.1)
            @time_it(logger=logger)
            def load_data_source(source_name, data, file_format='csv'):
                """Load and validate data from a source."""
                logger.info(f"Loading data from {source_name}")
                
                # Save to file first (simulating file-based sources)
                source_file = temp_dir / f"{source_name}.{file_format}"
                file_handler.save_dataframe(data, source_file)
                
                # Load from file
                loaded_data = file_handler.load_dataframe(source_file)
                
                # Add metadata
                loaded_data['data_source'] = source_name
                loaded_data['load_timestamp'] = pd.Timestamp.now()
                
                logger.metric(f"{source_name}_rows", len(loaded_data))
                logger.metric(f"{source_name}_columns", len(loaded_data.columns))
                
                return loaded_data
            
            @time_it(logger=logger)
            def harmonize_schemas(data_sources_dict):
                """Harmonize schemas across different data sources."""
                logger.info("Harmonizing schemas across data sources")
                
                all_columns = set()
                for source_data in data_sources_dict.values():
                    all_columns.update(source_data.columns)
                
                harmonized_sources = {}
                
                for source_name, source_data in data_sources_dict.items():
                    harmonized = source_data.copy()
                    
                    # Add missing columns with default values
                    for col in all_columns:
                        if col not in harmonized.columns:
                            if col in ['text']:
                                harmonized[col] = 'missing'
                            elif col in ['extra_col']:
                                harmonized[col] = 0.0
                            elif col in ['source']:
                                harmonized[col] = source_name
                            else:
                                harmonized[col] = None
                    
                    # Standardize categorical mappings
                    if 'categorical' in harmonized.columns:
                        mapping = {'X': 'A', 'Y': 'B', 'Z': 'C'}  # Reverse mapping for source_c
                        harmonized['categorical'] = harmonized['categorical'].replace(mapping)
                    
                    harmonized_sources[source_name] = harmonized
                    logger.info(f"Harmonized {source_name}: {harmonized.shape}")
                
                return harmonized_sources
            
            @time_it(logger=logger)
            def integrate_data_sources(harmonized_sources):
                """Integrate harmonized data sources."""
                logger.info("Integrating data sources")
                
                # Combine all sources
                integrated_data = pd.concat(harmonized_sources.values(), ignore_index=True)
                
                # Data quality checks
                initial_count = len(integrated_data)
                
                # Remove duplicates (keeping first occurrence)
                integrated_data = integrated_data.drop_duplicates(subset=['numeric', 'categorical'], keep='first')
                
                # Handle missing values
                integrated_data = integrated_data.dropna(subset=['numeric'])  # Keep only rows with numeric data
                
                final_count = len(integrated_data)
                
                logger.metric("integration_initial_rows", initial_count)
                logger.metric("integration_final_rows", final_count)
                logger.metric("integration_removed_rows", initial_count - final_count)
                
                return integrated_data
            
            # Run data integration pipeline
            logger.info("Starting multi-source data integration")
            
            # Load all data sources
            loaded_sources = {}
            for source_name, source_data in data_sources.items():
                loaded_sources[source_name] = load_data_source(source_name, source_data)
            
            # Harmonize schemas
            harmonized_sources = harmonize_schemas(loaded_sources)
            
            # Integrate data
            integrated_dataset = integrate_data_sources(harmonized_sources)
            
            # Save integrated dataset
            output_file = temp_dir / "integrated_dataset.csv"
            file_handler.save_dataframe(integrated_dataset, output_file)
            
            # Generate integration report
            integration_report = {
                'sources_processed': len(data_sources),
                'total_rows_input': sum(len(data) for data in data_sources.values()),
                'total_rows_output': len(integrated_dataset),
                'columns_final': len(integrated_dataset.columns),
                'data_sources': list(integrated_dataset['data_source'].unique()),
                'schema_harmonization': {
                    'common_columns': len(set.intersection(*[set(df.columns) for df in data_sources.values()])),
                    'total_unique_columns': len(set.union(*[set(df.columns) for df in data_sources.values()]))
                }
            }
            
            report_file = temp_dir / "integration_report.json"
            with open(report_file, 'w') as f:
                json.dump(integration_report, f, indent=2)
            
            logger.info("Data integration completed successfully")
            
            # Verify integration results
            assert output_file.exists()
            assert report_file.exists()
            assert len(integrated_dataset) > 0
            assert 'data_source' in integrated_dataset.columns
            assert len(integrated_dataset['data_source'].unique()) == len(data_sources)
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")


@pytest.mark.integration
class TestErrorRecoveryScenarios:
    """Test error recovery in realistic failure scenarios."""
    
    def test_network_failure_simulation(self, temp_dir, sample_dataframe):
        """Test pipeline behavior during simulated network failures."""
        try:
            from refunc.exceptions import retry_on_failure, RefuncError
            from refunc.logging import MLLogger
            from refunc.utils import FileHandler
            from refunc.decorators import time_it
            
            logger = MLLogger("network_failure_test", log_dir=str(temp_dir))
            file_handler = FileHandler(logger=logger)
            
            # Simulate network failure state
            network_failure_simulation = {
                'failure_rate': 0.3,  # 30% failure rate
                'consecutive_failures': 0,
                'max_consecutive_failures': 2
            }
            
            @retry_on_failure(max_attempts=4, delay=0.2)
            @time_it(logger=logger)
            def simulate_network_operation(operation_name, data=None):
                """Simulate network operation that may fail."""
                logger.info(f"Attempting network operation: {operation_name}")
                
                # Simulate network failure
                if np.random.random() < network_failure_simulation['failure_rate']:
                    network_failure_simulation['consecutive_failures'] += 1
                    
                    if network_failure_simulation['consecutive_failures'] <= network_failure_simulation['max_consecutive_failures']:
                        logger.warning(f"Network failure simulated for {operation_name}")
                        raise ConnectionError(f"Simulated network failure for {operation_name}")
                
                # Reset consecutive failures on success
                network_failure_simulation['consecutive_failures'] = 0
                
                # Simulate successful operation
                time.sleep(0.1)  # Simulate network latency
                
                if operation_name == 'data_download':
                    logger.info("Data download successful")
                    return sample_dataframe.copy()
                elif operation_name == 'data_upload':
                    logger.info("Data upload successful")
                    return {'status': 'uploaded', 'rows': len(data)}
                elif operation_name == 'model_sync':
                    logger.info("Model sync successful")
                    return {'status': 'synced', 'version': '1.0.0'}
                
                return {'status': 'success'}
            
            @time_it(logger=logger)
            def resilient_data_pipeline():
                """Data pipeline with network failure resilience."""
                logger.info("Starting resilient data pipeline")
                
                pipeline_results = {
                    'operations_attempted': 0,
                    'operations_successful': 0,
                    'operations_failed': 0,
                    'retry_attempts': 0
                }
                
                try:
                    # Step 1: Download data (may fail)
                    pipeline_results['operations_attempted'] += 1
                    downloaded_data = simulate_network_operation('data_download')
                    pipeline_results['operations_successful'] += 1
                    
                    # Step 2: Process data locally (should not fail)
                    processed_data = downloaded_data.copy()
                    processed_data['processed'] = True
                    processed_data['process_time'] = pd.Timestamp.now()
                    
                    # Step 3: Upload processed data (may fail)
                    pipeline_results['operations_attempted'] += 1
                    upload_result = simulate_network_operation('data_upload', processed_data)
                    pipeline_results['operations_successful'] += 1
                    
                    # Step 4: Sync model (may fail)
                    pipeline_results['operations_attempted'] += 1
                    sync_result = simulate_network_operation('model_sync')
                    pipeline_results['operations_successful'] += 1
                    
                    # Save local backup regardless of network operations
                    backup_file = temp_dir / "local_backup.csv"
                    file_handler.save_dataframe(processed_data, backup_file)
                    
                    pipeline_results['local_backup_saved'] = True
                    
                except Exception as e:
                    pipeline_results['operations_failed'] += 1
                    logger.error(f"Pipeline operation failed: {str(e)}")
                    
                    # Try to save what we have locally
                    try:
                        if 'processed_data' in locals():
                            emergency_backup = temp_dir / "emergency_backup.csv"
                            file_handler.save_dataframe(processed_data, emergency_backup)
                            pipeline_results['emergency_backup_saved'] = True
                    except Exception as backup_error:
                        logger.error(f"Emergency backup failed: {str(backup_error)}")
                
                return pipeline_results
            
            # Run resilient pipeline multiple times to test failure recovery
            all_results = []
            for run in range(5):
                logger.info(f"Pipeline run {run + 1}")
                run_results = resilient_data_pipeline()
                run_results['run_number'] = run + 1
                all_results.append(run_results)
            
            # Analyze failure recovery patterns
            total_operations = sum(r['operations_attempted'] for r in all_results)
            total_successful = sum(r['operations_successful'] for r in all_results)
            total_failed = sum(r['operations_failed'] for r in all_results)
            
            recovery_analysis = {
                'total_pipeline_runs': len(all_results),
                'total_operations_attempted': total_operations,
                'total_operations_successful': total_successful,
                'total_operations_failed': total_failed,
                'success_rate': total_successful / total_operations if total_operations > 0 else 0,
                'runs_with_backups': sum(1 for r in all_results if r.get('local_backup_saved', False)),
                'runs_with_emergency_backups': sum(1 for r in all_results if r.get('emergency_backup_saved', False))
            }
            
            # Save recovery analysis
            analysis_file = temp_dir / "failure_recovery_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump({
                    'recovery_analysis': recovery_analysis,
                    'individual_runs': all_results
                }, f, indent=2)
            
            logger.info("Network failure simulation completed")
            
            # Verify failure recovery worked
            assert analysis_file.exists()
            assert recovery_analysis['total_pipeline_runs'] == 5
            assert recovery_analysis['success_rate'] > 0.5  # Should have some success despite failures
            assert recovery_analysis['runs_with_backups'] > 0  # Should have saved some backups
            
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")