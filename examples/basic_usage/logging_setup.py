#!/usr/bin/env python3
"""
Logging Setup Examples - Refunc ML Logging

This example demonstrates the MLLogger system for experiment tracking,
structured logging, and ML workflow monitoring.

Key Features Demonstrated:
- MLLogger basic setup and usage
- Experiment tracking and metadata
- Metrics and parameter logging
- Progress tracking
- Multiple output formats
- Integration patterns
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Handle missing dependencies gracefully
try:
    from refunc.logging import (
        MLLogger, get_logger, setup_logging,
        ExperimentTracker, ProgressTracker,
        info, debug, warning, error, metric
    )
    REFUNC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Refunc not fully installed: {e}")
    print("This example requires: pip install refunc")
    print("Showing example code structure...\n")
    REFUNC_AVAILABLE = False


def basic_logging_setup():
    """Demonstrate basic MLLogger setup and usage."""
    print("🔧 Basic MLLogger Setup")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Basic MLLogger usage:
from refunc.logging import MLLogger, setup_logging

# Initialize logger for experiment
logger = MLLogger("my_experiment", log_dir="./logs")

# Log different types of information
logger.info("Starting experiment")
logger.debug("Debug information")
logger.warning("Warning message")
logger.error("Error occurred")

# Log metrics and parameters
logger.log_metrics({
    "epoch": 1,
    "accuracy": 0.85,
    "loss": 0.234
})

logger.log_params({
    "learning_rate": 0.001,
    "batch_size": 32,
    "model_type": "neural_network"
})
        """)
        return
    
    # Create temporary directory for logs
    temp_dir = Path(tempfile.mkdtemp(prefix="refunc_logging_"))
    print(f"📁 Log directory: {temp_dir}")
    
    try:
        # Initialize MLLogger
        logger = MLLogger("demo_experiment", log_dir=str(temp_dir))
        print("✓ MLLogger initialized")
        
        # Basic logging
        logger.info("🚀 Starting demonstration experiment")
        logger.debug("Debug: System initialized")
        logger.warning("⚠️  This is a demo warning")
        
        # Log experiment metadata
        logger.log_params({
            "experiment_type": "demonstration",
            "framework": "refunc",
            "version": "0.1.0",
            "environment": "development"
        })
        print("✓ Parameters logged")
        
        # Log some metrics
        for epoch in range(1, 4):
            # Simulate training metrics
            accuracy = 0.7 + (epoch * 0.05)
            loss = 0.5 - (epoch * 0.08)
            
            logger.log_metrics({
                "epoch": epoch,
                "accuracy": accuracy,
                "loss": loss,
                "learning_rate": 0.001 * (0.9 ** epoch)
            })
            
            logger.info(f"Epoch {epoch}: accuracy={accuracy:.3f}, loss={loss:.3f}")
            time.sleep(0.1)  # Simulate training time
        
        print("✓ Metrics logged for 3 epochs")
        
        # Log final results
        logger.info("✅ Experiment completed successfully")
        
        # Check what was created
        log_files = list(temp_dir.rglob("*"))
        print(f"📄 Created {len(log_files)} log files:")
        for log_file in log_files:
            print(f"   - {log_file.name} ({log_file.stat().st_size} bytes)")
            
    except Exception as e:
        print(f"❌ Error in basic logging: {e}")
    finally:
        # Note: We'll keep temp files for other examples to see
        pass


def experiment_tracking_example():
    """Demonstrate comprehensive experiment tracking."""
    print("\n🧪 Experiment Tracking")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Experiment tracking example:
from refunc.logging import ExperimentTracker, experiment_context

# Initialize experiment tracker
tracker = ExperimentTracker("ml_pipeline")

# Use context manager for automatic tracking
with experiment_context("model_training", tracker=tracker) as exp:
    # Log experiment setup
    exp.log_params({
        "model": "random_forest",
        "n_estimators": 100,
        "max_depth": 10
    })
    
    # Training loop
    for epoch in range(10):
        accuracy = train_epoch()
        exp.log_metric("accuracy", accuracy, step=epoch)
    
    # Log artifacts
    exp.log_artifact("model.pkl", model_path)
    exp.log_artifact("results.csv", results_path)
        """)
        return
    
    temp_dir = Path(tempfile.mkdtemp(prefix="refunc_experiment_"))
    
    try:
        # Initialize experiment tracker
        tracker = ExperimentTracker("ml_pipeline_demo", log_dir=str(temp_dir))
        print("✓ Experiment tracker initialized")
        
        # Start experiment
        experiment_name = "random_forest_classification"
        exp_id = tracker.start_experiment(experiment_name)
        print(f"✓ Started experiment: {experiment_name} (ID: {exp_id})")
        
        # Log experiment parameters
        params = {
            "model_type": "RandomForestClassifier",
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "dataset": "synthetic_classification",
            "features": 20,
            "samples": 1000
        }
        tracker.log_params(params)
        print("✓ Parameters logged")
        
        # Simulate training with metrics
        print("🏃 Simulating training...")
        for step in range(5):
            # Simulate model training metrics
            train_acc = 0.6 + (step * 0.08)
            val_acc = 0.55 + (step * 0.07)
            train_loss = 0.8 - (step * 0.12)
            val_loss = 0.85 - (step * 0.1)
            
            metrics = {
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "step": step
            }
            
            tracker.log_metrics(metrics, step=step)
            print(f"   Step {step}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
            time.sleep(0.1)
        
        # Log final model evaluation
        final_metrics = {
            "final_accuracy": 0.92,
            "precision": 0.89,
            "recall": 0.94,
            "f1_score": 0.91
        }
        tracker.log_metrics(final_metrics)
        print("✓ Final metrics logged")
        
        # Create and log artifacts
        model_info = {
            "model_type": "RandomForestClassifier",
            "training_time": "2.3s",
            "final_accuracy": 0.92
        }
        
        artifact_path = temp_dir / "model_info.json"
        import json
        artifact_path.write_text(json.dumps(model_info, indent=2))
        
        tracker.log_artifact("model_info", str(artifact_path))
        print("✓ Artifacts logged")
        
        # End experiment
        tracker.end_experiment()
        print("✓ Experiment ended")
        
        # Show experiment summary
        experiments = tracker.list_experiments()
        print(f"📊 Total experiments: {len(experiments)}")
        
    except Exception as e:
        print(f"❌ Error in experiment tracking: {e}")
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def progress_tracking_example():
    """Demonstrate progress tracking capabilities."""
    print("\n📈 Progress Tracking")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Progress tracking example:
from refunc.logging import ProgressTracker, progress_context

# Initialize progress tracker
tracker = ProgressTracker("data_processing")

# Track progress with context manager
with progress_context("Processing files", total=100) as progress:
    for i in range(100):
        # Do work
        process_file(i)
        
        # Update progress
        progress.update(1, description=f"Processed file {i}")
        
        # Log intermediate results
        if i % 10 == 0:
            progress.log_metric("processed_count", i)

# Epoch-based progress tracking
from refunc.logging import epoch_context

with epoch_context("model_training", epochs=10) as epoch_tracker:
    for epoch in range(10):
        with epoch_tracker.epoch(epoch) as ep:
            # Training phase
            train_loss = train_model()
            ep.log_metric("train_loss", train_loss)
            
            # Validation phase
            val_loss = validate_model()
            ep.log_metric("val_loss", val_loss)
        """)
        return
    
    temp_dir = Path(tempfile.mkdtemp(prefix="refunc_progress_"))
    
    try:
        # Initialize progress tracker
        logger = MLLogger("progress_demo", log_dir=str(temp_dir))
        tracker = ProgressTracker("data_processing", logger=logger)
        print("✓ Progress tracker initialized")
        
        # Simulate data processing with progress
        total_items = 20
        print(f"📊 Processing {total_items} items...")
        
        tracker.start(total=total_items, description="Processing dataset")
        
        for i in range(total_items):
            # Simulate processing time
            time.sleep(0.1)
            
            # Update progress
            tracker.update(1, description=f"Processed item {i+1}")
            
            # Log metrics every 5 items
            if (i + 1) % 5 == 0:
                metrics = {
                    "items_processed": i + 1,
                    "progress_percent": ((i + 1) / total_items) * 100,
                    "processing_rate": (i + 1) / ((i + 1) * 0.1)  # items per second
                }
                tracker.log_metrics(metrics)
                print(f"   Checkpoint: {i+1}/{total_items} items processed")
        
        tracker.finish("Processing completed successfully")
        print("✅ Progress tracking completed")
        
        # Demonstrate epoch tracking
        print("\n🔄 Epoch-based training simulation:")
        
        epoch_tracker = ProgressTracker("model_training", logger=logger)
        epochs = 5
        
        for epoch in range(epochs):
            print(f"   Epoch {epoch + 1}/{epochs}")
            
            # Simulate training
            train_loss = 1.0 - (epoch * 0.15)
            val_loss = 1.1 - (epoch * 0.12)
            
            epoch_tracker.log_metrics({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss
            })
            
            time.sleep(0.2)  # Simulate epoch time
        
        print("✅ Epoch tracking completed")
        
    except Exception as e:
        print(f"❌ Error in progress tracking: {e}")
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def structured_logging_example():
    """Demonstrate structured logging with different formats."""
    print("\n📝 Structured Logging")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Structured logging example:
from refunc.logging import setup_logging, get_logger
from refunc.logging import ColoredFormatter, JSONFormatter

# Setup logging with custom formatters
setup_logging(
    level="INFO",
    format="json",  # or "colored", "compact"
    handlers=["console", "file"]
)

# Get logger instance
logger = get_logger("my_module")

# Structured logging with context
logger.info("Model training started", extra={
    "model_type": "neural_network",
    "dataset_size": 10000,
    "batch_size": 32
})

# Use convenience functions
from refunc.logging import info, debug, warning, error, metric

info("Training started")
metric("accuracy", 0.85, extra={"epoch": 10})
warning("Memory usage high", extra={"memory_gb": 8.2})
        """)
        return
    
    temp_dir = Path(tempfile.mkdtemp(prefix="refunc_structured_"))
    
    try:
        # Setup structured logging
        log_file = temp_dir / "structured.log"
        
        setup_logging(
            level="DEBUG",
            handlers=["console", "file"],
            file_path=str(log_file)
        )
        
        # Get module logger
        logger = get_logger("ml_pipeline")
        print("✓ Structured logger setup")
        
        # Log with structured data
        logger.info("Pipeline started", extra={
            "pipeline_id": "demo_001",
            "timestamp": time.time(),
            "config": {
                "model": "random_forest",
                "features": 20
            }
        })
        
        # Use convenience functions with context
        info("Data loading phase", extra={"phase": "data_loading", "files": 5})
        debug("Feature engineering", extra={"features_created": 15})
        warning("High memory usage detected", extra={"memory_gb": 7.8})
        
        # Log metrics with context
        metric("data_quality_score", 0.92, extra={
            "phase": "preprocessing",
            "samples_processed": 10000
        })
        
        metric("model_accuracy", 0.87, extra={
            "model": "random_forest",
            "cross_validation": True
        })
        
        error("Validation failed", extra={
            "error_type": "schema_mismatch",
            "expected_columns": 20,
            "actual_columns": 18
        })
        
        info("Pipeline completed successfully", extra={
            "total_time": "45.2s",
            "final_accuracy": 0.87
        })
        
        print("✓ Structured logging completed")
        
        # Show log contents
        if log_file.exists():
            log_content = log_file.read_text()
            lines = log_content.strip().split('\n')
            print(f"📄 Generated {len(lines)} log entries")
            print("   Sample log entry:")
            if lines:
                print(f"   {lines[0][:100]}...")
        
    except Exception as e:
        print(f"❌ Error in structured logging: {e}")
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def integration_patterns():
    """Demonstrate logging integration patterns."""
    print("\n🔗 Integration Patterns")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Integration patterns:
from refunc.logging import MLLogger
from refunc.decorators import time_it, memory_profile

# Logger integration with decorators
logger = MLLogger("performance_tracking")

@time_it(logger=logger)
@memory_profile(logger=logger)
def ml_training_function(data):
    # Training code here
    return trained_model

# Exception handling integration
from refunc.exceptions import retry_on_failure

@retry_on_failure(max_attempts=3, logger=logger)
def robust_data_loading(file_path):
    return load_data(file_path)

# Progress tracking integration
from refunc.logging import progress_context

with progress_context("Training", total=epochs, logger=logger) as progress:
    for epoch in range(epochs):
        loss = train_epoch()
        progress.update(1)
        progress.log_metric("loss", loss)
        """)
        return
    
    print("✨ Integration patterns demonstration:")
    print("   - MLLogger works seamlessly with decorators")
    print("   - Exception handling can log retry attempts")
    print("   - Progress tracking integrates with metrics")
    print("   - Experiment tracking provides complete workflow logs")
    print("   - Multiple output formats (console, file, JSON)")
    print("   - Context managers for automatic cleanup")
    
    print("\n📚 Key integration benefits:")
    print("   • Unified logging across all ML components")
    print("   • Automatic performance metrics collection")
    print("   • Structured experiment tracking")
    print("   • Error handling with context")
    print("   • Progress visualization and monitoring")


def main():
    """Run all logging examples."""
    print("🚀 Refunc ML Logging Examples")
    print("=" * 60)
    
    if not REFUNC_AVAILABLE:
        print("ℹ️  Running in demo mode (showing code structure)")
    else:
        print("ℹ️  Running with full Refunc functionality")
    
    print()
    
    # Run examples
    basic_logging_setup()
    experiment_tracking_example()
    progress_tracking_example()
    structured_logging_example()
    integration_patterns()
    
    print("\n✅ Logging examples completed!")
    print("\n📖 Next steps:")
    print("- Integrate MLLogger into your ML workflows")
    print("- Explore experiment tracking for model comparison")
    print("- Check out error_handling.py for robust error management")
    print("- See performance_monitoring.py for decorator integration")


if __name__ == "__main__":
    main()