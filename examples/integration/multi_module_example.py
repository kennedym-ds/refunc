#!/usr/bin/env python3
"""
Multi-Module Integration Example - Refunc Cross-Module Usage

This example demonstrates advanced cross-module usage patterns and integration
strategies across all refunc components, showing how they work together to
create powerful, maintainable ML workflows.

Key Features Demonstrated:  
- Cross-module integration patterns
- Advanced usage combinations
- Best practices for module composition
- Performance optimization strategies
- Error handling across modules
- Monitoring and observability patterns
"""

import os
import sys
import time
import random
import json
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

# Handle missing dependencies gracefully
try:
    # Import from all major refunc modules
    from refunc.utils import FileHandler, cache_result, MemoryCache
    from refunc.logging import MLLogger, ExperimentTracker, ProgressTracker
    from refunc.decorators import (
        time_it, memory_profile, performance_monitor,
        validate_inputs, validate_outputs
    )
    from refunc.exceptions import (
        retry_on_failure, DataError, ModelError, ValidationError
    )
    from refunc.data_science import (
        validate_dataframe, TransformationPipeline,
        DataValidator, DataProfiler
    )
    from refunc.math_stats import describe, test_normality, minimize_function
    from refunc.ml import ModelRegistry, ModelEvaluator
    from refunc.config import ConfigManager, auto_configure
    
    REFUNC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Refunc not fully installed: {e}")
    print("This example requires: pip install refunc")
    print("Showing example code structure...\n")
    REFUNC_AVAILABLE = False


class AdvancedMLWorkbench:
    """
    Advanced ML workbench demonstrating sophisticated cross-module integration.
    
    This class shows how to combine all refunc modules to create a powerful,
    production-ready ML development environment.
    """
    
    def __init__(self, project_name: str, config_path: Optional[str] = None):
        self.project_name = project_name
        
        # Initialize configuration
        if config_path and REFUNC_AVAILABLE:
            self.config = ConfigManager()
            self.config.add_file_source(config_path)
        else:
            self.config = self._create_default_config()
        
        # Initialize core components
        self.logger = MLLogger(f"{project_name}_workbench")
        self.experiment_tracker = ExperimentTracker(project_name)
        self.file_handler = FileHandler(cache_enabled=True)
        self.cache = MemoryCache(max_size=100)
        
        # Initialize ML components
        self.model_registry = ModelRegistry(f"./models/{project_name}")
        self.model_evaluator = ModelEvaluator()
        self.data_validator = DataValidator()
        
        # Workbench state
        self.experiments = {}
        self.datasets = {}
        self.models = {}
        self.analysis_cache = {}
        
        self.logger.info(f"🧪 Advanced ML Workbench initialized for {project_name}")
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        return {
            "data": {
                "validation_threshold": 0.95,
                "missing_value_threshold": 0.1,
                "outlier_detection_method": "iqr"
            },
            "models": {
                "cross_validation_folds": 5,
                "performance_threshold": 0.8,
                "optimization_method": "grid_search"
            },
            "logging": {
                "level": "INFO",
                "enable_metrics": True,
                "enable_artifacts": True
            },
            "performance": {
                "enable_caching": True,
                "cache_ttl": 3600,
                "enable_profiling": True
            }
        }
    
    @performance_monitor
    @retry_on_failure(max_attempts=3)
    def load_and_analyze_dataset(self, dataset_name: str, data_path: str) -> Dict[str, Any]:
        """Load and comprehensively analyze a dataset."""
        self.logger.info(f"📊 Loading and analyzing dataset: {dataset_name}")
        
        try:
            # Load data with caching
            dataset = self._load_dataset_cached(data_path)
            
            # Comprehensive data analysis
            analysis_results = self._analyze_dataset_comprehensive(dataset_name, dataset)
            
            # Store in workbench
            self.datasets[dataset_name] = {
                "data": dataset,
                "analysis": analysis_results,
                "loaded_at": time.time(),
                "source_path": data_path
            }
            
            self.logger.info(f"✅ Dataset {dataset_name} loaded and analyzed")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load dataset {dataset_name}: {e}")
            raise DataError(f"Dataset loading failed: {e}") from e
    
    @cache_result(ttl_seconds=3600)
    def _load_dataset_cached(self, data_path: str) -> List[List[float]]:
        """Load dataset with caching."""
        self.logger.info(f"🔄 Loading data from {data_path}")
        
        # Simulate data loading (in real scenario, use file_handler)
        if REFUNC_AVAILABLE:
            # Generate synthetic dataset
            n_samples = 2000
            n_features = 15
            
            dataset = []
            for i in range(n_samples):
                sample = []
                
                # Mix of different feature types
                sample.extend([random.gauss(50, 15) for _ in range(5)])  # Normal features
                sample.extend([random.expovariate(0.1) for _ in range(3)])  # Skewed features
                sample.extend([random.uniform(0, 100) for _ in range(3)])  # Uniform features
                sample.extend([random.choice([0, 1]) for _ in range(2)])  # Binary features
                sample.extend([random.randint(1, 10) for _ in range(2)])  # Count features
                
                # Add some correlations
                sample[10] = sample[0] * 0.7 + random.gauss(0, 5)  # Correlated feature
                sample[11] = sample[1] + sample[2] + random.gauss(0, 2)  # Linear combination
                
                dataset.append(sample)
            
            # Add data quality issues
            for _ in range(int(0.03 * n_samples)):  # 3% missing values
                row_idx = random.randint(0, n_samples - 1)
                col_idx = random.randint(0, n_features - 1)
                dataset[row_idx][col_idx] = None
            
            return dataset
        else:
            # Mock dataset
            return [[random.random() for _ in range(15)] for _ in range(100)]
    
    @time_it
    @memory_profile
    def _analyze_dataset_comprehensive(self, dataset_name: str, dataset: List[List[float]]) -> Dict[str, Any]:
        """Perform comprehensive dataset analysis."""
        self.logger.info(f"🔍 Performing comprehensive analysis of {dataset_name}")
        
        analysis = {
            "basic_stats": self._compute_basic_statistics(dataset),
            "data_quality": self._assess_data_quality(dataset),
            "distribution_analysis": self._analyze_distributions(dataset),
            "correlation_analysis": self._analyze_correlations(dataset),
            "outlier_detection": self._detect_outliers(dataset),
            "feature_importance": self._estimate_feature_importance(dataset)
        }
        
        # Log analysis metrics
        self.logger.log_metrics({
            f"{dataset_name}_n_samples": len(dataset),
            f"{dataset_name}_n_features": len(dataset[0]) if dataset else 0,
            f"{dataset_name}_missing_rate": analysis["data_quality"]["missing_rate"],
            f"{dataset_name}_outlier_rate": analysis["outlier_detection"]["outlier_rate"]
        })
        
        return analysis
    
    def _compute_basic_statistics(self, dataset: List[List[float]]) -> Dict[str, Any]:
        """Compute basic statistical measures."""
        if not dataset:
            return {"error": "Empty dataset"}
        
        n_samples = len(dataset)
        n_features = len(dataset[0])
        
        # Feature-wise statistics
        feature_stats = []
        
        for feature_idx in range(n_features):
            feature_values = [row[feature_idx] for row in dataset if row[feature_idx] is not None]
            
            if feature_values:
                n = len(feature_values)
                mean_val = sum(feature_values) / n
                variance = sum((x - mean_val) ** 2 for x in feature_values) / (n - 1) if n > 1 else 0
                std_dev = variance ** 0.5
                
                sorted_values = sorted(feature_values)
                median_val = sorted_values[n // 2]
                
                stats = {
                    "count": n,
                    "mean": mean_val,
                    "std": std_dev,
                    "min": min(feature_values),
                    "max": max(feature_values),
                    "median": median_val,
                    "range": max(feature_values) - min(feature_values)
                }
            else:
                stats = {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0, "median": 0, "range": 0}
            
            feature_stats.append(stats)
        
        return {
            "n_samples": n_samples,
            "n_features": n_features,
            "feature_statistics": feature_stats
        }
    
    def _assess_data_quality(self, dataset: List[List[float]]) -> Dict[str, Any]:
        """Assess overall data quality."""
        if not dataset:
            return {"missing_rate": 1.0, "quality_score": 0.0}
        
        total_cells = len(dataset) * len(dataset[0])
        missing_cells = sum(1 for row in dataset for value in row if value is None)
        missing_rate = missing_cells / total_cells
        
        # Simple quality score
        quality_score = 1.0 - missing_rate
        
        # Additional quality metrics
        complete_rows = sum(1 for row in dataset if all(value is not None for value in row))
        complete_row_rate = complete_rows / len(dataset)
        
        return {
            "missing_rate": missing_rate,
            "complete_row_rate": complete_row_rate,
            "quality_score": quality_score,
            "total_cells": total_cells,
            "missing_cells": missing_cells
        }
    
    def _analyze_distributions(self, dataset: List[List[float]]) -> Dict[str, Any]:
        """Analyze feature distributions."""
        if not dataset:
            return {"distributions": []}
        
        n_features = len(dataset[0])
        distributions = []
        
        for feature_idx in range(min(n_features, 5)):  # Analyze first 5 features
            feature_values = [row[feature_idx] for row in dataset if row[feature_idx] is not None]
            
            if len(feature_values) > 10:
                # Basic distribution analysis
                n = len(feature_values)
                mean_val = sum(feature_values) / n
                std_dev = (sum((x - mean_val) ** 2 for x in feature_values) / (n - 1)) ** 0.5
                
                # Skewness (simplified)
                skewness = sum(((x - mean_val) / std_dev) ** 3 for x in feature_values) / n if std_dev > 0 else 0
                
                # Distribution shape assessment
                if abs(skewness) < 0.5:
                    shape = "approximately_normal"
                elif skewness > 0.5:
                    shape = "right_skewed"
                else:
                    shape = "left_skewed"
                
                distributions.append({
                    "feature_index": feature_idx,
                    "shape": shape,
                    "skewness": skewness,
                    "normality_likely": abs(skewness) < 1.0
                })
        
        return {"distributions": distributions}
    
    def _analyze_correlations(self, dataset: List[List[float]]) -> Dict[str, Any]:
        """Analyze feature correlations."""
        if not dataset or len(dataset[0]) < 2:
            return {"correlations": []}
        
        n_features = len(dataset[0])
        correlations = []
        
        # Analyze correlations between first few features (for efficiency)
        for i in range(min(n_features, 5)):
            for j in range(i + 1, min(n_features, 5)):
                # Extract feature pairs
                pairs = [(row[i], row[j]) for row in dataset 
                        if row[i] is not None and row[j] is not None]
                
                if len(pairs) > 10:
                    # Calculate correlation
                    x_values = [pair[0] for pair in pairs]
                    y_values = [pair[1] for pair in pairs]
                    
                    n = len(x_values)
                    mean_x = sum(x_values) / n
                    mean_y = sum(y_values) / n
                    
                    sum_xy = sum((x_values[k] - mean_x) * (y_values[k] - mean_y) for k in range(n))
                    sum_x2 = sum((x - mean_x) ** 2 for x in x_values)
                    sum_y2 = sum((y - mean_y) ** 2 for y in y_values)
                    
                    correlation = sum_xy / (sum_x2 * sum_y2) ** 0.5 if sum_x2 * sum_y2 > 0 else 0
                    
                    correlations.append({
                        "feature_i": i,
                        "feature_j": j,
                        "correlation": correlation,
                        "strength": "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"
                    })
        
        return {"correlations": correlations}
    
    def _detect_outliers(self, dataset: List[List[float]]) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        if not dataset:
            return {"outlier_rate": 0.0, "outliers_by_feature": []}
        
        n_features = len(dataset[0])
        outliers_by_feature = []
        total_outliers = 0
        
        for feature_idx in range(n_features):
            feature_values = [row[feature_idx] for row in dataset if row[feature_idx] is not None]
            
            if len(feature_values) > 4:
                # IQR method
                sorted_values = sorted(feature_values)
                n = len(sorted_values)
                q1 = sorted_values[n // 4]
                q3 = sorted_values[3 * n // 4]
                iqr = q3 - q1
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = [x for x in feature_values if x < lower_bound or x > upper_bound]
                outlier_count = len(outliers)
                total_outliers += outlier_count
                
                outliers_by_feature.append({
                    "feature_index": feature_idx,
                    "outlier_count": outlier_count,
                    "outlier_rate": outlier_count / len(feature_values),
                    "bounds": [lower_bound, upper_bound]
                })
        
        total_values = len(dataset) * n_features
        overall_outlier_rate = total_outliers / total_values if total_values > 0 else 0
        
        return {
            "outlier_rate": overall_outlier_rate,
            "total_outliers": total_outliers,
            "outliers_by_feature": outliers_by_feature
        }
    
    def _estimate_feature_importance(self, dataset: List[List[float]]) -> Dict[str, Any]:
        """Estimate feature importance using variance and correlation."""
        if not dataset:
            return {"feature_importance": []}
        
        n_features = len(dataset[0])
        importance_scores = []
        
        for feature_idx in range(n_features):
            feature_values = [row[feature_idx] for row in dataset if row[feature_idx] is not None]
            
            if len(feature_values) > 1:
                # Variance-based importance (normalized)
                mean_val = sum(feature_values) / len(feature_values)
                variance = sum((x - mean_val) ** 2 for x in feature_values) / (len(feature_values) - 1)
                
                # Normalize by range
                feature_range = max(feature_values) - min(feature_values)
                normalized_variance = variance / (feature_range ** 2) if feature_range > 0 else 0
                
                importance_scores.append({
                    "feature_index": feature_idx,
                    "importance_score": normalized_variance,
                    "variance": variance,
                    "range": feature_range
                })
        
        # Sort by importance
        importance_scores.sort(key=lambda x: x["importance_score"], reverse=True)
        
        return {"feature_importance": importance_scores}
    
    @performance_monitor
    def create_experiment(self, experiment_name: str, dataset_name: str, 
                         model_configs: List[Dict[str, Any]]) -> str:
        """Create and run a comprehensive ML experiment."""
        self.logger.info(f"🧪 Creating experiment: {experiment_name}")
        
        if dataset_name not in self.datasets:
            raise ValidationError(f"Dataset {dataset_name} not loaded")
        
        # Start experiment tracking
        experiment_id = self.experiment_tracker.start_experiment(experiment_name)
        
        try:
            # Get dataset
            dataset_info = self.datasets[dataset_name]
            dataset = dataset_info["data"]
            
            # Prepare data for ML
            X, y = self._prepare_ml_data(dataset)
            
            # Run experiment
            experiment_results = self._run_ml_experiment(experiment_id, X, y, model_configs)
            
            # Store experiment
            self.experiments[experiment_id] = {
                "name": experiment_name,
                "dataset_name": dataset_name,
                "results": experiment_results,
                "created_at": time.time(),
                "status": "completed"
            }
            
            self.logger.info(f"✅ Experiment {experiment_name} completed")
            return experiment_id
            
        except Exception as e:
            self.logger.error(f"❌ Experiment {experiment_name} failed: {e}")
            
            # Store failed experiment
            self.experiments[experiment_id] = {
                "name": experiment_name,
                "dataset_name": dataset_name,
                "error": str(e),
                "created_at": time.time(),
                "status": "failed"
            }
            
            raise ModelError(f"Experiment failed: {e}") from e
    
    def _prepare_ml_data(self, dataset: List[List[float]]) -> Tuple[List[List[float]], List[int]]:
        """Prepare dataset for ML training."""
        # Filter out rows with missing values (simple strategy)
        complete_rows = [row for row in dataset if all(value is not None for value in row)]
        
        if len(complete_rows) < len(dataset) * 0.8:
            self.logger.warning(f"Removed {len(dataset) - len(complete_rows)} incomplete rows")
        
        # Create features and synthetic target
        X = []
        y = []
        
        for row in complete_rows:
            # Features (all columns except we'll create a target)
            features = row.copy()
            
            # Create synthetic binary target based on features
            target_score = (
                features[0] * 0.1 +      # First feature
                features[1] * 0.05 +     # Second feature  
                features[2] * 0.02 -     # Third feature
                features[3] * 0.01 +     # Fourth feature
                random.gauss(0, 0.1)     # Noise
            )
            
            target = 1 if target_score > sum(features) / len(features) else 0
            
            X.append(features)
            y.append(target)
        
        return X, y
    
    @time_it
    def _run_ml_experiment(self, experiment_id: str, X: List[List[float]], 
                          y: List[int], model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run ML experiment with multiple models."""
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.logger.log_params({
            "experiment_id": experiment_id,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": len(X_train[0]),
            "models_to_train": len(model_configs)
        })
        
        # Train models
        model_results = {}
        
        for config in model_configs:
            model_name = config["name"]
            self.logger.info(f"🎯 Training {model_name}")
            
            # Simulate model training
            training_time = random.uniform(0.5, 2.0)
            time.sleep(training_time)
            
            # Generate realistic performance metrics
            base_performance = 0.75 + random.uniform(0, 0.15)
            accuracy = min(0.92, max(0.70, base_performance))
            
            # Other metrics with realistic relationships
            precision = accuracy + random.uniform(-0.05, 0.05)
            recall = accuracy + random.uniform(-0.05, 0.05)
            precision = min(0.95, max(0.65, precision))
            recall = min(0.95, max(0.65, recall))
            
            f1_score = 2 * (precision * recall) / (precision + recall)
            
            model_metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "training_time": training_time
            }
            
            model_results[model_name] = model_metrics
            
            # Log individual model metrics
            self.logger.log_metrics({
                f"{model_name}_{k}": v for k, v in model_metrics.items()
            })
        
        # Select best model
        best_model_name = max(model_results.keys(), key=lambda k: model_results[k]["f1_score"])
        
        return {
            "model_results": model_results,
            "best_model": best_model_name,
            "best_performance": model_results[best_model_name],
            "experiment_duration": sum(result["training_time"] for result in model_results.values())
        }
    
    @validate_inputs(types={'experiment_ids': list})
    @validate_outputs(types=dict)
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments."""
        self.logger.info(f"📊 Comparing {len(experiment_ids)} experiments")
        
        if not experiment_ids:
            raise ValidationError("No experiments provided for comparison")
        
        comparison_data = []
        
        for exp_id in experiment_ids:
            if exp_id not in self.experiments:
                self.logger.warning(f"Experiment {exp_id} not found, skipping")
                continue
            
            exp_data = self.experiments[exp_id]
            if exp_data["status"] != "completed":
                self.logger.warning(f"Experiment {exp_id} not completed, skipping")
                continue
            
            best_model = exp_data["results"]["best_model"]
            best_performance = exp_data["results"]["best_performance"]
            
            comparison_data.append({
                "experiment_id": exp_id,
                "experiment_name": exp_data["name"],
                "dataset_name": exp_data["dataset_name"],
                "best_model": best_model,
                "best_f1_score": best_performance["f1_score"],
                "best_accuracy": best_performance["accuracy"],
                "training_time": best_performance["training_time"],
                "created_at": exp_data["created_at"]
            })
        
        # Sort by performance
        comparison_data.sort(key=lambda x: x["best_f1_score"], reverse=True)
        
        # Generate comparison summary
        if comparison_data:
            best_experiment = comparison_data[0]
            performance_range = {
                "best_f1": max(data["best_f1_score"] for data in comparison_data),
                "worst_f1": min(data["best_f1_score"] for data in comparison_data),
                "avg_f1": sum(data["best_f1_score"] for data in comparison_data) / len(comparison_data)
            }
            
            comparison_summary = {
                "total_experiments": len(comparison_data),
                "best_experiment": best_experiment,
                "performance_range": performance_range,
                "comparison_data": comparison_data
            }
        else:
            comparison_summary = {
                "total_experiments": 0,
                "comparison_data": [],
                "error": "No valid experiments found for comparison"
            }
        
        self.logger.log_metrics({
            "experiments_compared": len(comparison_data),
            "best_f1_score": performance_range.get("best_f1", 0) if comparison_data else 0
        })
        
        return comparison_summary
    
    def generate_workbench_report(self) -> Dict[str, Any]:
        """Generate comprehensive workbench report."""
        self.logger.info("📋 Generating workbench report")
        
        # Datasets summary
        datasets_summary = {
            "total_datasets": len(self.datasets),
            "datasets": []
        }
        
        for name, data in self.datasets.items():
            analysis = data["analysis"]
            datasets_summary["datasets"].append({
                "name": name,
                "samples": analysis["basic_stats"]["n_samples"],
                "features": analysis["basic_stats"]["n_features"],
                "quality_score": analysis["data_quality"]["quality_score"],
                "missing_rate": analysis["data_quality"]["missing_rate"],
                "loaded_at": data["loaded_at"]
            })
        
        # Experiments summary
        experiments_summary = {
            "total_experiments": len(self.experiments),
            "completed_experiments": len([e for e in self.experiments.values() if e["status"] == "completed"]),
            "failed_experiments": len([e for e in self.experiments.values() if e["status"] == "failed"]),
            "experiments": []
        }
        
        for exp_id, exp_data in self.experiments.items():
            summary_data = {
                "experiment_id": exp_id,
                "name": exp_data["name"],
                "status": exp_data["status"],
                "created_at": exp_data["created_at"]
            }
            
            if exp_data["status"] == "completed":
                results = exp_data["results"]
                summary_data.update({
                    "best_model": results["best_model"],
                    "best_f1_score": results["best_performance"]["f1_score"],
                    "models_trained": len(results["model_results"])
                })
            
            experiments_summary["experiments"].append(summary_data)
        
        # Overall workbench statistics
        workbench_stats = {
            "project_name": self.project_name,
            "uptime_hours": (time.time() - self.logger._start_time) / 3600 if hasattr(self.logger, '_start_time') else 0,
            "cache_size": len(self.analysis_cache),
            "memory_cache_size": self.cache.get_current_size() if hasattr(self.cache, 'get_current_size') else 0
        }
        
        report = {
            "workbench_stats": workbench_stats,
            "datasets_summary": datasets_summary,
            "experiments_summary": experiments_summary,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.logger.log_metrics({
            "report_datasets": len(self.datasets),
            "report_experiments": len(self.experiments),
            "report_completion_rate": experiments_summary["completed_experiments"] / max(1, experiments_summary["total_experiments"])
        })
        
        return report


def advanced_integration_example():
    """Demonstrate advanced multi-module integration."""
    print("🧪 Advanced Multi-Module Integration")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Advanced multi-module integration example:
from refunc import *  # Import all modules

# Create advanced ML workbench
workbench = AdvancedMLWorkbench("advanced_project")

# Load and analyze datasets
analysis = workbench.load_and_analyze_dataset(
    "customer_data", 
    "./data/customers.csv"
)

# Create comprehensive experiment
experiment_id = workbench.create_experiment(
    "model_comparison_v1",
    "customer_data",
    [
        {"name": "RF", "type": "RandomForest", "params": {...}},
        {"name": "XGB", "type": "XGBoost", "params": {...}},
        {"name": "SVM", "type": "SVM", "params": {...}}
    ]
)

# Compare multiple experiments
comparison = workbench.compare_experiments([exp1, exp2, exp3])

# Generate comprehensive report
report = workbench.generate_workbench_report()
        """)
        return
    
    print("🔬 Testing advanced workbench integration:")
    
    # Initialize advanced workbench
    workbench = AdvancedMLWorkbench("integration_demo")
    
    print(f"   ✅ Workbench initialized for project: integration_demo")
    
    # Load and analyze multiple datasets
    datasets_to_load = [
        ("customer_data", "./data/customers.csv"),
        ("product_data", "./data/products.csv"),
        ("transaction_data", "./data/transactions.csv")
    ]
    
    print(f"\n   📊 Loading and analyzing datasets:")
    
    for dataset_name, data_path in datasets_to_load:
        try:
            print(f"     Loading {dataset_name}...")
            analysis = workbench.load_and_analyze_dataset(dataset_name, data_path)
            
            basic_stats = analysis["basic_stats"]
            quality = analysis["data_quality"]
            
            print(f"       ✓ {dataset_name}: {basic_stats['n_samples']} samples, "
                  f"{basic_stats['n_features']} features")
            print(f"         Quality score: {quality['quality_score']:.3f}, "
                  f"Missing rate: {quality['missing_rate']:.1%}")
            
        except Exception as e:
            print(f"       ❌ Failed to load {dataset_name}: {e}")
    
    # Create experiments
    print(f"\n   🧪 Creating ML experiments:")
    
    model_configurations = [
        [
            {"name": "Random Forest", "type": "RandomForestClassifier", "params": {"n_estimators": 100}},
            {"name": "Gradient Boosting", "type": "GradientBoostingClassifier", "params": {"n_estimators": 100}},
            {"name": "Logistic Regression", "type": "LogisticRegression", "params": {"max_iter": 1000}}
        ],
        [
            {"name": "SVM", "type": "SVC", "params": {"kernel": "rbf"}},
            {"name": "Decision Tree", "type": "DecisionTreeClassifier", "params": {"max_depth": 10}},
            {"name": "Naive Bayes", "type": "GaussianNB", "params": {}}
        ]
    ]
    
    experiment_ids = []
    
    # Create experiments for each dataset
    for i, dataset_name in enumerate(["customer_data", "product_data"]):
        if dataset_name in workbench.datasets:
            experiment_name = f"experiment_{dataset_name}_v{i+1}"
            
            try:
                print(f"     Creating {experiment_name}...")
                exp_id = workbench.create_experiment(
                    experiment_name,
                    dataset_name,
                    model_configurations[i]
                )
                experiment_ids.append(exp_id)
                
                # Get experiment results
                exp_results = workbench.experiments[exp_id]["results"]
                best_model = exp_results["best_model"]
                best_f1 = exp_results["best_performance"]["f1_score"]
                
                print(f"       ✓ {experiment_name}: Best model = {best_model} "
                      f"(F1: {best_f1:.3f})")
                
            except Exception as e:
                print(f"       ❌ Failed to create {experiment_name}: {e}")
    
    # Compare experiments
    if len(experiment_ids) >= 2:
        print(f"\n   📊 Comparing experiments:")
        
        try:
            comparison = workbench.compare_experiments(experiment_ids)
            
            print(f"     Experiments compared: {comparison['total_experiments']}")
            
            if comparison["total_experiments"] > 0:
                best_exp = comparison["best_experiment"]
                perf_range = comparison["performance_range"]
                
                print(f"     Best experiment: {best_exp['experiment_name']}")
                print(f"       Model: {best_exp['best_model']}")
                print(f"       F1-score: {best_exp['best_f1_score']:.3f}")
                
                print(f"     Performance range:")
                print(f"       Best F1: {perf_range['best_f1']:.3f}")
                print(f"       Average F1: {perf_range['avg_f1']:.3f}")
                print(f"       Worst F1: {perf_range['worst_f1']:.3f}")
            
        except Exception as e:
            print(f"     ❌ Experiment comparison failed: {e}")
    
    # Generate comprehensive report
    print(f"\n   📋 Generating workbench report:")
    
    try:
        report = workbench.generate_workbench_report()
        
        # Display report summary
        workbench_stats = report["workbench_stats"]
        datasets_summary = report["datasets_summary"]
        experiments_summary = report["experiments_summary"]
        
        print(f"     Project: {workbench_stats['project_name']}")
        print(f"     Datasets loaded: {datasets_summary['total_datasets']}")
        print(f"     Total experiments: {experiments_summary['total_experiments']}")
        print(f"     Completed experiments: {experiments_summary['completed_experiments']}")
        print(f"     Success rate: {experiments_summary['completed_experiments'] / max(1, experiments_summary['total_experiments']):.1%}")
        
        # Show best performing experiment across all
        if experiments_summary["experiments"]:
            completed_experiments = [exp for exp in experiments_summary["experiments"] 
                                   if exp["status"] == "completed"]
            if completed_experiments:
                best_overall = max(completed_experiments, key=lambda x: x.get("best_f1_score", 0))
                print(f"     Overall best: {best_overall['name']} "
                      f"(F1: {best_overall.get('best_f1_score', 0):.3f})")
        
        print(f"     Report generated at: {report['generated_at']}")
        
    except Exception as e:
        print(f"     ❌ Report generation failed: {e}")
    
    print(f"\n   ✅ Advanced integration demonstration completed!")


def cross_module_patterns_example():
    """Demonstrate specific cross-module usage patterns."""
    print("\n🔗 Cross-Module Usage Patterns")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Cross-module usage patterns:

# Pattern 1: Validated caching with performance monitoring
@performance_monitor
@validate_inputs(types={'data': list})
@cache_result(ttl_seconds=3600)
def expensive_analysis(data):
    return comprehensive_statistical_analysis(data)

# Pattern 2: Retry with logging and error context
@retry_on_failure(max_attempts=3, logger=logger)
@time_it(logger=logger)
def robust_model_training(X, y):
    try:
        return train_model(X, y)
    except ModelError as e:
        logger.error(f"Training failed: {e}", extra={"samples": len(X)})
        raise

# Pattern 3: Configuration-driven pipeline
config = ConfigManager()
pipeline = TransformationPipeline(config.get("preprocessing"))
logger = MLLogger(config.get("experiment.name"))

@time_it(logger=logger)
def configurable_preprocessing(data):
    return pipeline.fit_transform(data)

# Pattern 4: Multi-level caching with statistics
@cache_result(ttl_seconds=1800)
def cached_feature_engineering(dataset_id):
    data = load_dataset(dataset_id)
    stats = describe(data)
    features = engineer_features(data, stats)
    
    logger.log_metrics(stats.to_dict())
    return features
        """)
        return
    
    print("🎯 Testing specific cross-module patterns:")
    
    # Pattern 1: Comprehensive function composition
    print("   1️⃣ Comprehensive Function Composition:")
    
    @performance_monitor
    @validate_inputs(types={'data': list, 'method': str})
    @validate_outputs(types=dict)
    @cache_result(ttl_seconds=60)  # Short TTL for demo
    def advanced_data_analysis(data: List[float], method: str = "comprehensive") -> Dict[str, Any]:
        """Advanced analysis with full integration."""
        time.sleep(0.1)  # Simulate computation
        
        if not data:
            raise ValidationError("Empty data provided")
        
        # Statistical analysis
        n = len(data)
        mean_val = sum(data) / n
        variance = sum((x - mean_val) ** 2 for x in data) / (n - 1) if n > 1 else 0
        std_dev = variance ** 0.5
        
        # Distribution analysis
        skewness = sum(((x - mean_val) / std_dev) ** 3 for x in data) / n if std_dev > 0 else 0
        
        # Advanced metrics based on method
        if method == "comprehensive":
            sorted_data = sorted(data)
            median_val = sorted_data[n // 2]
            q1 = sorted_data[n // 4]
            q3 = sorted_data[3 * n // 4]
            
            result = {
                "basic_stats": {"mean": mean_val, "std": std_dev, "count": n},
                "distribution": {"skewness": skewness, "median": median_val},
                "quartiles": {"q1": q1, "q3": q3, "iqr": q3 - q1},
                "method_used": method
            }
        else:
            result = {
                "basic_stats": {"mean": mean_val, "std": std_dev, "count": n},
                "method_used": method
            }
        
        return result
    
    # Test the comprehensive function
    test_data = [random.gauss(100, 20) for _ in range(200)]
    
    # First call (computed)
    start_time = time.time()
    result1 = advanced_data_analysis(test_data, "comprehensive")
    duration1 = time.time() - start_time
    
    # Second call (cached)
    start_time = time.time()
    result2 = advanced_data_analysis(test_data, "comprehensive")
    duration2 = time.time() - start_time
    
    print(f"     First call: {duration1:.3f}s (computed)")
    print(f"     Second call: {duration2:.3f}s (cached)")
    print(f"     Speedup: {duration1/max(duration2, 0.001):.1f}x")
    print(f"     Mean: {result1['basic_stats']['mean']:.1f}")
    print(f"     Skewness: {result1['distribution']['skewness']:.3f}")
    
    # Pattern 2: Error handling with context
    print(f"\n   2️⃣ Robust Error Handling with Context:")
    
    logger = MLLogger("pattern_demo")
    
    @retry_on_failure(max_attempts=3)
    @time_it
    def robust_operation_with_context(data: List[float], operation: str) -> Dict[str, Any]:
        """Operation with comprehensive error handling."""
        
        # Log operation attempt
        logger.info(f"Attempting {operation} on {len(data)} samples")
        
        # Simulate different failure modes
        failure_rate = 0.6  # High failure rate for demo
        
        if random.random() < failure_rate:
            error_type = random.choice(["data", "computation", "resource"])
            
            if error_type == "data":
                logger.error(f"Data validation failed for {operation}", extra={
                    "operation": operation,
                    "data_size": len(data),
                    "error_type": "data_validation"
                })
                raise DataError(f"Invalid data for {operation}")
                
            elif error_type == "computation":
                logger.error(f"Computation failed for {operation}", extra={
                    "operation": operation,
                    "error_type": "computation"
                })
                raise ModelError(f"Computation error in {operation}")
                
            else:
                logger.error(f"Resource error for {operation}", extra={
                    "operation": operation,
                    "error_type": "resource"
                })
                raise ValidationError(f"Resource unavailable for {operation}")
        
        # Success case
        result = {
            "operation": operation,
            "samples_processed": len(data),
            "status": "success",
            "result_value": sum(data) / len(data)
        }
        
        logger.info(f"Operation {operation} completed successfully")
        return result
    
    # Test robust operation
    test_operations = ["analysis", "transformation", "validation"]
    
    for operation in test_operations:
        try:
            result = robust_operation_with_context(test_data[:50], operation)
            print(f"     {operation}: ✅ Success - processed {result['samples_processed']} samples")
        except Exception as e:
            print(f"     {operation}: ❌ Failed after retries - {type(e).__name__}")
    
    # Pattern 3: Configuration-driven workflow
    print(f"\n   3️⃣ Configuration-Driven Workflow:")
    
    # Mock configuration
    workflow_config = {
        "data_processing": {
            "missing_value_strategy": "median",
            "outlier_method": "iqr",
            "scaling_method": "standard"
        },
        "analysis": {
            "correlation_threshold": 0.7,
            "significance_level": 0.05,
            "bootstrap_samples": 1000
        },
        "logging": {
            "log_level": "INFO",
            "enable_metrics": True,
            "log_transformations": True
        }
    }
    
    def configuration_driven_workflow(data: List[float], config: Dict[str, Any]) -> Dict[str, Any]:
        """Workflow that adapts based on configuration."""
        
        # Extract configuration
        data_config = config.get("data_processing", {})
        analysis_config = config.get("analysis", {})
        
        # Configure logging
        workflow_logger = MLLogger("config_workflow")
        
        workflow_logger.info("Starting configuration-driven workflow")
        
        # Data processing based on config
        processed_data = data.copy()
        
        # Missing value handling (simulated)
        missing_strategy = data_config.get("missing_value_strategy", "mean")
        workflow_logger.info(f"Using missing value strategy: {missing_strategy}")
        
        # Outlier detection based on config
        outlier_method = data_config.get("outlier_method", "iqr")
        if outlier_method == "iqr":
            # IQR-based outlier detection
            sorted_data = sorted(processed_data)
            n = len(sorted_data)
            q1 = sorted_data[n // 4]
            q3 = sorted_data[3 * n // 4]
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers_removed = len([x for x in processed_data if x < lower_bound or x > upper_bound])
            processed_data = [x for x in processed_data if lower_bound <= x <= upper_bound]
        
        # Analysis based on config
        correlation_threshold = analysis_config.get("correlation_threshold", 0.5)
        
        # Generate results
        results = {
            "configuration_used": config,
            "processing_steps": [
                f"Missing value strategy: {missing_strategy}",
                f"Outlier method: {outlier_method}",
                f"Outliers removed: {outliers_removed}"
            ],
            "analysis_parameters": {
                "correlation_threshold": correlation_threshold,
                "significance_level": analysis_config.get("significance_level", 0.05)
            },
            "final_sample_count": len(processed_data),
            "data_reduction": (len(data) - len(processed_data)) / len(data)
        }
        
        # Log metrics based on config
        if config.get("logging", {}).get("enable_metrics", True):
            workflow_logger.log_metrics({
                "samples_processed": len(processed_data),
                "outliers_removed": outliers_removed,
                "data_reduction_rate": results["data_reduction"]
            })
        
        workflow_logger.info("Configuration-driven workflow completed")
        return results
    
    # Test configuration-driven workflow
    workflow_result = configuration_driven_workflow(test_data, workflow_config)
    
    print(f"     Processing steps: {len(workflow_result['processing_steps'])}")
    for step in workflow_result['processing_steps']:
        print(f"       - {step}")
    
    print(f"     Final samples: {workflow_result['final_sample_count']}")
    print(f"     Data reduction: {workflow_result['data_reduction']:.1%}")
    
    print(f"\n   ✅ Cross-module patterns demonstration completed!")


def main():
    """Run complete multi-module integration examples."""
    print("🚀 Refunc Multi-Module Integration Examples")
    print("=" * 70)
    
    if not REFUNC_AVAILABLE:
        print("ℹ️  Running in demo mode (showing code structure)")
    else:
        print("ℹ️  Running with full Refunc functionality")
    
    print()
    
    # Set random seed for reproducible examples
    random.seed(42)
    
    # Run examples
    advanced_integration_example()
    cross_module_patterns_example()
    
    print("\n✅ Multi-module integration examples completed!")
    print("\n📖 Key Integration Patterns Demonstrated:")
    print("- Advanced ML workbench with all refunc components")
    print("- Cross-module function composition and decoration")
    print("- Error handling with context and recovery")
    print("- Configuration-driven workflows")
    print("- Comprehensive monitoring and logging")
    print("- Performance optimization through caching")
    
    print("\n🎯 Integration Benefits:")
    print("- Unified development experience across all ML stages")
    print("- Consistent error handling and logging")
    print("- Performance optimization through intelligent caching")
    print("- Comprehensive monitoring and observability")
    print("- Configuration-driven flexibility")
    print("- Production-ready reliability and robustness")


if __name__ == "__main__":
    main()