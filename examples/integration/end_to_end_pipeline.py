#!/usr/bin/env python3
"""
End-to-End ML Pipeline Example - Refunc Integration

This example demonstrates a complete machine learning pipeline integrating
all major refunc components: data validation, preprocessing, model training,
evaluation, logging, error handling, and monitoring.

Key Features Demonstrated:
- Complete ML workflow orchestration
- Integration of all refunc modules
- Error handling and recovery
- Comprehensive logging and monitoring
- Performance optimization
- Production-ready pipeline design
"""

import os
import sys
import time
import random
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Handle missing dependencies gracefully
try:
    # Import all major refunc components
    from refunc.data_science import (
        validate_dataframe, TransformationPipeline,
        DataValidator, create_basic_pipeline
    )
    from refunc.ml import (
        ModelRegistry, ModelEvaluator, 
        HyperparameterOptimizer
    )
    from refunc.logging import MLLogger, ExperimentTracker
    from refunc.decorators import (
        time_it, memory_profile, performance_monitor,
        validate_inputs, validate_outputs
    )
    from refunc.exceptions import (
        retry_on_failure, DataError, ModelError,
        ValidationError
    )
    from refunc.utils import FileHandler, cache_result
    from refunc.math_stats import describe, test_normality
    
    REFUNC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Refunc not fully installed: {e}")
    print("This example requires: pip install refunc")
    print("Showing example code structure...\n")
    REFUNC_AVAILABLE = False


class MLPipelineOrchestrator:
    """Complete ML pipeline orchestrator integrating all refunc components."""
    
    def __init__(self, project_name: str, experiment_id: str):
        self.project_name = project_name
        self.experiment_id = experiment_id
        
        # Initialize logging and tracking
        self.logger = MLLogger(f"{project_name}_{experiment_id}")
        self.experiment_tracker = ExperimentTracker(project_name)
        
        # Initialize components
        self.file_handler = FileHandler(cache_enabled=True)
        self.data_validator = DataValidator()
        self.model_registry = ModelRegistry(f"./models/{project_name}")
        self.model_evaluator = ModelEvaluator()
        
        # Pipeline state
        self.pipeline_state = {
            "started_at": time.time(),
            "current_stage": "initialized",
            "completed_stages": [],
            "errors": [],
            "metrics": {}
        }
        
        self.logger.info(f"🚀 Pipeline initialized for {project_name}")
    
    @time_it
    @memory_profile
    @retry_on_failure(max_attempts=3)
    def load_and_validate_data(self, data_path: str) -> Tuple[List[List[float]], List[int]]:
        """Load and validate input data with comprehensive error handling."""
        self.pipeline_state["current_stage"] = "data_loading"
        self.logger.info(f"📊 Loading data from {data_path}")
        
        try:
            # Simulate data loading (in real scenario, use file_handler)
            if REFUNC_AVAILABLE:
                # Generate synthetic dataset for demonstration
                n_samples = 1500
                n_features = 12
                
                X = []
                y = []
                
                for i in range(n_samples):
                    # Generate realistic feature data
                    sample = [
                        random.gauss(50, 15),    # age
                        random.gauss(65000, 20000),  # income
                        random.randint(300, 850),    # credit_score
                        random.uniform(0, 50000),    # account_balance
                        random.randint(0, 20),       # years_customer
                        random.randint(0, 100),      # transaction_count
                        random.choice([0, 1]),       # is_premium
                        random.uniform(0, 1),        # engagement_score
                        random.randint(0, 10),       # support_tickets
                        random.uniform(0, 365),      # days_since_last_login
                        random.gauss(0, 1),          # feature_1 (normalized)
                        random.gauss(0, 1)           # feature_2 (normalized)
                    ]
                    
                    # Generate target based on features
                    target_score = (
                        sample[0] * 0.01 +      # age
                        sample[1] * 0.00001 +   # income
                        sample[2] * 0.001 +     # credit_score
                        sample[6] * 0.5 +       # is_premium
                        sample[7] * 0.3 -       # engagement_score
                        sample[8] * 0.1 +       # support_tickets (negative impact)
                        random.gauss(0, 0.1)    # noise
                    )
                    
                    X.append(sample)
                    y.append(1 if target_score > 1.2 else 0)
                
                # Add some data quality issues for validation
                for _ in range(int(0.05 * n_samples)):  # 5% missing values
                    sample_idx = random.randint(0, n_samples - 1)
                    feature_idx = random.randint(0, n_features - 1)
                    X[sample_idx][feature_idx] = None
                
                # Add some outliers
                for _ in range(int(0.02 * n_samples)):  # 2% outliers
                    sample_idx = random.randint(0, n_samples - 1)
                    X[sample_idx][1] = random.choice([500000, -10000])  # Income outliers
            
            else:
                # Mock data for demo mode
                X = [[random.random() for _ in range(12)] for _ in range(100)]
                y = [random.choice([0, 1]) for _ in range(100)]
            
            # Data validation
            self.logger.info("🔍 Validating data quality")
            
            # Basic data validation
            n_samples = len(X)
            n_features = len(X[0]) if X else 0
            n_missing = sum(1 for sample in X for value in sample if value is None)
            missing_rate = n_missing / (n_samples * n_features) if n_samples > 0 else 0
            
            # Class distribution
            class_counts = {0: y.count(0), 1: y.count(1)}
            class_balance = min(class_counts.values()) / max(class_counts.values()) if max(class_counts.values()) > 0 else 0
            
            # Log validation metrics
            validation_metrics = {
                "n_samples": n_samples,
                "n_features": n_features,
                "missing_rate": missing_rate,
                "class_balance": class_balance
            }
            
            self.logger.log_metrics(validation_metrics)
            
            # Validation checks
            if n_samples < 100:
                raise DataError("Insufficient data: need at least 100 samples")
            
            if missing_rate > 0.1:
                self.logger.warning(f"High missing rate: {missing_rate:.1%}")
            
            if class_balance < 0.1:
                self.logger.warning(f"Severe class imbalance: {class_balance:.2f}")
            
            self.pipeline_state["completed_stages"].append("data_loading")
            self.pipeline_state["metrics"].update(validation_metrics)
            
            self.logger.info(f"✅ Data loaded: {n_samples} samples, {n_features} features")
            return X, y
            
        except Exception as e:
            self.pipeline_state["errors"].append(f"Data loading failed: {str(e)}")
            self.logger.error(f"❌ Data loading failed: {e}")
            raise DataError(f"Failed to load data: {e}") from e
    
    @performance_monitor
    def preprocess_data(self, X: List[List[float]], y: List[int]) -> Tuple[List[List[float]], List[int]]:
        """Comprehensive data preprocessing pipeline."""
        self.pipeline_state["current_stage"] = "preprocessing"
        self.logger.info("🔧 Starting data preprocessing")
        
        try:
            # Create preprocessing pipeline
            processed_X = []
            processed_y = y.copy()
            
            # Step 1: Handle missing values
            self.logger.info("1️⃣ Handling missing values")
            
            # Calculate column-wise statistics for imputation
            n_features = len(X[0]) if X else 0
            feature_stats = []
            
            for feature_idx in range(n_features):
                feature_values = [sample[feature_idx] for sample in X if sample[feature_idx] is not None]
                
                if feature_values:
                    if feature_idx in [0, 1, 2, 3, 4, 5, 7, 9]:  # Numeric features
                        median_val = sorted(feature_values)[len(feature_values) // 2]
                        feature_stats.append(("median", median_val))
                    else:  # Categorical features
                        mode_val = max(set(feature_values), key=feature_values.count)
                        feature_stats.append(("mode", mode_val))
                else:
                    feature_stats.append(("default", 0))
            
            # Apply imputation
            imputed_count = 0
            for sample in X:
                imputed_sample = []
                for feature_idx, value in enumerate(sample):
                    if value is None:
                        impute_method, impute_value = feature_stats[feature_idx]
                        imputed_sample.append(impute_value)
                        imputed_count += 1
                    else:
                        imputed_sample.append(value)
                processed_X.append(imputed_sample)
            
            self.logger.info(f"   Imputed {imputed_count} missing values")
            
            # Step 2: Outlier handling
            self.logger.info("2️⃣ Handling outliers")
            
            outlier_count = 0
            for feature_idx in [0, 1, 2, 3, 4, 5]:  # Numeric features only
                feature_values = [sample[feature_idx] for sample in processed_X]
                
                # Calculate IQR bounds
                sorted_values = sorted(feature_values)
                n = len(sorted_values)
                q1 = sorted_values[n // 4]
                q3 = sorted_values[3 * n // 4]
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Cap outliers
                for sample in processed_X:
                    if sample[feature_idx] < lower_bound:
                        sample[feature_idx] = lower_bound
                        outlier_count += 1
                    elif sample[feature_idx] > upper_bound:
                        sample[feature_idx] = upper_bound
                        outlier_count += 1
            
            self.logger.info(f"   Capped {outlier_count} outliers")
            
            # Step 3: Feature scaling
            self.logger.info("3️⃣ Scaling features")
            
            # Calculate scaling parameters for numeric features
            scaling_params = []
            for feature_idx in range(n_features):
                if feature_idx in [0, 1, 2, 3, 4, 5, 7, 9, 10, 11]:  # Numeric features
                    feature_values = [sample[feature_idx] for sample in processed_X]
                    mean_val = sum(feature_values) / len(feature_values)
                    
                    variance = sum((x - mean_val) ** 2 for x in feature_values) / (len(feature_values) - 1)
                    std_val = variance ** 0.5
                    
                    scaling_params.append(("standard", mean_val, std_val))
                else:
                    scaling_params.append(("none", 0, 1))
            
            # Apply scaling
            for sample in processed_X:
                for feature_idx, (method, param1, param2) in enumerate(scaling_params):
                    if method == "standard" and param2 > 0:
                        sample[feature_idx] = (sample[feature_idx] - param1) / param2
            
            # Step 4: Feature engineering
            self.logger.info("4️⃣ Engineering features")
            
            # Add engineered features
            for sample in processed_X:
                # Age-income interaction
                age_income_interaction = sample[0] * sample[1] / 10000
                sample.append(age_income_interaction)
                
                # Customer loyalty score
                loyalty_score = sample[4] * 0.3 + sample[5] * 0.01  # years + transactions
                sample.append(loyalty_score)
                
                # Risk score
                risk_score = sample[8] * 0.1 - sample[7] * 0.2  # tickets - engagement
                sample.append(risk_score)
            
            preprocessing_metrics = {
                "imputed_values": imputed_count,
                "capped_outliers": outlier_count,
                "original_features": n_features,
                "engineered_features": 3,
                "final_features": n_features + 3
            }
            
            self.logger.log_metrics(preprocessing_metrics)
            self.pipeline_state["completed_stages"].append("preprocessing")
            self.pipeline_state["metrics"].update(preprocessing_metrics)
            
            self.logger.info(f"✅ Preprocessing completed: "
                           f"{len(processed_X)} samples, {len(processed_X[0])} features")
            
            return processed_X, processed_y
            
        except Exception as e:
            self.pipeline_state["errors"].append(f"Preprocessing failed: {str(e)}")
            self.logger.error(f"❌ Preprocessing failed: {e}")
            raise
    
    @time_it
    def train_and_evaluate_models(self, X: List[List[float]], y: List[int]) -> Dict[str, Any]:
        """Train multiple models and select the best one."""
        self.pipeline_state["current_stage"] = "model_training"
        self.logger.info("🤖 Training and evaluating models")
        
        try:
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            self.logger.info(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Model configurations
            model_configs = [
                {
                    "name": "Random Forest",
                    "type": "RandomForestClassifier",
                    "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42}
                },
                {
                    "name": "Gradient Boosting",
                    "type": "GradientBoostingClassifier", 
                    "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}
                },
                {
                    "name": "Logistic Regression",
                    "type": "LogisticRegression",
                    "params": {"max_iter": 1000, "random_state": 42}
                }
            ]
            
            model_results = {}
            
            # Train each model
            for config in model_configs:
                model_name = config["name"]
                self.logger.info(f"🎯 Training {model_name}")
                
                start_time = time.time()
                
                # Simulate model training with realistic performance
                training_time = random.uniform(0.5, 2.0)
                time.sleep(training_time)
                
                # Generate realistic performance metrics
                base_accuracy = 0.80 + random.uniform(0, 0.12)
                accuracy = min(0.94, max(0.75, base_accuracy))
                
                precision = accuracy + random.uniform(-0.03, 0.03)
                recall = accuracy + random.uniform(-0.03, 0.03)
                
                # Ensure realistic bounds
                precision = min(0.98, max(0.70, precision))
                recall = min(0.98, max(0.70, recall))
                
                f1_score = 2 * (precision * recall) / (precision + recall)
                
                # Additional metrics
                auc_score = accuracy + random.uniform(0, 0.05)
                auc_score = min(0.99, max(0.75, auc_score))
                
                model_metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "auc_score": auc_score,
                    "training_time": training_time,
                    "model_type": config["type"]
                }
                
                model_results[model_name] = model_metrics
                
                # Log model metrics
                self.logger.log_params({
                    f"{model_name}_params": config["params"],
                    f"{model_name}_type": config["type"]
                })
                
                self.logger.log_metrics({
                    f"{model_name}_{k}": v for k, v in model_metrics.items()
                })
                
                self.logger.info(f"   {model_name}: Accuracy={accuracy:.3f}, F1={f1_score:.3f}")
            
            # Select best model
            best_model_name = max(model_results.keys(), 
                                key=lambda k: model_results[k]['f1_score'])
            best_model_metrics = model_results[best_model_name]
            
            # Cross-validation for best model
            self.logger.info(f"🔄 Cross-validating best model: {best_model_name}")
            
            cv_scores = []
            for fold in range(5):
                # Simulate CV score with some variance
                cv_score = best_model_metrics["accuracy"] + random.gauss(0, 0.02)
                cv_score = max(0.6, min(0.99, cv_score))
                cv_scores.append(cv_score)
            
            cv_mean = sum(cv_scores) / len(cv_scores)
            cv_std = (sum((s - cv_mean) ** 2 for s in cv_scores) / (len(cv_scores) - 1)) ** 0.5
            
            # Model selection results
            model_selection_results = {
                "models_trained": len(model_configs),
                "best_model": best_model_name,
                "best_model_metrics": best_model_metrics,
                "all_model_results": model_results,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "cv_scores": cv_scores
            }
            
            self.logger.log_metrics({
                "best_model_f1": best_model_metrics["f1_score"],
                "best_model_accuracy": best_model_metrics["accuracy"],
                "cv_mean": cv_mean,
                "cv_std": cv_std
            })
            
            self.pipeline_state["completed_stages"].append("model_training")
            self.pipeline_state["metrics"].update({
                "best_model": best_model_name,
                "best_f1_score": best_model_metrics["f1_score"]
            })
            
            self.logger.info(f"✅ Best model: {best_model_name} "
                           f"(F1: {best_model_metrics['f1_score']:.3f})")
            
            return model_selection_results
            
        except Exception as e:
            self.pipeline_state["errors"].append(f"Model training failed: {str(e)}")
            self.logger.error(f"❌ Model training failed: {e}")
            raise ModelError(f"Model training failed: {e}") from e
    
    @validate_outputs(types=dict)
    def finalize_pipeline(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize pipeline with model registration and reporting."""
        self.pipeline_state["current_stage"] = "finalization"
        self.logger.info("🏁 Finalizing pipeline")
        
        try:
            # Model registration simulation
            best_model_name = model_results["best_model"]
            best_metrics = model_results["best_model_metrics"]
            
            model_id = f"{best_model_name.lower().replace(' ', '_')}_v1"
            
            # Register model
            registration_metadata = {
                "model_id": model_id,
                "model_name": best_model_name,
                "version": "1.0.0",
                "performance_metrics": best_metrics,
                "cross_validation": {
                    "cv_mean": model_results["cv_mean"],
                    "cv_std": model_results["cv_std"]
                },
                "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pipeline_id": self.experiment_id,
                "data_hash": "abc123def456",  # Would be actual data hash
                "feature_count": self.pipeline_state["metrics"].get("final_features", 0)
            }
            
            self.logger.info(f"📦 Registered model: {model_id}")
            
            # Pipeline summary
            end_time = time.time()
            total_duration = end_time - self.pipeline_state["started_at"]
            
            pipeline_summary = {
                "pipeline_id": self.experiment_id,
                "project_name": self.project_name,
                "status": "completed" if not self.pipeline_state["errors"] else "completed_with_warnings",
                "total_duration": total_duration,
                "stages_completed": self.pipeline_state["completed_stages"],
                "errors": self.pipeline_state["errors"],
                "final_metrics": self.pipeline_state["metrics"],
                "model_registration": registration_metadata,
                "best_model_performance": best_metrics,
                "models_compared": model_results["models_trained"]
            }
            
            # Log final metrics
            self.logger.log_metrics({
                "pipeline_duration": total_duration,
                "stages_completed": len(self.pipeline_state["completed_stages"]),
                "total_errors": len(self.pipeline_state["errors"]),
                "final_model_f1": best_metrics["f1_score"]
            })
            
            self.pipeline_state["completed_stages"].append("finalization")
            
            self.logger.info(f"✅ Pipeline completed successfully in {total_duration:.1f}s")
            self.logger.info(f"🏆 Best model: {best_model_name} "
                           f"(F1: {best_metrics['f1_score']:.3f})")
            
            return pipeline_summary
            
        except Exception as e:
            self.pipeline_state["errors"].append(f"Finalization failed: {str(e)}")
            self.logger.error(f"❌ Pipeline finalization failed: {e}")
            raise
    
    def run_complete_pipeline(self, data_path: str) -> Dict[str, Any]:
        """Execute the complete end-to-end ML pipeline."""
        self.logger.info(f"🚀 Starting complete ML pipeline: {self.project_name}")
        
        try:
            # Stage 1: Data loading and validation
            X, y = self.load_and_validate_data(data_path)
            
            # Stage 2: Data preprocessing
            X_processed, y_processed = self.preprocess_data(X, y)
            
            # Stage 3: Model training and evaluation
            model_results = self.train_and_evaluate_models(X_processed, y_processed)
            
            # Stage 4: Pipeline finalization
            pipeline_summary = self.finalize_pipeline(model_results)
            
            return pipeline_summary
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            
            # Generate failure report
            failure_summary = {
                "pipeline_id": self.experiment_id,
                "status": "failed",
                "error": str(e),
                "completed_stages": self.pipeline_state["completed_stages"],
                "all_errors": self.pipeline_state["errors"],
                "duration": time.time() - self.pipeline_state["started_at"]
            }
            
            return failure_summary


def end_to_end_pipeline_example():
    """Demonstrate complete end-to-end ML pipeline."""
    print("🚀 End-to-End ML Pipeline")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Complete end-to-end ML pipeline:
from refunc.integration import MLPipelineOrchestrator

# Initialize pipeline orchestrator
pipeline = MLPipelineOrchestrator(
    project_name="customer_churn_prediction",
    experiment_id="exp_001"
)

# Run complete pipeline
results = pipeline.run_complete_pipeline("./data/customer_data.csv")

# Pipeline includes:
# 1. Data loading and validation
# 2. Comprehensive preprocessing
# 3. Model training and comparison
# 4. Evaluation and selection
# 5. Registration and deployment prep

print(f"Pipeline Status: {results['status']}")
print(f"Best Model: {results['best_model_performance']}")
print(f"Duration: {results['total_duration']:.1f}s")
        """)
        return
    
    print("🔄 Running complete ML pipeline integration:")
    
    # Initialize pipeline
    project_name = "customer_segmentation"
    experiment_id = f"exp_{int(time.time())}"
    
    print(f"   📋 Project: {project_name}")
    print(f"   🔬 Experiment: {experiment_id}")
    
    # Create pipeline orchestrator
    pipeline = MLPipelineOrchestrator(project_name, experiment_id)
    
    # Run complete pipeline
    print(f"\n   ⚡ Executing pipeline stages:")
    
    # Simulate data path
    data_path = "./data/synthetic_customer_data.csv"
    
    try:
        # Execute complete pipeline
        results = pipeline.run_complete_pipeline(data_path)
        
        # Display results
        print(f"\n   📊 Pipeline Results:")
        print(f"     Status: {results['status']}")
        print(f"     Duration: {results['total_duration']:.1f} seconds")
        print(f"     Stages Completed: {len(results['stages_completed'])}")
        
        if results['status'] == 'completed':
            best_model = results['model_registration']
            print(f"     Best Model: {best_model['model_name']}")
            print(f"     Model F1-Score: {best_model['performance_metrics']['f1_score']:.3f}")
            print(f"     Model Accuracy: {best_model['performance_metrics']['accuracy']:.3f}")
            print(f"     Models Compared: {results['models_compared']}")
            
            # Display key metrics
            final_metrics = results['final_metrics']
            print(f"\n   📈 Key Metrics:")
            print(f"     Samples Processed: {final_metrics.get('n_samples', 'N/A')}")
            print(f"     Final Features: {final_metrics.get('final_features', 'N/A')}")
            print(f"     Missing Values Imputed: {final_metrics.get('imputed_values', 'N/A')}")
            print(f"     Outliers Handled: {final_metrics.get('capped_outliers', 'N/A')}")
            
            print(f"\n   ✅ Pipeline completed successfully!")
            
        else:
            print(f"     Errors: {len(results.get('all_errors', []))}")
            if results.get('all_errors'):
                print(f"     Last Error: {results['all_errors'][-1]}")
    
    except Exception as e:
        print(f"   ❌ Pipeline execution failed: {e}")
        return


def integration_patterns_example():
    """Demonstrate integration patterns across refunc modules."""
    print("\n🔗 Integration Patterns")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Integration patterns across refunc modules:

# 1. Logging + Decorators + Error Handling
from refunc.logging import MLLogger
from refunc.decorators import time_it, memory_profile
from refunc.exceptions import retry_on_failure

logger = MLLogger("integration_demo")

@time_it(logger=logger)
@memory_profile(logger=logger)
@retry_on_failure(max_attempts=3, logger=logger)
def robust_ml_function(data):
    # Function automatically logs timing, memory, and retry attempts
    return process_data(data)

# 2. Data Science + ML + Validation
from refunc.data_science import TransformationPipeline
from refunc.ml import ModelEvaluator
from refunc.decorators import validate_inputs

@validate_inputs(types={'X': list, 'y': list})
def integrated_ml_workflow(X, y):
    # Data preprocessing
    pipeline = TransformationPipeline()
    X_processed = pipeline.fit_transform(X)
    
    # Model evaluation
    evaluator = ModelEvaluator()
    results = evaluator.evaluate(model, X_processed, y)
    
    return results

# 3. Utils + Math/Stats + Logging
from refunc.utils import cache_result, FileHandler
from refunc.math_stats import describe
from refunc.logging import MLLogger

@cache_result(ttl_seconds=3600)
def cached_statistical_analysis(file_path):
    handler = FileHandler()
    data = handler.load_auto(file_path)
    
    stats = describe(data)
    logger.log_metrics(stats.to_dict())
    
    return stats
        """)
        return
    
    print("🧩 Testing cross-module integration patterns:")
    
    # Pattern 1: Comprehensive function decoration
    print("   1️⃣ Comprehensive Function Decoration:")
    
    @time_it
    @memory_profile 
    @retry_on_failure(max_attempts=2)
    @validate_inputs(types={'data': list, 'threshold': (int, float)})
    @validate_outputs(types=dict)
    def comprehensive_analysis(data: List[float], threshold: float = 0.5) -> Dict[str, Any]:
        """Function with multiple decorators demonstrating integration."""
        # Simulate some processing time
        time.sleep(0.1)
        
        # Basic statistical analysis
        if not data:
            return {"error": "Empty data"}
        
        n = len(data)
        mean_val = sum(data) / n
        variance = sum((x - mean_val) ** 2 for x in data) / (n - 1) if n > 1 else 0
        std_dev = variance ** 0.5
        
        # Apply threshold
        filtered_data = [x for x in data if x > threshold]
        
        return {
            "original_count": n,
            "filtered_count": len(filtered_data),
            "mean": mean_val,
            "std_dev": std_dev,
            "threshold_used": threshold,
            "filter_ratio": len(filtered_data) / n if n > 0 else 0
        }
    
    # Test comprehensive function
    test_data = [random.gauss(10, 3) for _ in range(100)]
    result = comprehensive_analysis(test_data, 8.0)
    
    print(f"     Function executed with multiple decorators")
    print(f"     Original samples: {result['original_count']}")
    print(f"     Filtered samples: {result['filtered_count']}")  
    print(f"     Filter ratio: {result['filter_ratio']:.2f}")
    
    # Pattern 2: Error handling with logging
    print(f"\n   2️⃣ Error Handling with Logging:")
    
    logger = MLLogger("integration_pattern_demo")
    
    @retry_on_failure(max_attempts=3)
    def flaky_data_operation(success_rate: float = 0.7):
        """Simulate a flaky operation that sometimes fails."""
        if random.random() < success_rate:
            # Success case
            result = {"status": "success", "data_processed": 1000}
            logger.info("Data operation completed successfully")
            return result
        else:
            # Failure case
            logger.error("Data operation failed - will retry")
            raise DataError("Simulated data processing failure")
    
    try:
        operation_result = flaky_data_operation(0.4)  # Low success rate
        print(f"     Operation result: {operation_result['status']}")
        print(f"     Data processed: {operation_result['data_processed']}")
    except DataError as e:
        print(f"     Operation failed after retries: {e}")
    
    # Pattern 3: Cached statistical analysis
    print(f"\n   3️⃣ Cached Statistical Analysis:")
    
    @cache_result(ttl_seconds=30)
    def cached_stats_analysis(dataset_id: str) -> Dict[str, float]:
        """Cached statistical analysis to avoid recomputation."""
        print(f"       Computing statistics for {dataset_id}...")
        time.sleep(0.1)  # Simulate computation time
        
        # Generate mock dataset
        data = [random.gauss(50, 15) for _ in range(200)]
        
        # Statistical analysis
        n = len(data)
        mean_val = sum(data) / n
        variance = sum((x - mean_val) ** 2 for x in data) / (n - 1)
        std_dev = variance ** 0.5
        
        # Quartiles
        sorted_data = sorted(data)
        q1 = sorted_data[n // 4]
        q3 = sorted_data[3 * n // 4]
        
        stats = {
            "count": n,
            "mean": mean_val,
            "std_dev": std_dev,
            "min": min(data),
            "max": max(data),
            "q1": q1,
            "q3": q3
        }
        
        # Log the analysis
        logger.log_metrics({f"stats_{dataset_id}_{k}": v for k, v in stats.items()})
        
        return stats
    
    # Test caching effectiveness
    datasets = ["dataset_A", "dataset_B", "dataset_A"]  # dataset_A repeated
    
    start_time = time.time()
    for dataset_id in datasets:
        print(f"     Analyzing {dataset_id}:")
        stats = cached_stats_analysis(dataset_id)
        print(f"       Mean: {stats['mean']:.1f}, Std: {stats['std_dev']:.1f}")
    
    total_time = time.time() - start_time
    print(f"     Total analysis time: {total_time:.2f}s (with caching speedup)")
    
    # Pattern 4: Complete workflow integration
    print(f"\n   4️⃣ Complete Workflow Integration:")
    
    class IntegratedWorkflow:
        """Demonstrates complete integration of all refunc components."""
        
        def __init__(self, workflow_name: str):
            self.workflow_name = workflow_name
            self.logger = MLLogger(workflow_name)
            self.file_handler = FileHandler(cache_enabled=True)
            
        @time_it
        @memory_profile
        def execute_workflow(self, data: List[float]) -> Dict[str, Any]:
            """Execute integrated workflow with all components."""
            self.logger.info(f"🚀 Starting {self.workflow_name}")
            
            # Step 1: Data validation
            if not data or len(data) < 10:
                raise ValidationError("Insufficient data for analysis")
            
            # Step 2: Statistical analysis
            stats = self.analyze_data(data)
            
            # Step 3: Data transformation
            transformed_data = self.transform_data(data, stats)
            
            # Step 4: Results compilation
            results = {
                "workflow_name": self.workflow_name,
                "original_stats": stats,
                "transformed_count": len(transformed_data),
                "transformation_applied": "z-score normalization",
                "completion_status": "success"
            }
            
            self.logger.log_metrics({
                "workflow_data_count": len(data),
                "workflow_mean": stats["mean"],
                "workflow_std": stats["std_dev"]
            })
            
            self.logger.info(f"✅ {self.workflow_name} completed successfully")
            return results
        
        def analyze_data(self, data: List[float]) -> Dict[str, float]:
            """Analyze data with error handling."""
            try:
                n = len(data)
                mean_val = sum(data) / n
                variance = sum((x - mean_val) ** 2 for x in data) / (n - 1)
                std_dev = variance ** 0.5
                
                return {
                    "count": n,
                    "mean": mean_val,
                    "std_dev": std_dev,
                    "min": min(data),
                    "max": max(data)
                }
            except Exception as e:
                self.logger.error(f"Data analysis failed: {e}")
                raise
        
        def transform_data(self, data: List[float], stats: Dict[str, float]) -> List[float]:
            """Transform data using z-score normalization."""
            mean_val = stats["mean"]
            std_dev = stats["std_dev"]
            
            if std_dev == 0:
                return data  # No transformation if no variance
            
            return [(x - mean_val) / std_dev for x in data]
    
    # Test integrated workflow
    workflow = IntegratedWorkflow("demo_workflow")
    test_workflow_data = [random.gauss(100, 20) for _ in range(150)]
    
    workflow_results = workflow.execute_workflow(test_workflow_data)
    
    print(f"     Workflow: {workflow_results['workflow_name']}")
    print(f"     Status: {workflow_results['completion_status']}")
    print(f"     Original mean: {workflow_results['original_stats']['mean']:.1f}")
    print(f"     Transformed samples: {workflow_results['transformed_count']}")
    
    print(f"\n   ✅ Integration patterns demonstration completed!")


def main():
    """Run complete end-to-end pipeline examples."""
    print("🚀 Refunc End-to-End Pipeline Examples")
    print("=" * 65)
    
    if not REFUNC_AVAILABLE:
        print("ℹ️  Running in demo mode (showing code structure)")
    else:
        print("ℹ️  Running with full Refunc functionality")
    
    print()
    
    # Set random seed for reproducible examples
    random.seed(42)
    
    # Run examples
    end_to_end_pipeline_example()
    integration_patterns_example()
    
    print("\n✅ End-to-end pipeline examples completed!")
    print("\n📖 Summary:")
    print("- Complete ML pipeline orchestration with all refunc modules")
    print("- Error handling and recovery throughout the workflow")
    print("- Comprehensive logging and monitoring integration")
    print("- Production-ready pipeline design patterns")
    print("- Cross-module integration demonstrations")
    
    print("\n🎯 Use these patterns to:")
    print("- Build robust, production-ready ML pipelines")
    print("- Integrate multiple refunc components seamlessly")
    print("- Implement comprehensive monitoring and logging")
    print("- Handle errors gracefully with recovery strategies")


if __name__ == "__main__":
    main()