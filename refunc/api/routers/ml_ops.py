"""
Machine Learning operations router.

Provides endpoints for:
- Model evaluation
- Feature engineering
- Model training utilities
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import io
import json
from pydantic import BaseModel

from ...ml.evaluation import ModelEvaluator
from ...ml.features import FeatureEngineer
from ..models.responses import ProcessingRequest

router = APIRouter()

class ModelEvaluationRequest(BaseModel):
    """Request model for model evaluation."""
    y_true: List[float]
    y_pred: List[float]
    task_type: str  # "classification" or "regression"
    metrics: Optional[List[str]] = None
    options: Optional[Dict[str, Any]] = None

class FeatureEngineeringRequest(BaseModel):
    """Request model for feature engineering."""
    data: List[Dict[str, Any]]
    operations: List[str]  # List of feature engineering operations
    options: Optional[Dict[str, Any]] = None

class TrainingDataRequest(BaseModel):
    """Request model for training data preparation."""
    features: List[Dict[str, Any]]
    target: List[float]
    test_size: float = 0.2
    random_state: Optional[int] = None
    stratify: bool = False

@router.post("/evaluate")
async def evaluate_model(request: ModelEvaluationRequest) -> Dict[str, Any]:
    """
    Evaluate model performance with various metrics.
    
    Args:
        request: Model evaluation configuration
    
    Returns:
        Evaluation metrics and results
    """
    try:
        y_true = np.array(request.y_true)
        y_pred = np.array(request.y_pred)
        
        evaluator = ModelEvaluator()
        options = request.options or {}
        
        if request.task_type == "classification":
            # Convert to integer labels if needed
            if y_true.dtype != int:
                y_true = y_true.astype(int)
            if y_pred.dtype != int:
                y_pred = y_pred.astype(int)
            
            result = evaluator.evaluate_classification(
                y_true=y_true,
                y_pred=y_pred,
                metrics=request.metrics,
                **options
            )
        
        elif request.task_type == "regression":
            result = evaluator.evaluate_regression(
                y_true=y_true,
                y_pred=y_pred,
                metrics=request.metrics,
                **options
            )
        
        else:
            raise ValueError(f"Unknown task type: {request.task_type}")
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        return {
            "task_type": request.task_type,
            "metrics": convert_numpy(result)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model evaluation failed: {str(e)}")

@router.post("/features/engineer")
async def engineer_features(request: FeatureEngineeringRequest) -> Dict[str, Any]:
    """
    Apply feature engineering operations to data.
    
    Args:
        request: Feature engineering configuration
    
    Returns:
        Engineered features and transformation information
    """
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Create feature engineer
        engineer = FeatureEngineer()
        options = request.options or {}
        
        # Apply requested operations
        results = {}
        transformed_df = df.copy()
        
        for operation in request.operations:
            if operation == "polynomial":
                degree = options.get("polynomial_degree", 2)
                poly_features = engineer.create_polynomial_features(
                    df.select_dtypes(include=[np.number]), 
                    degree=degree
                )
                results["polynomial"] = {
                    "shape": poly_features.shape,
                    "feature_names": getattr(engineer, "_poly_feature_names", [])
                }
            
            elif operation == "interaction":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    interaction_features = engineer.create_interaction_features(
                        df[numeric_cols]
                    )
                    results["interaction"] = {
                        "shape": interaction_features.shape,
                        "feature_names": getattr(engineer, "_interaction_feature_names", [])
                    }
            
            elif operation == "binning":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    n_bins = options.get("n_bins", 5)
                    binned = engineer.create_binned_features(df[[col]], n_bins=n_bins)
                    transformed_df[f"{col}_binned"] = binned
                
                results["binning"] = {
                    "columns_binned": list(numeric_cols),
                    "n_bins": options.get("n_bins", 5)
                }
            
            elif operation == "encoding":
                categorical_cols = df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    method = options.get("encoding_method", "onehot")
                    encoded = engineer.encode_categorical(df[[col]], method=method)
                    # Add encoded columns to transformed_df
                    if encoded.shape[1] > 1:
                        for i in range(encoded.shape[1]):
                            transformed_df[f"{col}_encoded_{i}"] = encoded[:, i]
                    else:
                        transformed_df[f"{col}_encoded"] = encoded.flatten()
                
                results["encoding"] = {
                    "columns_encoded": list(categorical_cols),
                    "method": options.get("encoding_method", "onehot")
                }
            
            elif operation == "scaling":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                method = options.get("scaling_method", "standard")
                scaled = engineer.scale_features(df[numeric_cols], method=method)
                for i, col in enumerate(numeric_cols):
                    transformed_df[f"{col}_scaled"] = scaled[:, i]
                
                results["scaling"] = {
                    "columns_scaled": list(numeric_cols),
                    "method": method
                }
        
        return {
            "operations_applied": request.operations,
            "original_shape": list(df.shape),
            "transformed_shape": list(transformed_df.shape),
            "results": results,
            "sample_data": transformed_df.head(5).to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature engineering failed: {str(e)}")

@router.post("/features/engineer-file")
async def engineer_features_from_file(
    file: UploadFile = File(...),
    operations: str = Form(...),
    options: Optional[str] = Form(None)
) -> Dict[str, Any]:
    """
    Apply feature engineering to uploaded data file.
    
    Args:
        file: Data file (CSV, JSON, Excel)
        operations: JSON string of operations to apply
        options: Optional JSON string of operation options
    
    Returns:
        Engineered features and transformation information
    """
    try:
        # Read the uploaded file
        content = await file.read()
        df = _read_file_content(content, file.filename)
        
        # Parse operations and options
        operation_list = json.loads(operations)
        option_dict = json.loads(options) if options else {}
        
        # Create request object
        request = FeatureEngineeringRequest(
            data=df.to_dict('records'),
            operations=operation_list,
            options=option_dict
        )
        
        # Apply feature engineering
        return await engineer_features(request)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File feature engineering failed: {str(e)}")

@router.post("/training/split")
async def split_training_data(request: TrainingDataRequest) -> Dict[str, Any]:
    """
    Split data into training and test sets.
    
    Args:
        request: Training data split configuration
    
    Returns:
        Split data information and statistics
    """
    try:
        # Convert to arrays
        X = pd.DataFrame(request.features)
        y = np.array(request.target)
        
        # Import train_test_split
        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            raise HTTPException(
                status_code=500, 
                detail="scikit-learn is required for data splitting"
            )
        
        # Prepare split parameters
        split_params = {
            "test_size": request.test_size,
            "random_state": request.random_state
        }
        
        if request.stratify and len(np.unique(y)) > 1:
            split_params["stratify"] = y
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, **split_params)
        
        return {
            "split_info": {
                "total_samples": len(X),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "test_size": request.test_size,
                "stratified": request.stratify
            },
            "train_data": {
                "features": X_train.to_dict('records')[:10],  # Sample only
                "target": y_train[:10].tolist(),
                "shape": list(X_train.shape)
            },
            "test_data": {
                "features": X_test.to_dict('records')[:10],  # Sample only
                "target": y_test[:10].tolist(),
                "shape": list(X_test.shape)
            },
            "target_distribution": {
                "train": dict(zip(*np.unique(y_train, return_counts=True))),
                "test": dict(zip(*np.unique(y_test, return_counts=True)))
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data splitting failed: {str(e)}")

@router.get("/metrics/available")
async def get_available_metrics() -> Dict[str, Any]:
    """
    Get available evaluation metrics for different task types.
    
    Returns:
        Dictionary of available metrics by task type
    """
    return {
        "classification": {
            "accuracy": "Overall accuracy score",
            "precision": "Precision score (macro/micro/weighted average)",
            "recall": "Recall score (macro/micro/weighted average)",
            "f1": "F1 score (macro/micro/weighted average)",
            "roc_auc": "ROC AUC score",
            "confusion_matrix": "Confusion matrix",
            "classification_report": "Detailed classification report"
        },
        "regression": {
            "mse": "Mean Squared Error",
            "rmse": "Root Mean Squared Error",
            "mae": "Mean Absolute Error",
            "r2": "R-squared score",
            "explained_variance": "Explained variance score",
            "median_absolute_error": "Median absolute error"
        },
        "feature_engineering": {
            "polynomial": "Create polynomial features",
            "interaction": "Create interaction features",
            "binning": "Create binned features",
            "encoding": "Encode categorical features",
            "scaling": "Scale numerical features"
        },
        "encoding_methods": ["onehot", "label", "ordinal", "target"],
        "scaling_methods": ["standard", "minmax", "robust", "normalizer"]
    }

def _read_file_content(content: bytes, filename: str) -> pd.DataFrame:
    """Helper function to read file content into DataFrame."""
    try:
        if filename.endswith('.csv'):
            return pd.read_csv(io.BytesIO(content))
        elif filename.endswith('.json'):
            return pd.read_json(io.BytesIO(content))
        elif filename.endswith(('.xlsx', '.xls')):
            return pd.read_excel(io.BytesIO(content))
        elif filename.endswith('.parquet'):
            return pd.read_parquet(io.BytesIO(content))
        else:
            # Try CSV as default
            return pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise ValueError(f"Unable to read file {filename}: {str(e)}")