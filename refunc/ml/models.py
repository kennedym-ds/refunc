"""
Machine Learning model management utilities.

This module provides comprehensive model management functionality including
base model classes, model registry, persistence, versioning, and metadata tracking
for organized ML model lifecycle management.
"""

import json
import pickle
import joblib
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Type
from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..exceptions import RefuncError, ValidationError
from ..logging import get_logger


@dataclass
class ModelMetadata:
    """Metadata for a machine learning model."""
    
    name: str
    version: str
    description: str = ""
    created_at: Optional[datetime] = None
    model_type: str = ""
    parameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    tags: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.parameters is None:
            self.parameters = {}
        if self.metrics is None:
            self.metrics = {}
        if self.tags is None:
            self.tags = []


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.metadata = ModelMetadata(name=self.name, version="1.0")
        self._model = None
        self._is_fitted = False
        
    @abstractmethod
    def fit(self, X, y=None, **kwargs) -> 'BaseModel':
        """Fit the model to training data."""
        pass
        
    @abstractmethod
    def predict(self, X, **kwargs):
        """Make predictions on input data."""
        pass
        
    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted
        
    def save(self, path: Union[str, Path]) -> None:
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseModel':
        """Load model from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class SklearnModel(BaseModel):
    """Wrapper for scikit-learn models."""
    
    def __init__(self, model, name: Optional[str] = None):
        super().__init__(name)
        self._model = model
        self.metadata.model_type = f"sklearn.{model.__class__.__name__}"
        
    def fit(self, X, y=None, **kwargs) -> 'SklearnModel':
        """Fit the sklearn model."""
        self._model.fit(X, y, **kwargs)
        self._is_fitted = True
        return self
        
    def predict(self, X, **kwargs):
        """Make predictions using sklearn model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self._model.predict(X, **kwargs)
        
    def predict_proba(self, X, **kwargs):
        """Predict class probabilities if supported."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        if not hasattr(self._model, 'predict_proba'):
            raise AttributeError("Model does not support probability predictions")
        return self._model.predict_proba(X, **kwargs)


class ModelRegistry:
    """Registry for managing multiple models."""
    
    def __init__(self, storage_path: Optional[Union[str, Path]] = None):
        self.storage_path = Path(storage_path) if storage_path else Path("./models")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, BaseModel] = {}
        self._metadata_file = self.storage_path / "registry.json"
        self._load_metadata()
        
    def register(self, model: BaseModel, name: Optional[str] = None) -> str:
        """Register a model in the registry."""
        name = name or model.name
        model.name = name
        
        # Save model to disk
        model_path = self.storage_path / f"{name}.pkl"
        model.save(model_path)
        
        # Store in memory
        self._models[name] = model
        
        # Update metadata
        self._save_metadata()
        
        return name
        
    def get(self, name: str) -> BaseModel:
        """Get a model by name."""
        if name in self._models:
            return self._models[name]
            
        # Try to load from disk
        model_path = self.storage_path / f"{name}.pkl"
        if model_path.exists():
            model = BaseModel.load(model_path)
            self._models[name] = model
            return model
            
        raise KeyError(f"Model '{name}' not found in registry")
        
    def list_models(self) -> List[str]:
        """List all registered model names."""
        disk_models = [f.stem for f in self.storage_path.glob("*.pkl")]
        all_models = set(self._models.keys()) | set(disk_models)
        return sorted(all_models)
        
    def remove(self, name: str) -> None:
        """Remove a model from the registry."""
        if name in self._models:
            del self._models[name]
            
        model_path = self.storage_path / f"{name}.pkl"
        if model_path.exists():
            model_path.unlink()
            
        self._save_metadata()
        
    def _load_metadata(self) -> None:
        """Load registry metadata from disk."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, 'r') as f:
                    # Load metadata if needed
                    pass
            except Exception:
                pass
                
    def _save_metadata(self) -> None:
        """Save registry metadata to disk."""
        try:
            metadata = {
                "models": [name for name in self.list_models()],
                "last_updated": datetime.now().isoformat()
            }
            with open(self._metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception:
            pass


# Global registry instance
_default_registry = None

def get_default_registry() -> ModelRegistry:
    """Get or create the default model registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ModelRegistry()
    return _default_registry


# Convenience functions
def register_model(model: BaseModel, name: Optional[str] = None) -> str:
    """Register a model in the default registry."""
    return get_default_registry().register(model, name)


def get_model(name: str) -> BaseModel:
    """Get a model from the default registry."""
    return get_default_registry().get(name)


def list_models() -> List[str]:
    """List all models in the default registry."""
    return get_default_registry().list_models()