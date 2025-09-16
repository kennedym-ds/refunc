"""
Experiment tracking and management for ML workflows.

This module provides experiment tracking, metadata management,
and integration with popular ML experiment tracking platforms.
"""

import json
import time
import uuid
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    try:
        import mlflow
        import wandb
    except ImportError:
        pass
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
import pickle

from ..exceptions import RefuncError


@dataclass
class ExperimentMetadata:
    """Metadata for ML experiments."""
    
    experiment_id: str
    experiment_name: str
    run_id: str
    run_name: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    status: str = "created"  # created, running, completed, failed
    description: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    source_code: Optional[str] = None
    git_commit: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str, indent=2)


@dataclass
class MetricEntry:
    """Single metric entry with timestamp."""
    
    name: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    step: Optional[int] = None
    epoch: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ExperimentTracker:
    """
    Core experiment tracking functionality.
    
    Manages experiment lifecycle, parameters, metrics, and artifacts
    with automatic persistence and metadata management.
    """
    
    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        auto_save: bool = True,
        save_interval: float = 30.0  # seconds
    ):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd() / "experiments"
        self.auto_save = auto_save
        self.save_interval = save_interval
        
        # Current experiment state
        self._current_experiment: Optional[ExperimentMetadata] = None
        self._metrics_buffer: List[MetricEntry] = []
        self._last_save_time = 0.0
        
        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new experiment.
        
        Args:
            name: Experiment name
            description: Optional description
            tags: Optional tags
            parameters: Optional initial parameters
        
        Returns:
            Experiment ID
        """
        experiment_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        
        self._current_experiment = ExperimentMetadata(
            experiment_id=experiment_id,
            experiment_name=name,
            run_id=run_id,
            description=description,
            tags=tags or {},
            parameters=parameters or {},
        )
        
        # Get environment info
        self._current_experiment.environment = self._get_environment_info()
        
        # Get git info if available
        self._current_experiment.git_commit = self._get_git_commit()
        
        # Create experiment directory
        exp_dir = self.base_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save initial metadata
        self._save_metadata()
        
        return experiment_id
    
    def start_run(
        self,
        experiment_id: Optional[str] = None,
        run_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new run within an experiment.
        
        Args:
            experiment_id: Existing experiment ID (creates new if None)
            run_name: Optional run name
            parameters: Optional run parameters
        
        Returns:
            Run ID
        """
        if experiment_id:
            # Load existing experiment
            self.load_experiment(experiment_id)
            if not self._current_experiment:
                raise RefuncError(f"Experiment {experiment_id} not found")
        elif not self._current_experiment:
            # Create new experiment
            experiment_id = self.create_experiment(f"experiment_{int(time.time())}")
        
        # Create new run
        run_id = str(uuid.uuid4())
        
        if self._current_experiment:
            self._current_experiment.run_id = run_id
            self._current_experiment.run_name = run_name
            self._current_experiment.status = "running"
            self._current_experiment.updated_at = time.time()
            
            if parameters:
                self._current_experiment.parameters.update(parameters)
            
            # Clear metrics buffer for new run
            self._metrics_buffer.clear()
            
            # Save metadata
            self._save_metadata()
        
        return run_id
    
    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """Log experiment parameters."""
        if not self._current_experiment:
            raise RefuncError("No active experiment")
        
        self._current_experiment.parameters.update(parameters)
        self._current_experiment.updated_at = time.time()
        
        if self.auto_save:
            self._auto_save()
    
    def log_metric(
        self,
        name: str,
        value: Any,
        step: Optional[int] = None,
        epoch: Optional[int] = None
    ) -> None:
        """Log a single metric."""
        if not self._current_experiment:
            raise RefuncError("No active experiment")
        
        # Create metric entry
        metric_entry = MetricEntry(
            name=name,
            value=value,
            step=step,
            epoch=epoch
        )
        
        self._metrics_buffer.append(metric_entry)
        
        # Update current metrics (latest values)
        self._current_experiment.metrics[name] = value
        self._current_experiment.updated_at = time.time()
        
        if self.auto_save:
            self._auto_save()
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        epoch: Optional[int] = None
    ) -> None:
        """Log multiple metrics."""
        for name, value in metrics.items():
            self.log_metric(name, value, step, epoch)
    
    def log_artifact(
        self,
        file_path: Union[str, Path],
        artifact_name: Optional[str] = None,
        copy: bool = True
    ) -> str:
        """
        Log an artifact file.
        
        Args:
            file_path: Path to the artifact file
            artifact_name: Optional name for the artifact
            copy: Whether to copy the file to experiment directory
        
        Returns:
            Path to the stored artifact
        """
        if not self._current_experiment:
            raise RefuncError("No active experiment")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise RefuncError(f"Artifact file not found: {file_path}")
        
        artifact_name = artifact_name or file_path.name
        exp_dir = self.base_dir / self._current_experiment.experiment_id
        artifacts_dir = exp_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        if copy:
            # Copy file to artifacts directory
            artifact_path = artifacts_dir / artifact_name
            import shutil
            shutil.copy2(file_path, artifact_path)
            stored_path = str(artifact_path.relative_to(exp_dir))
        else:
            # Just store the reference
            stored_path = str(file_path.absolute())
        
        self._current_experiment.artifacts.append(stored_path)
        self._current_experiment.updated_at = time.time()
        
        if self.auto_save:
            self._auto_save()
        
        return stored_path
    
    def log_model(
        self,
        model: Any,
        model_name: str = "model",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a model object.
        
        Args:
            model: Model object to save
            model_name: Name for the model
            metadata: Optional model metadata
        
        Returns:
            Path to the saved model
        """
        if not self._current_experiment:
            raise RefuncError("No active experiment")
        
        exp_dir = self.base_dir / self._current_experiment.experiment_id
        models_dir = exp_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = models_dir / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata if provided
        if metadata:
            metadata_path = models_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        stored_path = str(model_path.relative_to(exp_dir))
        self._current_experiment.artifacts.append(stored_path)
        self._current_experiment.updated_at = time.time()
        
        if self.auto_save:
            self._auto_save()
        
        return stored_path
    
    def add_tag(self, key: str, value: str) -> None:
        """Add a tag to the current experiment."""
        if not self._current_experiment:
            raise RefuncError("No active experiment")
        
        self._current_experiment.tags[key] = value
        self._current_experiment.updated_at = time.time()
        
        if self.auto_save:
            self._auto_save()
    
    def set_status(self, status: str) -> None:
        """Set experiment status."""
        if not self._current_experiment:
            raise RefuncError("No active experiment")
        
        self._current_experiment.status = status
        self._current_experiment.updated_at = time.time()
        
        if self.auto_save:
            self._auto_save()
    
    def end_experiment(self, status: str = "completed") -> None:
        """End the current experiment."""
        if not self._current_experiment:
            return
        
        self._current_experiment.status = status
        self._current_experiment.updated_at = time.time()
        
        # Save final state
        self._save_metadata()
        self._save_metrics()
        
        self._current_experiment = None
        self._metrics_buffer.clear()
    
    def load_experiment(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """Load an existing experiment."""
        exp_dir = self.base_dir / experiment_id
        metadata_file = exp_dir / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            self._current_experiment = ExperimentMetadata(**data)
            return self._current_experiment
        
        except Exception as e:
            raise RefuncError(f"Failed to load experiment {experiment_id}: {e}")
    
    def list_experiments(self) -> List[ExperimentMetadata]:
        """List all experiments."""
        experiments = []
        
        for exp_dir in self.base_dir.iterdir():
            if exp_dir.is_dir():
                metadata_file = exp_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            data = json.load(f)
                        experiments.append(ExperimentMetadata(**data))
                    except Exception:
                        continue
        
        return experiments
    
    def get_experiment_metrics(self, experiment_id: str) -> List[MetricEntry]:
        """Get all metrics for an experiment."""
        exp_dir = self.base_dir / experiment_id
        metrics_file = exp_dir / "metrics.jsonl"
        
        if not metrics_file.exists():
            return []
        
        metrics = []
        try:
            with open(metrics_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    metrics.append(MetricEntry(**data))
        except Exception as e:
            raise RefuncError(f"Failed to load metrics for {experiment_id}: {e}")
        
        return metrics
    
    def get_current_experiment(self) -> Optional[ExperimentMetadata]:
        """Get the current experiment metadata."""
        return self._current_experiment
    
    def _save_metadata(self) -> None:
        """Save experiment metadata."""
        if not self._current_experiment:
            return
        
        exp_dir = self.base_dir / self._current_experiment.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_file = exp_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            f.write(self._current_experiment.to_json())
    
    def _save_metrics(self) -> None:
        """Save buffered metrics."""
        if not self._current_experiment or not self._metrics_buffer:
            return
        
        exp_dir = self.base_dir / self._current_experiment.experiment_id
        metrics_file = exp_dir / "metrics.jsonl"
        
        with open(metrics_file, 'a') as f:
            for metric in self._metrics_buffer:
                f.write(json.dumps(metric.to_dict(), default=str) + '\n')
        
        self._metrics_buffer.clear()
    
    def _auto_save(self) -> None:
        """Auto-save if interval has passed."""
        current_time = time.time()
        
        if (current_time - self._last_save_time) >= self.save_interval:
            self._save_metadata()
            self._save_metrics()
            self._last_save_time = current_time
    
    def _get_environment_info(self) -> Dict[str, str]:
        """Get environment information."""
        import platform
        import sys
        
        env_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'hostname': platform.node(),
        }
        
        # Add package versions if available
        try:
            import pkg_resources
            working_set = getattr(pkg_resources, 'working_set', None)
            if working_set is not None:
                installed_packages = []
                for d in working_set:
                    try:
                        installed_packages.append(f"{d.project_name}=={d.version}")
                    except AttributeError:
                        continue
                env_info['packages'] = ', '.join(installed_packages[:50])  # Limit to avoid huge metadata
        except (ImportError, AttributeError):
            pass
        
        return env_info
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        
        return None


class MLflowIntegration:
    """Integration with MLflow tracking."""
    
    def __init__(self, tracking_uri: Optional[str] = None):
        self.mlflow: Any = None
        try:
            import mlflow
            self.mlflow = mlflow
            
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                
            self.available = True
        except ImportError:
            self.available = False
    
    def start_run(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None
    ) -> Optional[str]:
        """Start MLflow run."""
        if not self.available:
            return None
        
        if experiment_name:
            self.mlflow.set_experiment(experiment_name)
        
        run = self.mlflow.start_run(run_name=run_name)
        return run.info.run_id if run else None
    
    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        if not self.available:
            return
        
        self.mlflow.log_params(parameters)
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None
    ) -> None:
        """Log metrics to MLflow."""
        if not self.available:
            return
        
        self.mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, file_path: Union[str, Path]) -> None:
        """Log artifact to MLflow."""
        if not self.available:
            return
        
        self.mlflow.log_artifact(str(file_path))
    
    def end_run(self) -> None:
        """End MLflow run."""
        if not self.available:
            return
        
        self.mlflow.end_run()


class WandBIntegration:
    """Integration with Weights & Biases."""
    
    def __init__(self, project: Optional[str] = None):
        self.wandb: Any = None
        try:
            import wandb
            self.wandb = wandb
            self.project = project
            self.available = True
        except ImportError:
            self.available = False
    
    def start_run(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Start W&B run."""
        if not self.available:
            return None
        
        run = self.wandb.init(
            project=project or self.project,
            name=name,
            config=config
        )
        
        return run.id if run else None
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None
    ) -> None:
        """Log metrics to W&B."""
        if not self.available:
            return
        
        log_data = dict(metrics)
        if step is not None:
            log_data['_step'] = step
        
        self.wandb.log(log_data)
    
    def log_artifact(self, file_path: Union[str, Path], name: Optional[str] = None) -> None:
        """Log artifact to W&B."""
        if not self.available:
            return
        
        artifact = self.wandb.Artifact(name or Path(file_path).name, type='file')
        artifact.add_file(str(file_path))
        self.wandb.log_artifact(artifact)
    
    def end_run(self) -> None:
        """End W&B run."""
        if not self.available:
            return
        
        self.wandb.finish()


class MultiTracker:
    """
    Multi-backend experiment tracker.
    
    Coordinates tracking across multiple backends (local, MLflow, W&B).
    """
    
    def __init__(
        self,
        local_tracker: Optional[ExperimentTracker] = None,
        mlflow_config: Optional[Dict[str, Any]] = None,
        wandb_config: Optional[Dict[str, Any]] = None
    ):
        # Local tracker
        self.local_tracker = local_tracker or ExperimentTracker()
        
        # External trackers
        self.mlflow = None
        self.wandb = None
        
        if mlflow_config:
            self.mlflow = MLflowIntegration(**mlflow_config)
        
        if wandb_config:
            self.wandb = WandBIntegration(**wandb_config)
        
        self._active_trackers: List[Any] = [self.local_tracker]
        if self.mlflow and self.mlflow.available:
            self._active_trackers.append(self.mlflow)
        if self.wandb and self.wandb.available:
            self._active_trackers.append(self.wandb)
    
    def start_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Start experiment across all trackers."""
        # Start local experiment
        experiment_id = self.local_tracker.create_experiment(
            name=name,
            description=description,
            parameters=parameters
        )
        
        run_id = self.local_tracker.start_run()
        
        # Start external trackers
        if self.mlflow and self.mlflow.available:
            self.mlflow.start_run(experiment_name=name, **kwargs)
            if parameters:
                self.mlflow.log_parameters(parameters)
        
        if self.wandb and self.wandb.available:
            self.wandb.start_run(
                project=kwargs.get('project'),
                name=name,
                config=parameters
            )
        
        return experiment_id
    
    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """Log parameters across all trackers."""
        self.local_tracker.log_parameters(parameters)
        
        if self.mlflow and self.mlflow.available:
            self.mlflow.log_parameters(parameters)
        
        # W&B parameters are logged during init
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        epoch: Optional[int] = None
    ) -> None:
        """Log metrics across all trackers."""
        self.local_tracker.log_metrics(metrics, step=step, epoch=epoch)
        
        if self.mlflow and self.mlflow.available:
            self.mlflow.log_metrics(metrics, step=step)
        
        if self.wandb and self.wandb.available:
            self.wandb.log_metrics(metrics, step=step)
    
    def log_artifact(self, file_path: Union[str, Path], **kwargs) -> str:
        """Log artifact across all trackers."""
        stored_path = self.local_tracker.log_artifact(file_path, **kwargs)
        
        if self.mlflow and self.mlflow.available:
            self.mlflow.log_artifact(file_path)
        
        if self.wandb and self.wandb.available:
            self.wandb.log_artifact(file_path, **kwargs)
        
        return stored_path
    
    def end_experiment(self, status: str = "completed") -> None:
        """End experiment across all trackers."""
        self.local_tracker.end_experiment(status)
        
        if self.mlflow and self.mlflow.available:
            self.mlflow.end_run()
        
        if self.wandb and self.wandb.available:
            self.wandb.end_run()


@contextmanager
def experiment_context(
    name: str,
    tracker: Optional[Union[ExperimentTracker, MultiTracker]] = None,
    **kwargs
):
    """
    Context manager for experiments.
    
    Args:
        name: Experiment name
        tracker: Tracker instance (creates default if None)
        **kwargs: Additional experiment parameters
    
    Yields:
        Tracker instance
    """
    if tracker is None:
        tracker = ExperimentTracker()
    
    # Handle different tracker types
    if isinstance(tracker, MultiTracker):
        experiment_id = tracker.start_experiment(name=name, **kwargs)
    else:
        # ExperimentTracker
        experiment_id = tracker.create_experiment(name=name, **kwargs)
        tracker.start_run()
    
    try:
        yield tracker
    except Exception as e:
        tracker.end_experiment(status="failed")
        raise
    else:
        tracker.end_experiment(status="completed")