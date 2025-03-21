"""MLFlow configuration and tracking functionality."""

import os
import sys
import json
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import mlflow

# Add to Python path
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parentdir)

# Local imports
from modules.monitoring import UnifiedMonitor, MetricConfig

logger = logging.getLogger(__name__)

@dataclass
class MLflowConfig:
    """Configuration for MLflow tracking."""
    
    # Basic MLflow settings
    tracking_uri: Optional[str] = None
    experiment_name: str = "default"
    run_name: Optional[str] = None
    
    # Artifact settings
    artifact_location: Optional[str] = None
    log_artifacts: bool = True
    
    # Metric settings
    log_system_metrics: bool = True
    log_gpu_metrics: bool = True
    metric_step_frequency: int = 10
    
    # Model tracking
    log_model_artifacts: bool = True
    register_model: bool = False
    model_name: Optional[str] = None
    
    # Tags and metadata
    tags: Dict[str, str] = field(default_factory=dict)
    custom_metrics: Dict[str, MetricConfig] = field(default_factory=dict)
    
    def setup(self):
        """Set up MLflow tracking."""
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
            
        # Set up experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                name=self.experiment_name,
                artifact_location=self.artifact_location
            )
        else:
            experiment_id = experiment.experiment_id
            
        return experiment_id

class MLflowCallback:
    """Callback for tracking training metrics in MLflow."""
    
    def __init__(self, monitor: UnifiedMonitor):
        """
        Initialize the MLflow callback.
        
        Args:
            monitor: UnifiedMonitor instance for tracking
        """
        self.monitor = monitor
    
    def on_init(self, args, state, control, **kwargs):
        """Log initial training parameters."""
        self.monitor.log_metrics({
            "train_batch_size": args.per_device_train_batch_size,
            "eval_batch_size": args.per_device_eval_batch_size,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_grad_norm": args.max_grad_norm
        })
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Log epoch start."""
        self.monitor.log_metric(f"epoch_{state.epoch}", state.epoch)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Log epoch metrics."""
        epoch_duration = time.time() - state.epoch_start_time
        self.monitor.log_metric(f"epoch_{state.epoch}_duration", epoch_duration)
        
        # Log hardware metrics
        self.monitor.log_hardware_metrics(state.global_step)
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Log step start time."""
        state.step_start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        """Log step metrics."""
        step_duration = time.time() - state.step_start_time
        self.monitor.log_metric(f"step_{state.global_step}_duration", step_duration)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation metrics."""
        if metrics:
            self.monitor.log_metrics({f"eval_{k}": v for k, v in metrics.items()})
    
    def on_save(self, args, state, control, **kwargs):
        """Log model checkpoint."""
        if state.best_model_checkpoint:
            self.monitor.log_artifact(state.best_model_checkpoint, f"checkpoint_{state.global_step}")

class MLflowTracker:
    """MLflow experiment tracking manager."""
    
    def __init__(self, config: MLflowConfig):
        """
        Initialize MLflow tracker.
        
        Args:
            config: MLflow configuration
        """
        self.config = config
        self.experiment_id = None
        self.run_id = None
        self.monitor = None
        
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional dictionary of tags to log
        """
        # Set up experiment if not already done
        if self.experiment_id is None:
            self.experiment_id = self.config.setup()
            
        # Start run
        run_name = run_name or self.config.run_name
        tags = {**self.config.tags, **(tags or {})}
        
        mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags
        )
        self.run_id = mlflow.active_run().info.run_id
        
        # Initialize monitor
        self.monitor = UnifiedMonitor(
            experiment_id=self.experiment_id,
            run_id=self.run_id,
            custom_metrics=self.config.custom_metrics
        )
        
        return MLflowCallback(self.monitor)
        
    def end_run(self):
        """End the current MLflow run."""
        if mlflow.active_run():
            mlflow.end_run()
            
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        if not mlflow.active_run():
            logger.warning("No active MLflow run. Start a run before logging parameters.")
            return
            
        mlflow.log_params(params)
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        if not mlflow.active_run():
            logger.warning("No active MLflow run. Start a run before logging metrics.")
            return
            
        mlflow.log_metrics(metrics, step=step)
        
    def log_model(self, model: Any, artifact_path: str):
        """Log model to MLflow."""
        if not mlflow.active_run():
            logger.warning("No active MLflow run. Start a run before logging model.")
            return
            
        if not self.config.log_model_artifacts:
            return
            
        mlflow.pytorch.log_model(model, artifact_path)
        
        if self.config.register_model and self.config.model_name:
            mlflow.register_model(
                f"runs:/{self.run_id}/{artifact_path}",
                self.config.model_name
            )
            
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact to MLflow."""
        if not mlflow.active_run():
            logger.warning("No active MLflow run. Start a run before logging artifacts.")
            return
            
        if not self.config.log_artifacts:
            return
            
        mlflow.log_artifact(local_path, artifact_path) 