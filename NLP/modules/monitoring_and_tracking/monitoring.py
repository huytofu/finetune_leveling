"""
Unified monitoring module that integrates with MLflow while adding extended capabilities.
Eliminates redundancy while preserving advanced features like hardware monitoring and alerts.
"""

import os
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import torch
import numpy as np
import psutil
import mlflow
from mlflow.tracking import MlflowClient
from torch.utils.tensorboard import SummaryWriter

@dataclass
class MetricConfig:
    """Extended metric configuration that works with MLflow."""
    name: str
    unit: str = ""
    higher_is_better: bool = True
    visualization_type: str = "line"  # line, histogram, distribution
    alert_threshold: Optional[float] = None
    alert_condition: str = "above"  # above, below
    track_gradients: bool = False
    track_hardware: bool = False

class UnifiedMonitor:
    """Unified monitoring that extends MLflow with additional capabilities."""
    
    def __init__(self, 
                 experiment_name: str,
                 config: Dict[str, any],
                 metrics_config: Dict[str, MetricConfig],
                 tracking_uri: Optional[str] = None,
                 log_dir: str = "logs",
                 enable_tensorboard: bool = False,
                 run_id: Optional[str] = None):
        """
        Initialize unified monitoring.
        
        Args:
            experiment_name: Name of the experiment
            config: Training configuration to track
            metrics_config: Metric configuration
            tracking_uri: MLflow tracking URI
            log_dir: Local logging directory
            enable_tensorboard: Whether to enable TensorBoard (for local viz)
            run_id: Existing MLflow run ID to resume tracking
        """
        # Setup MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.experiment = mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        
        # Either resume existing run or start new one
        if run_id:
            self.run = mlflow.start_run(run_id=run_id)
        else:
            self.run = mlflow.start_run()
        
        # Log configuration if starting new run
        if not run_id:
            mlflow.log_params(self._flatten_dict(config))
        
        self.metrics_config = metrics_config
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup local logging
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(os.path.join(log_dir, f"{experiment_name}.log"))
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # Optional TensorBoard for local visualization
        self.tensorboard = None
        if enable_tensorboard:
            self.tensorboard = SummaryWriter(os.path.join(log_dir, "tensorboard"))
        
        # Initialize hardware monitoring
        self.hardware_metrics = {}
        
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary for MLflow params."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, str(v)))
        return dict(items)

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric with unified tracking."""
        # Log to MLflow
        mlflow.log_metric(name, value, step=step)
        
        # Check alerts
        if self._check_alert(name, value):
            self.logger.warning(f"Alert for {name}: {value} {self.metrics_config[name].alert_condition} {self.metrics_config[name].alert_threshold}")
        
        # Log to TensorBoard if enabled
        if self.tensorboard and step is not None:
            self.tensorboard.add_scalar(name, value, step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics."""
        # Log to MLflow
        mlflow.log_metrics(metrics, step=step)
        
        # Process each metric
        for name, value in metrics.items():
            # Check alerts
            if name in self.metrics_config and self._check_alert(name, value):
                self.logger.warning(f"Alert for {name}: {value} {self.metrics_config[name].alert_condition} {self.metrics_config[name].alert_threshold}")
            
            # Log to TensorBoard if enabled
            if self.tensorboard and step is not None:
                self.tensorboard.add_scalar(name, value, step)

    def log_model(self, model: torch.nn.Module, artifact_path: str):
        """Log model with extended tracking."""
        # Log model to MLflow
        mlflow.pytorch.log_model(model, artifact_path)
        
        # Log model graph to TensorBoard if enabled
        if self.tensorboard:
            try:
                dummy_input = torch.zeros(1, 3, 224, 224)  # Adjust size as needed
                self.tensorboard.add_graph(model, dummy_input)
            except Exception as e:
                self.logger.warning(f"Could not log model graph to TensorBoard: {e}")

    def log_gradients(self, model: torch.nn.Module, step: Optional[int] = None):
        """Track gradient statistics."""
        grad_norms = {}
        grad_means = {}
        grad_vars = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad.detach()
                grad_norms[f"grad_norm/{name}"] = grad.norm().item()
                grad_means[f"grad_mean/{name}"] = grad.mean().item()
                grad_vars[f"grad_var/{name}"] = grad.var().item()
                
                # Log detailed histogram to TensorBoard if enabled
                if self.tensorboard and step is not None:
                    self.tensorboard.add_histogram(f"gradients/{name}", grad, step)
        
        # Log summary statistics to MLflow
        mlflow.log_metrics(grad_norms, step=step)
        mlflow.log_metrics(grad_means, step=step)
        mlflow.log_metrics(grad_vars, step=step)

    def log_hardware_metrics(self, step: Optional[int] = None):
        """Track hardware utilization."""
        metrics = {}
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics["cpu_utilization"] = cpu_percent
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics["memory_used_gb"] = memory.used / (1024 ** 3)
        metrics["memory_percent"] = memory.percent
        
        # GPU metrics if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                metrics[f"gpu_{i}_memory_used"] = torch.cuda.memory_allocated(i) / (1024 ** 3)
                metrics[f"gpu_{i}_utilization"] = torch.cuda.utilization(i)
        
        # Log to MLflow
        mlflow.log_metrics(metrics, step=step)
        
        # Update hardware metrics history
        self.hardware_metrics[step] = metrics if step is not None else metrics
        
        # Log to TensorBoard if enabled
        if self.tensorboard and step is not None:
            for name, value in metrics.items():
                self.tensorboard.add_scalar(f"hardware/{name}", value, step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact file or directory."""
        mlflow.log_artifact(local_path, artifact_path)

    def _check_alert(self, name: str, value: float) -> bool:
        """Check if metric value triggers an alert."""
        if name not in self.metrics_config:
            return False
            
        config = self.metrics_config[name]
        if config.alert_threshold is None:
            return False
            
        if config.alert_condition == "above":
            return value > config.alert_threshold
        else:
            return value < config.alert_threshold

    def end_run(self):
        """End the monitoring run."""
        if self.tensorboard:
            self.tensorboard.close()
        mlflow.end_run()

    @property
    def run_id(self) -> str:
        """Get the current run ID."""
        return self.run.info.run_id

    @classmethod
    def resume_run(cls, run_id: str, **kwargs):
        """Resume monitoring from an existing MLflow run."""
        return cls(run_id=run_id, **kwargs)

# Example usage:
if __name__ == "__main__":
    # Define metrics configuration
    metrics_config = {
        "train_loss": MetricConfig(
            name="Training Loss",
            unit="",
            higher_is_better=False,
            visualization_type="line"
        ),
        "val_loss": MetricConfig(
            name="Validation Loss",
            unit="",
            higher_is_better=False,
            visualization_type="line"
        ),
        "accuracy": MetricConfig(
            name="Accuracy",
            unit="%",
            higher_is_better=True,
            visualization_type="line",
            alert_threshold=0.95,
            alert_condition="above"
        ),
        "gpu_memory": MetricConfig(
            name="GPU Memory Usage",
            unit="GB",
            higher_is_better=False,
            visualization_type="line",
            alert_threshold=0.90,
            alert_condition="above",
            track_hardware=True
        )
    }
    
    # Example configuration
    config = {
        "model": {"name": "gpt2", "dtype": "fp16"},
        "training": {"lr": 2e-5, "batch_size": 32}
    }
    
    # Create monitor
    monitor = UnifiedMonitor(
        experiment_name="test_experiment",
        config=config,
        metrics_config=metrics_config,
        enable_tensorboard=True
    )
    
    # Simulate training loop
    for step in range(100):
        metrics = {
            "train_loss": 1.0 - 0.5 * (step / 100),
            "val_loss": 1.2 - 0.6 * (step / 100),
            "accuracy": 0.5 + 0.4 * (step / 100)
        }
        monitor.log_metrics(metrics, step)
        monitor.log_hardware_metrics(step)
    
    monitor.end_run() 