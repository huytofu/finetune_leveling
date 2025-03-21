from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

@dataclass
class CustomizationConfig:
    """Configuration for custom finetuning and evaluation strategies."""
    
    # Finetuning customizations
    custom_learning_rate_schedule: Optional[Callable] = None
    custom_optimizer: Optional[Callable] = None
    custom_loss_function: Optional[Callable] = None
    custom_gradient_clipping: Optional[Callable] = None
    custom_batch_sampler: Optional[Callable] = None
    
    # Evaluation customizations
    custom_metrics: List[Callable] = field(default_factory=list)
    custom_validation_strategy: Optional[Callable] = None
    custom_model_selection: Optional[Callable] = None
    custom_checkpointing: Optional[Callable] = None
    custom_early_stopping: Optional[Callable] = None
    
    # Additional parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

class CustomizationManager:
    """Manages custom finetuning and evaluation strategies."""
    
    def __init__(self, config: CustomizationConfig):
        self.config = config
        self._validate_config()
        
    def _validate_config(self):
        """Validate customization configuration."""
        if self.config.custom_learning_rate_schedule and not callable(self.config.custom_learning_rate_schedule):
            raise ValueError("custom_learning_rate_schedule must be callable")
        if self.config.custom_optimizer and not callable(self.config.custom_optimizer):
            raise ValueError("custom_optimizer must be callable")
        if self.config.custom_loss_function and not callable(self.config.custom_loss_function):
            raise ValueError("custom_loss_function must be callable")
            
    def get_learning_rate_schedule(self, optimizer: Optimizer) -> _LRScheduler:
        """Get custom or default learning rate schedule."""
        if self.config.custom_learning_rate_schedule:
            return self.config.custom_learning_rate_schedule(optimizer)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    def get_optimizer(self, model: nn.Module) -> Optimizer:
        """Get custom or default optimizer."""
        if self.config.custom_optimizer:
            return self.config.custom_optimizer(model)
        return torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    def get_loss_function(self) -> nn.Module:
        """Get custom or default loss function."""
        if self.config.custom_loss_function:
            return self.config.custom_loss_function()
        return nn.CrossEntropyLoss()
    
    def apply_gradient_clipping(self, model: nn.Module, max_norm: float = 1.0):
        """Apply custom or default gradient clipping."""
        if self.config.custom_gradient_clipping:
            self.config.custom_gradient_clipping(model, max_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    def get_batch_sampler(self, dataset_size: int, batch_size: int):
        """Get custom or default batch sampler."""
        if self.config.custom_batch_sampler:
            return self.config.custom_batch_sampler(dataset_size, batch_size)
        return None  # Use default DataLoader behavior
    
    def evaluate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Evaluate custom metrics."""
        metrics = {}
        for metric_fn in self.config.custom_metrics:
            metric_name = metric_fn.__name__
            metrics[metric_name] = metric_fn(predictions, targets)
        return metrics
    
    def should_stop_training(self, validation_metrics: Dict[str, float]) -> bool:
        """Check if training should stop based on custom criteria."""
        if self.config.custom_early_stopping:
            return self.config.custom_early_stopping(validation_metrics)
        return False
    
    def select_best_model(self, model_checkpoints: List[Dict]) -> Dict:
        """Select best model based on custom criteria."""
        if self.config.custom_model_selection:
            return self.config.custom_model_selection(model_checkpoints)
        # Default: select model with best validation loss
        return min(model_checkpoints, key=lambda x: x['validation_loss'])
    
    def should_save_checkpoint(self, epoch: int, validation_metrics: Dict[str, float]) -> bool:
        """Determine if checkpoint should be saved based on custom criteria."""
        if self.config.custom_checkpointing:
            return self.config.custom_checkpointing(epoch, validation_metrics)
        return epoch % 5 == 0  # Save every 5 epochs by default

class CustomizationTemplates:
    """Predefined customization templates."""
    
    @staticmethod
    def get_linear_warmup_schedule(warmup_steps: int = 1000):
        """Create linear warmup learning rate schedule."""
        def create_schedule(optimizer):
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
        return create_schedule
    
    @staticmethod
    def get_cosine_annealing_schedule(epochs: int = 100):
        """Create cosine annealing learning rate schedule."""
        def create_schedule(optimizer):
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs
            )
        return create_schedule
    
    @staticmethod
    def get_custom_metrics():
        """Get common custom metrics."""
        def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
            return (predictions.argmax(dim=1) == targets).float().mean().item()
            
        def f1_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
            pred_labels = predictions.argmax(dim=1)
            true_positives = ((pred_labels == 1) & (targets == 1)).sum().item()
            false_positives = ((pred_labels == 1) & (targets == 0)).sum().item()
            false_negatives = ((pred_labels == 0) & (targets == 1)).sum().item()
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
        return [accuracy, f1_score]
    
    @staticmethod
    def get_early_stopping(patience: int = 5, min_delta: float = 0.001):
        """Create early stopping strategy."""
        def should_stop(validation_metrics: Dict[str, float]) -> bool:
            if not hasattr(should_stop, 'best_loss'):
                should_stop.best_loss = float('inf')
                should_stop.patience_counter = 0
                
            current_loss = validation_metrics.get('loss', float('inf'))
            if current_loss < should_stop.best_loss - min_delta:
                should_stop.best_loss = current_loss
                should_stop.patience_counter = 0
            else:
                should_stop.patience_counter += 1
                
            return should_stop.patience_counter >= patience
            
        return should_stop 