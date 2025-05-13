"""
Trainer customization integration module.

This module provides integration between the customization system and trainer classes.
It ensures consistent initialization of optimizers, schedulers, and loss functions
across all trainer types while respecting the customization configuration.
"""

from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from .customization import CustomizationConfig, CustomizationManager

class TrainerCustomizationMixin:
    """
    Mixin class to provide customization capabilities to trainer classes.
    
    This mixin provides methods for initializing optimizers, schedulers,
    and loss functions according to the customization configuration.
    """
    
    def setup_customization(self, config: Optional[CustomizationConfig] = None):
        """
        Set up customization for the trainer.
        
        Args:
            config: Optional customization configuration. If None, uses default behavior.
        """
        self.customization_config = config
        if config:
            self.customization_manager = CustomizationManager(config)
        else:
            self.customization_manager = None
            
    def configure_optimizers(self) -> Union[Optimizer, Dict[str, Any]]:
        """
        Configure optimizer with customization support.
        
        Returns:
            Either an optimizer or a dict containing optimizer and scheduler config.
        """
        # Get trainable parameters based on PEFT status
        if hasattr(self, 'is_peft_model') and self.is_peft_model:
            params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            params = self.model.parameters()
            
        # Create optimizer using customization if available
        if self.customization_manager:
            optimizer = self.customization_manager.get_optimizer(self.model)
        else:
            # Default optimizer configuration
            optimizer = torch.optim.AdamW(
                params,
                lr=self.specs.get('learning_rate', 2e-5),
                weight_decay=self.specs.get('weight_decay', 0.01),
            )
            
        # Configure scheduler if specified
        if self.specs.get('use_lr_scheduler', False):
            if self.customization_manager:
                scheduler = self.customization_manager.get_learning_rate_schedule(optimizer)
            else:
                scheduler = self.prepare_scheduler(optimizer)
                
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
        
        return optimizer
        
    def prepare_scheduler(self, optimizer: Optimizer) -> _LRScheduler:
        """
        Prepare learning rate scheduler with customization support.
        
        Args:
            optimizer: The optimizer to create a scheduler for.
            
        Returns:
            The learning rate scheduler.
        """
        if self.customization_manager:
            return self.customization_manager.get_learning_rate_schedule(optimizer)
            
        # Default scheduler configuration
        num_training_steps = self.calculate_training_steps()
        scheduler_type = self.specs.get('scheduler_type', 'linear')
        warmup_ratio = self.specs.get('warmup_ratio', 0.1)
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        
        if scheduler_type == 'linear':
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type == 'cosine':
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
            
        return scheduler
        
    def get_loss_function(self) -> nn.Module:
        """
        Get loss function with customization support.
        
        Returns:
            The loss function to use.
        """
        if self.customization_manager:
            return self.customization_manager.get_loss_function()
            
        # Default loss function
        return nn.CrossEntropyLoss(ignore_index=-100)
        
    def should_stop_training(self, validation_metrics: Dict[str, float]) -> bool:
        """
        Check if training should stop based on early stopping criteria.
        
        Args:
            validation_metrics: The current validation metrics.
            
        Returns:
            Whether training should stop.
        """
        if self.customization_manager:
            return self.customization_manager.should_stop_training(validation_metrics)
            
        # Default: no early stopping
        return False 