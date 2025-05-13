"""
Centralized utilities for trainer implementations.
This module provides common functionality used across different trainer classes.
"""

import os
import json
import torch
import logging
import evaluate
from typing import Optional, Dict, Any, Union, List, Tuple
from dataclasses import dataclass
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)

@dataclass
class PeftConfig:
    """Configuration for PEFT models with optimization settings."""
    peft_type: str
    task_type: str
    inference_mode: bool = False
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = None
    layers_to_transform: List[int] = None
    fan_in_fan_out: bool = False
    modules_to_save: List[str] = None
    init_lora_weights: bool = True
    use_gradient_checkpointing: bool = True
    use_cache: bool = False
    quantization_type: Optional[str] = None
    quantization_bits: Optional[int] = None

    @classmethod
    def from_pretrained(cls, model):
        """Create config from a pretrained PEFT model."""
        if not hasattr(model, "peft_config"):
            return None
        config = model.peft_config
        return cls(
            peft_type=getattr(config, "peft_type", "unknown"),
            task_type=getattr(config, "task_type", "unknown"),
            r=getattr(config, "r", 16),
            lora_alpha=getattr(config, "lora_alpha", 32),
            lora_dropout=getattr(config, "lora_dropout", 0.05),
            bias=getattr(config, "bias", "none"),
            target_modules=getattr(config, "target_modules", None),
            layers_to_transform=getattr(config, "layers_to_transform", None),
            fan_in_fan_out=getattr(config, "fan_in_fan_out", False),
            modules_to_save=getattr(config, "modules_to_save", None),
            init_lora_weights=getattr(config, "init_lora_weights", True),
            use_gradient_checkpointing=True,
            use_cache=False
        )

def check_is_peft_model(model) -> bool:
    """Check if the model is a PEFT model."""
    try:
        from peft import PeftModel
        is_peft = isinstance(model, PeftModel)
        if is_peft:
            logger.info(f"Detected PEFT model: {type(model).__name__}")
            log_peft_params(model)
        return is_peft
    except ImportError:
        logger.info("PEFT not installed, continuing with standard training")
        return False

def log_peft_params(model):
    """Log PEFT-specific parameter information."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"PEFT model has {trainable_params} trainable parameters out of {total_params} total parameters")
    
    if hasattr(model, "peft_config"):
        config = model.peft_config
        if hasattr(config, "r"):  # LoRA
            logger.info(f"LoRA rank: {config.r}")
        elif hasattr(config, "num_virtual_tokens"):  # Prefix Tuning
            logger.info(f"Number of prefix tokens: {config.num_virtual_tokens}")

def prepare_scheduler(optimizer: torch.optim.Optimizer,
                     num_training_steps: int,
                     scheduler_type: str = 'linear',
                     warmup_ratio: float = 0.1) -> torch.optim.lr_scheduler._LRScheduler:
    """Prepare learning rate scheduler."""
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    
    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
    logger.info(f"Created {scheduler_type} scheduler with {num_warmup_steps} warmup steps")
    return scheduler

def calculate_training_steps(train_dataset_size: int,
                           batch_size: int = 8,
                           grad_accumulation: int = 1,
                           num_epochs: int = 3) -> int:
    """Calculate total training steps."""
    steps_per_epoch = train_dataset_size // (batch_size * grad_accumulation)
    total_steps = steps_per_epoch * num_epochs
    return total_steps

def configure_optimizer(model,
                       is_peft_model: bool,
                       learning_rate: float = 2e-5,
                       weight_decay: float = 0.01) -> torch.optim.Optimizer:
    """Configure optimizer with proper parameter handling."""
    # Get trainable parameters based on PEFT status
    if is_peft_model:
        params = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"Configuring optimizer for PEFT model with {len(params)} trainable parameters")
    else:
        params = model.parameters()
        logger.info("Configuring optimizer for full model")

    # Create optimizer with specified parameters
    return torch.optim.AdamW(
        params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

def clip_gradients(model,
                   max_grad_norm: float,
                   is_peft_model: bool = False,
                   parameters: Optional[torch.nn.Parameter] = None) -> float:
    """Clip gradients with PEFT-aware handling."""
    if parameters is None:
        parameters = model.parameters()
        
    # Special handling for PEFT models
    if is_peft_model:
        # Scale gradients differently for PEFT parameters
        if hasattr(model, "peft_type"):
            if model.peft_type == "LORA":
                return _clip_lora_gradients(parameters, max_grad_norm)
            elif model.peft_type in ["PREFIX", "PROMPT"]:
                return _clip_prefix_gradients(parameters, max_grad_norm)
    
    # Standard gradient clipping
    return torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)

def _clip_lora_gradients(parameters, max_grad_norm: float) -> float:
    """Clip gradients for LoRA parameters."""
    grad_norm = torch.norm(torch.stack([
        torch.norm(p.grad.detach(), 2.0) 
        for p in parameters 
        if p.grad is not None and "lora_" in p.name
    ]))
    
    clip_coef = max_grad_norm / (grad_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None and "lora_" in p.name:
                p.grad.detach().mul_(clip_coef)
    return grad_norm

def _clip_prefix_gradients(parameters, max_grad_norm: float) -> float:
    """Clip gradients for prefix tuning parameters."""
    grad_norm = torch.norm(torch.stack([
        torch.norm(p.grad.detach(), 2.0) 
        for p in parameters 
        if p.grad is not None and "prefix" in p.name
    ]))
    
    clip_coef = max_grad_norm / (grad_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None and "prefix" in p.name:
                p.grad.detach().mul_(clip_coef)
    return grad_norm

def save_model_checkpoint(model,
                         tokenizer,
                         output_dir: str,
                         is_peft_model: bool = False) -> str:
    """Save model checkpoint with PEFT handling."""
    os.makedirs(output_dir, exist_ok=True)
    
    if is_peft_model:
        try:
            logger.info(f"Saving PEFT adapter to {output_dir}")
            # Save only the adapter parameters
            model.save_pretrained(output_dir)
        except Exception as e:
            logger.error(f"Error saving PEFT model: {e}")
            # Fall back to standard save
            model.save_pretrained(output_dir)
    else:
        # Standard save for non-PEFT models
        model.save_pretrained(output_dir)
    
    # Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    
    return output_dir

def load_model_checkpoint(model,
                         checkpoint_path: str,
                         is_peft_model: bool = False) -> Dict[str, Any]:
    """Load model checkpoint with PEFT handling."""
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    if is_peft_model:
        try:
            logger.info(f"Loading PEFT adapter from {checkpoint_path}")
            model.load_adapter(checkpoint_path)
        except Exception as e:
            logger.error(f"Error loading PEFT adapter: {e}")
            # Fall back to standard load
            model.load_state_dict(torch.load(os.path.join(checkpoint_path, "pytorch_model.bin")))
    else:
        # Standard load for non-PEFT models
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, "pytorch_model.bin")))
    
    return {"success": True, "path": checkpoint_path}

def get_default_metrics(task_type: str):
    """Get default metrics based on task type."""
    if task_type in ["classification", "token-classification"]:
        return evaluate.load("accuracy")
    elif task_type in ["summarization", "translation"]:
        return evaluate.load("rouge")
    elif task_type == "question-answering":
        return evaluate.load("squad")
    return None

def setup_accelerate_integration(specs: Dict[str, Any]):
    """Set up Accelerate integration."""
    try:
        from accelerate import Accelerator
        accelerator = Accelerator(
            gradient_accumulation_steps=specs.get('gradient_accumulation_steps', 1),
            mixed_precision=specs.get('fp16', True) and 'fp16' or 'no'
        )
        logger.info(f"Accelerate setup complete. Using device: {accelerator.device} with mixed precision: {accelerator.mixed_precision}")
        return accelerator
    except ImportError:
        logger.warning("Accelerate not installed - continuing without acceleration")
        return None

def optimize_memory_settings(model, use_gradient_checkpointing: bool = True):
    """Optimize memory usage settings."""
    if use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    if hasattr(model, "config"):
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
            logger.info("Model cache disabled for training") 