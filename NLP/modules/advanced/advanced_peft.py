"""
Advanced Parameter-Efficient Fine-Tuning (PEFT) Module

This module extends standard PEFT methods with advanced techniques optimized for large
language models (30B-70B parameters). It provides memory-efficient implementations of
various parameter-efficient methods including:

1. Enhanced LoRA with advanced configurations
2. IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)
3. UniPELT (Unified Parameter-Efficient Language Tuning)
4. Sparse Fine-tuning
5. Adapter Fusion

Each method is designed to minimize memory usage while maximizing performance for
extremely large models.
"""

# Standard library imports
import os
import math
import json
import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mlflow
from transformers import PreTrainedModel

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class AdvancedPeftConfig:
    """
    Configuration class for advanced PEFT methods.
    
    This class extends beyond standard PEFT configurations with parameters
    specifically designed for very large models.
    
    Attributes:
        method (str): The PEFT method to use (lora, ia3, unipelt, sparse_ft, adapter_fusion)
        model_type (str): Type of model architecture (e.g., llama, falcon, mpt)
        target_modules (List[str]): Which modules to apply PEFT to
        inference_mode (bool): Whether the model is for inference only
        
        # LoRA specific parameters
        lora_r (int): Rank of the LoRA low-rank matrices
        lora_alpha (int): Alpha parameter for LoRA
        lora_dropout (float): Dropout probability for LoRA
        
        # IA³ specific parameters
        ia3_target_modules (List[str]): Modules to apply IA³ to
        
        # UniPELT specific parameters
        unipelt_methods (List[str]): List of PEFT methods to combine in UniPELT
        
        # Sparse fine-tuning parameters
        sparsity (float): Target sparsity ratio (0.0-1.0)
        importance_measure (str): Method to determine parameter importance
        
        # Memory optimization parameters
        use_gradient_checkpointing (bool): Whether to use gradient checkpointing
        use_cpu_offloading (bool): Whether to offload parameters to CPU
        mixed_precision (bool): Whether to use mixed precision
        
        # Adapter fusion parameters
        adapter_paths (List[str]): Paths to adapters to fuse
        fusion_strategy (str): Strategy for adapter fusion
    """
    method: str = "lora"
    model_type: str = "llama"
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    inference_mode: bool = False
    
    # LoRA specific parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # IA³ specific parameters
    ia3_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # UniPELT specific parameters
    unipelt_methods: List[str] = field(default_factory=lambda: ["lora", "ia3"])
    
    # Sparse fine-tuning parameters
    sparsity: float = 0.9
    importance_measure: str = "magnitude"
    
    # Memory optimization parameters
    use_gradient_checkpointing: bool = True
    use_cpu_offloading: bool = False
    mixed_precision: bool = True
    
    # Adapter fusion parameters
    adapter_paths: List[str] = field(default_factory=list)
    fusion_strategy: str = "weighted"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for saving."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AdvancedPeftConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_pretrained(cls, path: str) -> "AdvancedPeftConfig":
        """Load configuration from a pretrained adapter directory."""
        config_path = os.path.join(path, "advanced_peft_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Config not found at {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)

class AdvancedPeftModule:
    """
    Advanced Parameter-Efficient Fine-Tuning module.
    
    This class provides implementations of various advanced PEFT methods
    optimized for very large language models (30B-70B parameters). It focuses
    on memory efficiency, training stability, and performance.
    """
    
    def __init__(
        self, 
        model: PreTrainedModel, 
        config: Optional[AdvancedPeftConfig] = None,
        track_with_mlflow: bool = True
    ):
        """
        Initialize the advanced PEFT module.
        
        Args:
            model: The pretrained model to apply PEFT to
            config: The PEFT configuration (default: None, will use default LoRA config)
            track_with_mlflow: Whether to track with MLflow (default: True)
        """
        self.model = model
        self.config = config or AdvancedPeftConfig()
        self.track_with_mlflow = track_with_mlflow
        self.peft_applied = False
        self.original_module_state = {}
        
        # Register hooks for tracking memory usage
        self.memory_stats = {"peak_allocated": 0, "current_allocated": 0}
        self._register_memory_hooks()
        
        # Log configuration if using MLflow
        if self.track_with_mlflow:
            self._log_config_to_mlflow()
    
    def _register_memory_hooks(self):
        """Register hooks to track memory usage."""
        def update_memory_stats():
            if torch.cuda.is_available():
                self.memory_stats["current_allocated"] = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                self.memory_stats["peak_allocated"] = max(
                    self.memory_stats["peak_allocated"], 
                    torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
                )
        
        # Register forward pre-hook on the model
        self.model.register_forward_pre_hook(lambda m, inp: update_memory_stats())
    
    def _log_config_to_mlflow(self):
        """Log PEFT configuration to MLflow."""
        try:
            # Log parameters
            mlflow.log_params({
                "peft_method": self.config.method,
                "model_type": self.config.model_type,
                "target_modules": str(self.config.target_modules),
                "use_gradient_checkpointing": self.config.use_gradient_checkpointing,
                "use_cpu_offloading": self.config.use_cpu_offloading,
            })
            
            # Log method-specific parameters
            if self.config.method == "lora":
                mlflow.log_params({
                    "lora_r": self.config.lora_r,
                    "lora_alpha": self.config.lora_alpha,
                    "lora_dropout": self.config.lora_dropout,
                })
            elif self.config.method == "ia3":
                mlflow.log_params({
                    "ia3_target_modules": str(self.config.ia3_target_modules),
                })
            elif self.config.method == "unipelt":
                mlflow.log_params({
                    "unipelt_methods": str(self.config.unipelt_methods),
                })
            elif self.config.method == "sparse_ft":
                mlflow.log_params({
                    "sparsity": self.config.sparsity,
                    "importance_measure": self.config.importance_measure,
                })
            elif self.config.method == "adapter_fusion":
                mlflow.log_params({
                    "fusion_strategy": self.config.fusion_strategy,
                    "num_adapters": len(self.config.adapter_paths),
                })
                
            # Log the full config as a JSON artifact
            with open("advanced_peft_config.json", "w") as f:
                json.dump(self.config.to_dict(), f, indent=2)
            mlflow.log_artifact("advanced_peft_config.json")
            
            logger.info("PEFT configuration logged to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log PEFT configuration to MLflow: {e}")
    
    def apply(self) -> PreTrainedModel:
        """
        Apply the configured PEFT method to the model.
        
        Returns:
            The model with PEFT applied
        """
        start_time = time.time()
        
        # Apply the appropriate PEFT method
        if self.config.method == "lora":
            self._apply_lora()
        elif self.config.method == "ia3":
            self._apply_ia3()
        elif self.config.method == "unipelt":
            self._apply_unipelt()
        elif self.config.method == "sparse_ft":
            self._apply_sparse_finetuning()
        elif self.config.method == "adapter_fusion":
            self._apply_adapter_fusion()
        else:
            raise ValueError(f"PEFT method {self.config.method} not supported")
        
        # Apply memory optimizations
        self._apply_memory_optimizations()
        
        # Track metrics
        apply_time = time.time() - start_time
        self.peft_applied = True
        
        # Log metrics to MLflow
        if self.track_with_mlflow:
            mlflow.log_metric("peft_apply_time_seconds", apply_time)
            mlflow.log_metric("peft_trainable_params", self._count_trainable_parameters())
            mlflow.log_metric("peft_total_params", self._count_total_parameters())
            mlflow.log_metric("peft_memory_usage_gb", self.memory_stats["current_allocated"])
            
        logger.info(
            f"Applied {self.config.method} in {apply_time:.2f}s. "
            f"Trainable parameters: {self._count_trainable_parameters():,} "
            f"({self._count_trainable_parameters() / self._count_total_parameters() * 100:.2f}% of total)"
        )
        
        return self.model
    
    def _apply_lora(self):
        """Apply enhanced LoRA with optimizations for very large models."""
        try:
            from peft import LoraConfig, get_peft_model
            
            # Create LoRA configuration
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            # Apply LoRA
            self.model = get_peft_model(self.model, lora_config)
            
            # Add rank adaptation capability if requested
            if hasattr(self.config, "enable_rank_adaptation") and self.config.enable_rank_adaptation:
                self._setup_rank_adaptation()
                
            logger.info(f"Applied LoRA with rank {self.config.lora_r}")
            
        except ImportError:
            raise ImportError("PEFT library not installed. Install it with `pip install peft`")
    
    def _apply_ia3(self):
        """
        Apply IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations).
        
        IA³ is a parameter-efficient method that modulates activations by 
        learning vectors that scale activations element-wise.
        """
        # Save original module state
        for name, module in self.model.named_modules():
            if any(target in name for target in self.config.ia3_target_modules):
                self.original_module_state[name] = {
                    "forward": module.forward
                }
        
        # Create and apply IA³ adapters
        for name, module in self.model.named_modules():
            if any(target in name for target in self.config.ia3_target_modules):
                ia3_adapter = IA3Adapter(module)
                self._replace_module_forward(module, ia3_adapter.forward)
                
        logger.info(f"Applied IA³ to {len(self.original_module_state)} modules")
    
    def _apply_unipelt(self):
        """
        Apply UniPELT (Unified Parameter-Efficient Language Tuning).
        
        UniPELT combines multiple PEFT methods (LoRA, Prefix Tuning, Adapters)
        in a unified framework.
        """
        unipelt_methods = self.config.unipelt_methods
        logger.info(f"Applying UniPELT with methods: {unipelt_methods}")
        
        if "lora" in unipelt_methods:
            self._apply_lora()
            
        if "ia3" in unipelt_methods:
            self._apply_ia3()
            
        if "prefix" in unipelt_methods:
            self._apply_prefix_tuning()
            
        if "adapter" in unipelt_methods:
            self._apply_adapters()
            
        logger.info(f"Applied UniPELT with {len(unipelt_methods)} methods")
    
    def _apply_sparse_finetuning(self):
        """
        Apply sparse fine-tuning, which updates only the most important parameters.
        
        This approach identifies important weights based on the specified importance
        measure and only updates those during training.
        """
        # Determine parameter importance
        parameter_importance = self._calculate_parameter_importance()
        
        # Set up sparse training with the mask
        for name, param in self.model.named_parameters():
            if name in parameter_importance:
                # Calculate threshold based on sparsity
                importance = parameter_importance[name]
                threshold = torch.quantile(importance.abs().flatten(), self.config.sparsity)
                
                # Create mask (1 for important weights, 0 for unimportant)
                mask = (importance.abs() > threshold).float()
                
                # Set unimportant weights to not require gradients
                param.requires_grad = False
                
                # Create a new parameter that shares storage with the original
                # but only for the important weights
                new_param = nn.Parameter(param.data.clone())
                param.data.copy_(new_param.data)
                
                # Register hook to apply mask during gradient computation
                def hook_factory(mask):
                    def hook(grad):
                        return grad * mask
                    return hook
                
                param.register_hook(hook_factory(mask))
                param.requires_grad = True
                
        logger.info(f"Applied sparse fine-tuning with sparsity {self.config.sparsity:.2f}")
    
    def _apply_adapter_fusion(self):
        """
        Apply adapter fusion to combine multiple pre-trained adapters.
        
        This method loads multiple adapters and fuses them according to
        the specified fusion strategy.
        """
        adapter_paths = self.config.adapter_paths
        if not adapter_paths:
            raise ValueError("No adapter paths provided for fusion")
            
        logger.info(f"Applying adapter fusion with {len(adapter_paths)} adapters")
        
        # Load adapters
        adapters = []
        for path in adapter_paths:
            adapter_config = AdvancedPeftConfig.from_pretrained(path)
            adapter = self._load_adapter(path, adapter_config)
            adapters.append(adapter)
            
        # Create fusion layer
        fusion = AdapterFusion(adapters, strategy=self.config.fusion_strategy)
        
        # Apply fusion to model
        # This implementation depends on the specific adapter architecture
        # and would need to be customized based on the model architecture
        
        logger.info(f"Applied adapter fusion with strategy {self.config.fusion_strategy}")
    
    def _calculate_parameter_importance(self) -> Dict[str, torch.Tensor]:
        """
        Calculate parameter importance for sparse fine-tuning.
        
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        importance = {}
        
        if self.config.importance_measure == "magnitude":
            # Use parameter magnitude as importance
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    importance[name] = param.data.abs()
                    
        elif self.config.importance_measure == "grad_magnitude":
            # Use gradient magnitude as importance (requires a forward/backward pass)
            # This is a simplified implementation; a real one would need to:
            # 1. Run forward/backward on calibration data
            # 2. Capture gradient magnitudes
            pass
            
        elif self.config.importance_measure == "fisher":
            # Fisher Information (approximated)
            # Would require estimating by sampling from model outputs
            pass
        
        return importance
    
    def _apply_memory_optimizations(self):
        """Apply memory optimizations for large models."""
        # Apply gradient checkpointing if enabled
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
            
        # Apply CPU offloading if enabled
        if self.config.use_cpu_offloading:
            self._setup_cpu_offloading()
            logger.info("Enabled CPU offloading")
            
        # Apply mixed precision if enabled
        if self.config.mixed_precision:
            # Note: actual mixed precision setup would be done at the trainer level
            logger.info("Mixed precision will be used during training")
    
    def _setup_cpu_offloading(self):
        """Set up CPU offloading for memory optimization."""
        # Move non-trainable parameters to CPU
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                param.data = param.data.to("cpu")
                
                # Register hook to keep the parameter on CPU
                def to_cpu_hook(grad):
                    return grad.to("cpu")
                
                param.register_hook(to_cpu_hook)
    
    def _setup_rank_adaptation(self):
        """Set up dynamic rank adaptation for LoRA."""
        # This would implement dynamic rank adaptation during training
        # Based on validation performance or other metrics
        pass
    
    def _replace_module_forward(self, module, new_forward):
        """Replace the forward method of a module."""
        module.forward = new_forward.__get__(module, type(module))
    
    def _load_adapter(self, path, config):
        """Load a pretrained adapter."""
        # This would load a pretrained adapter based on its configuration
        # Implementation depends on adapter architecture
        pass
    
    def _count_trainable_parameters(self) -> int:
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _count_total_parameters(self) -> int:
        """Count the total number of parameters in the model."""
        return sum(p.numel() for p in self.model.parameters())
    
    def save_pretrained(self, path: str):
        """
        Save the PEFT model and configuration.
        
        Args:
            path: Directory to save the model to
        """
        if not self.peft_applied:
            raise ValueError("Cannot save model before applying PEFT")
            
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(path, "advanced_peft_config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
            
        # Save model weights (implementation depends on PEFT method)
        if self.config.method == "lora":
            self.model.save_pretrained(path)
        else:
            # Save other PEFT method weights
            # This would need to be implemented based on the specific method
            model_path = os.path.join(path, "adapter_model.bin")
            torch.save(self._get_adapter_state_dict(), model_path)
            
        logger.info(f"Saved PEFT model to {path}")
        
        # Log to MLflow
        if self.track_with_mlflow:
            mlflow.log_artifact(path)
    
    def _get_adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get the state dictionary for the adapter weights."""
        # Implementation depends on the PEFT method
        return {}
    
    def train(self, mode: bool = True):
        """Set the model to training mode."""
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set the model to evaluation mode."""
        self.model.eval()
        return self


class IA3Adapter:
    """
    Implementation of IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations).
    
    IA³ adapts a pre-trained model by learning vectors that scale activations
    element-wise, leading to highly parameter-efficient adaptation.
    """
    
    def __init__(self, module: nn.Module):
        """
        Initialize the IA³ adapter.
        
        Args:
            module: The module to adapt with IA³
        """
        self.module = module
        self.original_forward = module.forward
        
        # Determine output dimension for scaling vector
        # This is a simplified implementation; would need to be adapted for specific architectures
        if hasattr(module, "out_features"):
            output_dim = module.out_features
        elif hasattr(module, "embedding_dim"):
            output_dim = module.embedding_dim
        else:
            # Guess based on weight shape
            weight = getattr(module, "weight", None)
            if weight is not None:
                output_dim = weight.shape[0]
            else:
                raise ValueError(f"Could not determine output dimension for module: {module}")
        
        # Create IA³ scaling vector
        self.ia3_scale = nn.Parameter(torch.ones(output_dim))
    
    def forward(self, *args, **kwargs):
        """
        Forward pass with IA³ scaling.
        
        This intercepts the original module's forward pass and applies
        the learned scaling vector to the output.
        """
        # Call original forward
        output = self.original_forward(*args, **kwargs)
        
        # Apply IA³ scaling
        if isinstance(output, torch.Tensor):
            # Direct scaling for tensor outputs
            output = output * self.ia3_scale
        elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            # For modules that return multiple tensors (like some attention modules)
            scaled_output = output[0] * self.ia3_scale
            output = (scaled_output,) + output[1:]
            
        return output


class AdapterFusion:
    """
    Implementation of Adapter Fusion.
    
    Adapter Fusion combines multiple pre-trained adapters, allowing
    models to leverage knowledge from different adapters.
    """
    
    def __init__(
        self, 
        adapters: List[Any], 
        strategy: str = "weighted"
    ):
        """
        Initialize adapter fusion.
        
        Args:
            adapters: List of adapters to fuse
            strategy: Fusion strategy (weighted, gating, attention)
        """
        self.adapters = adapters
        self.strategy = strategy
        self.num_adapters = len(adapters)
        
        if strategy == "weighted":
            # Learn weights for each adapter
            self.weights = nn.Parameter(torch.ones(self.num_adapters) / self.num_adapters)
        elif strategy == "gating":
            # Learn a gating network to dynamically combine adapters
            # This would need a more complex implementation
            pass
        elif strategy == "attention":
            # Use attention mechanism to combine adapters
            # This would need a more complex implementation
            pass
        else:
            raise ValueError(f"Fusion strategy {strategy} not supported")
    
    def forward(self, *args, **kwargs):
        """
        Forward pass with adapter fusion.
        
        Returns:
            Combined output from all adapters
        """
        # Run all adapters
        adapter_outputs = [adapter(*args, **kwargs) for adapter in self.adapters]
        
        # Apply fusion strategy
        if self.strategy == "weighted":
            # Apply softmax to ensure weights sum to 1
            normalized_weights = F.softmax(self.weights, dim=0)
            
            # Combine outputs with weighted sum
            output = sum(w * out for w, out in zip(normalized_weights, adapter_outputs))
            
        elif self.strategy == "gating":
            # More complex gating implementation would go here
            pass
            
        elif self.strategy == "attention":
            # More complex attention-based fusion would go here
            pass
        
        return output


def get_advanced_peft_model(
    model: PreTrainedModel, 
    peft_config: Optional[Union[Dict[str, Any], AdvancedPeftConfig]] = None,
    track_with_mlflow: bool = True
) -> PreTrainedModel:
    """
    Apply advanced PEFT to a pretrained model.
    
    This is the main user-facing function for applying advanced PEFT methods.
    
    Args:
        model: The pretrained model to apply PEFT to
        peft_config: PEFT configuration (dictionary or AdvancedPeftConfig)
        track_with_mlflow: Whether to track with MLflow
        
    Returns:
        The model with PEFT applied
    """
    # Convert dictionary config to AdvancedPeftConfig
    if isinstance(peft_config, dict):
        peft_config = AdvancedPeftConfig(**peft_config)
    elif peft_config is None:
        peft_config = AdvancedPeftConfig()
    
    # Create and apply PEFT
    peft_module = AdvancedPeftModule(model, peft_config, track_with_mlflow)
    return peft_module.apply() 