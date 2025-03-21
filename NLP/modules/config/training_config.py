"""Configuration classes for training pipelines."""

import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union

# Add to Python path
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parentdir)

# Local imports
from modules.advanced_training import UnslothIntegration, validate_advanced_config

logger = logging.getLogger(__name__)

@dataclass
class FineTuneConfig:
    """Configuration for fine-tuning pipeline."""
    model_name: str
    task_type: str
    
    # Basic training parameters
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    max_length: int = 512
    
    # Framework options
    use_lightning: bool = False
    use_accelerate: bool = False
    
    # Advanced optimization techniques
    use_mixed_precision: bool = False
    use_dynamic_batching: bool = False
    use_curriculum_learning: bool = False
    use_few_shot: bool = False
    
    # Large model optimization (for 10B-100B models)
    use_gradient_checkpointing: bool = False
    use_flash_attention: bool = False
    use_deepspeed: bool = False
    deepspeed_stage: int = 2
    deepspeed_offload_optimizer: bool = False
    deepspeed_offload_parameters: bool = False
    
    # Quantization options
    use_quantization: bool = False
    quantization_bits: int = 8  # 4 or 8 bits supported
    quantization_method: str = "bitsandbytes"  # "bitsandbytes", "auto_gptq" or "awq"
    awq_zero_point: bool = True  # AWQ-specific setting
    awq_group_size: int = 128    # AWQ-specific setting
    
    # Advanced training features
    use_advanced_training: bool = False
    
    # Multi-node training (Axolotl based)
    use_multi_node: bool = False
    num_nodes: int = 1
    node_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "29500"
    
    # Hardware optimization (Unsloth based)
    use_unsloth: bool = False
    unsloth_max_seq_length: int = 2048
    
    # RLHF (TRL based)
    use_rlhf: bool = False
    rlhf_method: str = "ppo"  # Options: "ppo", "dpo", "orpo"
    reward_model_name: Optional[str] = None
    num_ppo_epochs: int = 1
    kl_penalty_coefficient: float = 0.1
    beta: float = 0.1  # DPO specific
    
    # PEFT configuration
    use_peft: bool = False
    peft_method: str = "lora"  # Primary: "lora", "prefix", "prompt", "adapter"
    secondary_peft_method: str = None  # For mixing methods (optional)
    
    # PEFT parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    prefix_length: int = 30
    adapter_size: int = 64
    
    # Pruning config
    use_pruning: bool = False
    pruning_sparsity: float = 0.3
    pruning_method: str = "magnitude"  # "magnitude", "structured", or "movement"
    
    # Distillation config
    use_distillation: bool = False
    teacher_model_name: str = None
    distillation_alpha: float = 0.5
    distillation_temperature: float = 2.0
    
    # Model merging config
    use_model_merging: bool = False
    merge_method: str = "weighted_average"  # "weighted_average", "selective", "frankenstein"
    merge_models: List[str] = field(default_factory=list)  # List of model names to merge
    merge_weights: Dict[str, float] = field(default_factory=dict)  # Weights for weighted average
    merge_layer_mapping: Dict[str, Union[List[str], str]] = field(default_factory=dict)  # For selective/frankenstein
    verify_merged_model: bool = True
    
    # LoRA specific settings
    lora_target_modules: Optional[List[str]] = None
    
    # Prefix-tuning settings
    prefix_projection: bool = False
    
    # Prompt-tuning settings
    prompt_initialization: str = "random"  # Options: "random", "text", "embedding"
    
    # Mixed precision settings
    mixed_precision_dtype: str = "float16"
    mixed_precision_loss_scale: str = "dynamic"
    
    # Dynamic batching settings
    dynamic_batch_size_range: Tuple[int, int] = (16, 128)
    dynamic_batch_growth_factor: float = 1.5
    dynamic_batch_memory_threshold: float = 0.8
    
    # Curriculum learning settings
    curriculum_difficulty_metric: str = "length"
    curriculum_steps: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    curriculum_scoring_function: str = "linear"
    
    # Few-shot settings
    few_shot_examples: int = 5
    few_shot_metric: str = "similarity"
    few_shot_selection: str = "kmeans"
    
    # MLflow tracking
    enable_mlflow: bool = True
    mlflow_experiment_name: Optional[str] = None
    
    def get_peft_config(self) -> Dict[str, Any]:
        """Get PEFT configuration based on selected method."""
        if not self.use_peft:
            return None
            
        if self.peft_method == "lora":
            return {
                "r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "target_modules": self.lora_target_modules,
                "bias": "none",
                "task_type": self.task_type
            }
        elif self.peft_method == "prefix":
            return {
                "num_virtual_tokens": self.prefix_length,
                "projection": self.prefix_projection,
                "task_type": self.task_type
            }
        elif self.peft_method == "prompt":
            return {
                "num_virtual_tokens": self.prefix_length,
                "prompt_initialization": self.prompt_initialization,
                "task_type": self.task_type
            }
        elif self.peft_method == "adapter":
            return {
                "adapter_size": self.adapter_size,
                "adapter_dropout": self.adapter_dropout,
                "adapter_scaling": self.adapter_scaling,
                "task_type": self.task_type
            }
        else:
            raise ValueError(f"Unsupported PEFT method: {self.peft_method}")

    def _validate_config(self):
        """Validate configuration."""
        # Validate framework options
        if self.use_lightning and not self._is_lightning_available():
            logger.warning("PyTorch Lightning not available. Lightning training disabled.")
            self.use_lightning = False
        
        if self.use_accelerate and not self._is_accelerate_available():
            logger.warning("Accelerate not available. Accelerate training disabled.")
            self.use_accelerate = False
            
        # Validate RLHF compatibility
        if self.use_rlhf:
            # RLHF is not compatible with Lightning or Accelerate
            if self.use_lightning:
                logger.warning("RLHF is not compatible with PyTorch Lightning. Disabling Lightning.")
                self.use_lightning = False
            if self.use_accelerate:
                logger.warning("RLHF is not compatible with Accelerate. Disabling Accelerate.")
                self.use_accelerate = False
        
        # Validate Unsloth compatibility
        if self.use_unsloth:
            # Check if model is supported by Unsloth
            if not UnslothIntegration.is_model_supported(self.model_name):
                logger.warning(f"Model {self.model_name} is not supported by Unsloth. Disabling Unsloth.")
                self.use_unsloth = False
        
        # Ensure advanced training config is validated
        if self.use_advanced_training:
            validate_advanced_config(self)

    @staticmethod
    def _is_lightning_available():
        """Check if PyTorch Lightning is available."""
        try:
            import pytorch_lightning
            return True
        except ImportError:
            return False
            
    @staticmethod
    def _is_accelerate_available():
        """Check if Accelerate is available."""
        try:
            import accelerate
            return True
        except ImportError:
            return False 