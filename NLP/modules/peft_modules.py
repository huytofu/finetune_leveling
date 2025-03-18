import os
import sys
import torch
from typing import Dict, List, Optional, Union
from peft import (
    get_peft_model, 
    LoraConfig, 
    PrefixTuningConfig, 
    PromptTuningConfig,
    PromptEncoderConfig,
    TaskType,
    PeftType,
    PeftModel
)
from transformers import PreTrainedModel

class PEFTModules:
    """Module for Parameter-Efficient Fine-Tuning methods."""
    
    def __init__(self):
        """Initialize the PEFT modules."""
        self.task_type_mapping = {
            "masked_language_modeling": TaskType.CAUSAL_LM,
            "token_classification": TaskType.TOKEN_CLS,
            "translation": TaskType.SEQ_2_SEQ_LM,
            "summarization": TaskType.SEQ_2_SEQ_LM,
            "text_generation": TaskType.CAUSAL_LM,
            "question_answering": TaskType.QUESTION_ANS,
            "text_classification": TaskType.SEQ_CLS
        }
    
    def get_task_type(self, task_type: str) -> TaskType:
        """Map task type string to PEFT TaskType."""
        return self.task_type_mapping.get(task_type, TaskType.CAUSAL_LM)
    
    def apply_lora(
        self, 
        model: PreTrainedModel, 
        task_type: str,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[Union[List[str], str]] = None,
        bias: str = "none",
        modules_to_save: Optional[List[str]] = None
    ) -> PeftModel:
        """Apply LoRA to a model.
        
        Args:
            model: The model to apply LoRA to
            task_type: The type of task
            r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout probability for LoRA layers
            target_modules: Which modules to apply LoRA to (if None, will try to infer)
            bias: Bias type ("none", "all", or "lora_only")
            modules_to_save: List of modules apart from LoRA layers to be set as trainable
            
        Returns:
            The model with LoRA applied
        """
        # If target_modules is None, try to infer based on model architecture
        if target_modules is None:
            # For different model architectures, we need different target modules
            if hasattr(model, "config"):
                model_type = getattr(model.config, "model_type", "")
                if "llama" in model_type.lower():
                    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
                elif "mistral" in model_type.lower():
                    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
                elif "gpt" in model_type.lower():
                    target_modules = ["c_attn", "c_proj", "w1", "w2"]
                elif "bert" in model_type.lower():
                    target_modules = ["query", "key", "value"]
                elif "t5" in model_type.lower():
                    target_modules = ["q", "v", "k", "o"]
                else:
                    # Default to query, key, value projection matrices
                    target_modules = ["query", "key", "value"]
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=self.get_task_type(task_type),
            modules_to_save=modules_to_save
        )
        
        # Apply LoRA to the model
        peft_model = get_peft_model(model, lora_config)
        
        # Print trainable parameters info
        peft_model.print_trainable_parameters()
        
        return peft_model
    
    def apply_qlora(
        self, 
        model: PreTrainedModel, 
        task_type: str,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[Union[List[str], str]] = None,
        bias: str = "none",
        modules_to_save: Optional[List[str]] = None,
        quantization_config: Optional[Dict] = None
    ) -> PeftModel:
        """Apply QLoRA to a model.
        
        Args:
            model: The model to apply QLoRA to
            task_type: The type of task
            r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout probability for LoRA layers
            target_modules: Which modules to apply LoRA to (if None, will try to infer)
            bias: Bias type ("none", "all", or "lora_only")
            modules_to_save: List of modules apart from LoRA layers to be set as trainable
            quantization_config: Quantization configuration
            
        Returns:
            The model with QLoRA applied
        """
        # If quantization_config is None, use default 4-bit quantization
        if quantization_config is None:
            quantization_config = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            }
        
        # Apply LoRA to the quantized model
        return self.apply_lora(
            model=model,
            task_type=task_type,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
            modules_to_save=modules_to_save
        )
    
    def apply_prefix_tuning(
        self, 
        model: PreTrainedModel, 
        task_type: str,
        num_virtual_tokens: int = 20,
        encoder_hidden_size: Optional[int] = None,
        prefix_projection: bool = False
    ) -> PeftModel:
        """Apply Prefix Tuning to a model.
        
        Args:
            model: The model to apply Prefix Tuning to
            task_type: The type of task
            num_virtual_tokens: Number of virtual tokens
            encoder_hidden_size: Hidden size of the encoder
            prefix_projection: Whether to project the prefix
            
        Returns:
            The model with Prefix Tuning applied
        """
        # Create Prefix Tuning config
        prefix_config = PrefixTuningConfig(
            task_type=self.get_task_type(task_type),
            num_virtual_tokens=num_virtual_tokens,
            encoder_hidden_size=encoder_hidden_size,
            prefix_projection=prefix_projection
        )
        
        # Apply Prefix Tuning to the model
        peft_model = get_peft_model(model, prefix_config)
        
        # Print trainable parameters info
        peft_model.print_trainable_parameters()
        
        return peft_model
    
    def apply_prompt_tuning(
        self, 
        model: PreTrainedModel, 
        task_type: str,
        num_virtual_tokens: int = 20,
        prompt_tuning_init_text: str = "Perform the task:"
    ) -> PeftModel:
        """Apply Prompt Tuning to a model.
        
        Args:
            model: The model to apply Prompt Tuning to
            task_type: The type of task
            num_virtual_tokens: Number of virtual tokens
            prompt_tuning_init_text: Initial text for prompt tuning
            
        Returns:
            The model with Prompt Tuning applied
        """
        # Create Prompt Tuning config
        prompt_config = PromptTuningConfig(
            task_type=self.get_task_type(task_type),
            num_virtual_tokens=num_virtual_tokens,
            prompt_tuning_init_text=prompt_tuning_init_text
        )
        
        # Apply Prompt Tuning to the model
        peft_model = get_peft_model(model, prompt_config)
        
        # Print trainable parameters info
        peft_model.print_trainable_parameters()
        
        return peft_model
    
    def apply_p_tuning(
        self, 
        model: PreTrainedModel, 
        task_type: str,
        num_virtual_tokens: int = 20,
        encoder_hidden_size: Optional[int] = None,
        encoder_reparameterization_type: str = "MLP"
    ) -> PeftModel:
        """Apply P-Tuning to a model.
        
        Args:
            model: The model to apply P-Tuning to
            task_type: The type of task
            num_virtual_tokens: Number of virtual tokens
            encoder_hidden_size: Hidden size of the encoder
            encoder_reparameterization_type: Type of reparameterization for the encoder
            
        Returns:
            The model with P-Tuning applied
        """
        # Create P-Tuning config
        ptuning_config = PromptEncoderConfig(
            task_type=self.get_task_type(task_type),
            num_virtual_tokens=num_virtual_tokens,
            encoder_hidden_size=encoder_hidden_size,
            encoder_reparameterization_type=encoder_reparameterization_type
        )
        
        # Apply P-Tuning to the model
        peft_model = get_peft_model(model, ptuning_config)
        
        # Print trainable parameters info
        peft_model.print_trainable_parameters()
        
        return peft_model
    
    def apply_adapter(
        self, 
        model: PreTrainedModel, 
        task_type: str,
        adapter_size: int = 64,
        adapter_dropout: float = 0.1,
        adapter_init_scale: float = 1e-3,
        adapter_non_linearity: str = "relu"
    ) -> PeftModel:
        """Apply Adapters to a model.
        
        Args:
            model: The model to apply Adapters to
            task_type: The type of task
            adapter_size: Size of the adapter layers
            adapter_dropout: Dropout probability for adapter layers
            adapter_init_scale: Scale for the initialization of adapter weights
            adapter_non_linearity: Non-linearity to use in adapter layers
            
        Returns:
            The model with Adapters applied
        """
        from peft import AdapterConfig
        
        # Create Adapter config
        adapter_config = AdapterConfig(
            task_type=self.get_task_type(task_type),
            adapter_size=adapter_size,
            adapter_dropout=adapter_dropout,
            adapter_init_scale=adapter_init_scale,
            adapter_non_linearity=adapter_non_linearity
        )
        
        # Apply Adapters to the model
        peft_model = get_peft_model(model, adapter_config)
        
        # Print trainable parameters info
        peft_model.print_trainable_parameters()
        
        return peft_model
    
    def save_peft_model(self, model: PeftModel, output_dir: str) -> None:
        """Save a PEFT model.
        
        Args:
            model: The PEFT model to save
            output_dir: Directory to save the model to
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the model
        model.save_pretrained(output_dir)
    
    def load_peft_model(self, model: PreTrainedModel, peft_model_path: str) -> PeftModel:
        """Load a PEFT model.
        
        Args:
            model: The base model
            peft_model_path: Path to the PEFT model
            
        Returns:
            The loaded PEFT model
        """
        # Load the PEFT model
        peft_model = PeftModel.from_pretrained(model, peft_model_path)
        
        return peft_model 