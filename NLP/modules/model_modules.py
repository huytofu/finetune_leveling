from transformers import BertForMaskedLM, BertForTokenClassification
from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
from transformers import BitsAndBytesConfig
import torch
from typing import Dict, Optional, Union, List

class ModelModules():
    """
    A class to manage model loading and application of Parameter-Efficient Fine-Tuning (PEFT) methods.
    """
    
    def __init__(self, checkpoint, use_bert=False):
        """
        Initialize the ModelModules.
        
        Args:
            checkpoint (str): Path to the model checkpoint.
            use_bert (bool): Whether to use BERT-specific models.
        """
        # Store the checkpoint and BERT usage flag
        self.checkpoint = checkpoint
        self.use_bert = use_bert
        
    def load_model(self, task_type, quantization=None):
        """
        Load a model with optional quantization.
        
        Args:
            task_type (str): The type of task (e.g., 'masked_language_modeling', 'token_classification').
            quantization (Optional[str]): Quantization type ('4bit', '8bit', or None).
            
        Returns:
            model: The loaded model, potentially with quantization applied.
        """
        # Prepare quantization config if needed
        quantization_config = None
        if quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        elif quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
        # Load the appropriate model based on task type
        if task_type == "masked_language_modeling":
            if self.use_bert:
                model = BertForMaskedLM.from_pretrained(self.checkpoint)
            else:
                model = AutoModelForMaskedLM.from_pretrained(
                    self.checkpoint,
                    quantization_config=quantization_config
                )
        elif task_type == "token_classification":
            if self.use_bert:
                model = BertForTokenClassification.from_pretrained(self.checkpoint)
            else:
                model = AutoModelForTokenClassification.from_pretrained(
                    self.checkpoint,
                    quantization_config=quantization_config
                )
        elif task_type == "translation":
            if self.use_bert:
                model = None
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.checkpoint,
                    quantization_config=quantization_config
                )
        elif task_type == "summarization":
            if self.use_bert:
                model = None
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.checkpoint,
                    quantization_config=quantization_config
                )
        elif task_type == "text_generation":
            if self.use_bert:
                model = None
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.checkpoint,
                    quantization_config=quantization_config
                )
        elif task_type == "question_answering":
            if self.use_bert:
                model = None
            else:
                model = AutoModelForQuestionAnswering.from_pretrained(
                    self.checkpoint,
                    quantization_config=quantization_config
                )
        else:
            model = None
            
        # Enable gradient checkpointing for memory efficiency if available
        if model is not None and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            
        return model
        
    def apply_peft(self, model, tokenizer, peft_method, peft_config, task_type):
        """
        Apply a Parameter-Efficient Fine-Tuning (PEFT) method to the model.
        
        Args:
            model: The model to apply PEFT to.
            tokenizer: The tokenizer associated with the model.
            peft_method (str): The PEFT method to use (e.g., 'lora', 'qlora', 'prefix_tuning').
            peft_config (dict): Configuration for the PEFT method.
            task_type (str): The type of task.
            
        Returns:
            model: The model with PEFT applied.
            tokenizer: The tokenizer, potentially modified by the PEFT method.
        """
        from .peft_modules import PEFTModules
        
        peft_modules = PEFTModules()
        
        if peft_method == "lora":
            model = peft_modules.apply_lora(
                model=model,
                task_type=task_type,
                r=peft_config.get("r", 16),
                lora_alpha=peft_config.get("lora_alpha", 32),
                lora_dropout=peft_config.get("lora_dropout", 0.05),
                target_modules=peft_config.get("target_modules", None),
                bias=peft_config.get("bias", "none"),
                modules_to_save=peft_config.get("modules_to_save", None)
            )
        elif peft_method == "qlora":
            model = peft_modules.apply_qlora(
                model=model,
                task_type=task_type,
                r=peft_config.get("r", 16),
                lora_alpha=peft_config.get("lora_alpha", 32),
                lora_dropout=peft_config.get("lora_dropout", 0.05),
                target_modules=peft_config.get("target_modules", None),
                bias=peft_config.get("bias", "none"),
                modules_to_save=peft_config.get("modules_to_save", None),
                quantization_config=peft_config.get("quantization_config", None)
            )
        elif peft_method == "prefix_tuning":
            model = peft_modules.apply_prefix_tuning(
                model=model,
                task_type=task_type,
                num_virtual_tokens=peft_config.get("num_virtual_tokens", 20),
                encoder_hidden_size=peft_config.get("encoder_hidden_size", None),
                prefix_projection=peft_config.get("prefix_projection", False)
            )
        elif peft_method == "prompt_tuning":
            model = peft_modules.apply_prompt_tuning(
                model=model,
                task_type=task_type,
                num_virtual_tokens=peft_config.get("num_virtual_tokens", 20),
                prompt_tuning_init_text=peft_config.get("prompt_tuning_init_text", "Perform the task:")
            )
        elif peft_method == "p_tuning":
            model = peft_modules.apply_p_tuning(
                model=model,
                task_type=task_type,
                num_virtual_tokens=peft_config.get("num_virtual_tokens", 20),
                encoder_hidden_size=peft_config.get("encoder_hidden_size", None),
                encoder_reparameterization_type=peft_config.get("encoder_reparameterization_type", "MLP")
            )
        elif peft_method == "adapter":
            model = peft_modules.apply_adapter(
                model=model,
                task_type=task_type,
                adapter_size=peft_config.get("adapter_size", 64),
                adapter_dropout=peft_config.get("adapter_dropout", 0.1),
                adapter_init_scale=peft_config.get("adapter_init_scale", 1e-3),
                adapter_non_linearity=peft_config.get("adapter_non_linearity", "relu")
            )
        else:
            raise ValueError(f"Unsupported PEFT method: {peft_method}")
        
        return model, tokenizer
