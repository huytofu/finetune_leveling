import os
import torch
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

class QuantizationManager:
    """
    Manages loading, saving, and configuration of quantized models with PEFT compatibility.
    This class provides consistent and optimized handling of quantized models across different types.
    """
    
    def __init__(self):
        """Initialize the QuantizationManager."""
        self.supported_quant_types = ["4bit", "8bit", "nf4", "fp4"]
        self.quant_config_map = {
            "4bit": {"load_in_4bit": True},
            "8bit": {"load_in_8bit": True},
            "nf4": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"},
            "fp4": {"load_in_4bit": True, "bnb_4bit_quant_type": "fp4"}
        }
        self.optimized_modules = {
            "gptq": ["Linear4bit"],
            "bitsandbytes": ["Linear8bitLt", "Linear4bit"],
            "autoawq": ["WQLinear"],
            "gptq-for-llama": ["QuantLinear"]
        }
    
    def prepare_model_for_quantization(
        self, 
        model_name_or_path: str,
        quant_type: str,
        device_map: Optional[Union[str, Dict[str, Any]]] = "auto",
        custom_quantization_config: Optional[Dict[str, Any]] = None,
        **model_kwargs
    ) -> Dict[str, Any]:
        """
        Prepare configuration for loading a quantized model.
        
        Args:
            model_name_or_path: The path or name of the model
            quant_type: Type of quantization ("4bit", "8bit", "nf4", "fp4")
            device_map: Device mapping strategy for model loading
            custom_quantization_config: Optional custom quantization configuration
            **model_kwargs: Additional model loading arguments
            
        Returns:
            Dict with quantized model loading configuration
        """
        if quant_type not in self.supported_quant_types:
            logger.warning(f"Unsupported quantization type: {quant_type}. "
                         f"Supported types are: {self.supported_quant_types}")
            return model_kwargs
        
        # Get base quantization config
        quant_config = self.quant_config_map.get(quant_type, {}).copy()
        
        # Apply custom configuration
        if custom_quantization_config:
            quant_config.update(custom_quantization_config)
        
        # Set compute dtype to float16 if not specified otherwise
        if "bnb_4bit_compute_dtype" not in quant_config and quant_type in ["4bit", "nf4", "fp4"]:
            quant_config["bnb_4bit_compute_dtype"] = torch.float16
        
        # Enable double quantization for 4-bit types by default
        if "bnb_4bit_use_double_quant" not in quant_config and quant_type in ["4bit", "nf4", "fp4"]:
            quant_config["bnb_4bit_use_double_quant"] = True
        
        # Add device map if not already specified
        if "device_map" not in quant_config and device_map is not None:
            quant_config["device_map"] = device_map
        
        # Merge with model_kwargs and return
        model_kwargs.update(quant_config)
        return model_kwargs
    
    def optimize_model_for_peft(
        self, 
        model: PreTrainedModel,
        quant_type: str,
        peft_method: str = "lora",
    ) -> PreTrainedModel:
        """
        Optimize a quantized model for PEFT compatibility.
        
        Args:
            model: The quantized model
            quant_type: Type of quantization used
            peft_method: PEFT method to optimize for
            
        Returns:
            The optimized model ready for PEFT
        """
        try:
            # Import necessary libraries on demand
            import bitsandbytes as bnb
            from peft import prepare_model_for_kbit_training
            
            logger.info(f"Optimizing {quant_type} quantized model for {peft_method} PEFT method")
            
            # Check model type
            if quant_type in ["4bit", "8bit", "nf4", "fp4"]:
                # For bitsandbytes quantization
                model = prepare_model_for_kbit_training(
                    model,
                    use_gradient_checkpointing=getattr(model.config, "use_gradient_checkpointing", False)
                )
                
                # Special handling for LoRA with modules that need to be marked
                if peft_method.lower() == "lora":
                    modules_to_mark = self._find_modules_to_mark(model)
                    if modules_to_mark:
                        self._mark_lora_targets(model, modules_to_mark)
            
            # Log trainable parameters info
            self._log_trainable_params(model)
            
            return model
            
        except ImportError as e:
            logger.warning(f"Error optimizing model for PEFT: {e}. "
                         f"Make sure you have installed the required libraries.")
            return model
    
    def _find_modules_to_mark(self, model: PreTrainedModel) -> List[str]:
        """
        Find modules that need special marking for LoRA with quantized models.
        
        Args:
            model: The model to analyze
            
        Returns:
            List of module types that need marking
        """
        modules_to_mark = []
        
        # Check for various quantized module types in the model
        for module_type in ["Linear4bit", "Linear8bitLt", "WQLinear", "QuantLinear"]:
            # Check if this module type exists in the model
            if any(isinstance(module, eval(f"type('{module_type}', (), {{}})")) 
                   for _, module in model.named_modules() if module_type in str(type(module))):
                modules_to_mark.append(module_type)
        
        return modules_to_mark
    
    def _mark_lora_targets(self, model: PreTrainedModel, module_types: List[str]) -> None:
        """
        Mark modules as LoRA targets for quantized models.
        
        Args:
            model: The model to modify
            module_types: List of module types to mark
        """
        # Find modules of the specified types
        for name, module in model.named_modules():
            if any(module_type in str(type(module)) for module_type in module_types):
                module.is_target = True
                logger.debug(f"Marked {name} as LoRA target")
    
    def _log_trainable_params(self, model: PreTrainedModel) -> None:
        """
        Log information about trainable parameters in the model.
        
        Args:
            model: The model to analyze
        """
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        if total_params > 0:
            trainable_percent = 100 * trainable_params / total_params
        else:
            trainable_percent = 0
            
        logger.info(f"Trainable parameters: {trainable_params:,d} ({trainable_percent:.2f}% of {total_params:,d} total)")
    
    def auto_adjust_batch_size(
        self, 
        base_batch_size: int,
        quant_type: str,
        peft_method: str = None,
        **kwargs
    ) -> int:
        """
        Automatically adjust batch size based on quantization and PEFT method.
        
        Args:
            base_batch_size: The base batch size to adjust
            quant_type: Type of quantization
            peft_method: Optional PEFT method being used
            **kwargs: Additional parameters affecting batch size
            
        Returns:
            Adjusted batch size for optimal training
        """
        # Base adjustment factors
        adjustment_factor = 1.0
        
        # Adjust based on quantization type
        if quant_type == "4bit" or quant_type == "nf4" or quant_type == "fp4":
            # 4-bit quantization can often use larger batch sizes due to reduced memory
            adjustment_factor *= 1.2
        elif quant_type == "8bit":
            # 8-bit is between full precision and 4-bit
            adjustment_factor *= 1.1
            
        # Further adjust based on PEFT method
        if peft_method is not None:
            if peft_method.lower() in ["lora", "qlora"]:
                # LoRA/QLoRA often allows for larger batch sizes
                adjustment_factor *= 1.3
            elif peft_method.lower() in ["prefix_tuning", "p_tuning"]:
                # Prefix tuning can be more memory intensive
                adjustment_factor *= 0.8
                
        # Apply length adjustment if sequence length is provided
        if "max_length" in kwargs:
            max_length = kwargs["max_length"]
            # Longer sequences reduce possible batch size
            length_factor = 512 / max_length if max_length > 0 else 1.0
            adjustment_factor *= min(length_factor, 2.0)  # Cap at 2x
            
        # Apply model size adjustment if model parameter count is provided
        if "model_params_billion" in kwargs:
            # Larger models need smaller batches
            size_factor = 1.0 / (1.0 + 0.1 * kwargs["model_params_billion"])
            adjustment_factor *= size_factor
            
        # Calculate adjusted batch size
        adjusted_batch_size = max(1, int(base_batch_size * adjustment_factor))
        
        logger.info(f"Adjusted batch size from {base_batch_size} to {adjusted_batch_size} "
                   f"(factor: {adjustment_factor:.2f})")
                   
        return adjusted_batch_size
    
    def auto_configure_memory(
        self,
        model: PreTrainedModel,
        quant_type: str,
        peft_method: str = None,
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        **kwargs
    ) -> Dict[Union[int, str], str]:
        """
        Automatically configure memory settings for optimal performance.
        
        Args:
            model: The model being configured
            quant_type: Type of quantization
            peft_method: Optional PEFT method being used
            max_memory: Optional user-provided memory configuration
            **kwargs: Additional configuration parameters
            
        Returns:
            Memory configuration dictionary
        """
        if max_memory is not None:
            return max_memory
            
        # Get available devices
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_memory = {}
            
            # Get memory for each GPU
            for i in range(device_count):
                mem_info = torch.cuda.get_device_properties(i).total_memory
                # Convert to GB with safety margin (90% of available)
                mem_gb = int(mem_info * 0.9 / (1024 ** 3))
                
                # Adjust based on quantization and PEFT
                if quant_type in ["4bit", "nf4", "fp4"]:
                    # 4-bit models need less memory, so we can reserve more for activations
                    mem_gb = int(mem_gb * 0.8)  # 80% of device memory
                elif quant_type == "8bit":
                    mem_gb = int(mem_gb * 0.85)  # 85% of device memory
                else:
                    mem_gb = int(mem_gb * 0.9)  # 90% of device memory
                
                # Convert to string format used by device_map
                device_memory[i] = f"{mem_gb}GiB"
            
            # Add CPU memory
            device_memory["cpu"] = "24GiB"  # Default fallback
            
            return device_memory
        else:
            # No CUDA available, use CPU only
            return {"cpu": "24GiB"}
    
    def is_quantized_model(self, model: PreTrainedModel) -> bool:
        """
        Check if a model is quantized.
        
        Args:
            model: The model to check
            
        Returns:
            Boolean indicating if the model is quantized
        """
        # Check for bitsandbytes quantization
        has_bitsandbytes = any("Linear8bitLt" in str(type(module)) or "Linear4bit" in str(type(module))
                              for _, module in model.named_modules())
        
        # Check for GPTQ
        has_gptq = any("QuantLinear" in str(type(module)) for _, module in model.named_modules())
        
        # Check for AWQ
        has_awq = any("WQLinear" in str(type(module)) for _, module in model.named_modules())
        
        # Check for quantization config
        has_quant_config = hasattr(model, "config") and hasattr(model.config, "quantization_config")
        
        return has_bitsandbytes or has_gptq or has_awq or has_quant_config
    
    def get_quantization_type(self, model: PreTrainedModel) -> str:
        """
        Determine the quantization type of a model.
        
        Args:
            model: The model to analyze
            
        Returns:
            String indicating the quantization type, or "none" if not quantized
        """
        # Check for bitsandbytes 4-bit
        if any("Linear4bit" in str(type(module)) for _, module in model.named_modules()):
            # Determine 4-bit type (nf4 or fp4)
            if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
                quant_type = model.config.quantization_config.get("bnb_4bit_quant_type", "nf4")
                return f"{quant_type}"
            else:
                return "4bit"
                
        # Check for bitsandbytes 8-bit
        if any("Linear8bitLt" in str(type(module)) for _, module in model.named_modules()):
            return "8bit"
            
        # Check for GPTQ
        if any("QuantLinear" in str(type(module)) for _, module in model.named_modules()):
            return "gptq"
            
        # Check for AWQ
        if any("WQLinear" in str(type(module)) for _, module in model.named_modules()):
            return "awq"
            
        # Check quantization config
        if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
            if model.config.quantization_config.get("load_in_4bit", False):
                quant_type = model.config.quantization_config.get("bnb_4bit_quant_type", "nf4")
                return f"{quant_type}"
            elif model.config.quantization_config.get("load_in_8bit", False):
                return "8bit"
                
        return "none"
    
    def check_peft_compatible(
        self, 
        model: PreTrainedModel,
        peft_method: str
    ) -> Tuple[bool, str]:
        """
        Check if a model is compatible with a specific PEFT method.
        
        Args:
            model: The model to check
            peft_method: The PEFT method to check compatibility for
            
        Returns:
            Tuple of (is_compatible, reason)
        """
        quant_type = self.get_quantization_type(model)
        
        # Check if quantized
        if quant_type == "none":
            # Not quantized, so compatible with any PEFT method
            return True, "Model is not quantized, compatible with PEFT"
            
        # Map of compatible PEFT methods for different quantization types
        compatibility_map = {
            "4bit": ["lora", "qlora"],
            "8bit": ["lora", "qlora", "prefix_tuning", "p_tuning"],
            "nf4": ["lora", "qlora"],
            "fp4": ["lora", "qlora"],
            "gptq": ["lora"],
            "awq": ["lora"]
        }
        
        # Check compatibility
        compatible_methods = compatibility_map.get(quant_type, [])
        
        if peft_method.lower() in [m.lower() for m in compatible_methods]:
            return True, f"{peft_method} is compatible with {quant_type} quantization"
        else:
            return False, f"{peft_method} is not compatible with {quant_type} quantization. " \
                        f"Compatible methods are: {compatible_methods}" 