import os
import torch
import logging
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

class TypeUtils:
    """
    Utilities for type checking, conversion, and validation in PEFT models.
    This class provides methods to ensure proper type compatibility
    and conversion across different precision formats and devices.
    """
    
    def __init__(self):
        """Initialize the TypeUtils class."""
        self.supported_dtypes = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "int8": torch.int8,
            "int4": None  # No direct PyTorch dtype for int4
        }
        
        self.device_priorities = {
            "cuda": 0,
            "mps": 1,
            "cpu": 2
        }
    
    def check_model_dtypes(self, model: PreTrainedModel) -> Dict[str, Any]:
        """
        Check the data types used in a model.
        
        Args:
            model: The model to check
            
        Returns:
            Dict with summary of model dtypes
        """
        dtype_counts = {}
        params_count = {}
        total_params = 0
        
        # Check all parameters
        for name, param in model.named_parameters():
            dtype_str = str(param.dtype).split(".")[-1]
            
            if dtype_str not in dtype_counts:
                dtype_counts[dtype_str] = 0
                params_count[dtype_str] = 0
            
            dtype_counts[dtype_str] += 1
            params_count[dtype_str] += param.numel()
            total_params += param.numel()
        
        # Calculate percentages
        dtype_percentages = {
            dtype: count / total_params * 100 
            for dtype, count in params_count.items()
        }
        
        # Detect if model is in mixed precision
        is_mixed_precision = len(dtype_counts) > 1
        
        # Determine primary dtype (most common)
        primary_dtype = max(params_count.items(), key=lambda x: x[1])[0] if params_count else None
        
        return {
            "dtype_counts": dtype_counts,
            "params_count": params_count,
            "total_params": total_params,
            "dtype_percentages": dtype_percentages,
            "is_mixed_precision": is_mixed_precision,
            "primary_dtype": primary_dtype
        }
    
    def convert_model_dtype(
        self, 
        model: PreTrainedModel,
        target_dtype: str,
        exclude_modules: Optional[List[str]] = None
    ) -> PreTrainedModel:
        """
        Convert model parameters to the target data type.
        
        Args:
            model: The model to convert
            target_dtype: Target data type (fp32, fp16, bf16)
            exclude_modules: List of module names to exclude from conversion
            
        Returns:
            Model with converted dtype
        """
        if target_dtype not in self.supported_dtypes:
            logger.warning(f"Unsupported dtype: {target_dtype}. "
                         f"Supported types: {list(self.supported_dtypes.keys())}")
            return model
            
        pytorch_dtype = self.supported_dtypes[target_dtype]
        
        # Skip if trying to convert to int4/int8 (requires specialized quantization)
        if target_dtype in ["int8", "int4"]:
            logger.warning(f"Cannot directly convert to {target_dtype}. "
                         f"Use the QuantizationManager for quantization.")
            return model
        
        # Exclude modules if specified
        excluded = set(exclude_modules) if exclude_modules else set()
        
        try:
            from peft import PeftModel
            
            # Special handling for PEFT models
            if isinstance(model, PeftModel):
                logger.info(f"Converting PEFT model to {target_dtype}")
                
                # For PEFT models, only convert the trainable adapter parameters by default
                # unless specific exclusions are provided
                if not exclude_modules:
                    # Count trainable parameters before conversion
                    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    
                    # Convert only trainable parameters
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            parent_module_name = ".".join(name.split(".")[:-1])
                            if parent_module_name not in excluded:
                                param.data = param.data.to(pytorch_dtype)
                    
                    # Verify trainable parameters after conversion
                    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    logger.info(f"Converted {trainable_after} trainable parameters to {target_dtype}")
                    assert trainable_before == trainable_after, "Trainable parameter count changed during conversion"
                    
                    # Return early to avoid base model conversion
                    return model
            
            # Standard conversion for non-PEFT models or PEFT models with explicit exclusions
            params_converted = 0
            for name, param in model.named_parameters():
                parent_module_name = ".".join(name.split(".")[:-1])
                if parent_module_name not in excluded:
                    param.data = param.data.to(pytorch_dtype)
                    params_converted += param.numel()
            
            logger.info(f"Converted {params_converted} parameters to {target_dtype}")
            return model
            
        except (ImportError, RuntimeError) as e:
            logger.warning(f"Error during model conversion: {e}")
            # Fallback to standard PyTorch conversion
            try:
                if pytorch_dtype == torch.float16:
                    model = model.half()
                elif pytorch_dtype == torch.bfloat16:
                    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                        model = model.to(pytorch_dtype)
                    else:
                        logger.warning("BF16 not supported on this device, keeping original dtype")
                elif pytorch_dtype == torch.float32:
                    model = model.float()
                
                logger.info(f"Converted model to {target_dtype} using PyTorch conversion")
                return model
                
            except RuntimeError as e:
                logger.error(f"Failed to convert model dtype: {e}")
                return model
    
    def optimize_memory_layout(
        self, 
        model: PreTrainedModel,
        peft_method: Optional[str] = None
    ) -> PreTrainedModel:
        """
        Optimize memory layout for the model, with special handling for PEFT models.
        
        Args:
            model: The model to optimize
            peft_method: Optional PEFT method to optimize for
            
        Returns:
            Model with optimized memory layout
        """
        try:
            # Default is to apply torch.compile (PyTorch 2.0+) for non-adapter layers
            if hasattr(torch, "compile") and not peft_method:
                logger.info("Applying torch.compile for memory and performance optimization")
                # Avoid applying to adapter components
                model = torch.compile(model)
                
            # Apply memory efficient attention if available
            if hasattr(model, "config"):
                if hasattr(model.config, "use_cache"):
                    # Disable KV cache during training to save memory
                    model.config.use_cache = False
                    logger.info("Disabled KV cache for training to save memory")
                
                # For Flash Attention, check if the model supports it
                if hasattr(model, "enable_xformers_memory_efficient_attention"):
                    model.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xformers memory efficient attention")
                    
            return model
            
        except Exception as e:
            logger.warning(f"Error optimizing memory layout: {e}")
            return model
    
    def validate_device_compatibility(
        self, 
        model: PreTrainedModel
    ) -> Dict[str, Any]:
        """
        Validate device compatibility and consistency for a model.
        
        Args:
            model: The model to validate
            
        Returns:
            Dict with device compatibility information
        """
        devices = {}
        inconsistent_devices = False
        primary_device = None
        
        # Check module devices
        for name, module in model.named_modules():
            if list(module.parameters()):
                # Get the device of the first parameter
                param = next(module.parameters())
                device = param.device
                
                if device not in devices:
                    devices[device] = []
                
                devices[device].append(name)
                
                # Track primary device
                if primary_device is None:
                    primary_device = device
                elif primary_device != device:
                    inconsistent_devices = True
        
        # Check for multiple devices
        is_distributed = len(devices) > 1
        
        # Sort devices by priority
        best_device = min(devices.keys(), key=lambda d: self.device_priorities.get(d.type, 999)) if devices else None
        
        return {
            "devices": devices,
            "inconsistent_devices": inconsistent_devices,
            "is_distributed": is_distributed,
            "primary_device": primary_device,
            "best_device": best_device
        }
    
    def convert_peft_format(
        self, 
        model: PreTrainedModel,
        source_peft_type: str,
        target_peft_type: str,
        adapter_name: str = "default"
    ) -> Tuple[bool, PreTrainedModel, str]:
        """
        Attempt to convert between different PEFT formats.
        This is experimental and may not work for all model types.
        
        Args:
            model: The PEFT model to convert
            source_peft_type: Source PEFT type (lora, prefix_tuning, etc.)
            target_peft_type: Target PEFT type
            adapter_name: Adapter name to convert
            
        Returns:
            Tuple of (success, model, message)
        """
        try:
            from peft import PeftModel, LoraConfig, PrefixTuningConfig, PromptTuningConfig
            
            if not isinstance(model, PeftModel):
                return False, model, "Model is not a PEFT model"
                
            # Currently, we only support conversion from LoRA to other types
            if source_peft_type.lower() != "lora":
                return False, model, f"Conversion from {source_peft_type} is not supported"
                
            # Get the adapter config
            if hasattr(model, "peft_config") and adapter_name in model.peft_config:
                # Get the config for the adapter
                config = model.peft_config[adapter_name]
                
                # Extract common parameters
                common_params = {
                    "task_type": getattr(config, "task_type", None),
                    "inference_mode": False,
                }
                
                # Get trainable parameters from the existing adapter
                lora_params = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and adapter_name in name:
                        lora_params[name] = param.data.clone()
                
                # Create a new config based on the target type
                if target_peft_type.lower() == "prefix_tuning":
                    # Extract model hidden size
                    hidden_size = getattr(model.base_model.config, "hidden_size", 768)
                    
                    # Create new config
                    new_config = PrefixTuningConfig(
                        num_virtual_tokens=20,  # Default value
                        prefix_projection=True,
                        **common_params
                    )
                elif target_peft_type.lower() == "prompt_tuning":
                    new_config = PromptTuningConfig(
                        num_virtual_tokens=20,  # Default value
                        **common_params
                    )
                else:
                    return False, model, f"Conversion to {target_peft_type} is not supported"
                
                # Store the current adapter state
                adapter_state = model.get_adapter_state_dict(adapter_name)
                
                # Create a new adapter with the new config
                model.add_adapter(f"{adapter_name}_new", new_config)
                
                logger.warning(f"Converted adapter from {source_peft_type} to {target_peft_type}. "
                             f"Note: This conversion is experimental and parameter values are initialized randomly.")
                
                return True, model, f"Successfully created new adapter {adapter_name}_new with type {target_peft_type}"
                
            else:
                return False, model, f"Adapter {adapter_name} not found in model"
                
        except (ImportError, AttributeError, TypeError) as e:
            logger.error(f"Error during PEFT format conversion: {e}")
            return False, model, f"Error: {str(e)}"
    
    def check_parameter_types(
        self,
        model: PreTrainedModel,
        check_fn: Optional[Callable] = None
    ) -> Dict[str, List[str]]:
        """
        Check parameter types throughout the model and identify potential issues.
        
        Args:
            model: Model to check
            check_fn: Optional custom validation function
            
        Returns:
            Dict with lists of parameter names that might have issues
        """
        results = {
            "nan_params": [],
            "inf_params": [],
            "zero_grad_params": [],
            "zero_params": [],
            "high_magnitude_params": [],
            "inconsistent_dtype_params": []
        }
        
        # Get the most common dtype
        dtype_info = self.check_model_dtypes(model)
        primary_dtype = dtype_info["primary_dtype"]
        
        # Check each parameter
        for name, param in model.named_parameters():
            # Check for NaN
            if torch.isnan(param).any():
                results["nan_params"].append(name)
                
            # Check for Inf
            if torch.isinf(param).any():
                results["inf_params"].append(name)
                
            # Check for zero gradients in trainable parameters
            if param.requires_grad and param.grad is not None:
                if (param.grad == 0).all():
                    results["zero_grad_params"].append(name)
                    
            # Check for all-zero parameters
            if (param == 0).all():
                results["zero_params"].append(name)
                
            # Check for high magnitude values
            if torch.abs(param).max() > 100:  # Arbitrary threshold
                results["high_magnitude_params"].append(name)
                
            # Check for inconsistent dtypes
            if str(param.dtype).split(".")[-1] != primary_dtype:
                results["inconsistent_dtype_params"].append(name)
                
            # Apply custom check if provided
            if check_fn is not None:
                try:
                    if not check_fn(name, param):
                        if "custom_check_failed" not in results:
                            results["custom_check_failed"] = []
                        results["custom_check_failed"].append(name)
                except Exception as e:
                    logger.warning(f"Custom check failed for {name}: {e}")
        
        # Log summary of issues
        for issue_type, params in results.items():
            if params:
                logger.warning(f"Found {len(params)} parameters with {issue_type.replace('_', ' ')}")
                
        return results
    
    def automatic_type_repair(
        self,
        model: PreTrainedModel,
        repair_nan: bool = True,
        repair_inf: bool = True,
        consolidate_dtypes: bool = False,
        nan_value: float = 0.0,
        inf_value: float = 1.0
    ) -> PreTrainedModel:
        """
        Automatically repair common type issues in a model.
        
        Args:
            model: The model to repair
            repair_nan: Whether to replace NaN values
            repair_inf: Whether to replace Inf values
            consolidate_dtypes: Whether to consolidate to a single dtype
            nan_value: Value to replace NaNs with
            inf_value: Value to replace Infs with
            
        Returns:
            Repaired model
        """
        # Check parameter types
        issues = self.check_parameter_types(model)
        
        # Fix NaN values
        if repair_nan and issues["nan_params"]:
            for name in issues["nan_params"]:
                param = model.get_parameter(name)
                param.data = torch.where(torch.isnan(param.data), 
                                         torch.tensor(nan_value, device=param.device, dtype=param.dtype), 
                                         param.data)
            logger.info(f"Replaced NaN values in {len(issues['nan_params'])} parameters")
        
        # Fix Inf values
        if repair_inf and issues["inf_params"]:
            for name in issues["inf_params"]:
                param = model.get_parameter(name)
                param.data = torch.where(torch.isinf(param.data), 
                                         torch.tensor(inf_value, device=param.device, dtype=param.dtype), 
                                         param.data)
            logger.info(f"Replaced Inf values in {len(issues['inf_params'])} parameters")
            
        # Consolidate dtypes if requested
        if consolidate_dtypes and issues["inconsistent_dtype_params"]:
            # Get the target dtype
            dtype_info = self.check_model_dtypes(model)
            primary_dtype = dtype_info["primary_dtype"]
            target_dtype = next((k for k, v in self.supported_dtypes.items() 
                              if str(v).split(".")[-1] == primary_dtype), "fp32")
            
            # Only convert inconsistent parameters
            self.convert_model_dtype(
                model, 
                target_dtype, 
                exclude_modules=[name for name, _ in model.named_parameters() 
                                if name not in issues["inconsistent_dtype_params"]]
            )
            logger.info(f"Consolidated {len(issues['inconsistent_dtype_params'])} parameters to {target_dtype}")
            
        return model 