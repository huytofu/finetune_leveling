import sys
import logging
from typing import Optional, Dict, Any, List

class ErrorHandler:
    """
    Centralized error handling with detailed messages and recovery suggestions.
    
    This class provides standardized error messages and recovery suggestions
    for common errors in the fine-tuning pipeline.
    """
    
    # Error code definitions with message templates
    ERROR_CODES = {
        "MODEL_NOT_FOUND": "Model '{0}' not found. Please check the model name or path.",
        "DATASET_LOAD_ERROR": "Failed to load dataset from '{0}'. Please check the dataset path and format.",
        "CONFIG_ERROR": "Configuration error: {0}",
        "PEFT_METHOD_ERROR": "PEFT method '{0}' is not supported. Supported methods: lora, qlora, prefix_tuning, prompt_tuning, p_tuning",
        "MEMORY_ERROR": "Not enough memory for the current configuration. Try reducing batch size or model size.",
        "TASK_TYPE_ERROR": "Task type '{0}' is not supported. Supported task types: text_classification, token_classification, etc.",
        "TRAINING_ERROR": "Error during training: {0}",
        "IO_ERROR": "I/O error: {0}",
        "VALIDATION_ERROR": "Validation error: {0}",
        "IMPORT_ERROR": "Failed to import module: {0}",
        "CUDA_ERROR": "CUDA error: {0}",
        "CHECKPOINT_ERROR": "Error with checkpoint: {0}"
    }
    
    # Recovery suggestions by error code
    RECOVERY_SUGGESTIONS = {
        "MEMORY_ERROR": [
            "Reduce batch size (per_device_train_batch_size)",
            "Use gradient accumulation (gradient_accumulation_steps)",
            "Use PEFT or quantization (use_peft=True, use_quantization=True)",
            "Use a smaller model",
            "Enable gradient checkpointing (use_gradient_checkpointing=True)"
        ],
        "MODEL_NOT_FOUND": [
            "Check that the model name is spelled correctly",
            "Ensure you have internet access to download the model",
            "Try using a local path if you have downloaded the model previously"
        ],
        "DATASET_LOAD_ERROR": [
            "Check that the dataset path exists",
            "Ensure the dataset format is correct (CSV, JSON, etc.)",
            "Verify that you have read permissions for the dataset files",
            "Try using a Hugging Face dataset directly"
        ],
        "CUDA_ERROR": [
            "Ensure your GPU drivers are up to date",
            "Check if you have enough GPU memory available",
            "Try running with CPU only (set device=cpu)",
            "Reduce model or batch size"
        ]
    }
    
    @staticmethod
    def handle_error(error_code: str, *args, exit_code: Optional[int] = None, 
                     logger: Optional[logging.Logger] = None) -> None:
        """
        Handle an error with a detailed message and optional exit.
        
        Args:
            error_code: The error code from ERROR_CODES
            args: Arguments to format the error message
            exit_code: If provided, exits with this code
            logger: Optional logger to log the error
        """
        if error_code in ErrorHandler.ERROR_CODES:
            message = ErrorHandler.ERROR_CODES[error_code].format(*args)
            
            # Print error message
            error_msg = f"ERROR: {message}"
            if logger:
                logger.error(error_msg)
            else:
                print(error_msg)
            
            # Print recovery suggestions
            if error_code in ErrorHandler.RECOVERY_SUGGESTIONS:
                suggestions = ErrorHandler.RECOVERY_SUGGESTIONS[error_code]
                if logger:
                    logger.info("\nSuggestions:")
                    for i, suggestion in enumerate(suggestions, 1):
                        logger.info(f"{i}. {suggestion}")
                else:
                    print("\nSuggestions:")
                    for i, suggestion in enumerate(suggestions, 1):
                        print(f"{i}. {suggestion}")
            
            if exit_code is not None:
                sys.exit(exit_code)
        else:
            unknown_error = f"Unknown error: {error_code}"
            if logger:
                logger.error(unknown_error)
            else:
                print(unknown_error)
    
    @staticmethod
    def get_suggestions(error_code: str) -> List[str]:
        """
        Get recovery suggestions for an error code.
        
        Args:
            error_code: The error code from ERROR_CODES
            
        Returns:
            List of recovery suggestions
        """
        return ErrorHandler.RECOVERY_SUGGESTIONS.get(error_code, []) 