import os
import json
from typing import Dict, Any, Optional

from .config_manager import ConfigurationManager
from .error_handler import ErrorHandler

class TaskFactory:
    """
    Factory for creating pre-configured task setups.
    
    This class provides easy-to-use factory methods for creating
    common fine-tuning tasks with sensible defaults.
    """
    
    # Task templates with sensible defaults
    TASK_TEMPLATES = {
        "masked_language_modeling": {
            "per_device_train_batch_size": 8,
            "num_train_epochs": 3,
            "learning_rate": 2e-5,
            "mlm_probability": 0.15,
            "max_length": 256
        },
        "text_classification": {
            "per_device_train_batch_size": 16,
            "num_train_epochs": 3,
            "learning_rate":
            2e-5,
            "max_length": 128
        },
        "token_classification": {
            "per_device_train_batch_size": 16,
            "num_train_epochs": 3,
            "learning_rate": 3e-5,
            "max_length": 128
        },
        "question_answering": {
            "per_device_train_batch_size": 8,
            "num_train_epochs": 2,
            "learning_rate": 3e-5,
            "max_length": 384,
            "doc_stride": 128
        },
        "summarization": {
            "per_device_train_batch_size": 4,
            "num_train_epochs": 4,
            "learning_rate": 5e-5,
            "max_length": 512,
            "min_summary_length": 30,
            "max_summary_length": 120
        },
        "translation": {
            "per_device_train_batch_size": 4,
            "num_train_epochs": 3,
            "learning_rate": 4e-5,
            "max_length": 256
        },
        "text_generation": {
            "per_device_train_batch_size": 2,
            "num_train_epochs": 3,
            "learning_rate": 5e-5,
            "max_length": 1024,
            "gradient_accumulation_steps": 4
        }
    }
    
    @staticmethod
    def create_task(task_type: str, model_name: str, dataset_path: str, 
                    output_dir: Optional[str] = None, 
                    use_peft: bool = False,
                    peft_method: Optional[str] = None,
                    use_lightning: bool = False,
                    use_accelerate: bool = False):
        """
        Create a pre-configured task setup based on task type.
        
        Args:
            task_type: The type of task (e.g., "text_classification")
            model_name: The name or path of the model
            dataset_path: Path to the dataset
            output_dir: Output directory (defaults to model_name_task_type)
            use_peft: Whether to use PEFT
            peft_method: PEFT method to use (default: "lora")
            use_lightning: Whether to use PyTorch Lightning
            use_accelerate: Whether to use Hugging Face Accelerate
            
        Returns:
            A configured FineTunePipeLine instance
        """
        # Import here to avoid circular imports
        from .pipelines import FineTunePipeLine
        
        # Validate task type
        if task_type not in TaskFactory.TASK_TEMPLATES:
            ErrorHandler.handle_error("TASK_TYPE_ERROR", task_type)
            task_type = "text_classification"  # Fallback to a common task
            print(f"Falling back to {task_type} task type")
        
        # Create configuration from template
        config = ConfigurationManager()
        
        # Set common configurations
        config.set("model_name", model_name)
        config.set("dataset_dir", dataset_path)
        config.set("output_dir", output_dir or f"{model_name.split('/')[-1]}_{task_type}_output")
        
        # Apply task-specific template
        for key, value in TaskFactory.TASK_TEMPLATES[task_type].items():
            config.set(key, value)
        
        # Set PEFT configurations if requested
        if use_peft:
            config.set("use_peft", True)
            config.set("peft_method", peft_method or "lora")
        
        # Create temporary config file
        config_path = os.path.join(os.path.dirname(__file__), f"tmp_{task_type}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config.get_config(), f, indent=2)
        
        # Create pipeline with configuration
        pipeline = FineTunePipeLine(
            args_dir=config_path,
            task_type=task_type,
            use_lightning=use_lightning,
            use_accelerate=use_accelerate,
            peft_method=peft_method if use_peft else None
        )
        
        # Remove temporary config file
        try:
            os.remove(config_path)
        except:
            pass
        
        return pipeline
    
    @staticmethod
    def get_task_templates():
        """Get all available task templates."""
        return TaskFactory.TASK_TEMPLATES.copy()
    
    @staticmethod
    def get_task_template(task_type: str) -> Dict[str, Any]:
        """
        Get a specific task template.
        
        Args:
            task_type: The type of task
            
        Returns:
            The task template configuration
        """
        if task_type not in TaskFactory.TASK_TEMPLATES:
            ErrorHandler.handle_error("TASK_TYPE_ERROR", task_type)
            return {}
        return TaskFactory.TASK_TEMPLATES[task_type].copy() 