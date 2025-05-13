"""
ExploreNLP - A flexible and easy-to-use package for fine-tuning language models.

This package provides tools for fine-tuning, adapting, and using language
models for various NLP tasks, with a focus on simplicity and flexibility.
"""

# Import commonly used modules
from .modules.pipelines import FineTunePipeLine, InferencePipeLine
from .modules.model_modules import ModelModules
from .modules.dataset_modules import DatasetModules
from .modules.tokenizer_modules import TokenizerModules
from .modules.adapter_manager import MultiAdapterManager
from .modules.task_factory import TaskFactory
from .modules.config_manager import ConfigurationManager
from .modules.error_handler import ErrorHandler

# Set package metadata
__version__ = "0.1.0"
__author__ = "Explorer Team"

def create_pipeline(task_type, model_name, dataset_path, output_dir=None, 
                    use_peft=False, peft_method=None, use_lightning=False, 
                    use_accelerate=False, quantization=None):
    """
    Create a fine-tuning pipeline with sensible defaults.
    
    Args:
        task_type (str): Type of task (e.g., text_classification, summarization)
        model_name (str): Name or path of the model
        dataset_path (str): Path to the dataset
        output_dir (str, optional): Output directory for the model
        use_peft (bool, optional): Whether to use PEFT
        peft_method (str, optional): PEFT method to use (default: lora)
        use_lightning (bool, optional): Whether to use PyTorch Lightning
        use_accelerate (bool, optional): Whether to use HF Accelerate
        quantization (str, optional): Quantization type (4bit, 8bit)
        
    Returns:
        A fine-tuning pipeline
    """
    return TaskFactory.create_task(
        task_type=task_type,
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=output_dir,
        use_peft=use_peft,
        peft_method=peft_method,
        use_lightning=use_lightning,
        use_accelerate=use_accelerate
    )

def get_task_templates():
    """
    Get all available task templates.
    
    Returns:
        A dictionary of task templates
    """
    return TaskFactory.get_task_templates()

def get_sample_config(task_type=None):
    """
    Get a sample configuration for a specific task type.
    
    Args:
        task_type (str, optional): The task type to get configuration for
        
    Returns:
        A sample configuration dictionary
    """
    if task_type:
        return TaskFactory.get_task_template(task_type)
    
    # Return a general config if no task type is specified
    from .configs.default_config import DEFAULT_SPECS
    return DEFAULT_SPECS 