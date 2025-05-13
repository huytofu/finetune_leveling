#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, Any

# Add parent directory to path to allow imports
currdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currdir)

from modules.task_factory import TaskFactory
from modules.config_manager import ConfigurationManager
from modules.error_handler import ErrorHandler

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--task", 
        type=str, 
        required=True, 
        choices=list(TaskFactory.TASK_TEMPLATES.keys()),
        help="Task type"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Model name or path"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        help="Dataset path"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Output directory"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=None, 
        help="Configuration file path"
    )
    
    # PEFT arguments
    parser.add_argument(
        "--use-peft", 
        action="store_true", 
        help="Use PEFT"
    )
    parser.add_argument(
        "--peft-method", 
        type=str, 
        default="lora", 
        choices=["lora", "qlora", "prefix_tuning", "prompt_tuning", "p_tuning"],
        help="PEFT method to use"
    )
    
    # Training framework arguments
    parser.add_argument(
        "--use-accelerate", 
        action="store_true", 
        help="Use HF Accelerate"
    )
    parser.add_argument(
        "--use-lightning", 
        action="store_true", 
        help="Use PyTorch Lightning"
    )
    
    # Advanced options
    parser.add_argument(
        "--quantization", 
        type=str, 
        choices=["4bit", "8bit", "none"], 
        default="none",
        help="Quantization type"
    )
    parser.add_argument(
        "--list-templates", 
        action="store_true", 
        help="List available task templates and exit"
    )
    
    return parser.parse_args()

def list_templates():
    """List all available task templates."""
    templates = TaskFactory.get_task_templates()
    print("Available task templates:")
    print("------------------------")
    for task, config in templates.items():
        print(f"\n{task}:")
        for key, value in config.items():
            print(f"  {key}: {value}")

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # List templates and exit if requested
    if args.list_templates:
        list_templates()
        return 0
    
    # Create the pipeline
    try:
        # Load config if provided
        config = {}
        if args.config:
            try:
                with open(args.config, 'r') as f:
                    config = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                ErrorHandler.handle_error("CONFIG_ERROR", str(e))
                return 1
        
        # Set quantization if provided
        quantization = None
        if args.quantization != "none":
            quantization = args.quantization
        
        # Create the pipeline
        pipeline = TaskFactory.create_task(
            task_type=args.task,
            model_name=args.model,
            dataset_path=args.dataset,
            output_dir=args.output,
            use_peft=args.use_peft,
            peft_method=args.peft_method,
            use_lightning=args.use_lightning,
            use_accelerate=args.use_accelerate
        )
        
        # Update with config if provided
        if config:
            pipeline.specs.update(config)
        
        # Set quantization
        if quantization:
            pipeline.quantization = quantization
            pipeline.specs["use_quantization"] = True
            pipeline.specs["quantization_type"] = quantization
        
        # Run the pipeline
        print(f"Starting fine-tuning for task: {args.task}")
        print(f"Model: {args.model}")
        print(f"Dataset: {args.dataset}")
        print(f"Output directory: {pipeline.specs['output_dir']}")
        
        model, tokenizer = pipeline.run()
        
        print(f"\nTraining complete!")
        print(f"Model saved to {pipeline.specs['output_dir']}")
        return 0
        
    except Exception as e:
        ErrorHandler.handle_error("TRAINING_ERROR", str(e))
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 