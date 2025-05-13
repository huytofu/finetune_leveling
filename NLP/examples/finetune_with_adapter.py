"""
Example script for fine-tuning a model with Adapters.

This script demonstrates how to use the Adapter parameter-efficient fine-tuning method
to fine-tune a model on a specific task. Adapters add small trainable modules between
layers of the model while keeping most of the original parameters frozen.
"""

import os
import sys
import json
import argparse

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from modules.pipelines import FineTunePipeLine

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model with Adapters")
    
    # Required arguments
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="The name of the dataset to use")
    parser.add_argument("--task_type", type=str, required=True, 
                        choices=["masked_language_modeling", "token_classification", "translation", 
                                "summarization", "text_generation", "question_answering"],
                        help="The type of task to fine-tune for")
    
    # Optional arguments
    parser.add_argument("--dataset_config_name", type=str, default=None,
                        help="The configuration name of the dataset to use")
    parser.add_argument("--text_column", type=str, default=None,
                        help="The name of the text column in the dataset")
    parser.add_argument("--summary_column", type=str, default=None,
                        help="The name of the summary column in the dataset")
    parser.add_argument("--output_dir", type=str, default="./adapter_model",
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--use_accelerate", action="store_true",
                        help="Whether to use Accelerate for training")
    parser.add_argument("--metric", type=str, default="accuracy",
                        help="The metric to use for evaluation")
    
    # Adapter-specific arguments
    parser.add_argument("--adapter_size", type=int, default=64,
                        help="The size of the adapter layers")
    parser.add_argument("--adapter_dropout", type=float, default=0.1,
                        help="The dropout probability for adapter layers")
    parser.add_argument("--adapter_init_scale", type=float, default=1e-3,
                        help="The scale for the initialization of adapter weights")
    parser.add_argument("--adapter_non_linearity", type=str, default="relu",
                        choices=["relu", "gelu", "swish"],
                        help="The non-linearity to use in adapter layers")
    
    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size per GPU/TPU core/CPU for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Batch size per GPU/TPU core/CPU for evaluation")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="The initial learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Create config file
    config = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "push_to_hub": False
    }
    
    # Create config directory if it doesn't exist
    os.makedirs("./configs", exist_ok=True)
    
    # Write config to file
    config_path = "./configs/adapter_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    # Prepare adapter config
    peft_config = {
        "adapter_size": args.adapter_size,
        "adapter_dropout": args.adapter_dropout,
        "adapter_init_scale": args.adapter_init_scale,
        "adapter_non_linearity": args.adapter_non_linearity
    }
    
    # Create and run fine-tuning pipeline
    pipeline = FineTunePipeLine(
        args_dir=config_path,
        task_type=args.task_type,
        checkpoint=args.model_name_or_path,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        text_column=args.text_column,
        summary_column=args.summary_column,
        use_bert=False,
        use_accelerate=args.use_accelerate,
        chosen_metric=args.metric,
        peft_method="adapter",
        peft_config=peft_config
    )
    
    # Run fine-tuning
    model, tokenizer = pipeline.run()
    
    print(f"Model fine-tuned with Adapters and saved to {args.output_dir}")

if __name__ == "__main__":
    main() 