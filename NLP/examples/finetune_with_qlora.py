"""
Example script for fine-tuning a model with QLoRA.

This script demonstrates how to use the QLoRA parameter-efficient fine-tuning method
to fine-tune a model on a specific task. QLoRA combines quantization with LoRA
for memory-efficient fine-tuning of large language models.
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
    parser = argparse.ArgumentParser(description="Fine-tune a model with QLoRA")
    
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
    parser.add_argument("--output_dir", type=str, default="./qlora_model",
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--use_accelerate", action="store_true",
                        help="Whether to use Accelerate for training")
    parser.add_argument("--metric", type=str, default="accuracy",
                        help="The metric to use for evaluation")
    
    # LoRA-specific arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="The rank of the LoRA matrices")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="The alpha parameter for LoRA scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="The dropout probability for LoRA layers")
    parser.add_argument("--target_modules", type=str, default=None,
                        help="Comma-separated list of modules to apply LoRA to")
    
    # Quantization arguments
    parser.add_argument("--quantization_type", type=str, default="4bit",
                        choices=["4bit", "8bit"],
                        help="Type of quantization to use")
    parser.add_argument("--double_quant", action="store_true",
                        help="Whether to use double quantization")
    parser.add_argument("--quant_type", type=str, default="nf4",
                        choices=["nf4", "fp4"],
                        help="Quantization data type to use")
    
    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size per GPU/TPU core/CPU for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Batch size per GPU/TPU core/CPU for evaluation")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
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
    config_path = "./configs/qlora_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    # Prepare LoRA config
    target_modules = args.target_modules.split(",") if args.target_modules else None
    peft_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules,
        "bias": "none"
    }
    
    # Prepare quantization config
    quantization_config = {
        "load_in_4bit": args.quantization_type == "4bit",
        "load_in_8bit": args.quantization_type == "8bit",
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_use_double_quant": args.double_quant,
        "bnb_4bit_quant_type": args.quant_type
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
        peft_method="qlora",
        peft_config=peft_config,
        quantization=args.quantization_type
    )
    
    # Run fine-tuning
    model, tokenizer = pipeline.run()
    
    print(f"Model fine-tuned with QLoRA and saved to {args.output_dir}")

if __name__ == "__main__":
    main() 