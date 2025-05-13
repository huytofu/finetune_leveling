"""
Example script for comparing different PEFT methods.

This script demonstrates how to use and compare different Parameter-Efficient Fine-Tuning
methods on the same task and dataset. It runs multiple fine-tuning jobs with different
PEFT methods and compares their performance, memory usage, and training speed.
"""

import os
import sys
import json
import time
import argparse
import torch
import psutil
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from modules.pipelines import FineTunePipeLine

def parse_args():
    parser = argparse.ArgumentParser(description="Compare different PEFT methods")
    
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
    parser.add_argument("--output_dir", type=str, default="./peft_comparison",
                        help="The output directory where the results will be saved")
    parser.add_argument("--use_accelerate", action="store_true",
                        help="Whether to use Accelerate for training")
    parser.add_argument("--metric", type=str, default="accuracy",
                        help="The metric to use for evaluation")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size per GPU/TPU core/CPU for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="The initial learning rate for AdamW")
    
    # PEFT methods to compare
    parser.add_argument("--methods", type=str, nargs="+", 
                        default=["full", "lora", "qlora", "adapter", "prefix_tuning"],
                        help="PEFT methods to compare")
    
    return parser.parse_args()

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def create_config_file(args, method, method_dir):
    """Create a config file for the specified method."""
    config = {
        "output_dir": method_dir,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_train_batch_size,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 1,
        "push_to_hub": False
    }
    
    # Create config directory if it doesn't exist
    os.makedirs(os.path.join(args.output_dir, "configs"), exist_ok=True)
    
    # Write config to file
    config_path = os.path.join(args.output_dir, "configs", f"{method}_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    return config_path

def get_peft_config(method):
    """Get PEFT configuration for the specified method."""
    if method == "lora":
        return {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none"
        }
    elif method == "qlora":
        return {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            }
        }
    elif method == "adapter":
        return {
            "adapter_size": 64,
            "adapter_dropout": 0.1,
            "adapter_init_scale": 1e-3,
            "adapter_non_linearity": "relu"
        }
    elif method == "prefix_tuning":
        return {
            "num_virtual_tokens": 20,
            "prefix_projection": False
        }
    else:
        return None

def run_method(args, method):
    """Run fine-tuning with the specified method."""
    print(f"\n{'='*50}")
    print(f"Running fine-tuning with method: {method}")
    print(f"{'='*50}\n")
    
    # Create method-specific output directory
    method_dir = os.path.join(args.output_dir, method)
    os.makedirs(method_dir, exist_ok=True)
    
    # Create config file
    config_path = create_config_file(args, method, method_dir)
    
    # Get PEFT configuration
    peft_config = get_peft_config(method)
    
    # Record start time and memory
    start_time = time.time()
    start_memory = get_memory_usage()
    
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
        peft_method=None if method == "full" else method,
        peft_config=peft_config,
        quantization="4bit" if method == "qlora" else None
    )
    
    # Run fine-tuning
    model, tokenizer = pipeline.run()
    
    # Record end time and memory
    end_time = time.time()
    end_memory = get_memory_usage()
    
    # Calculate metrics
    training_time = end_time - start_time
    memory_used = end_memory - start_memory
    
    # Save metrics
    metrics = {
        "method": method,
        "training_time_seconds": training_time,
        "memory_usage_mb": memory_used,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(method_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nFinished {method} fine-tuning:")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Memory used: {memory_used:.2f} MB")
    
    return metrics

def plot_comparison(metrics_list, output_dir):
    """Plot comparison of different methods."""
    methods = [m["method"] for m in metrics_list]
    training_times = [m["training_time_seconds"] for m in metrics_list]
    memory_usages = [m["memory_usage_mb"] for m in metrics_list]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot training time
    ax1.bar(methods, training_times, color='skyblue')
    ax1.set_title('Training Time Comparison')
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Time (seconds)')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot memory usage
    ax2.bar(methods, memory_usages, color='lightgreen')
    ax2.set_title('Memory Usage Comparison')
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Memory (MB)')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'peft_comparison.png'))
    plt.close()

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run each method and collect metrics
    metrics_list = []
    for method in args.methods:
        try:
            metrics = run_method(args, method)
            metrics_list.append(metrics)
        except Exception as e:
            print(f"Error running method {method}: {e}")
    
    # Save all metrics
    with open(os.path.join(args.output_dir, "all_metrics.json"), "w") as f:
        json.dump(metrics_list, f, indent=4)
    
    # Plot comparison
    if len(metrics_list) > 1:
        plot_comparison(metrics_list, args.output_dir)
        print(f"\nComparison plot saved to {os.path.join(args.output_dir, 'peft_comparison.png')}")
    
    print("\nPEFT comparison completed!")

if __name__ == "__main__":
    main() 