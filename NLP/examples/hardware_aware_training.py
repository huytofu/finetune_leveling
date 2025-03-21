import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..modules.training_optimizations import OptimizationConfig, HardwareAwareOptimizer

def main():
    # Initialize model and tokenizer
    model_name = "meta-llama/Llama-2-7b-hf"  # Example model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create optimization config (hardware detection happens automatically)
    config = OptimizationConfig(
        enable_mixed_precision=True,
        enable_gradient_checkpointing=True,
        enable_dynamic_batching=True,
        target_batch_size=8,
        max_batch_size=32
    )
    
    # Initialize hardware-aware optimizer
    optimizer = HardwareAwareOptimizer(model, config)
    
    # Print hardware and optimization summary
    print("\nHardware and Optimization Summary:")
    print(optimizer.get_optimization_summary())
    
    # Example training loop
    def train_step(batch):
        # Your training logic here
        pass
    
    # Example usage in training loop
    print("\nStarting training with hardware-aware optimizations...")
    try:
        # Your training loop here
        for epoch in range(config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
            # Training logic
            pass
    except Exception as e:
        print(f"Training error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 