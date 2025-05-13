# Customization Guide

This guide details all customization options available in the finetune_leveling library.

## Training Framework Selection

Choose between four training frameworks:

1. **Pure PyTorch (`NLPTrainer`)**
   - Maximum flexibility
   - Direct control over training loop
   - Manual optimization
   ```python
   from modules.trainers import NLPTrainer
   ```

2. **PyTorch Lightning (`LightningNLPTrainer`)**
   - Structured training
   - Automatic optimization
   - Built-in distributed training
   ```python
   from modules.trainers import LightningNLPTrainer
   ```

3. **Accelerate (`AcceleratedNLPTrainer`)**
   - Hardware optimization
   - Mixed precision training
   - Easy scaling
   ```python
   from modules.trainers import AcceleratedNLPTrainer
   ```

4. **Accelerate + Lightning (`AcceleratedLightningNLPTrainer`)**
   - Best of both worlds
   - Hardware optimization with structured training
   - Maximum performance
   ```python
   from modules.trainers import AcceleratedLightningNLPTrainer
   ```

## PEFT Methods

### First-Level PEFT Methods

1. **LoRA (Low-Rank Adaptation)**
   ```python
   peft_config = {
       "method": "lora",
       "r": 8,
       "alpha": 16,
       "dropout": 0.05,
       "target_modules": ["q_proj", "v_proj"]
   }
   ```

2. **QLoRA (Quantized LoRA)**
   ```python
   peft_config = {
       "method": "qlora",
       "quantization_bits": 4,
       "double_quant": True,
       "r": 16,
       "alpha": 32
   }
   ```

3. **Adapters**
   ```python
   peft_config = {
       "method": "adapter",
       "adapter_size": 64,
       "adapter_dropout": 0.1,
       "target_modules": ["attention", "mlp"]
   }
   ```

4. **Prefix Tuning**
   ```python
   peft_config = {
       "method": "prefix",
       "num_virtual_tokens": 20,
       "prefix_dropout": 0.1
   }
   ```

### Second-Level PEFT Methods

1. **P-Tuning**
   ```python
   peft_config = {
       "method": "p-tuning",
       "encoder_hidden_size": 512,
       "num_virtual_tokens": 20
   }
   ```

2. **Prompt Tuning**
   ```python
   peft_config = {
       "method": "prompt",
       "num_virtual_tokens": 100,
       "prompt_tuning_init": "TEXT"
   }
   ```

3. **IA3**
   ```python
   peft_config = {
       "method": "ia3",
       "target_modules": ["k_proj", "v_proj", "down_proj"],
       "feedforward_modules": ["down_proj"]
   }
   ```

4. **AdaLoRA**
   ```python
   peft_config = {
       "method": "adalora",
       "r": 8,
       "beta1": 0.85,
       "beta2": 0.95,
       "target_rank": 0.5
   }
   ```

## Training Paradigms

1. **Standard Supervised Learning**
   ```python
   training_config = {
       "paradigm": "supervised",
       "loss": "cross_entropy",
       "metrics": ["accuracy", "f1"]
   }
   ```

2. **RLHF (Reinforcement Learning from Human Feedback)**
   ```python
   training_config = {
       "paradigm": "rlhf",
       "reward_model": "reward_model_path",
       "ppo_config": {
           "num_rollouts": 128,
           "chunk_size": 128,
           "init_kl_coef": 0.2
       }
   }
   ```

3. **DPO (Direct Preference Optimization)**
   ```python
   training_config = {
       "paradigm": "dpo",
       "beta": 0.1,
       "reference_model": "reference_model_path",
       "preference_data": "preferences.json"
   }
   ```

## Hardware Optimization

1. **Mixed Precision Training**
   ```python
   hardware_config = {
       "mixed_precision": {
           "enabled": True,
           "dtype": "fp16",
           "dynamic_scale": True
       }
   }
   ```

2. **Multi-GPU Training**
   ```python
   hardware_config = {
       "distributed": {
           "strategy": "ddp",  # or "deepspeed", "fsdp"
           "find_unused_parameters": False,
           "sync_batch_norm": True
       }
   }
   ```

3. **Memory Management**
   ```python
   hardware_config = {
       "memory": {
           "gradient_checkpointing": True,
           "gradient_accumulation_steps": 4,
           "max_grad_norm": 1.0,
           "dynamic_batch_size": True
       }
   }
   ```

## Logging and Monitoring

1. **Basic Logging**
   ```python
   logging_config = {
       "level": "INFO",
       "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
       "handlers": ["console", "file"]
   }
   ```

2. **Advanced Monitoring**
   ```python
   monitoring_config = {
       "backends": ["tensorboard", "wandb", "mlflow"],
       "metrics": {
           "train_loss": {"type": "line", "alert_threshold": 0.5},
           "gpu_memory": {"type": "gauge", "alert_threshold": 0.9}
       }
   }
   ```

## Example: Full Configuration

```python
from modules.trainers import AcceleratedLightningNLPTrainer

# Create comprehensive configuration
config = {
    "model": {
        "name": "meta-llama/Llama-2-7b-hf",
        "tokenizer": "auto",
        "dtype": "bfloat16"
    },
    "peft": {
        "method": "qlora",
        "quantization_bits": 4,
        "r": 16,
        "alpha": 32,
        "dropout": 0.05
    },
    "training": {
        "paradigm": "dpo",
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.1,
        "max_steps": 1000
    },
    "hardware": {
        "mixed_precision": True,
        "gradient_checkpointing": True,
        "distributed_strategy": "fsdp"
    },
    "monitoring": {
        "backends": ["wandb"],
        "log_every_n_steps": 10,
        "eval_every_n_steps": 100
    }
}

# Initialize trainer
trainer = AcceleratedLightningNLPTrainer(config)

# Start training
trainer.train()
```

## Best Practices

1. **Memory Efficiency**
   - Use QLoRA for large models
   - Enable gradient checkpointing
   - Use dynamic batch sizing

2. **Training Speed**
   - Use mixed precision training
   - Enable flash attention when possible
   - Optimize data loading with smart prefetching

3. **Stability**
   - Start with conservative learning rates
   - Use gradient clipping
   - Monitor training metrics

4. **Customization**
   - Test configurations on small datasets first
   - Use the validation set for hyperparameter tuning
   - Monitor hardware utilization 