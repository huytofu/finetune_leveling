# Finetune Leveling Library

A comprehensive library for efficient fine-tuning and training of large language models with extensive customization options.

## Features

### Training Frameworks
- Pure PyTorch for maximum flexibility
- PyTorch Lightning for structured training
- Hugging Face Accelerate for hardware optimization
- Combined Accelerate + Lightning for best of both worlds

### PEFT Support
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- Adapters and Prefix Tuning
- Advanced PEFT methods (P-Tuning, IA3, AdaLoRA)

### Hardware Optimization
- Automatic hardware detection
- Multi-GPU training support
- Mixed precision training
- Dynamic resource allocation

### Memory Management
- Gradient checkpointing
- Dynamic batching
- Memory-mapped datasets
- Smart gradient accumulation

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic training
python cli.py train \
  --model "gpt2" \
  --dataset "imdb" \
  --framework "pytorch" \
  --peft "lora"

# Run advanced training with customization
python cli.py train \
  --model "meta-llama/Llama-2-7b-hf" \
  --dataset "samsum" \
  --framework "lightning" \
  --peft "qlora" \
  --advanced_mode
```

## Documentation Structure

- [Main Documentation](docs/README.md): Overview and getting started
- [PEFT Guide](README_PEFT.md): Detailed PEFT implementation
- [Training Guide](docs/training.md): Training configuration
- [Hardware Guide](docs/hardware.md): Hardware optimization
- [Customization Guide](docs/customization.md): All customization options
- [API Reference](docs/api/README.md): API documentation

## Customization Options

### Training Framework Selection
```python
from modules.trainers import (
    NLPTrainer,  # Pure PyTorch
    LightningNLPTrainer,  # PyTorch Lightning
    AcceleratedNLPTrainer,  # Accelerate
    AcceleratedLightningNLPTrainer  # Combined
)
```

### PEFT Configuration
```python
peft_config = {
    "method": "lora",  # or "qlora", "adapter", "prefix"
    "mode": "basic",   # or "advanced"
    "params": {
        "r": 8,
        "alpha": 16,
        "dropout": 0.05
    }
}
```

### Hardware Optimization
```python
hardware_config = {
    "mixed_precision": "fp16",
    "gradient_checkpointing": True,
    "gradient_accumulation_steps": 4,
    "dynamic_batch_size": True
}
```

## Logging and Monitoring

### Basic Logging
```python
from modules.logging import setup_logging

logger = setup_logging(
    log_level="INFO",
    log_file="training.log",
    console_output=True
)
```

### Advanced Monitoring
```python
from modules.monitoring import (
    TensorBoardLogger,
    WandbLogger,
    MLflowLogger
)

logger = WandbLogger(
    project="my_project",
    metrics=["loss", "accuracy", "gpu_utilization"],
    log_gradients=True
)
```

## Testing

Run comprehensive tests:
```bash
python -m pytest tests/
```

Run specific test categories:
```bash
python -m pytest tests/test_peft.py
python -m pytest tests/test_training.py
python -m pytest tests/test_hardware.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style
- Testing requirements
- Documentation standards
- Pull request process

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details. 