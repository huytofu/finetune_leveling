# Finetune Levelling

A flexible framework for exploring, fine-tuning, and deploying language models with a focus on efficiency and performance.

## Features

- **Modular Architecture**: Easily extensible components for models, tokenizers, datasets, and training
- **Task Flexibility**: Support for various NLP tasks including:
  - Masked Language Modeling
  - Token Classification
  - Translation
  - Summarization
  - Text Generation
  - Question Answering
- **Parameter-Efficient Fine-Tuning**: Integrated support for:
  - LoRA (Low-Rank Adaptation)
  - QLoRA (Quantized LoRA)
  - Adapters
  - Prefix Tuning
  - Prompt Tuning
  - P-Tuning
- **Quantization Support**: 4-bit and 8-bit quantization for memory-efficient training and inference
- **Accelerated Training**: Integration with Hugging Face Accelerate for distributed training

## Installation

```bash
git clone https://github.com/yourusername/finetune_leveling.git
cd finetune_leveling
pip install -r requirements.txt
```

## Quick Start

### Fine-tuning with LoRA

```bash
python NLP/examples/finetune_with_lora.py \
  --model_name_or_path "gpt2" \
  --dataset_name "imdb" \
  --task_type "text_generation" \
  --lora_r 8 \
  --lora_alpha 16 \
  --num_train_epochs 3
```

### Fine-tuning with QLoRA

```bash
python NLP/examples/finetune_with_qlora.py \
  --model_name_or_path "meta-llama/Llama-2-7b-hf" \
  --dataset_name "samsum" \
  --task_type "summarization" \
  --quantization_type "4bit" \
  --double_quant \
  --lora_r 16
```

### Fine-tuning with Adapters

```bash
python NLP/examples/finetune_with_adapter.py \
  --model_name_or_path "bert-base-uncased" \
  --dataset_name "glue" \
  --dataset_config_name "sst2" \
  --task_type "text_classification" \
  --adapter_size 64
```

### Comparing PEFT Methods

```bash
python NLP/examples/peft_comparison.py \
  --model_name_or_path "distilbert-base-uncased" \
  --dataset_name "glue" \
  --dataset_config_name "mrpc" \
  --task_type "token_classification" \
  --methods lora qlora adapter prefix_tuning
```

## Inference with Multiple Adapters

The `InferencePipeline` class allows you to run inference with multiple adapters and switch between them dynamically. Here's how you can use it:

### Running Inference with Adapters

To run inference with a specific adapter, you can specify the adapter ID when calling the `run` method:

```python
from NLP.modules.pipelines import InferencePipeLine

# Initialize the pipeline
pipeline = InferencePipeLine(
    task_type="text_generation",
    checkpoint="path/to/your/model",
    adapter_config={
        'cache_dir': './adapter_cache',
        'max_adapters': 5
    }
)

# Run inference with a specific adapter
result = pipeline.run("Your input text here", adapter_id="adapter_id")
print(result)
```

### Automatic Adapter Selection

You can let the system decide which adapter to use by enabling the `auto_adapter` option:

```python
result = pipeline.run("Your input text here", auto_adapter=True)
```

### Adapter Selection with LLM

To use the base model to decide the best adapter based on adapter names, enable the `select_adapter_with_llm` option:

```python
result = pipeline.run("Your input text here", auto_adapter=True, select_adapter_with_llm=True)
```

### Command Line Interface

You can also use the provided example script to interactively switch between adapters:

```bash
python NLP/examples/multi_adapter_inference.py \
  --model_name_or_path "path/to/your/model" \
  --task_type "text_generation" \
  --adapters_dir "./adapters" \
  --auto_adapter
```

This script allows you to switch adapters interactively and run inference with the selected adapter.

## Documentation

For more detailed documentation, see:

- [PEFT Methods Guide](NLP/README_PEFT.md)
- [API Reference](docs/api_reference.md)
- [Examples](NLP/examples/)

## Project Structure

```
finetune_leveling/
├── NLP/
│   ├── classes/
│   │   ├── accelerated_trainers.py
│   │   └── trainers.py
│   ├── configs/
│   │   └── default_config.py
│   ├── examples/
│   │   ├── finetune_with_adapter.py
│   │   ├── finetune_with_lora.py
│   │   ├── finetune_with_qlora.py
│   │   └── peft_comparison.py
│   ├── modules/
│   │   ├── dataset_modules.py
│   │   ├── model_modules.py
│   │   ├── peft_modules.py
│   │   ├── pipelines.py
│   │   ├── pretrain_modules.py
│   │   └── tokenizer_modules.py
│   └── README_PEFT.md
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
