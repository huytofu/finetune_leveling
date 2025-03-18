# Explore LLM

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
git clone https://github.com/yourusername/explore_llm.git
cd explore_llm
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

## Documentation

For more detailed documentation, see:

- [PEFT Methods Guide](NLP/README_PEFT.md)
- [API Reference](docs/api_reference.md)
- [Examples](NLP/examples/)

## Project Structure

```
explore_llm/
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
