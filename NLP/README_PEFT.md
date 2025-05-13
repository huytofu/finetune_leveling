# Parameter-Efficient Fine-Tuning (PEFT) in finetune_leveling

This document explains how to use the Parameter-Efficient Fine-Tuning (PEFT) features implemented in the `finetune_leveling` library.

## Overview

PEFT methods allow you to fine-tune large language models with significantly reduced memory requirements and computational costs. The following PEFT methods are supported:

- **LoRA (Low-Rank Adaptation)**: Adds trainable low-rank matrices to existing weights, keeping original weights frozen
- **QLoRA**: Combines quantization with LoRA for even more memory-efficient fine-tuning
- **Adapters**: Adds small trainable modules between layers of the model
- **Prefix Tuning**: Prepends trainable continuous vectors to the input of transformer layers

## Requirements

To use the PEFT features, you need to install the following dependencies:

```bash
pip install peft transformers accelerate bitsandbytes
```

## Usage Examples

### Fine-tuning with LoRA

```bash
python examples/finetune_with_lora.py \
  --model_name_or_path "gpt2" \
  --dataset_name "imdb" \
  --task_type "text_generation" \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --target_modules "c_attn,c_proj" \
  --num_train_epochs 3 \
  --learning_rate 5e-4 \
  --per_device_train_batch_size 4
```

### Fine-tuning with QLoRA

```bash
python examples/finetune_with_qlora.py \
  --model_name_or_path "meta-llama/Llama-2-7b-hf" \
  --dataset_name "samsum" \
  --task_type "summarization" \
  --quantization_type "4bit" \
  --double_quant \
  --quant_type "nf4" \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4
```

## Programmatic Usage

You can also use the PEFT features programmatically:

```python
from modules.pipelines import FineTunePipeLine

# Configure LoRA parameters
peft_config = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "bias": "none"
}

# Create and run fine-tuning pipeline
pipeline = FineTunePipeLine(
    args_dir="./configs/my_config.json",
    task_type="text_generation",
    checkpoint="gpt2",
    dataset_name="imdb",
    use_accelerate=True,
    peft_method="lora",
    peft_config=peft_config
)

# Run fine-tuning
model, tokenizer = pipeline.run()
```

## Supported Models

The PEFT methods have been tested with the following model architectures:

- BERT and RoBERTa-based models
- GPT-2 and GPT-Neo models
- T5 and BART models
- Llama and Llama-2 models

## Advanced Configuration

### Quantization Options

- **4-bit Quantization**: Reduces model size by 4x compared to FP16
- **8-bit Quantization**: Reduces model size by 2x compared to FP16
- **Double Quantization**: Further reduces memory usage by quantizing the quantization constants

### Target Modules

Different model architectures have different module names. Here are some common target modules for popular architectures:

- **BERT/RoBERTa**: `query`, `key`, `value`
- **GPT-2**: `c_attn`, `c_proj`
- **T5**: `q`, `k`, `v`, `o`
- **Llama/Llama-2**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `down_proj`, `up_proj`

## Troubleshooting

- If you encounter CUDA out-of-memory errors, try:
  - Reducing batch size
  - Using gradient accumulation
  - Switching to QLoRA instead of LoRA
  - Reducing the model size or LoRA rank (r)

- If training is unstable:
  - Try a lower learning rate
  - Increase LoRA alpha
  - Check if the target modules are appropriate for your model architecture 