DEFAULT_SPECS = {
    'output_dir': 'nameless_model',
    'evaluation_strategy': 'epoch',
    'scheduler_strategy': 'linear',
    'num_train_epochs': 3,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'push_to_hub': True,
    'per_device_train_batch_size': 8,
    'per_device_eval_batch_size': 8,
    'mlm_probability': 0.2,
    'max_length': 256,
    'chunk_size': 128,
    'stride': 64,
    'fp16': True,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    
    # PEFT-related settings
    'use_peft': False,
    'peft_method': 'lora',  # Options: lora, qlora, prefix_tuning, prompt_tuning, p_tuning
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'lora_bias': 'none',  # Options: none, all, lora_only
    
    # Quantization settings
    'use_quantization': False,
    'quantization_type': '4bit',  # Options: 4bit, 8bit
    
    # Optimization settings
    'use_gradient_checkpointing': True,
    'use_deepspeed': False,
    'deepspeed_stage': 2,  # Options: 1, 2, 3
    'use_flash_attention': False
}