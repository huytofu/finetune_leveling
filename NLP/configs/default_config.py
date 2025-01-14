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
    'fp16': True
}