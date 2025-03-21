import configargparse

class ConfigManager:
    """
    A class to manage configuration parsing and handling.
    """
    def __init__(self, config_file='config.yaml'):
        """
        Initialize the ConfigManager with a default configuration file.
        
        Args:
            config_file (str): Path to the default configuration file.
        """
        self.parser = configargparse.ArgParser(default_config_files=[config_file])
        self._add_arguments()

    def _add_arguments(self):
        """
        Add configuration arguments to the parser.
        """
        self.parser.add('--output_dir', type=str, help='Output directory for model checkpoints', default='nameless_model')
        self.parser.add('--evaluation_strategy', type=str, help='Evaluation strategy', default='epoch')
        self.parser.add('--scheduler_strategy', type=str, help='Scheduler strategy', default='linear')
        self.parser.add('--num_train_epochs', type=int, help='Number of training epochs', default=3)
        self.parser.add('--learning_rate', type=float, help='Learning rate', default=2e-5)
        self.parser.add('--weight_decay', type=float, help='Weight decay', default=0.01)
        self.parser.add('--push_to_hub', type=bool, help='Whether to push to hub', default=True)
        self.parser.add('--per_device_train_batch_size', type=int, help='Training batch size per device', default=8)
        self.parser.add('--per_device_eval_batch_size', type=int, help='Evaluation batch size per device', default=8)
        self.parser.add('--mlm_probability', type=float, help='Masked language modeling probability', default=0.2)
        self.parser.add('--max_length', type=int, help='Maximum sequence length', default=256)
        self.parser.add('--chunk_size', type=int, help='Chunk size for processing', default=128)
        self.parser.add('--stride', type=int, help='Stride for processing', default=64)
        self.parser.add('--fp16', type=bool, help='Use FP16 precision', default=True)
        self.parser.add('--gradient_accumulation_steps', type=int, help='Gradient accumulation steps', default=1)
        self.parser.add('--max_grad_norm', type=float, help='Maximum gradient norm', default=1.0)

    def parse_args(self):
        """
        Parse the configuration arguments.
        
        Returns:
            Namespace: Parsed arguments.
        """
        return self.parser.parse_args() 