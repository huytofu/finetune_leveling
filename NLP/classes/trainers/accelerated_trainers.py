import os
import sys
import json
import logging
import torch
from typing import Dict, Any, Optional, List
from transformers import PreTrainedModel, PreTrainedTokenizer
from accelerate import Accelerator

# Add to Python path
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parentdir)

# Local imports
from configs.default_config import DEFAULT_SPECS
from modules.customizations_and_optimizations.trainer_customization import TrainerCustomizationMixin
from modules.customizations_and_optimizations.trainer_utils import (
    PeftConfig,
    check_is_peft_model,
    prepare_scheduler,
    calculate_training_steps,
    configure_optimizer,
    clip_gradients,
    save_model_checkpoint,
    get_default_metrics,
    optimize_memory_settings,
    setup_accelerate_integration
)
from modules.monitoring_and_tracking.mlflow_tracking import MLflowTracker
from modules.config.training_config import FineTuneConfig

logger = logging.getLogger(__name__)

# Initialize the parser
p = configargparse.ArgParser(default_config_files=["config.yaml"])

# Add configuration arguments
p.add('--output_dir', type=str, help='Output directory for model checkpoints', default='nameless_model')
p.add('--evaluation_strategy', type=str, help='Evaluation strategy', default='epoch')
p.add('--scheduler_strategy', type=str, help='Scheduler strategy', default='linear')
p.add('--num_train_epochs', type=int, help='Number of training epochs', default=3)
p.add('--learning_rate', type=float, help='Learning rate', default=2e-5)
p.add('--weight_decay', type=float, help='Weight decay', default=0.01)
p.add('--push_to_hub', type=bool, help='Whether to push to hub', default=True)
p.add('--per_device_train_batch_size', type=int, help='Training batch size per device', default=8)
p.add('--per_device_eval_batch_size', type=int, help='Evaluation batch size per device', default=8)
p.add('--mlm_probability', type=float, help='Masked language modeling probability', default=0.2)
p.add('--max_length', type=int, help='Maximum sequence length', default=256)
p.add('--chunk_size', type=int, help='Chunk size for processing', default=128)
p.add('--stride', type=int, help='Stride for processing', default=64)
p.add('--fp16', type=bool, help='Use FP16 precision', default=True)
p.add('--gradient_accumulation_steps', type=int, help='Gradient accumulation steps', default=1)
p.add('--max_grad_norm', type=float, help='Maximum gradient norm', default=1.0)

# Parse arguments
args = p.parse_args()

class AcceleratedNLPTrainer(TrainerCustomizationMixin):
    """
    A trainer class for NLP models using the Hugging Face Accelerate library.
    
    This trainer handles the training loop, evaluation, and optimization for NLP models.
    It integrates with the Accelerate library for distributed training and optimization,
    and can be used independently or as part of a fine-tuning pipeline.
    
    Role in the Pipeline:
    - Manages the training loop and optimization process
    - Handles distributed training via Accelerate
    - Supports PEFT (Parameter-Efficient Fine-Tuning) methods
    - Provides hooks for callbacks and metrics tracking
    - Supports checkpoint saving and loading
    """
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, raw_dataset,
                task_type, optimizer=None, chosen_metric=None, 
                customization_config=None):
        """Initialize the AcceleratedNLPTrainer."""
        try:
            # Load configuration specifications
            specs = json.load(open(args_dir, 'r'))
            self.specs = {**DEFAULT_SPECS, **specs}
            
            # Store model and data components
            self.model = model
            self.tokenizer = tokenizer
            self.data_collator = data_collator
            self.task_type = task_type
            self.chosen_metric = chosen_metric
            
            # Initialize Accelerator
            self.accelerator = setup_accelerate_integration(self.specs)
            
            # Check if this is a PEFT model before initializing
            self.is_peft_model = check_is_peft_model(model)
            
            # Initialize customization
            self.setup_customization(customization_config)
            
            # Initialize optimizer and scheduler
            self.optimizer = self.configure_optimizers()
            if isinstance(self.optimizer, dict):
                self.optimizer = self.optimizer["optimizer"]
                self.scheduler = self.optimizer["lr_scheduler"]["scheduler"]
            else:
                self.scheduler = None
            
            # Prepare datasets and model with Accelerator
            self.train_dataset, self.eval_dataset = self.prepare_with_accelerator(train_dataset, eval_dataset)
            self.raw_dataset = raw_dataset
            
            # Initialize metrics
            self.losses = []
            self.current_loss = 0
            self.best_metric = float('inf')
            self.no_improvement_count = 0
            
            # Initialize metrics based on task type
            self.metric = get_default_metrics(task_type)
            
            # Optimize memory settings if needed
            if self.is_peft_model:
                optimize_memory_settings(
                    model=self.model,
                    use_gradient_checkpointing=self.specs.get('gradient_checkpointing', True)
                )
            
            self.setup_configuration()
            self.peft_config = PeftConfig.from_pretrained(model) if self.is_peft_model else None
            
            self.progress_bar = tqdm(range(self.num_training_steps))
            
            self._setup_gradient_accumulation()
            
            # Lightning coordination flag - set to False as this is Accelerate-only
            self.use_lightning = False
            self.use_accelerate = True
            
            logger.info("AcceleratedNLPTrainer initialized successfully.")
        except Exception as e:
            ErrorHandler.handle_error(e, "AcceleratedNLPTrainer initialization")

    def setup_configuration(self):
        """Set up configuration using ConfigManager."""
        config_manager = ConfigManager()
        self.specs = vars(config_manager.parse_args())

    def configure_optimizers(self):
        """Configure and return the optimizer with proper parameter handling."""
        optimizer = configure_optimizer(
            model=self.model,
            is_peft_model=self.is_peft_model,
            learning_rate=self.specs.get('learning_rate', 2e-5),
            weight_decay=self.specs.get('weight_decay', 0.01)
        )
        
        # Configure scheduler if specified
        if self.specs.get('use_lr_scheduler', False):
            num_training_steps = calculate_training_steps(
                train_dataset_size=len(self.train_dataset),
                batch_size=self.specs.get('per_device_train_batch_size', 8),
                grad_accumulation=self.specs.get('gradient_accumulation_steps', 1),
                num_epochs=self.specs.get('num_train_epochs', 3)
            )
            
            scheduler = prepare_scheduler(
                optimizer=optimizer,
                num_training_steps=num_training_steps,
                scheduler_type=self.specs.get('scheduler_type', 'linear'),
                warmup_ratio=self.specs.get('warmup_ratio', 0.1)
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
        
        return optimizer

    def prepare_with_accelerator(self, train_dataset, eval_dataset):
        """Prepare datasets with Accelerator."""
        train_dataloader = self.prepare_data_loader("train", train_dataset)
        eval_dataloader = self.prepare_data_loader("eval", eval_dataset)
        
        model, optimizer, train_dataloader, eval_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader, eval_dataloader
        )
        
        self.model = model
        self.optimizer = optimizer
        
        return train_dataloader, eval_dataloader

    def _setup_gradient_accumulation(self):
        """Set up gradient accumulation steps."""
        self.gradient_accumulation_steps = self.specs.get('gradient_accumulation_steps', 1)
        if self.gradient_accumulation_steps > 1:
            logger.info(f"Using gradient accumulation with {self.gradient_accumulation_steps} steps")

    def prepare_data_loader(self, slice_type, dataset):
        """Prepare data loader for training or evaluation."""
        if dataset is None:
            return None
            
        params = self.get_dataloader_params(slice_type)
        return DataLoader(dataset, **params)

    def get_dataloader_params(self, slice_type):
        """Get parameters for data loader configuration."""
        return {
            'batch_size': self.specs.get(f'per_device_{slice_type}_batch_size', 8),
            'collate_fn': self.data_collator,
            'num_workers': self.specs.get('dataloader_num_workers', 0),
            'shuffle': slice_type == "train"
        }

    def save_model(self, output_dir=None):
        """Save model with special handling for PEFT models."""
        output_dir = output_dir if output_dir is not None else self.specs.get('output_dir', 'model_output')
        
        return save_model_checkpoint(
            model=self.model,
            tokenizer=self.tokenizer,
            output_dir=output_dir,
            is_peft_model=self.is_peft_model
        )

    def training_step(self, batch, step):
        """Perform a training step."""
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # Handle gradients for PEFT models
        if self.is_peft_model:
            clip_gradients(
                model=self.model,
                max_grad_norm=self.specs.get('max_grad_norm', 1.0),
                is_peft_model=True
            )
        
        return loss

    def train(self):
        """Train the model."""
        self.model.train()
        total_batch_size = self.specs.get('per_device_train_batch_size', 8) * self.accelerator.num_processes
        logger.info(f"Running training with total batch size of {total_batch_size}")
        
        progress_bar = tqdm(range(self.num_training_steps), disable=not self.accelerator.is_local_main_process)
        completed_steps = 0
        
        for epoch in range(self.specs.get('num_train_epochs', 3)):
            for step, batch in enumerate(self.train_dataset):
                loss = self.training_step(batch, step)
                
                # Update progress
                progress_bar.update(1)
                completed_steps += 1
                
                if completed_steps >= self.num_training_steps:
                    break
            
            # Save checkpoint and evaluate
            if self.specs.get('save_strategy', 'epoch') == 'epoch':
                self.save_model()
                
            if self.specs.get('evaluation_strategy', 'epoch') == 'epoch':
                metrics = self._compute_metrics()
                self.accelerator.print(f"Epoch {epoch}: {metrics}")
        
        return self.save_model()

    def _compute_metrics(self):
        """Compute evaluation metrics."""
        self.model.eval()
        metrics = {}
        
        for step, batch in enumerate(self.eval_dataset):
            with torch.no_grad():
                outputs = self.model(**batch)
            
            predictions = outputs.logits.argmax(dim=-1) if hasattr(outputs, "logits") else outputs.predictions
            references = batch["labels"]
            
            # Update metrics
            if self.metric:
                metrics.update(
                    self.metric.compute(predictions=predictions, references=references)
                )
        
        return metrics

class AcceleratedNLPSeq2SeqTrainer(AcceleratedNLPTrainer):
    """
    A trainer class for sequence-to-sequence NLP models using Accelerate.
    
    This trainer extends AcceleratedNLPTrainer to handle sequence-to-sequence tasks
    like translation, summarization, and text generation.
    """
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, raw_dataset,
                task_type, optimizer=None, chosen_metric=None, 
                generation_config=None, customization_config=None):
        """Initialize the AcceleratedNLPSeq2SeqTrainer."""
        super().__init__(args_dir, model, tokenizer, data_collator, 
                        train_dataset, eval_dataset, raw_dataset,
                        task_type, optimizer, chosen_metric, 
                        customization_config)
        
        # Store generation config
        self.generation_config = generation_config or GenerationConfig(
            max_length=self.specs.get('max_length', 128),
            num_beams=self.specs.get('num_beams', 4),
            length_penalty=self.specs.get('length_penalty', 1.0),
            early_stopping=self.specs.get('early_stopping', True)
        )
        
        # Initialize sequence-specific metrics
        self.seq_metrics = get_default_metrics(task_type, is_seq2seq=True)
        
        # Optimize memory settings for sequence models
        if self.is_peft_model:
            optimize_memory_settings(
                model=self.model,
                use_gradient_checkpointing=self.specs.get('gradient_checkpointing', True),
                is_seq2seq=True
            )

    def training_step(self, batch, step):
        """Perform a training step with sequence-specific loss handling."""
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # Handle gradients for PEFT models with sequence-specific settings
        if self.is_peft_model:
            clip_gradients(
                model=self.model,
                max_grad_norm=self.specs.get('max_grad_norm', 1.0),
                is_peft_model=True,
                is_seq2seq=True
            )
        
        return loss

    def _compute_metrics(self):
        """Compute sequence-specific evaluation metrics."""
        self.model.eval()
        metrics = {}
        
        for step, batch in enumerate(self.eval_dataset):
            with torch.no_grad():
                # Generate sequences
                generated_ids = self.model.generate(
                    batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    **self.generation_config.to_dict()
                )
                
                # Decode generated and target sequences
                generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                target_texts = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                
                # Calculate sequence-specific metrics
                if self.seq_metrics:
                    for metric_name, metric_fn in self.seq_metrics.items():
                        score = metric_fn(predictions=generated_texts, references=target_texts)
                        metrics[metric_name] = metrics.get(metric_name, 0) + score
        
        # Average metrics
        for key in metrics:
            metrics[key] /= len(self.eval_dataset)
        
        return metrics

    def save_model(self, output_dir=None):
        """Save model with sequence-specific configurations."""
        output_dir = output_dir if output_dir is not None else self.specs.get('output_dir', 'model_output')
        
        return save_model_checkpoint(
            model=self.model,
            tokenizer=self.tokenizer,
            output_dir=output_dir,
            is_peft_model=self.is_peft_model,
            generation_config=self.generation_config
        )
