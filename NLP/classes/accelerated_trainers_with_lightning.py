import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

import torch
import nltk
import json
import evaluate
import collections
import numpy as np
import logging
import configargparse
import pytorch_lightning as pl
from nltk.translate.bleu_score import sentence_bleu
from transformers import GenerationConfig
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm
from configs.default_config import DEFAULT_SPECS
from .config_manager import ConfigManager
from .utils import Logger, ErrorHandler
from ..modules.trainer_customization import TrainerCustomizationMixin
from .trainer_utils import (
    check_is_peft_model,
    prepare_scheduler,
    calculate_training_steps,
    configure_optimizer,
    clip_gradients,
    save_model_checkpoint,
    get_default_metrics,
    optimize_memory_settings,
    setup_accelerate_integration,
    PeftConfig
)

# Set up logging
logging.basicConfig(level=logging.INFO)
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

class AcceleratedNLPTrainer(pl.LightningModule, TrainerCustomizationMixin):
    """
    A trainer class that combines PyTorch Lightning and Accelerate for enhanced training capabilities.
    
    This trainer provides seamless integration between Lightning's high-level training abstractions
    and Accelerate's distributed training optimizations. It preserves Lightning's lifecycle hooks
    while leveraging Accelerate's performance benefits.
    """
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, raw_dataset,
                task_type, optimizer=None, compute_metrics=None,
                model_init=None, callbacks=None, scheduler=None, 
                customization_config=None, **kwargs):
        """Initialize the AcceleratedNLPTrainer."""
        super().__init__()
        try:
            self.model = model
            self.tokenizer = tokenizer
            self.data_collator = data_collator
            self.datasets = {"train": train_dataset, "eval": eval_dataset}
            self.raw_dataset = raw_dataset
            self.task_type = task_type
            self.chosen_metric = compute_metrics
            
            # Load configuration specifications
            specs = json.load(open(args_dir, 'r'))
            self.specs = {**DEFAULT_SPECS, **specs}
            
            # Initialize Accelerator in a deferred way
            # Will be properly set up in setup() to ensure Lightning trainer is initialized
            self.accelerator = None
            
            # Store optimizer and scheduler
            self.optimizer_init = optimizer
            self.scheduler_init = scheduler
            
            # PEFT detection and configuration
            self.is_peft_model = check_is_peft_model(model)
            if self.is_peft_model:
                self.peft_config = PeftConfig.from_pretrained(model)
                optimize_memory_settings(
                    model=self.model,
                    use_gradient_checkpointing=self.specs.get('gradient_checkpointing', True)
                )
            
            # Initialize metrics based on task type
            self.metric = get_default_metrics(task_type)
            
            # Setup customization
            self.setup_customization(customization_config)
            
            self.save_hyperparameters(ignore=['model', 'tokenizer', 'data_collator'])
            logger.info("AcceleratedNLPTrainer initialized successfully.")
        except Exception as e:
            ErrorHandler.handle_error(e, "AcceleratedNLPTrainer initialization")

    def configure_optimizers(self):
        """Configure optimizers and schedulers with Lightning integration."""
        optimizer = configure_optimizer(
            model=self.model,
            is_peft_model=self.is_peft_model,
            learning_rate=self.specs.get('learning_rate', 2e-5),
            weight_decay=self.specs.get('weight_decay', 0.01)
        )
        
        # Configure scheduler if specified
        if self.specs.get('use_lr_scheduler', False):
            num_training_steps = calculate_training_steps(
                train_dataset_size=len(self.datasets['train']),
                batch_size=self.specs.get('per_device_train_batch_size', 8),
                grad_accumulation=self.trainer.accumulate_grad_batches,
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

    def setup(self, stage=None):
        """Lightning setup with Accelerate integration."""
        if stage == 'fit':
            # Initialize Accelerator after Lightning trainer is set up
            self.accelerator = setup_accelerate_integration(self.specs)
            
            # Prepare model and data with both frameworks
            if self.accelerator:
                self.model = self.accelerator.prepare(self.model)
                
                # Store original device for Lightning
                self._original_device = self.device
                # Let Accelerate set the device
                self.device = self.accelerator.device
                
                logger.info(f"Model prepared with Accelerator on {self.device}")

    def training_step(self, batch, batch_idx):
        """Training step with coordinated Lightning-Accelerate handling."""
        # Forward pass with Accelerator's autocast if available
        if self.accelerator:
            with self.accelerator.autocast():
                outputs = self.model(**batch)
                loss = outputs.loss if not isinstance(outputs, tuple) else outputs[0]
        else:
            outputs = self.model(**batch)
            loss = outputs.loss if not isinstance(outputs, tuple) else outputs[0]

        # Get loss function from customization or use model's loss
        loss_fn = self.get_loss_function()
        if loss_fn and not isinstance(outputs, tuple):
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            labels = batch.get('labels')
            if labels is not None:
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Handle gradients for PEFT models
        if self.is_peft_model:
            # Scale loss based on PEFT type
            if hasattr(self.model, "peft_config"):
                peft_type = self.model.peft_config.peft_type
                if peft_type == "LORA":
                    loss = loss / getattr(self.model.peft_config, "lora_alpha", 32)
                elif "PREFIX" in peft_type:
                    loss = loss * 1.2

            # Scale loss for gradient accumulation
            if self.trainer.accumulate_grad_batches > 1:
                loss = loss / self.trainer.accumulate_grad_batches

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with metric computation."""
        outputs = self.model(**batch)
        val_loss = outputs.loss if not isinstance(outputs, tuple) else outputs[0]
        
        self.log('val_loss', val_loss, prog_bar=True)
        
        # Compute metrics if available
        if self.metric:
            predictions = outputs.logits.argmax(dim=-1) if hasattr(outputs, "logits") else outputs.predictions
            references = batch["labels"]
            metrics = self.metric.compute(predictions=predictions, references=references)
            self.log_dict({f"val_{k}": v for k, v in metrics.items()}, prog_bar=True)
        
        return val_loss

    def on_save_checkpoint(self, checkpoint):
        """Enhanced checkpoint saving with PEFT state management."""
        if self.is_peft_model:
            # Save PEFT-specific states
            checkpoint['peft_config'] = self.peft_config
            checkpoint['peft_state'] = self.model.get_adapter_state_dict() if hasattr(self.model, "get_adapter_state_dict") else None
            
            # Save framework coordination states
            checkpoint['framework_state'] = {
                'accelerator_device': self.accelerator.device if self.accelerator else None,
                'original_device': getattr(self, '_original_device', None)
            }

    def on_load_checkpoint(self, checkpoint):
        """Enhanced checkpoint loading with PEFT state restoration."""
        if self.is_peft_model:
            # Restore PEFT-specific states
            if 'peft_config' in checkpoint:
                self.peft_config = checkpoint['peft_config']
            if 'peft_state' in checkpoint and checkpoint['peft_state']:
                self.model.load_adapter_state_dict(checkpoint['peft_state'])
            
            # Restore framework coordination states
            if 'framework_state' in checkpoint:
                if checkpoint['framework_state']['accelerator_device']:
                    self.device = checkpoint['framework_state']['accelerator_device']
                if checkpoint['framework_state']['original_device']:
                    self._original_device = checkpoint['framework_state']['original_device']

    def train_dataloader(self):
        """Get train dataloader."""
        return DataLoader(
            self.datasets["train"],
            batch_size=self.specs.get('per_device_train_batch_size', 8),
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.specs.get('dataloader_num_workers', 0)
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        if self.datasets["eval"] is None:
            return None
            
        return DataLoader(
            self.datasets["eval"],
            batch_size=self.specs.get('per_device_eval_batch_size', 8),
            collate_fn=self.data_collator,
            num_workers=self.specs.get('dataloader_num_workers', 0)
        )

class AcceleratedNLPSeq2SeqTrainer(AcceleratedNLPTrainer):
    """
    A trainer class that combines PyTorch Lightning and Accelerate for sequence-to-sequence tasks.
    
    This trainer extends AcceleratedNLPTrainer to handle sequence-to-sequence tasks while maintaining
    the seamless integration between Lightning and Accelerate frameworks.
    """
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, raw_dataset,
                task_type, optimizer=None, compute_metrics=None,
                model_init=None, callbacks=None, scheduler=None,
                customization_config=None, generation_config=None, **kwargs):
        """Initialize the AcceleratedNLPSeq2SeqTrainer."""
        super().__init__(args_dir, model, tokenizer, data_collator, 
                        train_dataset, eval_dataset, raw_dataset,
                        task_type, optimizer, compute_metrics,
                        model_init, callbacks, scheduler, 
                        customization_config, **kwargs)
        
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

    def validation_step(self, batch, batch_idx):
        """Validation step with sequence generation and metric computation."""
        # Standard loss calculation
        outputs = self.model(**batch)
        val_loss = outputs.loss if not isinstance(outputs, tuple) else outputs[0]
        self.log('val_loss', val_loss, prog_bar=True)
        
        # Generate sequences with framework coordination
        if self.generation_config:
            if self.accelerator:
                with self.accelerator.autocast():
                    generated_ids = self.model.generate(
                        batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        **self.generation_config.to_dict()
                    )
            else:
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
                    self.log(f'val_{metric_name}', score, prog_bar=True)
        
        return val_loss

    def on_save_checkpoint(self, checkpoint):
        """Enhanced checkpoint saving with sequence-specific configurations."""
        super().on_save_checkpoint(checkpoint)
        
        # Save sequence-specific configurations
        if self.generation_config:
            checkpoint['generation_config'] = self.generation_config.to_dict()
        if self.seq_metrics:
            checkpoint['seq_metrics'] = {
                name: metric.get_state() if hasattr(metric, 'get_state') else None
                for name, metric in self.seq_metrics.items()
            }

    def on_load_checkpoint(self, checkpoint):
        """Enhanced checkpoint loading with sequence-specific configurations."""
        super().on_load_checkpoint(checkpoint)
        
        # Restore sequence-specific configurations
        if 'generation_config' in checkpoint:
            self.generation_config = GenerationConfig(**checkpoint['generation_config'])
        if 'seq_metrics' in checkpoint:
            for name, state in checkpoint['seq_metrics'].items():
                if state and name in self.seq_metrics and hasattr(self.seq_metrics[name], 'set_state'):
                    self.seq_metrics[name].set_state(state)
