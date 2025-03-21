import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

import json
import evaluate
import pytorch_lightning as pl
import torch
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer, GenerationConfig
from configs.default_config import DEFAULT_SPECS
from torch.utils.data import DataLoader
import random
from ..modules.trainer_customization import TrainerCustomizationMixin
from .trainer_utils import (
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
import logging

logger = logging.getLogger(__name__)

@dataclass
class PeftConfig:
    """
    Configuration for PEFT (Parameter-Efficient Fine-Tuning) models with enhanced optimization settings.
    
    This configuration class extends the basic PEFT configuration with Lightning-specific
    settings for memory optimization and resource management.
    
    Attributes:
        peft_type (str): Type of PEFT method (e.g., "LORA", "PREFIX")
        task_type (str): Type of task being performed
        inference_mode (bool): Whether the model is in inference mode
        r (int): Rank for LoRA, aligned with default_config
        lora_alpha (int): Alpha parameter for LoRA
        lora_dropout (float): Dropout rate for LoRA, aligned with default_config
        bias (str): Bias handling strategy
        target_modules (List[str]): Specific modules to apply PEFT to
        layers_to_transform (List[int]): Specific layers to transform
        fan_in_fan_out (bool): Whether to use fan-in/fan-out rescaling
        modules_to_save (List[str]): Modules to save separately
        init_lora_weights (bool): Whether to initialize LoRA weights
        use_gradient_checkpointing (bool): Whether to use gradient checkpointing for memory efficiency
        use_cache (bool): Whether to use cache during forward pass
        quantization_type (str): Type of quantization to use
        quantization_bits (int): Number of bits for quantization
    """
    peft_type: str
    task_type: str
    inference_mode: bool = False
    r: int = 16  # Aligned with default_config
    lora_alpha: int = 32
    lora_dropout: float = 0.05  # Aligned with default_config
    bias: str = "none"
    target_modules: List[str] = None
    layers_to_transform: List[int] = None
    fan_in_fan_out: bool = False
    modules_to_save: List[str] = None
    init_lora_weights: bool = True
    
    # Memory optimization settings
    use_gradient_checkpointing: bool = True
    use_cache: bool = False
    
    # Quantization settings
    quantization_type: Optional[str] = None
    quantization_bits: Optional[int] = None
    
    @classmethod
    def from_pretrained(cls, model):
        """
        Create config from a pretrained PEFT model.
        
        Args:
            model: The PEFT model to extract configuration from
            
        Returns:
            PeftConfig: A configuration object based on the model's parameters
        """
        if not hasattr(model, "peft_config"):
            return None
        config = model.peft_config
        return cls(
            peft_type=getattr(config, "peft_type", "unknown"),
            task_type=getattr(config, "task_type", "unknown"),
            r=getattr(config, "r", 16),
            lora_alpha=getattr(config, "lora_alpha", 32),
            lora_dropout=getattr(config, "lora_dropout", 0.05),
            bias=getattr(config, "bias", "none"),
            target_modules=getattr(config, "target_modules", None),
            layers_to_transform=getattr(config, "layers_to_transform", None),
            fan_in_fan_out=getattr(config, "fan_in_fan_out", False),
            modules_to_save=getattr(config, "modules_to_save", None),
            init_lora_weights=getattr(config, "init_lora_weights", True),
            use_gradient_checkpointing=True,
            use_cache=False
        )

class NLPTrainer(pl.LightningModule, TrainerCustomizationMixin):
    """
    A PyTorch Lightning trainer class for NLP models.
    
    This trainer leverages PyTorch Lightning for training loop management, multi-GPU training,
    and optimization. It can be used independently or as part of the fine-tuning pipeline.
    
    Role in the Pipeline:
    - Provides a Lightning-based training implementation
    - Manages the training loop with Lightning's lifecycle hooks
    - Handles distributed training using Lightning's internal mechanisms
    - Supports PEFT methods with Lightning-specific optimizations
    - Can integrate with Accelerate for enhanced performance
    
    Lightning-Accelerate Coordination:
    When both Lightning and Accelerate are enabled, this class:
    1. Uses Lightning for the overall training loop and lifecycle management
    2. Delegates device/precision handling to Accelerate
    3. Uses Accelerate's optimized memory management with Lightning's training flow
    4. Maintains compatibility with Accelerate's distributed features
    """
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, 
                task_type, optimizer=None, compute_metrics=None,
                model_init=None, callbacks=None, scheduler=None, 
                customization_config=None, **kwargs):
        """Initialize the NLPTrainer."""
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.task_type = task_type
        self.losses = []

        # Load configuration specifications
        specs = json.load(open(args_dir, 'r'))
        self.specs = {**DEFAULT_SPECS, **specs}

        # Store optimizer and scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Add PEFT detection and configuration
        self.is_peft_model = check_is_peft_model(model)
        self.peft_config = PeftConfig.from_pretrained(model) if self.is_peft_model else None
        
        # Initialize customization
        self.setup_customization(customization_config)
        
        # Set flags for Lightning and Accelerate integration
        self.use_lightning = True
        self.use_accelerate = kwargs.get('use_accelerate', False)
        
        # Initialize Accelerate if both Lightning and Accelerate are enabled
        if self.use_accelerate:
            self.accelerator = setup_accelerate_integration(self.specs)
            if self.accelerator:
                # Save original device placement for Lightning
                self._original_device = self.device
                # Let Accelerate set the device
                self.device = self.accelerator.device

        # Initialize metrics based on task type
        self.metric = get_default_metrics(task_type)

        # Optimize memory settings if needed
        if self.is_peft_model:
            optimize_memory_settings(
                model=self.model,
                use_gradient_checkpointing=self.specs.get('gradient_checkpointing', True)
            )

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Configure optimizer
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

    def training_step(self, batch, batch_idx):
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
        
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        outputs = self.model(**batch)
        loss = outputs.loss
        
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def save_checkpoint(self, filepath):
        """Save model checkpoint."""
        return save_model_checkpoint(
            model=self.model,
            tokenizer=self.tokenizer,
            output_dir=filepath,
            is_peft_model=self.is_peft_model
        )

    def train_dataloader(self):
        """Get train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.specs.get('per_device_train_batch_size', 8),
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.specs.get('dataloader_num_workers', 0)
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        if self.eval_dataset is None:
            return None
            
        return DataLoader(
            self.eval_dataset,
            batch_size=self.specs.get('per_device_eval_batch_size', 8),
            collate_fn=self.data_collator,
            num_workers=self.specs.get('dataloader_num_workers', 0)
        )

class NLPSeq2SeqTrainer(NLPTrainer):
    """
    A PyTorch Lightning trainer class for sequence-to-sequence NLP models.
    
    This trainer extends NLPTrainer to handle sequence-to-sequence tasks like translation,
    summarization, and text generation. It includes specialized handling for generation
    configuration and sequence-specific metrics.
    """
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, 
                task_type, optimizer=None, compute_metrics=None,
                model_init=None, callbacks=None, scheduler=None,
                generation_config=None, customization_config=None, **kwargs):
        """Initialize the NLPSeq2SeqTrainer."""
        super().__init__(args_dir, model, tokenizer, data_collator, 
                        train_dataset, eval_dataset, task_type, 
                        optimizer, compute_metrics, model_init, 
                        callbacks, scheduler, customization_config, **kwargs)
                        
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

    def training_step(self, batch, batch_idx):
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
        
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a validation step with sequence generation."""
        # Standard loss calculation
        outputs = self.model(**batch)
        val_loss = outputs.loss
        self.log('val_loss', val_loss, prog_bar=True)
        
        # Generate sequences for evaluation
        if self.generation_config:
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

    def save_checkpoint(self, filepath):
        """Save model checkpoint with sequence-specific configurations."""
        save_model_checkpoint(
            model=self.model,
            tokenizer=self.tokenizer,
            output_dir=filepath,
            is_peft_model=self.is_peft_model,
            generation_config=self.generation_config
        )

    def print_args(self):
        """
        Print the training arguments.
        """
        print(self.specs)

    def print_trainer_type(self):
        """
        Print the type of trainer.
        """
        print("I am an NLPSeq2SeqTrainer!")

    def train(self, max_epochs=10, gpus=1):
        """
        Train the model using PyTorch Lightning's Trainer.
        
        Args:
            max_epochs (int): Maximum number of epochs to train.
            gpus (int): Number of GPUs to use for training.
        """
        trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus)
        trainer.fit(self, self.train_dataloader(), self.val_dataloader())

    def train_dataloader(self):
        """
        Create the training data loader.
        
        Returns:
            DataLoader: The training data loader.
        """
        return DataLoader(self.train_dataset, batch_size=self.specs['per_device_train_batch_size'], collate_fn=self.data_collator)

    def val_dataloader(self):
        """
        Create the validation data loader.
        
        Returns:
            DataLoader: The validation data loader.
        """
        return DataLoader(self.eval_dataset, batch_size=self.specs['per_device_eval_batch_size'], collate_fn=self.data_collator)

    def clip_gradients(self, parameters=None):
        """
        Apply PEFT-aware gradient clipping with seq2seq-specific adjustments.
        
        Args:
            parameters: Optional list of parameters to clip. If None, uses all model parameters.
        """
        if not self.specs.get('max_grad_norm', 0) > 0:
            return
            
        if parameters is None:
            # For seq2seq, separate encoder and decoder parameters
            encoder_params = []
            decoder_params = []
            shared_params = []
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if 'encoder' in name:
                    encoder_params.append(param)
                elif 'decoder' in name:
                    decoder_params.append(param)
                else:
                    shared_params.append(param)
            
            # Get PEFT-specific clipping configuration
            if self.is_peft_model and hasattr(self.model, "peft_config"):
                peft_type = self.model.peft_config.peft_type
                if peft_type == "LORA":
                    # Different clipping for encoder/decoder in LoRA
                    encoder_norm = self.specs['max_grad_norm'] * 0.7
                    decoder_norm = self.specs['max_grad_norm'] * 0.9
                    shared_norm = self.specs['max_grad_norm'] * 0.8
                elif "PREFIX" in peft_type:
                    # Different clipping for prefix tuning
                    encoder_norm = self.specs['max_grad_norm'] * 1.1
                    decoder_norm = self.specs['max_grad_norm'] * 1.3
                    shared_norm = self.specs['max_grad_norm'] * 1.2
                else:
                    encoder_norm = decoder_norm = shared_norm = self.specs['max_grad_norm']
            else:
                encoder_norm = decoder_norm = shared_norm = self.specs['max_grad_norm']
            
            # Apply clipping to each parameter group using Lightning's method if available
            if hasattr(self, 'trainer') and self.trainer is not None:
                if encoder_params:
                    self.trainer.gradient_clip_val = encoder_norm
                    self.trainer.gradient_clip_algorithm = 'norm'
                    torch.nn.utils.clip_grad_norm_(encoder_params, encoder_norm)
                if decoder_params:
                    self.trainer.gradient_clip_val = decoder_norm
                    self.trainer.gradient_clip_algorithm = 'norm'
                    torch.nn.utils.clip_grad_norm_(decoder_params, decoder_norm)
                if shared_params:
                    self.trainer.gradient_clip_val = shared_norm
                    self.trainer.gradient_clip_algorithm = 'norm'
                    torch.nn.utils.clip_grad_norm_(shared_params, shared_norm)
            else:
                # Fallback to PyTorch implementation
                if encoder_params:
                    torch.nn.utils.clip_grad_norm_(encoder_params, encoder_norm)
                if decoder_params:
                    torch.nn.utils.clip_grad_norm_(decoder_params, decoder_norm)
                if shared_params:
                    torch.nn.utils.clip_grad_norm_(shared_params, shared_norm)
                
            if self.specs.get('debug_gradients', False):
                # Log gradient norms for debugging
                for name, params, norm in [
                    ('encoder', encoder_params, encoder_norm),
                    ('decoder', decoder_params, decoder_norm),
                    ('shared', shared_params, shared_norm)
                ]:
                    if params:
                        grad_norm = torch.norm(torch.stack([
                            torch.norm(p.grad.detach()) 
                            for p in params 
                            if p.grad is not None
                        ]))
                        self.log(f'{name}_gradient_norm', grad_norm, prog_bar=True)
        else:
            # If parameters are provided, use parent implementation
            super().clip_gradients(parameters)