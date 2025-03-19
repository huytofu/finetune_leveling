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

@dataclass
class PeftConfig:
    """Configuration for PEFT models with enhanced optimization settings"""
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
        """Create config from a pretrained PEFT model"""
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

class NLPTrainer(pl.LightningModule):
    """
    A PyTorch Lightning trainer class for NLP models.
    
    Attributes:
        model: The model to be trained.
        specs: Configuration specifications for training.
        task_type: The type of task (e.g., classification, generation).
        losses: A list to store training losses.
    """
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, 
                task_type, optimizer=None, compute_metrics=None,
                model_init=None, callbacks=None, scheduler=None, **kwargs):
        """
        Initialize the NLPTrainer.
        
        Args:
            args_dir (str): Directory containing configuration arguments.
            model: The model to be trained.
            tokenizer: The tokenizer for text processing.
            data_collator: The data collator for batching.
            train_dataset: The training dataset.
            eval_dataset: The evaluation dataset.
            task_type (str): The type of task (e.g., classification, generation).
            optimizer: The optimizer for training.
            compute_metrics: Function to compute metrics during evaluation.
            model_init: Function to initialize the model.
            callbacks: List of callbacks to apply during training.
            scheduler: Learning rate scheduler.
            **kwargs: Additional keyword arguments.
        """
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
        self.is_peft_model = self._check_is_peft_model(model)
        self.peft_config = self._get_peft_config() if self.is_peft_model else None

    def _check_is_peft_model(self, model):
        """Check if the model is a PEFT model and get its type"""
        try:
            from peft import PeftModel
            is_peft = isinstance(model, PeftModel)
            if is_peft:
                peft_type = getattr(model.peft_config, "peft_type", "unknown")
                print(f"Detected PEFT model of type: {peft_type}")
                self._log_peft_params(model)
            return is_peft
        except ImportError:
            print("PEFT not installed, continuing with standard training")
            return False

    def _log_peft_params(self, model):
        """Log PEFT-specific parameter information"""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"PEFT model has {trainable_params} trainable parameters out of {total_params} total parameters")
        
        if hasattr(model, "peft_config"):
            config = model.peft_config
            if hasattr(config, "r"):  # LoRA
                print(f"LoRA rank: {config.r}")
            elif hasattr(config, "num_virtual_tokens"):  # Prefix Tuning
                print(f"Number of prefix tokens: {config.num_virtual_tokens}")

    def _get_peft_config(self):
        """Get PEFT-specific configuration"""
        if not self.is_peft_model:
            return None
            
        peft_config = PeftConfig(
            peft_type=getattr(self.model.peft_config, "peft_type", "unknown"),
            task_type=self.task_type,
            r=self.specs.get('lora_r', 8),
            lora_alpha=self.specs.get('lora_alpha', 32),
            lora_dropout=self.specs.get('lora_dropout', 0.1),
        )
        return peft_config

    def forward(self, **inputs):
        """
        Forward pass through the model.
        
        Args:
            **inputs: Input data for the model.
        
        Returns:
            Model outputs.
        """
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        """
        Enhanced training step with PEFT-aware gradient handling.
        
        Args:
            batch: The batch of data for the current step.
            batch_idx: The index of the batch.
        
        Returns:
            Loss from the current training step.
        """
        # Forward pass
        outputs = self.forward(**batch)
        loss = outputs.loss
        
        # Scale loss based on PEFT type
        if self.is_peft_model and hasattr(self.model, "peft_config"):
            peft_type = self.model.peft_config.peft_type
            if peft_type == "LORA":
                # Scale based on LoRA alpha
                loss = loss / getattr(self.model.peft_config, "lora_alpha", 32)
            elif "PREFIX" in peft_type:
                # Prefix tuning might need different scaling
                loss = loss * 1.2  # Slight upscaling for prefix tuning
        
        # Apply gradient clipping if specified
        if self.specs.get('max_grad_norm', 1.0) > 0 and self.is_peft_model:
            self.clip_gradients(loss)
        
        # Log metrics
        self.log('train_loss', loss)
        
        return loss

    def clip_gradients(self, loss):
        """
        Clip gradients with PEFT-specific handling.
        
        Args:
            loss: The loss value from the current step.
        """
        if not self.is_peft_model:
            return

        try:
            # Get trainable parameters
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            
            # Different clipping strategies based on PEFT type
            if hasattr(self.model, "peft_config"):
                peft_type = self.model.peft_config.peft_type
                if peft_type == "LORA":
                    # For LoRA, we can use more aggressive clipping
                    max_grad_norm = self.specs.get('max_grad_norm', 1.0) * 1.5
                elif "PREFIX" in peft_type:
                    # For prefix tuning, we need more conservative clipping
                    max_grad_norm = self.specs.get('max_grad_norm', 1.0) * 0.8
                else:
                    max_grad_norm = self.specs.get('max_grad_norm', 1.0)
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                
                # Log gradient norms if in debug mode
                if self.specs.get('debug_gradients', False):
                    grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in trainable_params]))
                    self.log('gradient_norm', grad_norm)
        except Exception as e:
            print(f"Error in gradient clipping: {e}")

    def on_before_optimizer_step(self, optimizer):
        """
        Handle gradient operations before optimizer step.
        
        Args:
            optimizer: The optimizer being used.
        """
        if not self.is_peft_model:
            return

        # Check for NaN or Inf gradients
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    print(f"Warning: Invalid gradients detected in {name}")
                    param.grad.data = torch.zeros_like(param.grad.data)

        # Gradient scaling for different PEFT types
        if hasattr(self.model, "peft_config"):
            peft_type = self.model.peft_config.peft_type
            if peft_type == "LORA":
                self._scale_lora_gradients()
            elif "PREFIX" in peft_type:
                self._scale_prefix_gradients()

    def _scale_lora_gradients(self):
        """Scale gradients specifically for LoRA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Scale LoRA gradients
                if 'lora_' in name:
                    if param.grad is not None:
                        param.grad.data = param.grad.data / self.model.peft_config.lora_alpha

    def _scale_prefix_gradients(self):
        """Scale gradients specifically for prefix tuning parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Scale prefix tuning gradients
                if 'prefix' in name:
                    if param.grad is not None:
                        # Apply more conservative updates to prefix parameters
                        param.grad.data = param.grad.data * 0.8

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.
        
        Args:
            batch: The batch of data for the current step.
            batch_idx: The index of the batch.
        
        Returns:
            Validation loss.
        """
        outputs = self.forward(**batch)
        val_loss = outputs.loss
        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self):
        """Enhanced optimizer configuration with PEFT-specific optimizations."""
        if self.optimizer is not None and self.scheduler is not None:
            return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}

        if self.is_peft_model:
            # Get trainable parameters for PEFT
            trainable_params = []
            all_param_size = 0
            trainable_param_size = 0
            
            for name, param in self.model.named_parameters():
                all_param_size += param.numel()
                if param.requires_grad:
                    trainable_params.append(param)
                    trainable_param_size += param.numel()
            
            # Log parameter statistics
            print(f"Total parameters: {all_param_size:,}")
            print(f"Trainable parameters: {trainable_param_size:,}")
            print(f"Parameter efficiency ratio: {100 * trainable_param_size / all_param_size:.2f}%")
            
            # Configure optimizer with PEFT-specific settings
            optimizer_kwargs = {
                'lr': self.specs['learning_rate'],
                'weight_decay': self.specs.get('weight_decay', 0.01),
                'eps': self.specs.get('adam_epsilon', 1e-8),
                'betas': (self.specs.get('adam_beta1', 0.9), self.specs.get('adam_beta2', 0.999))
            }
            
            # Use different optimizers based on PEFT type
            if hasattr(self.model, "peft_config"):
                peft_type = self.model.peft_config.peft_type
                if peft_type == "LORA":
                    # For LoRA, we can use a higher learning rate
                    optimizer_kwargs['lr'] *= 2
                elif "PREFIX" in peft_type:
                    # For prefix tuning, we need more conservative updates
                    optimizer_kwargs['lr'] *= 0.5
                    optimizer_kwargs['weight_decay'] *= 2
            
            optimizer = torch.optim.AdamW(trainable_params, **optimizer_kwargs)
            print(f"Configured optimizer with settings: {optimizer_kwargs}")
            
            # Configure scheduler if needed
            if self.specs.get('scheduler_strategy', 'linear') == 'linear':
                from transformers import get_linear_schedule_with_warmup
                num_training_steps = self.trainer.estimated_stepping_batches
                num_warmup_steps = int(num_training_steps * self.specs.get('warmup_ratio', 0.1))
                
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps
                )
                print(f"Configured linear scheduler with {num_warmup_steps} warmup steps")
                
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'step'
                    }
                }
            
            return optimizer
        
        # For non-PEFT models, use standard optimization
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.specs['learning_rate'],
            weight_decay=self.specs.get('weight_decay', 0.01)
        )
        return optimizer

    def on_fit_start(self):
        """Enhanced training setup with optimized memory handling for PEFT models."""
        super().on_fit_start()
        
        if self.is_peft_model:
            # Get PEFT configuration
            peft_config = self.peft_config or PeftConfig.from_pretrained(self.model)
            
            # Enable gradient checkpointing if specified
            if peft_config.use_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                print("Enabled gradient checkpointing for PEFT model")
            
            # Disable model caching if specified
            if not peft_config.use_cache and hasattr(self.model, "config"):
                self.model.config.use_cache = False
                print("Disabled model caching for memory efficiency")
            
            # Handle quantization settings
            if peft_config.quantization_type:
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=peft_config.quantization_type == "4bit",
                        load_in_8bit=peft_config.quantization_type == "8bit",
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    print(f"Applied {peft_config.quantization_type} quantization")
                except ImportError:
                    print("BitsAndBytes not available for quantization")
            
            # Adjust batch size based on PEFT type
            if hasattr(self.model, "peft_config"):
                peft_type = self.model.peft_config.peft_type
                if peft_type == "LORA":
                    self.trainer.accumulate_grad_batches = max(1, self.trainer.accumulate_grad_batches // 2)
                elif "PREFIX" in peft_type:
                    self.trainer.accumulate_grad_batches = self.trainer.accumulate_grad_batches * 2
                print(f"Adjusted gradient accumulation for {peft_type}: {self.trainer.accumulate_grad_batches} steps")

    def on_save_checkpoint(self, checkpoint):
        """Enhanced checkpoint saving with comprehensive PEFT state management."""
        checkpoint = super().on_save_checkpoint(checkpoint)
        
        if self.is_peft_model:
            try:
                # Save PEFT adapter state
                checkpoint['peft_adapter_state'] = self.model.get_adapter_state_dict()
                
                # Save PEFT configuration
                if hasattr(self.model, "peft_config"):
                    checkpoint['peft_config'] = self.model.peft_config
                    checkpoint['peft_type'] = self.model.peft_config.peft_type
                
                # Save quantization state if applicable
                if hasattr(self.model, "quantization_config"):
                    checkpoint['quantization_config'] = self.model.quantization_config
                
                # Save additional training state
                checkpoint['peft_training_state'] = {
                    'current_lora_alpha': getattr(self.model.peft_config, "lora_alpha", None),
                    'current_r': getattr(self.model.peft_config, "r", None),
                    'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                    'total_params': sum(p.numel() for p in self.model.parameters())
                }
                
                print("Saved complete PEFT state to checkpoint")
            except Exception as e:
                print(f"Error saving PEFT state: {str(e)}")
        
        return checkpoint
    
    def on_load_checkpoint(self, checkpoint):
        """Enhanced checkpoint loading with comprehensive PEFT state restoration."""
        super().on_load_checkpoint(checkpoint)
        
        if self.is_peft_model:
            try:
                # Restore PEFT adapter state
                if 'peft_adapter_state' in checkpoint:
                    self.model.load_adapter_state_dict(checkpoint['peft_adapter_state'])
                    print("Restored PEFT adapter state")
                
                # Restore PEFT configuration
                if 'peft_config' in checkpoint:
                    self.model.peft_config = checkpoint['peft_config']
                    print(f"Restored PEFT configuration for type: {checkpoint.get('peft_type', 'unknown')}")
                
                # Restore quantization configuration
                if 'quantization_config' in checkpoint and hasattr(self.model, "quantization_config"):
                    self.model.quantization_config = checkpoint['quantization_config']
                    print("Restored quantization configuration")
                
                # Verify training state
                if 'peft_training_state' in checkpoint:
                    current_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    saved_trainable = checkpoint['peft_training_state']['trainable_params']
                    if current_trainable != saved_trainable:
                        print(f"Warning: Current trainable parameters ({current_trainable}) "
                              f"differ from checkpoint ({saved_trainable})")
                
                print("Completed PEFT state restoration")
            except Exception as e:
                print(f"Error loading PEFT state: {str(e)}")

    def print_args(self):
        """
        Print the training arguments.
        """
        print(self.specs)

    def print_trainer_type(self):
        """
        Print the type of trainer.
        """
        print("I am an NLPTrainer!")

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

class NLPSeq2SeqTrainer(NLPTrainer):
    """
    A PyTorch Lightning trainer class for sequence-to-sequence NLP models.
    
    Attributes:
        model: The model to be trained.
        specs: Configuration specifications for training.
        task_type: The type of task (e.g., translation, summarization).
        losses: A list to store training losses.
    """
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, 
                task_type, optimizer=None, compute_metrics=None,
                model_init=None, callbacks=None, scheduler=None, **kwargs):
        """
        Initialize the NLPSeq2SeqTrainer.
        
        Args:
            args_dir (str): Directory containing configuration arguments.
            model: The model to be trained.
            tokenizer: The tokenizer for text processing.
            data_collator: The data collator for batching.
            train_dataset: The training dataset.
            eval_dataset: The evaluation dataset.
            task_type (str): The type of task (e.g., translation, summarization).
            optimizer: The optimizer for training.
            compute_metrics: Function to compute metrics during evaluation.
            model_init: Function to initialize the model.
            callbacks: List of callbacks to apply during training.
            scheduler: Learning rate scheduler.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(args_dir, model, tokenizer, 
                        data_collator, train_dataset, eval_dataset, 
                        task_type, optimizer, compute_metrics,
                        model_init, callbacks, scheduler, **kwargs)

        # Setup specialized configuration for seq2seq training
        self.setup_seq2seq_config()
        
        # Initialize metric tracking
        self.compute_metrics = compute_metrics
        self.current_val_metrics = {}

    def setup_seq2seq_config(self):
        """Setup specialized configuration for seq2seq training"""
        self.teacher_forcing_ratio = self.specs.get('teacher_forcing_ratio', 0.5)
        self.use_selective_activation_cache = self.specs.get('use_selective_activation_cache', True)
        
        # Setup generation config with memory optimization
        self.generation_config = GenerationConfig(
            max_length=self.specs.get('max_length', 128),
            num_beams=self.specs.get('num_beams', 4),
            length_penalty=self.specs.get('length_penalty', 1.0),
            early_stopping=True,
            use_cache=not self.is_peft_model,  # Disable KV cache for PEFT models
        )

    def configure_optimizers(self):
        """Enhanced optimizer configuration for seq2seq models"""
        # Get base optimizer configuration
        optimizer = super().configure_optimizers()
        
        if self.is_peft_model:
            # Adjust learning rates for encoder and decoder separately
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
            
            # Create parameter groups with different learning rates
            param_groups = [
                {'params': encoder_params, 'lr': self.specs['learning_rate'] * 0.8},
                {'params': decoder_params, 'lr': self.specs['learning_rate'] * 1.2},
                {'params': shared_params, 'lr': self.specs['learning_rate']}
            ]
            
            optimizer.param_groups = param_groups
            
        return optimizer

    def _prepare_inputs(self, batch):
        """Enhanced input preparation with specialized attention patterns"""
        inputs = super()._prepare_inputs(batch)
        
        # Handle attention masks differently for encoder and decoder
        if 'attention_mask' in inputs:
            encoder_attention_mask = inputs['attention_mask']
            # Create causal mask for decoder
            decoder_attention_mask = self._create_causal_mask(
                inputs.get('decoder_input_ids', None)
            )
            
            inputs['encoder_attention_mask'] = encoder_attention_mask
            inputs['decoder_attention_mask'] = decoder_attention_mask
            
        return inputs

    def _create_causal_mask(self, input_ids):
        """Create causal attention mask for decoder"""
        if input_ids is None:
            return None
            
        batch_size, seq_length = input_ids.shape
        mask = torch.triu(
            torch.ones((seq_length, seq_length), dtype=torch.bool, device=self.device),
            diagonal=1
        )
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        return mask

    def _setup_gradient_checkpointing(self):
        """Enhanced gradient checkpointing for seq2seq models"""
        if not self.specs.get('use_gradient_checkpointing', False):
            return
            
        # Enable gradient checkpointing
        if hasattr(self.model, 'encoder'):
            self.model.encoder.gradient_checkpointing_enable()
        if hasattr(self.model, 'decoder'):
            self.model.decoder.gradient_checkpointing_enable()
        
        # Selective activation caching
        if self.use_selective_activation_cache:
            if hasattr(self.model.encoder, "enable_selective_activation_cache"):
                self.model.encoder.enable_selective_activation_cache()
            if hasattr(self.model.decoder, "enable_selective_activation_cache"):
                self.model.decoder.enable_selective_activation_cache()

    def _optimize_beam_search(self):
        """Memory-efficient beam search implementation"""
        kwargs = {}
        
        if not self.is_peft_model:
            return kwargs
            
        # Optimize memory usage for PEFT models during beam search
        kwargs['use_cache'] = False  # Disable KV cache
        
        # Adjust beam size based on available memory
        if self.specs.get('num_beams', 4) > 4 and torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            if free_memory < 4 * 1024 * 1024 * 1024:  # Less than 4GB free
                kwargs['num_beams'] = min(self.specs.get('num_beams', 4), 4)
                
        return kwargs

    def training_step(self, batch, batch_idx):
        """Enhanced training step with specialized seq2seq handling"""
        # Setup gradient checkpointing
        self._setup_gradient_checkpointing()
        
        # Prepare inputs with attention patterns
        inputs = self._prepare_inputs(batch)
        
        # Forward pass with teacher forcing
        if random.random() < self.teacher_forcing_ratio:
            outputs = self.model(**inputs)
        else:
            with torch.no_grad():
                generated = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    **self._optimize_beam_search()
                )
            outputs = self.model(**inputs, decoder_input_ids=generated)
        
        loss = outputs.loss
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        
        if self.specs.get('debug_gradients', False):
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in trainable_params if p.grad is not None]))
            self.log('gradient_norm', grad_norm, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Enhanced validation step with memory optimizations"""
        inputs = self._prepare_inputs(batch)
        
        generation_kwargs = self._optimize_beam_search()
        
        # Generate with optimized memory usage
        with torch.no_grad():
            generated_tokens = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generation_kwargs,
            )
            
        # Compute loss
        outputs = self.model(**inputs)
        loss = outputs.loss
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        
        return {
            'val_loss': loss,
            'generated': generated_tokens,
            'labels': inputs.get('labels')
        }

    def validation_epoch_end(self, outputs):
        """Process all validation outputs at the end of the epoch"""
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        
        # Gather predictions and labels
        for output in outputs:
            all_predictions.extend(output['predictions'])
            # Decode labels if they're token IDs
            if isinstance(output['labels'], torch.Tensor):
                labels = self.tokenizer.batch_decode(output['labels'], skip_special_tokens=True)
            else:
                labels = output['labels']
            all_labels.extend(labels)
            total_loss += output['loss'].item()
            
        # Compute average loss
        avg_loss = total_loss / len(outputs)
        self.log('val_loss', avg_loss, prog_bar=True)
        
        # Compute sequence metrics
        metrics = self.compute_seq2seq_metrics(all_predictions, all_labels)
        self.current_val_metrics = metrics
        
        # Log all metrics
        for metric_name, value in metrics.items():
            self.log(f'val_{metric_name}', value, prog_bar=True)
            
        return {'val_loss': avg_loss, **metrics}

    def compute_seq2seq_metrics(self, predictions, labels):
        """Compute sequence-to-sequence specific metrics"""
        metrics = {}
        
        # Use provided compute_metrics function if available
        if self.compute_metrics is not None:
            metrics.update(self.compute_metrics(predictions, labels))
            
        # Add task-specific metrics
        if self.task_type == "translation":
            try:
                import sacrebleu
                bleu = sacrebleu.corpus_bleu(predictions, [labels])
                metrics['bleu'] = bleu.score
            except ImportError:
                print("sacrebleu not installed, skipping BLEU score computation")
                
        elif self.task_type == "summarization":
            try:
                rouge = evaluate.load('rouge')
                rouge_output = rouge.compute(predictions=predictions, references=labels)
                metrics.update(rouge_output)
            except Exception as e:
                print(f"Error computing ROUGE score: {e}")
                
        return metrics

    def postprocess_text(self, preds, labels):
        """Post-process generated text and labels"""
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        
        # Remove special tokens if any remain
        special_tokens = set([self.tokenizer.pad_token, self.tokenizer.eos_token, 
                            self.tokenizer.bos_token, self.tokenizer.sep_token, 
                            self.tokenizer.cls_token])
        special_tokens = {t for t in special_tokens if t is not None}
        
        for token in special_tokens:
            preds = [pred.replace(token, '') for pred in preds]
            labels = [label.replace(token, '') for label in labels]
            
        # Clean up whitespace
        preds = [" ".join(pred.split()) for pred in preds]
        labels = [" ".join(label.split()) for label in labels]
        
        return preds, labels

    def on_save_checkpoint(self, checkpoint):
        """Enhanced checkpoint saving with seq2seq state management"""
        super().on_save_checkpoint(checkpoint)
        
        if self.is_peft_model:
            # Save encoder and decoder states separately
            if hasattr(self.model, 'encoder'):
                checkpoint['encoder_state'] = self.model.encoder.state_dict()
            if hasattr(self.model, 'decoder'):
                checkpoint['decoder_state'] = self.model.decoder.state_dict()
            
            # Save generation config
            checkpoint['generation_config'] = self.generation_config.to_dict()
            
            # Save memory optimization settings
            checkpoint['memory_config'] = {
                'use_selective_activation_cache': self.use_selective_activation_cache,
                'teacher_forcing_ratio': self.teacher_forcing_ratio
            }

    def on_load_checkpoint(self, checkpoint):
        """Enhanced checkpoint loading with seq2seq state management"""
        super().on_load_checkpoint(checkpoint)
        
        if self.is_peft_model:
            # Load encoder and decoder states
            if 'encoder_state' in checkpoint and hasattr(self.model, 'encoder'):
                self.model.encoder.load_state_dict(checkpoint['encoder_state'])
            if 'decoder_state' in checkpoint and hasattr(self.model, 'decoder'):
                self.model.decoder.load_state_dict(checkpoint['decoder_state'])
            
            # Load generation config
            if 'generation_config' in checkpoint:
                self.generation_config = GenerationConfig(**checkpoint['generation_config'])
            
            # Load memory optimization settings
            if 'memory_config' in checkpoint:
                self.use_selective_activation_cache = checkpoint['memory_config']['use_selective_activation_cache']
                self.teacher_forcing_ratio = checkpoint['memory_config']['teacher_forcing_ratio']

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