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

@dataclass
class PeftConfig:
    """Configuration for PEFT models"""
    peft_type: str
    task_type: str
    inference_mode: bool = False
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    bias: str = "none"
    target_modules: List[str] = None
    layers_to_transform: List[int] = None
    fan_in_fan_out: bool = False
    modules_to_save: List[str] = None
    init_lora_weights: bool = True

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
        Perform a single training step.
        
        Args:
            batch: The batch of data for the current step.
            batch_idx: The index of the batch.
        
        Returns:
            Loss from the current training step.
        """
        outputs = self.forward(**batch)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

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
        """Configure optimizers with special handling for PEFT models."""
        if self.optimizer is not None and self.scheduler is not None:
            return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}

        # Check if this is a PEFT model
        try:
            from peft.utils import get_peft_model_state_dict
            # Try to access PEFT-specific attributes
            _ = get_peft_model_state_dict(self.model)
            # This is a PEFT model - only train the parameters that require gradients
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.specs['learning_rate'],
                weight_decay=self.specs.get('weight_decay', 0.01)
            )
            print(f"Configured optimizer for PEFT model with {len(trainable_params)} trainable parameters")
        except (ImportError, ValueError, AttributeError, TypeError):
            # For regular models, train all parameters
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.specs['learning_rate'],
                weight_decay=self.specs.get('weight_decay', 0.01)
            )
            print("Configured optimizer for standard model (non-PEFT)")

        return {'optimizer': optimizer}

    def on_fit_start(self):
        """Set up training-specific configurations for PEFT models."""
        super().on_fit_start()
        
        if self.is_peft_model:
            # Enable gradient checkpointing if available
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                print("Enabled gradient checkpointing for PEFT model")
            
            # Configure model for memory efficiency
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False

    def on_save_checkpoint(self, checkpoint):
        """Enhanced checkpoint saving for PEFT models"""
        # First call parent class method
        checkpoint = super().on_save_checkpoint(checkpoint)
        
        # Add PEFT-specific state
        try:
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                # Save PEFT-specific state
                checkpoint['peft_adapter_state'] = self.model.get_adapter_state_dict()
                print("Saved PEFT adapter state to checkpoint")
                
                # Save additional PEFT-specific information
                if hasattr(self.model, "peft_config"):
                    checkpoint['peft_config'] = self.model.peft_config
        except (ImportError, AttributeError):
            pass
            
        return checkpoint
    
    def on_load_checkpoint(self, checkpoint):
        """Enhanced checkpoint loading for PEFT models"""
        # First call parent class method
        super().on_load_checkpoint(checkpoint)
        
        # Load PEFT-specific state
        try:
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                if 'peft_adapter_state' in checkpoint:
                    self.model.load_adapter_state_dict(checkpoint['peft_adapter_state'])
                    print("Loaded PEFT adapter state from checkpoint")
                    
                if 'peft_config' in checkpoint:
                    self.model.peft_config = checkpoint['peft_config']
        except (ImportError, AttributeError):
            pass

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

        # Sequence generation parameters with PEFT awareness
        self.setup_generation_config()
        
        # Initialize metric tracking
        self.compute_metrics = compute_metrics
        self.current_val_metrics = {}

    def setup_generation_config(self):
        """Setup generation configuration with PEFT awareness"""
        self.generation_config = GenerationConfig(
            max_length=self.specs.get('max_length', 128),
            num_beams=self.specs.get('num_beams', 4),
            do_sample=self.specs.get('do_sample', False),
            temperature=self.specs.get('temperature', 1.0),
            top_p=self.specs.get('top_p', 1.0),
            top_k=self.specs.get('top_k', 50),
        )
        
        if self.is_peft_model:
            # Adjust generation parameters based on PEFT type
            if self.peft_config.peft_type == "LORA":
                # LoRA models might need different generation settings
                self.generation_config.temperature = 0.7
                self.generation_config.top_p = 0.9
            elif "PREFIX" in self.peft_config.peft_type:
                # Prefix tuning might need different settings
                self.generation_config.num_beams = max(2, self.generation_config.num_beams - 1)

    def validation_step(self, batch, batch_idx):
        """Specialized validation step for Seq2Seq models with PEFT support"""
        outputs = self.forward(**batch)
        val_loss = outputs.loss
        self.log('val_loss', val_loss)
        
        # Generate sequences with PEFT awareness
        generated_ids = self.generate(batch)
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        return {'loss': val_loss, 'predictions': generated_texts, 'labels': batch['labels']}

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

    def _prepare_inputs_for_generation(self, batch):
        """Prepare inputs for sequence generation"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        
        # Add encoder outputs for encoder-decoder models if available
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "forward"):
            with torch.no_grad():
                inputs["encoder_outputs"] = self.model.encoder(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_dict=True
                )
                
        return inputs

    def generate(self, batch):
        """Generate sequences with PEFT-aware handling"""
        generation_inputs = self._prepare_inputs_for_generation(batch)
        
        if self.is_peft_model:
            try:
                generation_kwargs = self.generation_config.to_dict()
                
                # Adjust settings based on PEFT type
                if self.peft_config.peft_type == "LORA":
                    generation_kwargs["max_new_tokens"] = min(
                        generation_kwargs.get("max_length", 128),
                        self.specs.get("max_new_tokens", 64)
                    )
                    # Add LoRA-specific generation settings
                    if hasattr(self.model, "disable_adapter"):
                        self.model.disable_adapter()
                        
                elif "PREFIX" in self.peft_config.peft_type:
                    generation_kwargs["length_penalty"] = 0.8
                    generation_kwargs["repetition_penalty"] = 1.2
                    
                generated_ids = self.model.generate(
                    **generation_inputs,
                    **generation_kwargs
                )
                
                # Re-enable adapter if it was disabled
                if hasattr(self.model, "enable_adapter"):
                    self.model.enable_adapter()
                    
                return generated_ids
                
            except Exception as e:
                print(f"Error in PEFT generation, falling back to default: {e}")
                
        # Default generation for non-PEFT models
        return self.model.generate(
            **generation_inputs,
            max_length=self.specs.get("max_length", 128),
            num_beams=self.specs.get("num_beams", 4),
            do_sample=self.specs.get("do_sample", False)
        )

    def on_save_checkpoint(self, checkpoint):
        """Enhanced checkpoint saving for Seq2Seq PEFT models"""
        # First call parent class method
        checkpoint = super().on_save_checkpoint(checkpoint)
        
        # Add Seq2Seq specific PEFT state
        try:
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                # Save generation config
                if hasattr(self, 'generation_config'):
                    checkpoint['generation_config'] = self.generation_config.to_dict()
                
                # Save any sequence-specific adapter states
                if hasattr(self.model, "get_encoder_adapter_state"):
                    checkpoint['encoder_adapter_state'] = self.model.get_encoder_adapter_state()
                if hasattr(self.model, "get_decoder_adapter_state"):
                    checkpoint['decoder_adapter_state'] = self.model.get_decoder_adapter_state()
        except (ImportError, AttributeError):
            pass
            
        return checkpoint
    
    def on_load_checkpoint(self, checkpoint):
        """Enhanced checkpoint loading for Seq2Seq PEFT models"""
        # First call parent class method
        super().on_load_checkpoint(checkpoint)
        
        # Load Seq2Seq specific PEFT state
        try:
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                # Load generation config
                if 'generation_config' in checkpoint:
                    self.generation_config = GenerationConfig(**checkpoint['generation_config'])
                
                # Load sequence-specific adapter states
                if 'encoder_adapter_state' in checkpoint and hasattr(self.model, "set_encoder_adapter_state"):
                    self.model.set_encoder_adapter_state(checkpoint['encoder_adapter_state'])
                if 'decoder_adapter_state' in checkpoint and hasattr(self.model, "set_decoder_adapter_state"):
                    self.model.set_decoder_adapter_state(checkpoint['decoder_adapter_state'])
        except (ImportError, AttributeError):
            pass

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