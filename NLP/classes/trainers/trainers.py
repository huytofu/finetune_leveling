import os
import sys
import json
import logging
from typing import Optional, Dict, Any, Union, List
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer, GenerationConfig

# Add to Python path
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parentdir)

# Local imports
from configs.default_config import DEFAULT_SPECS
from modules.monitoring_and_tracking.mlflow_tracking import MLflowCallback
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
    optimize_memory_settings
)

logger = logging.getLogger(__name__)

class NLPTrainer(Trainer, TrainerCustomizationMixin):
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, 
                task_type, optimizer=None, compute_metrics=None,
                model_init=None, callbacks=None, scheduler=None, 
                customization_config=None, **kwargs):

        specs = json.load(open(args_dir, 'r'))
        self.specs = {**DEFAULT_SPECS, **specs}

        # Set PEFT-specific training arguments
        if self.specs.get('use_peft', False):
            # Enable gradient checkpointing for memory efficiency with PEFT
            self.specs['gradient_checkpointing'] = self.specs.get('use_gradient_checkpointing', True)
            
            # Add adapter dropout if using PEFT
            if 'peft_method' in self.specs and self.specs['peft_method'] == 'lora':
                self.specs['lora_dropout'] = self.specs.get('lora_dropout', 0.05)
                
            # Modify optimizer settings for PEFT
            if not 'optim' in self.specs:
                self.specs['optim'] = 'adamw_torch'

        self.args = TrainingArguments(
            **self.specs
        )

        self.datasets = {"train": train_dataset, "eval": eval_dataset}
        self.task_type = task_type
        self.losses = []
        self.tokenizer = tokenizer
        self.model = model

        # Check if this is a PEFT model before initializing
        self.is_peft_model = check_is_peft_model(model)
        
        # Initialize customization
        self.setup_customization(customization_config)
        
        # Initialize optimizer and scheduler
        self.optimizer = self.configure_optimizers()
        if isinstance(self.optimizer, dict):
            optimizer = self.optimizer["optimizer"]
            scheduler = self.optimizer["lr_scheduler"]["scheduler"]
        else:
            optimizer = self.optimizer
            scheduler = None
            
        # Initialize metrics based on task type
        self.metric = get_default_metrics(task_type)
            
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            base_path=os.path.join("checkpoints", task_type),
            model_name=model.config.name_or_path
        )
        
        super().__init__(model, self.args, 
                        data_collator=data_collator, 
                        train_dataset=train_dataset, 
                        eval_dataset=eval_dataset,
                        tokenizer=tokenizer, 
                        optimizers=(optimizer, scheduler),
                        model_init=model_init, 
                        compute_metrics=compute_metrics or self.compute_metrics,
                        callbacks=callbacks, **kwargs)

    def prepare_scheduler(self, optimizer):
        """Prepare learning rate scheduler with proper configuration."""
        num_training_steps = calculate_training_steps(
            train_dataset_size=len(self.train_dataset),
            batch_size=self.specs.get('per_device_train_batch_size', 8),
            grad_accumulation=self.specs.get('gradient_accumulation_steps', 1),
            num_epochs=self.specs.get('num_train_epochs', 3)
        )
        
        return prepare_scheduler(
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            scheduler_type=self.specs.get('scheduler_type', 'linear'),
            warmup_ratio=self.specs.get('warmup_ratio', 0.1)
        )

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
            scheduler = self.prepare_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
        
        return optimizer

    def train(self):
        """Train the model, with special handling for PEFT models"""
        if self.is_peft_model:
            logger.info("Training PEFT model")
            # Log PEFT-specific info before training
            try:
                peft_config = getattr(self.model, "peft_config", None)
                if peft_config:
                    logger.info(f"PEFT config: {peft_config}")
                    
                    # Add PEFT-specific memory optimizations
                    optimize_memory_settings(
                        model=self.model,
                        use_gradient_checkpointing=getattr(self.args, "gradient_checkpointing", False)
                    )
                        
            except Exception as e:
                logger.warning(f"Error configuring PEFT model: {e}")
        
        return super().train()
    
    def save_model(self, output_dir=None, _internal_call=False):
        """Save model with special handling for PEFT models"""
        output_dir = output_dir if output_dir is not None else self.args.output_dir

        # Use centralized save function for PEFT models
        if self.is_peft_model:
            return save_model_checkpoint(
                model=self.model,
                tokenizer=self.tokenizer,
                output_dir=output_dir,
                is_peft_model=True
            )
        
        # Standard save for non-PEFT models
        return super().save_model(output_dir, _internal_call)
    
    def _get_train_sampler(self):
        """Get train sampler with PEFT-specific adjustments if needed"""
        sampler = super()._get_train_sampler()
        
        # For quantized PEFT models, we might need to adjust the sampler
        if self.is_peft_model and hasattr(self.model, 'quantization_config'):
            logger.info("Detected quantized PEFT model, checking sampler")
            
            # Check if we need to modify the batch size for quantized models
            if self.args.per_device_train_batch_size > 4 and getattr(self.model, 'quantization_config', {}).get('quantization_type', None) in ['4bit', '8bit']:
                logger.warning("Quantized PEFT model detected - consider reducing batch size if you encounter memory issues")
                
        return sampler
    
    def print_args(self):
        print(self.args)

    def print_trainer_type(self):
        print("I am a NLPTrainer!")
        if self.is_peft_model:
            print("(with PEFT support)")

    def _handle_gradients(self, loss):
        """Handle gradients with PEFT-specific optimizations"""
        if self.is_peft_model:
            # Use centralized gradient clipping for PEFT models
            clip_gradients(
                model=self.model,
                max_grad_norm=self.args.max_grad_norm,
                is_peft_model=True
            )
        
        return loss

    def _scale_gradients(self):
        """Apply PEFT-specific gradient scaling"""
        if not self.is_peft_model or not hasattr(self.model, "peft_config"):
            return

        peft_type = self.model.peft_config.peft_type
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            if peft_type == "LORA" and "lora_" in name:
                # Scale LoRA gradients
                param.grad.data = param.grad.data / self.model.peft_config.lora_alpha
            elif "PREFIX" in peft_type and "prefix" in name:
                # More conservative updates for prefix parameters
                param.grad.data = param.grad.data * 0.8

    def training_step(self, model, inputs, **kwargs):
        """Enhanced training step with PEFT-aware gradient handling"""
        outputs = super().training_step(model, inputs, **kwargs)
        
        if self.is_peft_model:
            # Handle gradients before optimizer step
            outputs["loss"] = self._handle_gradients(outputs["loss"])
            self._scale_gradients()
            
            # Log gradient norms in debug mode
            if self.specs.get('debug_gradients', False):
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in trainable_params]))
                self.log({"gradient_norm": grad_norm})
        
        return outputs

    def compute_metrics(self, eval_pred):
        """Compute task-specific evaluation metrics."""
        predictions, labels = eval_pred
        if self.task_type in ["classification", "token-classification"]:
            predictions = np.argmax(predictions, axis=-1)
            return self.metric.compute(predictions=predictions, references=labels)
            
        elif self.task_type in ["summarization", "translation"]:
            decoded_preds = self.tokenizer.batch_decode(
                predictions, skip_special_tokens=True
            )
            decoded_labels = self.tokenizer.batch_decode(
                labels, skip_special_tokens=True
            )
            
            # Rouge expects a newline after each sentence
            decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
            decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
            
            result = self.metric.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                use_stemmer=True
            )
            
            # Extract median scores
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
            return result
            
        elif self.task_type == "question-answering":
            decoded_preds = self.tokenizer.batch_decode(
                predictions, skip_special_tokens=True
            )
            decoded_labels = self.tokenizer.batch_decode(
                labels, skip_special_tokens=True
            )
            
            formatted_predictions = [{"id": str(i), "prediction_text": pred} for i, pred in enumerate(decoded_preds)]
            formatted_references = [{"id": str(i), "answers": {"text": [label], "answer_start": [0]}} for i, label in enumerate(decoded_labels)]
            
            result = self.metric.compute(
                predictions=formatted_predictions,
                references=formatted_references
            )
            return result
            
        return {}

    def post_process_predictions(self, predictions):
        """Post-process model predictions based on task type."""
        if self.task_type in ["classification", "token-classification"]:
            return np.argmax(predictions, axis=-1).tolist()
            
        elif self.task_type in ["summarization", "translation", "text-generation"]:
            return self.tokenizer.batch_decode(
                predictions,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
        elif self.task_type == "question-answering":
            start_logits, end_logits = predictions
            
            all_answers = []
            for start, end in zip(start_logits, end_logits):
                start_idx = np.argmax(start)
                end_idx = np.argmax(end[start_idx:]) + start_idx
                
                answer = {
                    "answer": self.tokenizer.decode(predictions[start_idx:end_idx+1]),
                    "start": int(start_idx),
                    "end": int(end_idx),
                    "score": float(start[start_idx] * end[end_idx])
                }
                all_answers.append(answer)
                
            return all_answers
            
        return predictions.tolist()

    def save_checkpoint(
        self,
        step: int,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Save training checkpoint with metadata."""
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            epoch=epoch,
            metrics=metrics
        )
        
        # Log checkpoint as MLflow artifact if callback exists
        mlflow_callback = next((cb for cb in self.callback_handler.callbacks if isinstance(cb, MLflowCallback)), None)
        if mlflow_callback:
            mlflow_callback.on_save(self.args, self.state, None)
            
        return checkpoint_path
        
    def load_checkpoint(
        self,
        checkpoint_path: str
    ) -> Dict[str, Any]:
        """Load checkpoint and return state dict."""
        return self.checkpoint_manager.load_checkpoint(
            model=self.model,
            checkpoint_path=checkpoint_path
        )

    def clip_gradients(self, parameters=None):
        """
        Apply PEFT-aware gradient clipping.
        
        Args:
            parameters: Optional list of parameters to clip. If None, uses all model parameters.
        """
        if not self.specs.get('max_grad_norm', 0) > 0:
            return
            
        if parameters is None:
            parameters = self.model.parameters()
            
        # Get PEFT-specific clipping configuration
        if self.is_peft_model and hasattr(self.model, "peft_config"):
            peft_type = self.model.peft_config.peft_type
            if peft_type == "LORA":
                # More aggressive clipping for LoRA to prevent instability
                max_grad_norm = self.specs['max_grad_norm'] * 0.8
                logger.info(f"Applying LoRA-specific gradient clipping with norm {max_grad_norm}")
            elif "PREFIX" in peft_type:
                # Less aggressive clipping for prefix tuning
                max_grad_norm = self.specs['max_grad_norm'] * 1.2
                logger.info(f"Applying Prefix-specific gradient clipping with norm {max_grad_norm}")
            else:
                max_grad_norm = self.specs['max_grad_norm']
        else:
            max_grad_norm = self.specs['max_grad_norm']
            
        # Apply clipping
        torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        
        if self.specs.get('debug_gradients', False):
            # Log gradient norms for debugging
            grad_norm = torch.norm(torch.stack([
                torch.norm(p.grad.detach()) 
                for p in parameters 
                if p.grad is not None
            ]))
            logger.debug(f"Gradient norm after clipping: {grad_norm}")

class NLPSeq2SeqTrainer(Seq2SeqTrainer, TrainerCustomizationMixin):
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, 
                task_type, optimizer=None, compute_metrics=None,
                model_init=None, callbacks=None, scheduler=None, 
                generation_config=None, customization_config=None, **kwargs):

        specs = json.load(open(args_dir, 'r'))
        self.specs = {**DEFAULT_SPECS, **specs}

        # Set PEFT-specific training arguments for Seq2Seq
        if self.specs.get('use_peft', False):
            # Enable gradient checkpointing for memory efficiency with PEFT
            self.specs['gradient_checkpointing'] = self.specs.get('use_gradient_checkpointing', True)
            
            # Add adapter dropout if using PEFT
            if 'peft_method' in self.specs and self.specs['peft_method'] == 'lora':
                self.specs['lora_dropout'] = self.specs.get('lora_dropout', 0.05)
                
            # Modify optimizer settings for PEFT
            if not 'optim' in self.specs:
                self.specs['optim'] = 'adamw_torch'
                
            # Ensure predict_with_generate is set for Seq2Seq PEFT models
            self.specs['predict_with_generate'] = True

        self.args = TrainingArguments(
            **self.specs,
            predict_with_generate=self.specs.get('predict_with_generate', True)
        )

        self.datasets = {"train": train_dataset, "eval": eval_dataset}
        self.task_type = task_type
        self.losses = []
        self.tokenizer = tokenizer
        self.generation_config = generation_config

        # Check if this is a PEFT model before initializing
        self.is_peft_model = check_is_peft_model(model)
        
        # Initialize customization
        self.setup_customization(customization_config)
        
        # Initialize optimizer and scheduler
        self.optimizer = self.configure_optimizers()
        if isinstance(self.optimizer, dict):
            optimizer = self.optimizer["optimizer"]
            scheduler = self.optimizer["lr_scheduler"]["scheduler"]
        else:
            optimizer = self.optimizer
            scheduler = None
        
        # Initialize metrics based on task type
        self.metric = get_default_metrics(task_type)
            
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            base_path=os.path.join("checkpoints", task_type),
            model_name=model.config.name_or_path
        )
        
        # Set up generation config
        self.generation_config = generation_config or GenerationConfig(
            max_length=self.specs.get("max_length", 128),
            num_beams=self.specs.get("num_beams", 4),
            length_penalty=self.specs.get("length_penalty", 1.0),
            early_stopping=self.specs.get("early_stopping", True)
        )
        
        super().__init__(model, self.args, 
                        data_collator=data_collator, 
                        train_dataset=train_dataset, 
                        eval_dataset=eval_dataset,
                        tokenizer=tokenizer, 
                        optimizers=(optimizer, scheduler),
                        model_init=model_init, 
                        compute_metrics=compute_metrics or self.compute_metrics,
                        callbacks=callbacks, **kwargs)

    def configure_optimizers(self):
        """Configure and return the optimizer with proper parameter handling."""
        # Get trainable parameters based on PEFT status
        if self.is_peft_model:
            params = [p for p in self.model.parameters() if p.requires_grad]
            logger.info(f"Configuring optimizer for PEFT model with {len(params)} trainable parameters")
        else:
            params = self.model.parameters()
            logger.info("Configuring optimizer for full model")

        # Create optimizer with specified or default parameters
        optimizer = torch.optim.AdamW(
            params,
            lr=self.specs.get('learning_rate', 2e-5),
            weight_decay=self.specs.get('weight_decay', 0.01),
        )
        
        return optimizer

    def train(self):
        """Train the model, with special handling for PEFT models"""
        if self.is_peft_model:
            logger.info("Training PEFT Seq2Seq model")
            # Log PEFT-specific info before training
            try:
                peft_config = getattr(self.model, "peft_config", None)
                if peft_config:
                    logger.info(f"PEFT config: {peft_config}")
                    
                    # Add PEFT-specific memory optimizations
                    optimize_memory_settings(
                        model=self.model,
                        use_gradient_checkpointing=getattr(self.args, "gradient_checkpointing", False)
                    )
                        
            except Exception as e:
                logger.warning(f"Error configuring PEFT model: {e}")
        
        return super().train()
    
    def save_model(self, output_dir=None, _internal_call=False):
        """Save model with special handling for PEFT models"""
        output_dir = output_dir if output_dir is not None else self.args.output_dir

        # Special handling for PEFT models
        if self.is_peft_model:
            try:
                logger.info(f"Saving PEFT adapter to {output_dir}")
                # Save only the adapter parameters
                self.model.save_pretrained(output_dir)
                
                # Also save the tokenizer
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(output_dir)
                    
                # Save generation config if available
                if self.generation_config is not None:
                    self.generation_config.save_pretrained(output_dir)
                    
                return output_dir
            except Exception as e:
                logger.error(f"Error saving PEFT model: {e}")
                # Fall back to standard save
        
        # Standard save for non-PEFT models
        return super().save_model(output_dir, _internal_call)
    
    def _get_train_sampler(self):
        """Get train sampler with PEFT-specific adjustments if needed"""
        sampler = super()._get_train_sampler()
        
        # For quantized PEFT models, we might need to adjust the sampler
        if self.is_peft_model and hasattr(self.model, 'quantization_config'):
            logger.info("Detected quantized PEFT model, checking sampler")
            
            # Check if we need to modify the batch size for quantized models
            if self.args.per_device_train_batch_size > 4 and getattr(self.model, 'quantization_config', {}).get('quantization_type', None) in ['4bit', '8bit']:
                logger.warning("Quantized PEFT model detected - consider reducing batch size if you encounter memory issues")
                
        return sampler
    
    def print_args(self):
        print(self.args)

    def print_trainer_type(self):
        print("I am a NLPSeq2SeqTrainer!")
        if self.is_peft_model:
            print("(with PEFT support)")

    def training_step(self, model, inputs, **kwargs):
        """Enhanced training step with sequence-to-sequence optimizations."""
        # Apply label smoothing if configured
        if self.args.label_smoothing_factor > 0:
            inputs = self._apply_label_smoothing(inputs)
            
        # Apply sequence length optimization
        inputs = self._optimize_sequence_length(inputs)
        
        # Forward pass with teacher forcing
        outputs = model(**inputs)
        
        # Handle gradients with PEFT awareness
        if self.is_peft_model:
            outputs["loss"] = self._handle_gradients(outputs["loss"])
            self._scale_gradients()
            
        return outputs

    def _apply_label_smoothing(self, inputs):
        """Apply label smoothing to target sequences."""
        if "labels" in inputs:
            labels = inputs["labels"].clone()
            ignore_index = -100
            
            # Create smoothing mask
            vocab_size = self.model.config.vocab_size
            smoothing_value = self.args.label_smoothing_factor / (vocab_size - 1)
            
            # Apply smoothing only to non-ignored indices
            mask = labels != ignore_index
            labels[mask] = ((1 - self.args.label_smoothing_factor) * torch.nn.functional.one_hot(labels[mask], vocab_size) + 
                          smoothing_value)
            
            inputs["labels"] = labels
            
        return inputs

    def _optimize_sequence_length(self, inputs):
        """Optimize sequence lengths for better memory usage."""
        max_length = self.model.config.max_position_embeddings
        
        # Dynamically adjust sequence length
        if "input_ids" in inputs and inputs["input_ids"].shape[1] > max_length:
            # Truncate while preserving important tokens
            inputs["input_ids"] = inputs["input_ids"][:, :max_length]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, :max_length]
            if "labels" in inputs:
                inputs["labels"] = inputs["labels"][:, :max_length]
                
        return inputs

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Enhanced prediction step with optimized generation."""
        if not prediction_loss_only:
            # Use optimized generation config
            generation_outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                generation_config=self.generation_config,
                synced_gpus=True if self.is_peft_model else False,
                **self._get_generation_kwargs()
            )
            
            # Compute loss
            with torch.no_grad():
                outputs = model(**inputs)
                
            return (outputs.loss, generation_outputs, inputs["labels"])
            
        # Standard loss computation for training
        return super().prediction_step(model, inputs, prediction_loss_loss_only, ignore_keys)

    def _get_generation_kwargs(self):
        """Get optimized generation kwargs based on task type."""
        kwargs = {}
        
        if self.task_type == "summarization":
            kwargs.update({
                "min_length": self.specs.get("min_length", 10),
                "max_length": self.specs.get("max_length", 128),
                "length_penalty": self.specs.get("length_penalty", 2.0),
                "no_repeat_ngram_size": self.specs.get("no_repeat_ngram_size", 3)
            })
        elif self.task_type == "translation":
            kwargs.update({
                "max_length": self.specs.get("max_length", 128),
                "num_beams": self.specs.get("num_beams", 4),
                "length_penalty": self.specs.get("length_penalty", 0.6)
            })
            
        return kwargs

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
            
            # Apply clipping to each parameter group
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
                        logger.debug(f"{name} gradient norm after clipping: {grad_norm} (max: {norm})")
        else:
            # If parameters are provided, use parent implementation
            super().clip_gradients(parameters)