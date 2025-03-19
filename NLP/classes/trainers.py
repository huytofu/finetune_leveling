import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

import json
import logging
import torch
from typing import Optional, Dict, Any, Union, List
import evaluate
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer
from configs.default_config import DEFAULT_SPECS

logger = logging.getLogger(__name__)

class NLPTrainer(Trainer):
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, 
                task_type, optimizer=None, compute_metrics=None,
                model_init=None, callbacks=None, scheduler=None, **kwargs):

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

        # Check if this is a PEFT model before initializing
        self.is_peft_model = self._check_is_peft_model(model)
        
        # Setup optimizers with PEFT awareness
        if optimizer is None and self.is_peft_model:
            # Only optimize trainable parameters for PEFT models
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.specs.get('learning_rate', 2e-5),
                weight_decay=self.specs.get('weight_decay', 0.01),
            )
            logger.info(f"Created optimizer for PEFT model with {len(trainable_params)} trainable parameters")

        super().__init__(model, self.args, 
                        data_collator=data_collator, 
                        train_dataset=train_dataset, 
                        eval_dataset=eval_dataset,
                        tokenizer=tokenizer, 
                        optimizers=[optimizer, scheduler] if optimizer else None,
                        model_init=model_init, 
                        compute_metrics=compute_metrics,
                        callbacks=callbacks, **kwargs)

    def _check_is_peft_model(self, model):
        """Check if the model is a PEFT model"""
        try:
            from peft import PeftModel
            is_peft = isinstance(model, PeftModel)
            if is_peft:
                logger.info(f"Detected PEFT model: {type(model).__name__}")
                # Log trainable parameters for PEFT model
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"Number of trainable parameters: {trainable_params}")
            return is_peft
        except ImportError:
            logger.info("PEFT not installed, continuing with standard training")
            return False

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
                    if getattr(self.args, "gradient_checkpointing", False):
                        logger.info("Enabling gradient checkpointing for PEFT model")
                        self.model.gradient_checkpointing_enable()
                        
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
        print("I am a NLPTrainer!")
        if self.is_peft_model:
            print("(with PEFT support)")

    def _handle_gradients(self, loss):
        """Handle gradients with PEFT-specific optimizations"""
        if not self.is_peft_model:
            return loss

        # Scale loss based on PEFT type
        if hasattr(self.model, "peft_config"):
            peft_type = self.model.peft_config.peft_type
            if peft_type == "LORA":
                # Scale based on LoRA alpha
                loss = loss / getattr(self.model.peft_config, "lora_alpha", 32)
            elif "PREFIX" in peft_type:
                # Prefix tuning might need different scaling
                loss = loss * 1.2

        # Apply gradient clipping
        if self.args.max_grad_norm > 0:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if hasattr(self.model, "peft_config"):
                peft_type = self.model.peft_config.peft_type
                if peft_type == "LORA":
                    # More aggressive clipping for LoRA
                    max_grad_norm = self.args.max_grad_norm * 1.5
                elif "PREFIX" in peft_type:
                    # More conservative clipping for prefix tuning
                    max_grad_norm = self.args.max_grad_norm * 0.8
                else:
                    max_grad_norm = self.args.max_grad_norm
                    
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

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
            if self.args.debug:
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in trainable_params]))
                self.log({"gradient_norm": grad_norm})
        
        return outputs


class NLPSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, 
                task_type, optimizer=None, compute_metrics=None,
                model_init=None, callbacks=None, scheduler=None, 
                generation_config=None, **kwargs):

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
        self.is_peft_model = self._check_is_peft_model(model)
        
        # Setup optimizers with PEFT awareness
        if optimizer is None and self.is_peft_model:
            # Only optimize trainable parameters for PEFT models
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.specs.get('learning_rate', 2e-5),
                weight_decay=self.specs.get('weight_decay', 0.01),
            )
            logger.info(f"Created optimizer for PEFT model with {len(trainable_params)} trainable parameters")

        super().__init__(model, self.args, 
                        data_collator=data_collator, 
                        train_dataset=train_dataset, 
                        eval_dataset=eval_dataset,
                        tokenizer=tokenizer, 
                        optimizers=[optimizer, scheduler] if optimizer else None,
                        model_init=model_init, 
                        compute_metrics=compute_metrics,
                        callbacks=callbacks, **kwargs)

    def _check_is_peft_model(self, model):
        """Check if the model is a PEFT model"""
        try:
            from peft import PeftModel
            is_peft = isinstance(model, PeftModel)
            if is_peft:
                logger.info(f"Detected PEFT model: {type(model).__name__}")
                # Log trainable parameters for PEFT model
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"Number of trainable parameters: {trainable_params}")
            return is_peft
        except ImportError:
            logger.info("PEFT not installed, continuing with standard training")
            return False

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
                    if getattr(self.args, "gradient_checkpointing", False):
                        logger.info("Enabling gradient checkpointing for PEFT model")
                        self.model.gradient_checkpointing_enable()
                        
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
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """
        Override prediction_step to handle PEFT models with special generation parameters
        """
        if self.is_peft_model and not prediction_loss_only:
            # For generation with PEFT models, we need special handling
            try:
                # Extract the generation kwargs from the model
                generation_kwargs = {}
                
                # Set max length if not provided
                if not hasattr(self.generation_config, "max_length") and not hasattr(self.generation_config, "max_new_tokens"):
                    generation_kwargs["max_length"] = self.args.max_length
                
                # For LoRA in particular, some generation settings may need adjustment
                peft_type = getattr(getattr(self.model, "peft_config", None), "peft_type", None)
                if peft_type and "LORA" in str(peft_type):
                    # LoRA models might need adjusted temperature/top_p for best generation
                    if hasattr(self.generation_config, "temperature") and self.generation_config.temperature == 1.0:
                        # Only adjust if user hasn't explicitly set these
                        logger.info("Adjusting default temperature for LoRA generation")
                        generation_kwargs["temperature"] = 0.7
                
                logger.debug(f"Using generation kwargs for PEFT model: {generation_kwargs}")
                
                # Pass these generation configs to the parent method
                return super().prediction_step(
                    model, 
                    inputs, 
                    prediction_loss_only,
                    ignore_keys=ignore_keys,
                    **generation_kwargs
                )
            except Exception as e:
                logger.warning(f"Error in PEFT prediction_step, falling back to default: {e}")
                
        # For non-PEFT models or if PEFT handling fails, use default behavior
        return super().prediction_step(
            model, 
            inputs, 
            prediction_loss_only,
            ignore_keys=ignore_keys,
        )
    
    def print_args(self):
        print(self.args)

    def print_trainer_type(self):
        print("I am a NLPSeq2SeqTrainer!")
        if self.is_peft_model:
            print("(with PEFT support)")

    def training_step(self, model, inputs, **kwargs):
        """Enhanced Seq2Seq training step with PEFT-aware gradient handling"""
        outputs = super().training_step(model, inputs, **kwargs)
        
        if self.is_peft_model:
            # Handle gradients with Seq2Seq specific adjustments
            outputs["loss"] = self._handle_seq2seq_gradients(outputs["loss"])
            self._scale_seq2seq_gradients()
            
            # Log gradient norms in debug mode
            if self.args.debug:
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in trainable_params]))
                self.log({"gradient_norm": grad_norm})
        
        return outputs

    def _handle_seq2seq_gradients(self, loss):
        """Handle gradients specifically for Seq2Seq PEFT models"""
        if not self.is_peft_model:
            return loss

        # Scale loss based on PEFT type with Seq2Seq adjustments
        if hasattr(self.model, "peft_config"):
            peft_type = self.model.peft_config.peft_type
            if peft_type == "LORA":
                # Different scaling for Seq2Seq LoRA
                loss = loss / (getattr(self.model.peft_config, "lora_alpha", 32) * 1.2)
            elif "PREFIX" in peft_type:
                # Different scaling for Seq2Seq prefix tuning
                loss = loss * 1.1

        # Apply gradient clipping with Seq2Seq specific thresholds
        if self.args.max_grad_norm > 0:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if hasattr(self.model, "peft_config"):
                peft_type = self.model.peft_config.peft_type
                # More conservative clipping for Seq2Seq models
                max_grad_norm = self.args.max_grad_norm * 0.9
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

        return loss

    def _scale_seq2seq_gradients(self):
        """Scale gradients specifically for Seq2Seq PEFT models"""
        if not self.is_peft_model or not hasattr(self.model, "peft_config"):
            return

        # Get encoder and decoder parameters
        encoder_params = []
        decoder_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if 'encoder' in name:
                    encoder_params.append(param)
                elif 'decoder' in name:
                    decoder_params.append(param)

        # Apply different scaling to encoder and decoder gradients
        if self.model.peft_config.peft_type == "LORA":
            # Scale encoder gradients more conservatively
            for param in encoder_params:
                param.grad.data = param.grad.data * 0.9
            
            # Scale decoder gradients more aggressively
            for param in decoder_params:
                param.grad.data = param.grad.data * 1.1