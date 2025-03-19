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


# Set up logging
logging.basicConfig(level=logging.INFO)

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

class AcceleratedNLPTrainer(pl.LightningModule):
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, raw_dataset,
                task_type, optimizer=None, compute_metrics=None,
                model_init=None, callbacks=None, scheduler=None, **kwargs):
        super().__init__()
        try:
            self.model = model
            self.tokenizer = tokenizer
            self.data_collator = data_collator
            self.datasets = {"train": train_dataset, "eval": eval_dataset}
            self.raw_dataset = raw_dataset
            self.task_type = task_type
            self.chosen_metric = compute_metrics
            self.setup_configuration()
            
            # Initialize Accelerator in a deferred way
            # Will be properly set up in setup() to ensure Lightning trainer is initialized
            self.accelerator = None
            
            # Store optimizer and scheduler
            self.optimizer_init = optimizer
            self.scheduler_init = scheduler
            
            # PEFT detection and configuration
            self.is_peft_model = self._check_is_peft_model()
            self._setup_peft_config()
            
            self.save_hyperparameters(ignore=['model', 'tokenizer', 'data_collator'])
            Logger.info("AcceleratedNLPTrainer initialized successfully.")
        except Exception as e:
            ErrorHandler.handle_error(e, "AcceleratedNLPTrainer initialization")

    def _check_is_peft_model(self) -> bool:
        """Check if the model is a PEFT model"""
        try:
            from peft import PeftModel
            is_peft = isinstance(self.model, PeftModel)
            if is_peft:
                peft_type = getattr(self.model.peft_config, "peft_type", "unknown")
                Logger.info(f"Detected PEFT model of type: {peft_type}")
                self._log_peft_params()
            return is_peft
        except ImportError:
            Logger.info("PEFT not installed, continuing with standard training")
            return False

    def _log_peft_params(self):
        """Log PEFT-specific parameter information"""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        Logger.info(f"PEFT model has {trainable_params} trainable parameters out of {total_params} total parameters")
        
        if hasattr(self.model, "peft_config"):
            config = self.model.peft_config
            if hasattr(config, "r"):  # LoRA
                Logger.info(f"LoRA rank: {config.r}")
            elif hasattr(config, "num_virtual_tokens"):  # Prefix Tuning
                Logger.info(f"Number of prefix tokens: {config.num_virtual_tokens}")

    def _setup_peft_config(self):
        """Set up PEFT-specific configurations"""
        if not self.is_peft_model:
            return
            
        # Enable memory efficient training for PEFT
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
            
        # Configure gradient checkpointing
        if self.specs.get('use_gradient_checkpointing', True):
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                Logger.info("Enabled gradient checkpointing for PEFT model")
                
        # Disable caching for memory efficiency
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

    def configure_optimizers(self):
        """Configure optimizers with PEFT awareness"""
        if self.optimizer_init is not None and self.scheduler_init is not None:
            # Prepare optimizer and scheduler with Accelerator
            optimizer = self.accelerator.prepare_optimizer(self.optimizer_init)
            scheduler = self.accelerator.prepare_scheduler(self.scheduler_init)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }

        # Get trainable parameters
        if self.is_peft_model:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            Logger.info(f"Configuring optimizer for PEFT model with {len(trainable_params)} trainable parameters")
        else:
            trainable_params = self.model.parameters()
            
        # Create and prepare optimizer with Accelerator
        optimizer = self.accelerator.prepare_optimizer(
            torch.optim.AdamW(
                trainable_params,
                lr=self.specs['learning_rate'],
                weight_decay=self.specs.get('weight_decay', 0.01)
            )
        )
        
        # Create and prepare scheduler with Accelerator
        scheduler = self.accelerator.prepare_scheduler(
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.specs['learning_rate'],
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1
            )
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def on_before_optimizer_step(self, optimizer):
        """Handle gradient clipping with PEFT awareness"""
        if not self.specs.get('max_grad_norm'):
            return
            
        # Let Accelerator handle the gradient clipping
        if self.is_peft_model:
            # Only clip gradients of trainable parameters
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.accelerator.clip_grad_norm_(trainable_params, self.specs['max_grad_norm'])
        else:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.specs['max_grad_norm'])

    def training_step(self, batch, batch_idx):
        """Training step combining Lightning and Accelerator features"""
        # Use Accelerator's autocast while letting Lightning handle the backward pass
        with self.accelerator.autocast():
            outputs = self.model(**batch)
            loss = outputs.loss
            
        # Scale loss for gradient accumulation if needed
        if self.specs.get('gradient_accumulation_steps', 1) > 1:
            loss = loss / self.specs['gradient_accumulation_steps']
        
        # Log the loss using Lightning's logging
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with metric computation"""
        outputs = self.model(**batch)
        val_loss = outputs.loss
        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)
        
        # Handle predictions
        predictions = self.handle_predictions(outputs)
        return {'loss': val_loss, 'predictions': predictions, 'labels': batch['labels']}

    def validation_epoch_end(self, outputs):
        all_predictions = [x['predictions'] for x in outputs]
        all_labels = [x['labels'] for x in outputs]
        self.compute_metrics(all_predictions, all_labels)

    def handle_predictions(self, outputs):
        # Task-specific prediction handling
        if self.task_type == "classification":
            return torch.argmax(outputs.logits, dim=-1)
        # Add more task-specific logic as needed

    def compute_metrics(self, predictions, labels):
        # Task-specific metric computation
        if self.task_type == "classification":
            accuracy = (predictions == labels).float().mean()
            self.log('val_accuracy', accuracy)
        # Add more task-specific logic as needed

    def on_save_checkpoint(self, checkpoint):
        """Enhanced checkpoint saving with PEFT state"""
        if not self.is_peft_model:
            return checkpoint
            
        try:
            # Save PEFT adapter state
            checkpoint['peft_adapter_state'] = self.model.get_adapter_state_dict()
            
            # Save PEFT config
            if hasattr(self.model, "peft_config"):
                checkpoint['peft_config'] = self.model.peft_config
                
            # Save Accelerator state
            checkpoint['accelerator_state'] = self.accelerator.state
                
            Logger.info("Saved PEFT and Accelerator state to checkpoint")
        except Exception as e:
            Logger.warning(f"Error saving PEFT/Accelerator state: {e}")
            
        return checkpoint
    
    def on_load_checkpoint(self, checkpoint):
        """Load PEFT state properly"""
        if not self.is_peft_model:
            return
            
        try:
            if 'peft_adapter_state' in checkpoint:
                self.model.load_adapter_state_dict(checkpoint['peft_adapter_state'])
                
            if 'peft_config' in checkpoint:
                self.model.peft_config = checkpoint['peft_config']
                
            Logger.info("Loaded PEFT state from checkpoint")
        except Exception as e:
            Logger.warning(f"Error loading PEFT state: {e}")

    def setup(self, stage=None):
        """Initialize accelerator and other components at the setup stage."""
        # Initialize Accelerator with Lightning-compatible settings
        self.accelerator = Accelerator(
            fp16=self.specs.get('fp16', False),
            gradient_accumulation_steps=self.specs.get('gradient_accumulation_steps', 1),
            # Disable Accelerator's own distributed training to avoid conflict with Lightning
            kwargs_handlers=[{'no_cuda': False, 'local_rank': self.local_rank}]
        )
        
        # Prepare model with Accelerator while preserving Lightning's device management
        self.model = self.accelerator.prepare_model(self.model)
        Logger.info("Model prepared with Accelerator")
        
        # Initialize metric
        if hasattr(self, 'chosen_metric') and self.chosen_metric:
            self.metric = evaluate.load(self.chosen_metric)
            
        # Calculate steps for scheduler
        total_train_batch_size = (
            self.specs['per_device_train_batch_size'] 
            * self.specs['gradient_accumulation_steps']
            * self.trainer.num_devices
        )
        self.num_training_steps = len(self.datasets['train']) // total_train_batch_size * self.specs['num_train_epochs']

    def setup_configuration(self):
        config_manager = ConfigManager()
        self.specs = vars(config_manager.parse_args())

    def forward(self, **inputs):
        return self.model(**inputs)

    def train_dataloader(self):
        """Configure training dataloader with PEFT optimizations"""
        dataloader_config = {
            'batch_size': self.specs['per_device_train_batch_size'],
            'shuffle': True,
            'collate_fn': self.data_collator,
            'pin_memory': True,
            'num_workers': 4
        }
        
        if self.is_peft_model:
            # Optimize for PEFT models
            if hasattr(self.model, 'quantization_config'):
                # Reduce batch size for quantized models
                dataloader_config['batch_size'] = min(4, dataloader_config['batch_size'])
            
            # Enable persistent workers for better throughput
            dataloader_config['persistent_workers'] = True
            
        return DataLoader(self.datasets['train'], **dataloader_config)

    def val_dataloader(self):
        """Configure validation dataloader with PEFT optimizations"""
        dataloader_config = {
            'batch_size': self.specs['per_device_eval_batch_size'],
            'shuffle': False,
            'collate_fn': self.data_collator,
            'pin_memory': True,
            'num_workers': 4
        }
        
        if self.is_peft_model:
            # Optimize for PEFT models
            if hasattr(self.model, 'quantization_config'):
                # Reduce batch size for quantized models
                dataloader_config['batch_size'] = min(4, dataloader_config['batch_size'])
            
            # Enable persistent workers for better throughput
            dataloader_config['persistent_workers'] = True
            
        return DataLoader(self.datasets['eval'], **dataloader_config)

    def save_and_upload(self, epoch):
        # Split into save and upload methods
        self.save_checkpoint()
        self.upload_model(epoch)

    def upload_model(self, epoch):
        """
        Upload the model to the hub.
        """
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )
        if self.accelerator.is_main_process:
            self.tokenizer.push_to_hub(
                commit_message=f"Training in progress epoch {epoch}", blocking=False
            )

    def postprocess(self, predictions, labels):
        predictions = predictions.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        label_names = self.datasets["train"].features["labels"].feature.names

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return true_labels, true_predictions

    def compute_qna_metrics(self, start_logits, end_logits, eval_dataset, raw_eval_dataset, chosen_metric, max_answer_length):
        self.metric = evaluate.load_metric(chosen_metric)
        example_to_features = self.process_examples(eval_dataset)

        predicted_answers = []
        for example in raw_eval_dataset:
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = eval_dataset[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1 : -5 - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -5 - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                            end_index < start_index or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in raw_eval_dataset]
        result = self.metric.compute(predictions=predicted_answers, references=theoretical_answers)
        return result
    
    def process_examples(self, eval_dataset):
        """
        Process examples to map example IDs to feature indices.
        """
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(eval_dataset):
            example_to_features[feature["example_id"]].append(idx)
        return example_to_features

    def handle_predictions_and_metric(self, outputs, batch, metric):
        if self.task_type == "token_classification":
            predictions = outputs.logits.argmax(-1)
            labels = batch["labels"]
            predictions = self.accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            predictions = self.accelerator.gather(predictions)
            labels = self.accelerator.gather(labels)
            true_predictions, true_labels = self.postprocess(predictions, labels)
            if metric.name == "rouge":
                true_predictions = ["\n".join(nltk.sent_tokenize(pred)) for pred in true_predictions]
                true_labels = ["\n".join(nltk.sent_tokenize(label)) for label in true_labels]
            metric.add_batch(predictions=true_predictions, references=true_labels)
        elif self.task_type == "masked_language_modeling":
            # TO ADD LATER
            pass
        elif self.task_type in ["summarization", "translation", "text_generation"]:
            pass
        elif self.task_type == "question_answering":
            batch_start_logits = self.accelerator.gather(outputs.start_logits).cpu().numpy()
            batch_end_logits = self.accelerator.gather(outputs.end_logits).cpu().numpy()
            self.start_logits.append(batch_start_logits)
            self.end_logits.append(batch_end_logits)
        else: pass

    def handle_outputs(self, outputs, batch, batch_size, losses, metric):
        loss = outputs.loss
        # Log the loss
        self.accelerator.log({"loss": loss.item()})
        #Gather the losses
        losses.append(self.accelerator.gather(loss.repeat(batch_size)))

        #Gather the metrics
        if self.task_type == "token_classification":
            if batch["labels"] is not None:
                self.handle_predictions_and_metric(outputs, batch, metric)
        elif self.task_type == "masked_language_modeling":
            # TO ADD LATER
            pass

    def train(self):
        """
        Train the model with mixed precision and gradient accumulation.
        """
        for epoch in range(self.num_train_epochs):
            logging.info(f"Starting epoch {epoch}")

            self.model.train()
            for step, batch in enumerate(self.train_dataloader()):
                self._train_step(batch, step)
            
            self.model.eval()
            for step, batch in enumerate(self.val_dataloader()):
                self._eval_step(batch, step)

            self._compute_metrics()
            self.save_and_upload(epoch)
            logging.info(f"Completed epoch {epoch}")

    def _train_step(self, batch, step):
        """
        Lightning-compatible training step that doesn't use Accelerator.
        
        Args:
            batch: The batch of data for the current step.
            step: The current step number.
        """
        # Let Lightning handle mixed precision
        outputs = self.model(**batch)
        loss = outputs.loss / self.specs.get('gradient_accumulation_steps', 1)
        
        # Use PyTorch Lightning's manual optimization if needed
        if self.trainer.precision != 32:  # Using mixed precision
            self.manual_backward(loss)
        else:
            loss.backward()
            
        if (step + 1) % self.specs['gradient_accumulation_steps'] == 0:
            # Use the optimizer from Lightning
            if hasattr(self, 'trainer') and hasattr(self.trainer, 'optimizers'):
                optimizer = self.trainer.optimizers[0]
                optimizer.step()
                optimizer.zero_grad()
                
                # Update LR scheduler if available
                if hasattr(self.trainer, 'lr_schedulers') and self.trainer.lr_schedulers:
                    scheduler = self.trainer.lr_schedulers[0]['scheduler']
                    scheduler.step()
            else:
                # Fallback if trainer is not available
                if hasattr(self, 'optimizer'):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if hasattr(self, 'lr_scheduler'):
                        self.lr_scheduler.step()

    def _eval_step(self, batch, step):
        """
        Perform a single evaluation step.
        
        Args:
            batch: The batch of data for the current step.
            step: The current step number.
        """
        with torch.no_grad():
            outputs = self.model(**batch)
            self.handle_outputs(outputs, batch, self.specs['per_device_eval_batch_size'], 
                                self.losses, self.metric)
        logging.info(f"Finished evaluating step {step}")

    def _compute_metrics(self):
        """
        Compute and log metrics after evaluation.
        """
        losses = torch.cat(self.losses)
        losses = losses[: len(self.datasets['eval'])]
        logging.info(f"Losses: {losses}")

        if self.task_type == "question_answering":
            self.start_logits = np.concatenate(self.start_logits)
            self.end_logits = np.concatenate(self.end_logits)
            self.compute_qna_metrics(self.start_logits, self.end_logits, self.eval_dataset, self.raw_dataset['eval'])
        else:
            self.metric.compute()

    def print_args(self):
        print(self.specs)

    def print_trainer_type(self):
        print("I am an AcceleratedNLPTrainer!")


class AcceleratedNLPSeq2SeqTrainer(AcceleratedNLPTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_generation_config()
        
        if self.is_peft_model:
            self._setup_seq2seq_peft_config()

    def setup_generation_config(self):
        """Set up generation configuration with PEFT awareness"""
        self.generation_config = GenerationConfig(
            max_length=self.specs.get('max_length', 128),
            num_beams=self.specs.get('num_beams', 4),
            do_sample=self.specs.get('do_sample', False),
            temperature=self.specs.get('temperature', 1.0),
            top_p=self.specs.get('top_p', 1.0),
            top_k=self.specs.get('top_k', 50),
            repetition_penalty=self.specs.get('repetition_penalty', 1.0),
            length_penalty=self.specs.get('length_penalty', 1.0)
        )
        
        if self.is_peft_model:
            # Adjust generation parameters based on PEFT type
            peft_type = getattr(self.model.peft_config, "peft_type", "")
            if "LORA" in peft_type:
                self.generation_config.temperature = 0.7
                self.generation_config.top_p = 0.9
            elif "PREFIX" in peft_type:
                self.generation_config.num_beams = max(2, self.generation_config.num_beams - 1)
                self.generation_config.length_penalty = 0.8

    def _setup_seq2seq_peft_config(self):
        """Set up PEFT-specific configurations for seq2seq models"""
        super()._setup_peft_config()
        
        # Additional seq2seq-specific configurations
        if hasattr(self.model, "enable_memory_efficient_attention"):
            self.model.enable_memory_efficient_attention()
            Logger.info("Enabled memory efficient attention for PEFT seq2seq model")
        
        # Handle quantized models
        if hasattr(self.model, 'quantization_config'):
            Logger.info("Detected quantized PEFT model, adjusting configurations")
            self.specs['per_device_train_batch_size'] = min(
                4, self.specs['per_device_train_batch_size']
            )
            self.specs['per_device_eval_batch_size'] = min(
                4, self.specs['per_device_eval_batch_size']
            )

    def validation_step(self, batch, batch_idx):
        """Specialized validation step for seq2seq models"""
        # Use Accelerator's autocast for mixed precision
        with self.accelerator.autocast():
            outputs = self.model(**batch)
            val_loss = outputs.loss
            
            # Generate sequences
            generated_ids = self.generate(batch)
            
        # Process outputs
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Get reference texts
        if isinstance(batch['labels'], torch.Tensor):
            labels = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        else:
            labels = batch['labels']
            
        # Log metrics
        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)
        
        return {
            'loss': val_loss,
            'predictions': generated_texts,
            'labels': labels
        }

    def generate(self, batch):
        """Generate sequences with PEFT and Accelerator awareness"""
        generation_inputs = self._prepare_inputs_for_generation(batch)
        
        try:
            if self.is_peft_model:
                # Use Accelerator's unwrap_model to get the base model for generation
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                return self._generate_with_peft(unwrapped_model, generation_inputs)
            else:
                # Standard generation with Accelerator's autocast
                with self.accelerator.autocast():
                    return self.model.generate(
                        **generation_inputs,
                        **self.generation_config.to_dict()
                    )
        except Exception as e:
            Logger.warning(f"Error in generation: {e}")
            return self.model.generate(
                **generation_inputs,
                max_length=self.specs.get('max_length', 128)
            )

    def _prepare_inputs_for_generation(self, batch):
        """Prepare inputs for generation with proper device handling"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        
        # Add encoder outputs for encoder-decoder models
        if (hasattr(self.model, "encoder") and 
            hasattr(self.model.encoder, "forward")):
            with torch.no_grad():
                encoder_outputs = self.model.encoder(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_dict=True
                )
                inputs["encoder_outputs"] = encoder_outputs
                
        return inputs

    def _generate_with_peft(self, model, inputs):
        """Handle generation for PEFT models with proper memory management"""
        generation_kwargs = self.generation_config.to_dict()
        peft_type = getattr(model.peft_config, "peft_type", "")
        
        # Use Accelerator's autocast for mixed precision generation
        with self.accelerator.autocast():
            if "LORA" in peft_type:
                # Temporarily disable adapter for memory efficiency
                if hasattr(model, "disable_adapter"):
                    model.disable_adapter()
                    
                try:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=min(
                            generation_kwargs.get("max_length", 128),
                            self.specs.get("max_new_tokens", 64)
                        ),
                        **generation_kwargs
                    )
                finally:
                    if hasattr(model, "enable_adapter"):
                        model.enable_adapter()
                        
            elif "PREFIX" in peft_type:
                generation_kwargs["repetition_penalty"] = 1.2
                outputs = model.generate(
                    **inputs,
                    **generation_kwargs
                )
                
            else:
                outputs = model.generate(
                    **inputs,
                    **generation_kwargs
                )
                
        return outputs

    def on_save_checkpoint(self, checkpoint):
        """Enhanced checkpoint saving for seq2seq PEFT models"""
        checkpoint = super().on_save_checkpoint(checkpoint)
        
        if self.is_peft_model:
            try:
                # Save generation config
                if hasattr(self, 'generation_config'):
                    checkpoint['generation_config'] = self.generation_config.to_dict()
                
                # Save encoder/decoder states separately
                if hasattr(self.model, "get_encoder_adapter_state"):
                    checkpoint['encoder_adapter_state'] = self.model.get_encoder_adapter_state()
                if hasattr(self.model, "get_decoder_adapter_state"):
                    checkpoint['decoder_adapter_state'] = self.model.get_decoder_adapter_state()
                    
                Logger.info("Saved seq2seq PEFT state to checkpoint")
            except Exception as e:
                Logger.warning(f"Error saving seq2seq PEFT state: {e}")
                
        return checkpoint
    
    def on_load_checkpoint(self, checkpoint):
        """Enhanced checkpoint loading for seq2seq PEFT models"""
        super().on_load_checkpoint(checkpoint)
        
        if self.is_peft_model:
            try:
                # Load generation config
                if 'generation_config' in checkpoint:
                    self.generation_config = GenerationConfig(**checkpoint['generation_config'])
                
                # Load encoder/decoder states
                if ('encoder_adapter_state' in checkpoint and 
                    hasattr(self.model, "set_encoder_adapter_state")):
                    self.model.set_encoder_adapter_state(checkpoint['encoder_adapter_state'])
                    
                if ('decoder_adapter_state' in checkpoint and 
                    hasattr(self.model, "set_decoder_adapter_state")):
                    self.model.set_decoder_adapter_state(checkpoint['decoder_adapter_state'])
                    
                Logger.info("Loaded seq2seq PEFT state from checkpoint")
            except Exception as e:
                Logger.warning(f"Error loading seq2seq PEFT state: {e}")
