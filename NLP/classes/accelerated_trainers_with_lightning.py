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
                task_type, optimizer, chosen_metric):
        super(AcceleratedNLPTrainer, self).__init__()
        try:
            self.model = model
            self.tokenizer = tokenizer
            self.data_collator = data_collator
            self.datasets = {"train": train_dataset, "eval": eval_dataset}
            self.raw_dataset = raw_dataset
            self.task_type = task_type
            self.chosen_metric = chosen_metric
            self.setup_configuration()
            self.accelerator = None  # Will be initialized in setup
            self.save_hyperparameters()
            Logger.info("AcceleratedNLPTrainer initialized successfully.")
        except Exception as e:
            ErrorHandler.handle_error(e, "AcceleratedNLPTrainer initialization")

    def setup(self, stage=None):
        """Initialize accelerator and other components at the setup stage."""
        self.accelerator = Accelerator(
            fp16=self.specs.get('fp16', False),
            gradient_accumulation_steps=self.specs.get('gradient_accumulation_steps', 1)
        )
        Logger.info("Accelerator initialized in setup.")
        
        # Initialize metric
        if hasattr(self, 'chosen_metric') and self.chosen_metric:
            self.metric = evaluate.load(self.chosen_metric)
            
        # Initialize tracking variables
        self.losses = []
        if self.task_type == "question_answering":
            self.start_logits = []
            self.end_logits = []
            
        # Calculate number of training steps for scheduler
        total_train_batch_size = (
            self.specs['per_device_train_batch_size'] 
            * self.specs['gradient_accumulation_steps']
        )
        self.num_training_steps = len(self.datasets['train']) // total_train_batch_size * self.specs['num_train_epochs']
        
    def setup_configuration(self):
        config_manager = ConfigManager()
        self.specs = vars(config_manager.parse_args())

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step using Lightning's precision handling.
        
        Args:
            batch: The batch of data for the current step.
            batch_idx: The index of the current batch.
            
        Returns:
            The loss value for the current batch.
        """
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # Log the loss
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        
        # For gradient accumulation, scale the loss
        if self.specs.get('gradient_accumulation_steps', 1) > 1:
            loss = loss / self.specs['gradient_accumulation_steps']
        
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        predictions = self.handle_predictions(outputs)
        self.log('val_loss', outputs.loss)
        return {'predictions': predictions, 'labels': batch['labels']}

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

    def configure_optimizers(self):
        """
        Configure optimizers with special handling for PEFT models.
        
        Returns:
            A tuple of (optimizer, scheduler) to be used by PyTorch Lightning.
        """
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
            Logger.info(f"Configured optimizer for PEFT model with {len(trainable_params)} trainable parameters")
        except (ImportError, ValueError, AttributeError, TypeError) as e:
            # For regular models, train all parameters
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.specs['learning_rate'],
                weight_decay=self.specs.get('weight_decay', 0.01)
            )
            Logger.info("Configured optimizer for standard model (non-PEFT)")
        
        # Configure the scheduler
        scheduler = get_scheduler(
            self.specs['scheduler_strategy'],
            optimizer=optimizer,
            num_warmup_steps=self.specs.get('num_warmup_steps', 0),
            num_training_steps=self.num_training_steps
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def on_fit_start(self):
        """
        Set up training-specific configurations when the fit process starts.
        
        This is called by Lightning before training begins and is the ideal place
        to set up gradient clipping and other training-specific configurations.
        """
        # Set up gradient clipping if specified in specs
        if self.specs.get('max_grad_norm', 0) > 0:
            # In PyTorch Lightning, gradient clipping is set on the trainer
            # This will be automatically applied during training
            if hasattr(self, 'trainer'):
                max_grad_norm = self.specs.get('max_grad_norm')
                self.trainer.gradient_clip_val = max_grad_norm
                self.trainer.gradient_clip_algorithm = "norm"
                Logger.info(f"Gradient clipping enabled with max_norm={max_grad_norm}")
                
                # Special handling for PEFT models
                try:
                    from peft import PeftModel
                    if isinstance(self.model, PeftModel):
                        Logger.info("Applying specialized gradient clipping config for PEFT model")
                        # Some PEFT models may need different gradient clipping settings
                        # based on the specific method (e.g., LoRA vs. Prefix Tuning)
                        peft_type = getattr(self.model.peft_config, "peft_type", None)
                        if peft_type and "PREFIX" in str(peft_type):
                            # Prefix tuning might need different gradient clipping
                            prefix_norm = max(1.0, max_grad_norm / 2.0)  # More conservative clipping
                            self.trainer.gradient_clip_val = prefix_norm
                            Logger.info(f"Adjusted gradient clipping for prefix tuning: {prefix_norm}")
                except (ImportError, AttributeError):
                    pass
                    
    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        """
        Apply custom logic before the optimizer step.
        
        This hook is called by Lightning before each optimizer step, allowing for
        custom gradient manipulation, monitoring, or clipping before the actual update.
        
        Args:
            optimizer: The optimizer being used
            optimizer_idx: The index of the optimizer
        """
        # Additional custom gradient processing can be done here
        # This is a good place to add custom gradient scaling for PEFT models
        try:
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                peft_type = getattr(self.model.peft_config, "peft_type", None)
                
                # Handle specific PEFT types that might need special gradient processing
                if peft_type and "LORA" in str(peft_type):
                    # For LoRA, we can implement custom gradient preprocessing if needed
                    lora_gradient_monitoring = self.specs.get('lora_gradient_monitoring', False)
                    if lora_gradient_monitoring:
                        # Monitor gradients of LoRA parameters
                        lora_params = [n for n, p in self.model.named_parameters() 
                                     if 'lora_' in n and p.requires_grad]
                        if lora_params:
                            sample_param = lora_params[0]
                            sample_grad = self.model.get_parameter(sample_param).grad
                            if sample_grad is not None:
                                grad_norm = sample_grad.norm().item()
                                self.log('lora_grad_norm', grad_norm, prog_bar=False)
                                Logger.debug(f"LoRA gradient norm: {grad_norm}")
        except (ImportError, AttributeError, ValueError):
            pass

    def save_checkpoint(self, filepath=None):
        """
        Save model checkpoint with special handling for PEFT models.
        
        Args:
            filepath: Optional path to save the checkpoint. If None, uses the output_dir from specs.
        """
        if filepath is None:
            filepath = self.specs['output_dir']
            
        # Check if we're dealing with a PEFT model
        try:
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                Logger.info(f"Saving PEFT model adapter to {filepath}")
                # Save only the adapter
                self.model.save_pretrained(filepath)
                # Save tokenizer separately
                self.tokenizer.save_pretrained(filepath)
                return
        except (ImportError, AttributeError):
            pass
            
        # For non-PEFT models or if PEFT check fails, use the standard approach
        Logger.info(f"Saving model to {filepath}")
        try:
            # First try using save_pretrained if available (HuggingFace models)
            self.model.save_pretrained(filepath)
        except AttributeError:
            # Fall back to regular PyTorch saving
            torch.save(self.model.state_dict(), os.path.join(filepath, "pytorch_model.bin"))
        
        # Save tokenizer if available
        if hasattr(self, 'tokenizer'):
            self.tokenizer.save_pretrained(filepath)

    def train_dataloader(self):
        """
        Return the training dataloader with PEFT compatibility.
        
        The dataloader is configured with appropriate batch size, workers, and
        any special handling needed for PEFT models.
        
        Returns:
            DataLoader: The training data loader.
        """
        # Check if using a PEFT model to apply any special configurations
        try:
            from peft import PeftModel
            is_peft_model = isinstance(self.model, PeftModel)
        except ImportError:
            is_peft_model = False
            
        # Base configuration for the dataloader
        dataloader_config = {
            'batch_size': self.specs['per_device_train_batch_size'],
            'shuffle': True,
            'collate_fn': self.data_collator,
            'pin_memory': True,
            'num_workers': 4
        }
        
        # Apply any PEFT-specific dataloader modifications if needed
        if is_peft_model:
            Logger.info("Configuring training DataLoader for PEFT model")
            # PEFT models may benefit from different prefetch settings
            dataloader_config['prefetch_factor'] = 2
            
            # For certain PEFT methods like prefix tuning, we might need to adjust batch size
            peft_type = getattr(self.model.peft_config, "peft_type", None)
            if peft_type and "PREFIX" in str(peft_type):
                Logger.info("Detected prefix tuning PEFT type, adjusting dataloader settings")
                # Prefix tuning might benefit from these settings
                dataloader_config['persistent_workers'] = True
                
        return DataLoader(self.datasets['train'], **dataloader_config)

    def val_dataloader(self):
        """
        Return the validation dataloader with PEFT compatibility.
        
        The dataloader is configured with appropriate batch size, workers, and
        any special handling needed for PEFT models.
        
        Returns:
            DataLoader: The validation data loader.
        """
        # Check if using a PEFT model to apply any special configurations
        try:
            from peft import PeftModel
            is_peft_model = isinstance(self.model, PeftModel)
        except ImportError:
            is_peft_model = False
            
        # Base configuration for the dataloader
        dataloader_config = {
            'batch_size': self.specs['per_device_eval_batch_size'],
            'shuffle': False,
            'collate_fn': self.data_collator,
            'pin_memory': True,
            'num_workers': 4
        }
        
        # Apply any PEFT-specific dataloader modifications if needed
        if is_peft_model:
            Logger.info("Configuring validation DataLoader for PEFT model")
            # PEFT models may benefit from different prefetch settings
            dataloader_config['prefetch_factor'] = 2
            
            # For certain quantized PEFT methods, we might want to reduce batch size
            # to avoid memory issues during evaluation
            if hasattr(self.model, 'quantization_config'):
                Logger.info("Detected quantized PEFT model, adjusting dataloader settings")
                # For quantized models, we might need smaller batches for evaluation
                if dataloader_config['batch_size'] > 4:
                    # Only reduce if current batch size is large enough
                    # (avoid making it too small)
                    dataloader_config['batch_size'] = max(4, dataloader_config['batch_size'] // 2)
                
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

    def on_save_checkpoint(self, checkpoint):
        """
        Customize the checkpoint saving process.
        
        Args:
            checkpoint: The checkpoint dictionary being saved.
            
        Returns:
            The modified checkpoint dictionary.
        """
        # Check if this is a PEFT model
        try:
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                # Save PEFT-specific state
                checkpoint['peft_adapter_state'] = self.model.get_adapter_state_dict()
                Logger.info("Saved PEFT adapter state to checkpoint")
        except (ImportError, AttributeError):
            pass
            
        return checkpoint
    
    def on_load_checkpoint(self, checkpoint):
        """
        Customize the checkpoint loading process.
        
        Args:
            checkpoint: The checkpoint dictionary being loaded.
        """
        # Check if this is a PEFT model and if we have adapter state
        try:
            from peft import PeftModel
            if isinstance(self.model, PeftModel) and 'peft_adapter_state' in checkpoint:
                # Load PEFT-specific state
                self.model.load_adapter_state_dict(checkpoint['peft_adapter_state'])
                Logger.info("Loaded PEFT adapter state from checkpoint")
        except (ImportError, AttributeError):
            pass


class AcceleratedNLPSeq2SeqTrainer(pl.LightningModule):
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, raw_dataset,
                task_type, optimizer, chosen_metric):
        super(AcceleratedNLPSeq2SeqTrainer, self).__init__()
        try:
            self.model = model
            self.tokenizer = tokenizer
            self.data_collator = data_collator
            self.datasets = {"train": train_dataset, "eval": eval_dataset}
            self.raw_dataset = raw_dataset
            self.task_type = task_type
            self.chosen_metric = chosen_metric
            self.setup_configuration()
            self.accelerator = None  # Will be initialized in setup
            self.save_hyperparameters()
            Logger.info("AcceleratedNLPSeq2SeqTrainer initialized successfully.")
        except Exception as e:
            ErrorHandler.handle_error(e, "AcceleratedNLPSeq2SeqTrainer initialization")

    def setup(self, stage=None):
        """Initialize accelerator and other components at the setup stage."""
        self.accelerator = Accelerator(
            fp16=self.specs.get('fp16', False),
            gradient_accumulation_steps=self.specs.get('gradient_accumulation_steps', 1)
        )
        Logger.info("Accelerator initialized in setup.")
        
        # Initialize metric
        if hasattr(self, 'chosen_metric') and self.chosen_metric:
            self.metric = evaluate.load(self.chosen_metric)
            
        # Initialize tracking variables
        self.losses = []
        if self.task_type == "question_answering":
            self.start_logits = []
            self.end_logits = []
            
        # Calculate number of training steps for scheduler
        total_train_batch_size = (
            self.specs['per_device_train_batch_size'] 
            * self.specs['gradient_accumulation_steps']
        )
        self.num_training_steps = len(self.datasets['train']) // total_train_batch_size * self.specs['num_train_epochs']
        
    def setup_configuration(self):
        config_manager = ConfigManager()
        self.specs = vars(config_manager.parse_args())

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        predictions = self.handle_seq2seq_predictions(outputs)
        self.log('val_loss', outputs.loss)
        return {'predictions': predictions, 'labels': batch['labels']}

    def validation_epoch_end(self, outputs):
        all_predictions = [x['predictions'] for x in outputs]
        all_labels = [x['labels'] for x in outputs]
        self.compute_seq2seq_metrics(all_predictions, all_labels)

    def handle_seq2seq_predictions(self, outputs):
        # Sequence-to-sequence specific prediction handling
        return self.tokenizer.batch_decode(outputs.logits, skip_special_tokens=True)

    def compute_seq2seq_metrics(self, predictions, labels):
        # Sequence-to-sequence specific metric computation
        # Example: BLEU score
        bleu_score = sentence_bleu([labels], predictions)
        self.log('val_bleu', bleu_score)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.specs['learning_rate'])
        scheduler = get_scheduler(
            self.specs['scheduler_strategy'],
            optimizer=optimizer,
            num_warmup_steps=self.specs.get('num_warmup_steps', 0),
            num_training_steps=self.num_training_steps,
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.datasets['train'],
            batch_size=self.specs['per_device_train_batch_size'],
            shuffle=True,
            collate_fn=self.data_collator,
            pin_memory=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets['eval'],
            batch_size=self.specs['per_device_eval_batch_size'],
            shuffle=False,
            collate_fn=self.data_collator,
            pin_memory=True,
            num_workers=4
        )

    def save_and_upload(self, epoch):
        # Split into save and upload methods
        self.save_model(epoch)
        self.upload_model(epoch)

    def save_model(self, epoch):
        """
        Save the model to the specified directory.
        """
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(self.specs['output_dir'], save_function=self.accelerator.save)
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(self.specs['output_dir'])

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
        predictions = predictions.detach().cpu().clone().numpy()
        labels = labels.detach().cpu().clone().numpy()
        
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Remove ignored index (special tokens) and convert to labels
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        true_predictions = [pred.strip() for pred in decoded_preds]
        true_labels = [label.strip() for label in decoded_labels]

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
        predictions = self.accelerator.unwrap_model(self.model).generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=self.specs["max_length"],
        )
        labels = batch["labels"]
        predictions = self.accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
        predictions = self.accelerator.gather(predictions)
        labels = self.accelerator.gather(labels)
        true_predictions, true_labels = self.postprocess(predictions, labels)
        metric.add_batch(predictions=true_predictions, references=true_labels)

    def handle_outputs(self, outputs, batch, batch_size, losses, metric):
        loss = outputs.loss
        self.accelerator.log({"loss": loss.item()})
        #Gather the losses
        losses.append(self.accelerator.gather(loss.repeat(batch_size)))

        self.handle_predictions_and_metric(outputs, batch, metric)

    def train(self):
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
        Perform a single training step.
        
        Args:
            batch: The batch of data for the current step.
            step: The current step number.
        """
        with self.accelerator.autocast():  # Use mixed precision
            outputs = self.model(**batch)
            loss = outputs.loss / self.specs['gradient_accumulation_steps']
        self.accelerator.backward(loss)

        if (step + 1) % self.specs['gradient_accumulation_steps'] == 0:
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.progress_bar.update(1)

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
        print("I am an AcceleratedNLPSeq2SeqTrainer!")
