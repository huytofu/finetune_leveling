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
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass
import random

from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import get_scheduler, GenerationConfig
from tqdm.auto import tqdm
from configs.default_config import DEFAULT_SPECS
from .config_manager import ConfigManager
from .utils import Logger, ErrorHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PeftConfig:
    """
    Configuration for PEFT (Parameter-Efficient Fine-Tuning) models.
    
    This class defines the configuration parameters used for different PEFT methods
    like LoRA, Prefix Tuning, etc.
    
    Attributes:
        peft_type (str): Type of PEFT method (e.g., "LORA", "PREFIX")
        task_type (str): Type of task being performed
        inference_mode (bool): Whether the model is in inference mode
        r (int): Rank for LoRA
        lora_alpha (int): Alpha parameter for LoRA
        lora_dropout (float): Dropout rate for LoRA
        bias (str): Bias handling strategy
        target_modules (List[str]): Specific modules to apply PEFT to
        layers_to_transform (List[int]): Specific layers to transform
        fan_in_fan_out (bool): Whether to use fan-in/fan-out rescaling
        modules_to_save (List[str]): Modules to save separately
        init_lora_weights (bool): Whether to initialize LoRA weights
    """
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

class AcceleratedNLPTrainer():
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
    
    Attributes:
        model: The model to be trained
        optimizer: The optimizer for training
        data_collator: The data collator for batching
        datasets: A dictionary containing training and evaluation datasets
        raw_dataset: The raw dataset for processing
        tokenizer: The tokenizer for text processing
        specs: Configuration specifications for training
        task_type: The type of task (e.g., classification, generation)
        losses: A list to store training losses
        metric: The evaluation metric
        accelerator: The Accelerate instance for distributed training
    """
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, raw_dataset,
                task_type, optimizer=None, chosen_metric=None):
        """
        Initialize the AcceleratedNLPTrainer.
        
        Args:
            args_dir (str): Directory containing configuration arguments
            model: The model to be trained
            tokenizer: The tokenizer for text processing
            data_collator: The data collator for batching
            train_dataset: The training dataset
            eval_dataset: The evaluation dataset
            raw_dataset: The raw dataset for processing
            task_type (str): The type of task (e.g., classification, generation)
            optimizer: The optimizer for training
            chosen_metric (str): The metric for evaluation
        """
        try:
            self.model = model
            self.optimizer = optimizer
            self.data_collator = data_collator
            self.datasets = {"train": train_dataset, "eval": eval_dataset}
            self.raw_dataset = raw_dataset
            self.tokenizer = tokenizer
            
            self.setup_configuration()
            self.is_peft_model = self._check_is_peft_model(model)
            self.peft_config = self._get_peft_config() if self.is_peft_model else None
            
            self.optimizer = self.configure_optimizers()
            
            self.prepare_with_accelerator(train_dataset, eval_dataset)
            self.prepare_scheduler()

            self.progress_bar = tqdm(range(self.num_training_steps))
            self.task_type = task_type
            self.losses = []
            self.metric = evaluate.load(chosen_metric)
            
            self._setup_gradient_accumulation()
            
            # Lightning coordination flag - set to False as this is Accelerate-only
            self.use_lightning = False
            self.use_accelerate = True
            
            logger.info("AcceleratedNLPTrainer initialized successfully.")
        except Exception as e:
            ErrorHandler.handle_error(e, "AcceleratedNLPTrainer initialization")

    def setup_configuration(self):
        """
        Set up configuration using ConfigManager.
        
        This loads the configuration from the default specs and any custom settings.
        """
        config_manager = ConfigManager()
        self.specs = vars(config_manager.parse_args())

    def _check_is_peft_model(self, model):
        """
        Check if the model is a PEFT model and get its type.
        
        Args:
            model: The model to check
            
        Returns:
            bool: Whether the model is a PEFT model
        """
        try:
            from peft import PeftModel
            is_peft = isinstance(model, PeftModel)
            if is_peft:
                peft_type = getattr(model.peft_config, "peft_type", "unknown")
                logger.info(f"Detected PEFT model of type: {peft_type}")
                self._log_peft_params(model)
            return is_peft
        except ImportError:
            logger.info("PEFT not installed, continuing with standard training")
            return False

    def _log_peft_params(self, model):
        """
        Log PEFT-specific parameter information.
        
        Args:
            model: The PEFT model to log information about
        """
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"PEFT model has {trainable_params} trainable parameters out of {total_params} total parameters")
        
        # Log adapter-specific information
        if hasattr(model, "peft_config"):
            config = model.peft_config
            if hasattr(config, "r"):  # LoRA
                logger.info(f"LoRA rank: {config.r}")
            elif hasattr(config, "num_virtual_tokens"):  # Prefix Tuning
                logger.info(f"Number of prefix tokens: {config.num_virtual_tokens}")

    def _get_peft_config(self):
        """
        Get PEFT-specific configuration.
        
        Returns:
            PeftConfig: The PEFT configuration object
        """
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
        
        # Prepare optimizer with Accelerator if available
        if hasattr(self, 'accelerator'):
            optimizer = self.accelerator.prepare_optimizer(optimizer)
        
        # Configure scheduler if specified
        if self.specs.get('use_lr_scheduler', False):
            scheduler = self.prepare_scheduler(optimizer)
            return optimizer, scheduler
        
        return optimizer, None

    def prepare_with_accelerator(self, train_dataset, eval_dataset):
        """Prepare with Accelerator, including PEFT and mixed precision support"""
        try:
            # Configure mixed precision based on PEFT and model type
            mixed_precision = 'fp16' if self.specs.get('fp16', True) else 'no'
            if self.is_peft_model and hasattr(self.model, 'quantization_config'):
                if self.model.quantization_config.get('quantization_type') in ['4bit', '8bit']:
                    logger.info("Detected quantized PEFT model, adjusting mixed precision settings")
                    mixed_precision = 'bf16'  # Better compatibility with quantized models
            
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.specs.get('gradient_accumulation_steps', 1),
                mixed_precision=mixed_precision,
                log_with="all",
                project_dir=self.specs['output_dir']
            )
            
            # Prepare dataloaders with PEFT-aware batch sizes
            self.train_dataloader = self.prepare_data_loader("train", train_dataset)
            self.eval_dataloader = self.prepare_data_loader("eval", eval_dataset)

            # Prepare model and optimizer
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = \
                self.accelerator.prepare(self.model, self.optimizer, self.train_dataloader, self.eval_dataloader)
                
            logger.info(f"Model prepared with Accelerator using {mixed_precision} precision")
        except Exception as e:
            logger.error(f"Error in Accelerator preparation: {e}")
            raise

    def _setup_gradient_accumulation(self):
        """Setup gradient accumulation with PEFT-specific handling"""
        if self.is_peft_model:
            # Adjust gradient accumulation for different PEFT types
            if self.peft_config.peft_type == "LORA":
                # LoRA typically needs less accumulation
                self.effective_batch_size = self.specs['per_device_train_batch_size'] * \
                                         max(1, self.specs.get('gradient_accumulation_steps', 1) // 2)
            elif "PREFIX" in self.peft_config.peft_type:
                # Prefix tuning might need more accumulation
                self.effective_batch_size = self.specs['per_device_train_batch_size'] * \
                                         self.specs.get('gradient_accumulation_steps', 1) * 2
            else:
                self.effective_batch_size = self.specs['per_device_train_batch_size'] * \
                                         self.specs.get('gradient_accumulation_steps', 1)
        else:
            self.effective_batch_size = self.specs['per_device_train_batch_size'] * \
                                     self.specs.get('gradient_accumulation_steps', 1)

    def prepare_data_loader(self, slice_type, dataset):
        """
        Prepare the data loader with dynamic batching and efficient data loading.
        
        Args:
            slice_type (str): The type of dataset slice ('train' or 'eval').
            dataset: The dataset to load.
        
        Returns:
            DataLoader: The prepared data loader.
        """
        shuffle, batch_size = self.get_dataloader_params(slice_type)
        
        # Use dynamic padding to minimize padding token computation
        dataset.set_format('torch')
        self.dataloader = DataLoader(
            dataset, 
            shuffle=shuffle, 
            collate_fn=self.data_collator, 
            batch_size=batch_size,
            pin_memory=True,  # Pin memory for faster data transfer to GPU
            num_workers=4  # Use multiple workers for data loading
        )
        
        return self.dataloader
    
    def get_dataloader_params(self, slice_type):
        """
        Get DataLoader parameters based on slice type.
        """
        if slice_type == "train":
            return True, self.specs['per_device_train_batch_size']
        else:
            return False, self.specs['per_device_eval_batch_size']
    
    def prepare_scheduler(self):
        """Prepare learning rate scheduler with proper configuration."""
        num_training_steps = self.calculate_training_steps()
        
        # Get scheduler type and warmup ratio from specs
        scheduler_type = self.specs.get('scheduler_type', 'linear')
        warmup_ratio = self.specs.get('warmup_ratio', 0.1)
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        
        if scheduler_type == 'linear':
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type == 'cosine':
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type == 'one_cycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.specs.get('learning_rate', 2e-5),
                total_steps=num_training_steps,
                pct_start=warmup_ratio
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
            
        # Prepare scheduler with Accelerator if available
        if hasattr(self, 'accelerator'):
            scheduler = self.accelerator.prepare_scheduler(scheduler)
            
        logger.info(f"Created {scheduler_type} scheduler with {num_warmup_steps} warmup steps")
        return scheduler

    def calculate_training_steps(self):
        """Calculate total training steps."""
        train_dataset_size = len(self.datasets['train'])
        batch_size = self.specs.get('per_device_train_batch_size', 8)
        grad_accumulation = self.specs.get('gradient_accumulation_steps', 1)
        num_epochs = self.specs.get('num_train_epochs', 3)
        
        # Account for distributed training with Accelerator
        if hasattr(self, 'accelerator'):
            num_processes = self.accelerator.num_processes
            batch_size *= num_processes
            
        steps_per_epoch = train_dataset_size // (batch_size * grad_accumulation)
        total_steps = steps_per_epoch * num_epochs
        
        return total_steps

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
        
        if self.is_peft_model:
            try:
                logger.info(f"Saving PEFT adapter state to {self.specs['output_dir']}")
                # Save adapter state
                unwrapped_model.save_pretrained(self.specs['output_dir'])
                
                # Save additional PEFT-specific information
                if hasattr(unwrapped_model, "peft_config"):
                    peft_config_path = os.path.join(self.specs['output_dir'], "peft_config.json")
                    with open(peft_config_path, 'w') as f:
                        json.dump(unwrapped_model.peft_config, f)
                
                # Save tokenizer and generation config if available
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(self.specs['output_dir'])
                if hasattr(self, 'generation_config'):
                    self.generation_config.save_pretrained(self.specs['output_dir'])
                    
                return
            except Exception as e:
                logger.error(f"Error saving PEFT model: {e}")
        
        # Standard save for non-PEFT models
        unwrapped_model.save_pretrained(
            self.specs['output_dir'],
            save_function=self.accelerator.save
        )
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

    def _handle_gradients(self, loss, step):
        """Handle gradients with PEFT-specific optimizations and Accelerator integration"""
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

        # Scale loss for gradient accumulation
        if self.specs.get('gradient_accumulation_steps', 1) > 1:
            loss = loss / self.specs['gradient_accumulation_steps']

        # Backward pass with Accelerator
        self.accelerator.backward(loss)

        # Apply gradient clipping on accumulation step
        if (step + 1) % self.specs.get('gradient_accumulation_steps', 1) == 0:
            if self.specs.get('max_grad_norm', 1.0) > 0:
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                if hasattr(self.model, "peft_config"):
                    peft_type = self.model.peft_config.peft_type
                    if peft_type == "LORA":
                        # More aggressive clipping for LoRA
                        max_grad_norm = self.specs.get('max_grad_norm', 1.0) * 1.5
                    elif "PREFIX" in peft_type:
                        # More conservative clipping for prefix tuning
                        max_grad_norm = self.specs.get('max_grad_norm', 1.0) * 0.8
                    else:
                        max_grad_norm = self.specs.get('max_grad_norm', 1.0)
                        
                    self.accelerator.clip_grad_norm_(trainable_params, max_grad_norm)

            # Scale gradients based on PEFT type
            self._scale_gradients()

        return loss

    def _scale_gradients(self):
        """Apply PEFT-specific gradient scaling with Accelerator integration"""
        if not self.is_peft_model or not hasattr(self.model, "peft_config"):
            return

        peft_type = self.model.peft_config.peft_type
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            # Check for NaN or Inf gradients
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                logger.warning(f"Invalid gradients detected in {name}")
                param.grad.data = torch.zeros_like(param.grad.data)
                continue

            if peft_type == "LORA" and "lora_" in name:
                # Scale LoRA gradients
                param.grad.data = param.grad.data / self.model.peft_config.lora_alpha
            elif "PREFIX" in peft_type and "prefix" in name:
                # More conservative updates for prefix parameters
                param.grad.data = param.grad.data * 0.8

    def training_step(self, batch, step):
        """
        Execute a single training step with optimizations and loss handling.
        
        Args:
            batch: The current batch of data
            step: The current training step
            
        Returns:
            dict: Training metrics for this step including loss
        """
        try:
            self.model.train()
            
            # Apply memory optimizations
            if self.specs.get('use_gradient_checkpointing', False):
                self.model.gradient_checkpointing_enable()
                
            # Forward pass with automatic mixed precision
            with self.accelerator.autocast():
                outputs = self.model(**batch)
                loss = outputs.loss
                
            # Handle loss scaling based on PEFT type
            if self.is_peft_model and hasattr(self.model, "peft_config"):
                peft_type = self.model.peft_config.peft_type
                if peft_type == "LORA":
                    # Scale based on LoRA alpha
                    loss = loss / getattr(self.model.peft_config, "lora_alpha", 32)
                elif "PREFIX" in peft_type:
                    # Prefix tuning might need different scaling
                    loss = loss * 1.2
                
            # Handle gradient accumulation
            if self.specs.get('gradient_accumulation_steps', 1) > 1:
                loss = loss / self.specs.get('gradient_accumulation_steps')
                
            # Backward pass with gradient scaling
            self.accelerator.backward(loss)
            
            # Gradient clipping
            if self.specs.get('max_grad_norm', 0) > 0:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.specs['max_grad_norm']
                )
                
            # Scale gradients based on PEFT type
            self._scale_gradients()
                
            # Update weights if gradient accumulation complete
            if (step + 1) % self.specs.get('gradient_accumulation_steps', 1) == 0:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
            # Compute and return metrics
            metrics = {
                'loss': loss.item(),
                'learning_rate': self.lr_scheduler.get_last_lr()[0],
                'step': step,
                'epoch': step / self.num_training_steps
            }
            
            # Update progress bar
            self.progress_bar.update(1)
            self.progress_bar.set_postfix(**metrics)
            
            return metrics
            
        except Exception as e:
            ErrorHandler.handle_error(e, "training_step")
            raise

    def train(self):
        """
        Execute the training loop with all optimizations.
        
        Returns:
            dict: Final training metrics
        """
        try:
            logger.info("Starting optimized training loop")
            self.model.train()
            total_loss = 0
            
            # Enable memory optimizations
            if self.specs.get('use_gradient_checkpointing', False):
                logger.info("Enabling gradient checkpointing")
                self.model.gradient_checkpointing_enable()
                
            # Main training loop
            for step, batch in enumerate(self.train_dataloader):
                # Training step with optimizations
                metrics = self.training_step(batch, step)
                total_loss += metrics['loss']
                
                # Evaluation if needed
                if step > 0 and step % self.specs.get('eval_steps', 500) == 0:
                    eval_metrics = self._eval_step(batch, step)
                    metrics.update(eval_metrics)
                    
                # Save checkpoint if needed
                if step > 0 and step % self.specs.get('save_steps', 500) == 0:
                    self.save_model(step)
                    
            # Final evaluation
            final_metrics = self._compute_metrics()
            
            # Clean up and return results
            self.accelerator.free_memory()
            return {
                'train_loss': total_loss / len(self.train_dataloader),
                **final_metrics
            }
            
        except Exception as e:
            ErrorHandler.handle_error(e, "train")
            raise

    def _eval_step(self, batch, step):
        """
        Execute a single evaluation step with optimizations.
        
        Args:
            batch: The current batch of data
            step: The current training step
            
        Returns:
            dict: Evaluation metrics for this step
        """
        try:
            self.model.eval()
            metrics = {}
            
            with torch.no_grad():
                # Forward pass with automatic mixed precision
                with self.accelerator.autocast():
                    outputs = self.model(**batch)
                    
                # Compute metrics
                if self.metric is not None:
                    metrics = self.handle_predictions_and_metric(outputs, batch, self.metric)
                    
            return metrics
            
        except Exception as e:
            ErrorHandler.handle_error(e, "_eval_step")
            raise

    def _compute_metrics(self):
        """
        Compute final metrics with optimizations.
        
        Returns:
            dict: Computed metrics
        """
        try:
            self.model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in self.eval_dataloader:
                    with self.accelerator.autocast():
                        outputs = self.model(**batch)
                        predictions = outputs.logits
                        labels = batch['labels']
                        
                    # Gather predictions and labels from all processes
                    predictions = self.accelerator.gather(predictions)
                    labels = self.accelerator.gather(labels)
                    
                    all_preds.append(predictions)
                    all_labels.append(labels)
                    
            # Concatenate all predictions and labels
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Post-process predictions if needed
            if hasattr(self, 'post_process_predictions'):
                all_preds = self.post_process_predictions(all_preds)
                
            # Compute metrics
            if self.metric is not None:
                metrics = self.metric.compute(
                    predictions=all_preds,
                    references=all_labels
                )
                return metrics
                
            return {}
            
        except Exception as e:
            ErrorHandler.handle_error(e, "_compute_metrics")
            raise

    def print_args(self):
        print(self.specs)

    def print_trainer_type(self):
        print("I am an AcceleratedNLPTrainer!")


class AcceleratedNLPSeq2SeqTrainer(AcceleratedNLPTrainer):
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, raw_dataset,
                task_type, optimizer=None, chosen_metric=None):
        super().__init__(args_dir, model, tokenizer, 
                        data_collator, train_dataset, eval_dataset, raw_dataset,
                        task_type, optimizer, chosen_metric)
        self.setup_seq2seq_config()
        
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

    def _prepare_inputs(self, inputs):
        """Enhanced input preparation with specialized attention patterns"""
        prepared_inputs = super()._prepare_inputs(inputs)
        
        # Handle attention masks differently for encoder and decoder
        if 'attention_mask' in prepared_inputs:
            encoder_attention_mask = prepared_inputs['attention_mask']
            # Create causal mask for decoder
            decoder_attention_mask = self._create_causal_mask(
                prepared_inputs.get('decoder_input_ids', None)
            )
            
            prepared_inputs['encoder_attention_mask'] = encoder_attention_mask
            prepared_inputs['decoder_attention_mask'] = decoder_attention_mask
            
        return prepared_inputs

    def _create_causal_mask(self, input_ids):
        """Create causal attention mask for decoder"""
        if input_ids is None:
            return None
            
        batch_size, seq_length = input_ids.shape
        mask = torch.triu(
            torch.ones((seq_length, seq_length), dtype=torch.bool, device=self.accelerator.device),
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

        # Scale loss for gradient accumulation if needed
        if self.specs.get('gradient_accumulation_steps', 1) > 1:
            loss = loss / self.specs['gradient_accumulation_steps']

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
                # Check for NaN or Inf gradients
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    logger.warning(f"Invalid gradients detected in {name}")
                    param.grad.data = torch.zeros_like(param.grad.data)
                    continue

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

    def training_step(self, batch, step):
        """
        Enhanced training step with specialized seq2seq handling and loss computation.
        
        Args:
            batch: The current batch of data
            step: The current training step
            
        Returns:
            dict: Training metrics for this step including loss
        """
        try:
            # Setup gradient checkpointing
            self._setup_gradient_checkpointing()
            
            # Prepare inputs with attention patterns
            inputs = self._prepare_inputs(batch)
            
            # Forward pass with teacher forcing
            with self.accelerator.autocast():
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
            
            # Handle loss scaling based on PEFT type
            if self.is_peft_model and hasattr(self.model, "peft_config"):
                peft_type = self.model.peft_config.peft_type
                if peft_type == "LORA":
                    # Different scaling for Seq2Seq LoRA
                    loss = loss / (getattr(self.model.peft_config, "lora_alpha", 32) * 1.2)
                elif "PREFIX" in peft_type:
                    # Different scaling for Seq2Seq prefix tuning
                    loss = loss * 1.1
            
            # Handle gradient accumulation
            if self.specs.get('gradient_accumulation_steps', 1) > 1:
                loss = loss / self.specs.get('gradient_accumulation_steps')
            
            # Backward pass
            self.accelerator.backward(loss)
            
            # Scale gradients
            self._scale_seq2seq_gradients()
            
            # Update weights if gradient accumulation complete
            if (step + 1) % self.specs.get('gradient_accumulation_steps', 1) == 0:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
            # Compute metrics
            metrics = {
                'loss': loss.item(),
                'learning_rate': self.lr_scheduler.get_last_lr()[0],
                'step': step,
                'epoch': step / self.num_training_steps
            }
            
            # Log gradient norm if in debug mode
            if self.specs.get('debug_gradients', False):
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in trainable_params if p.grad is not None]))
                metrics['gradient_norm'] = grad_norm.item()
            
            # Update progress bar
            self.progress_bar.update(1)
            self.progress_bar.set_postfix(**metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in Seq2Seq training step {step}: {e}")
            raise e

    def save_model(self, output_dir: str = None):
        """Enhanced model saving with seq2seq state management"""
        output_dir = output_dir or self.specs.get('output_dir', 'model')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Save PEFT-specific states
        if self.is_peft_model:
            # Save encoder and decoder states separately
            if hasattr(unwrapped_model, 'encoder'):
                encoder_state = unwrapped_model.encoder.state_dict()
                torch.save(encoder_state, os.path.join(output_dir, 'encoder_state.bin'))
            if hasattr(unwrapped_model, 'decoder'):
                decoder_state = unwrapped_model.decoder.state_dict()
                torch.save(decoder_state, os.path.join(output_dir, 'decoder_state.bin'))
            
            # Save generation config
            if hasattr(self, 'generation_config'):
                self.generation_config.save_pretrained(output_dir)
            
            # Save memory optimization settings
            memory_config = {
                'use_selective_activation_cache': self.use_selective_activation_cache,
                'teacher_forcing_ratio': self.teacher_forcing_ratio
            }
            with open(os.path.join(output_dir, 'memory_config.json'), 'w') as f:
                json.dump(memory_config, f)
        
        # Save the full model
        unwrapped_model.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")

    def load_model(self, model_path: str):
        """Enhanced model loading with seq2seq state management"""
        # Load the base model
        super().load_model(model_path)
        
        if self.is_peft_model:
            # Load encoder and decoder states
            encoder_path = os.path.join(model_path, 'encoder_state.bin')
            if os.path.exists(encoder_path) and hasattr(self.model, 'encoder'):
                encoder_state = torch.load(encoder_path, map_location=self.accelerator.device)
                self.model.encoder.load_state_dict(encoder_state)
                
            decoder_path = os.path.join(model_path, 'decoder_state.bin')
            if os.path.exists(decoder_path) and hasattr(self.model, 'decoder'):
                decoder_state = torch.load(decoder_path, map_location=self.accelerator.device)
                self.model.decoder.load_state_dict(decoder_state)
            
            # Load generation config
            generation_config_path = os.path.join(model_path, 'generation_config.json')
            if os.path.exists(generation_config_path):
                self.generation_config = GenerationConfig.from_pretrained(model_path)
            
            # Load memory optimization settings
            memory_config_path = os.path.join(model_path, 'memory_config.json')
            if os.path.exists(memory_config_path):
                with open(memory_config_path, 'r') as f:
                    memory_config = json.load(f)
                self.use_selective_activation_cache = memory_config['use_selective_activation_cache']
                self.teacher_forcing_ratio = memory_config['teacher_forcing_ratio']

    def print_trainer_type(self):
        print("I am an AcceleratedNLPSeq2SeqTrainer!")
