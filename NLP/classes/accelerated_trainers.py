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

from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm
from configs.default_config import DEFAULT_SPECS
from .config_manager import ConfigManager
from .utils import Logger, ErrorHandler

# Set up logging
logging.basicConfig(level=logging.INFO)

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
    A trainer class for NLP models using the Accelerate library.
    
    Attributes:
        model: The model to be trained.
        optimizer: The optimizer for training.
        data_collator: The data collator for batching.
        datasets: A dictionary containing training and evaluation datasets.
        raw_dataset: The raw dataset for processing.
        tokenizer: The tokenizer for text processing.
        specs: Configuration specifications for training.
        task_type: The type of task (e.g., classification, generation).
        losses: A list to store training losses.
        metric: The evaluation metric.
    """
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, raw_dataset,
                task_type, optimizer, chosen_metric):
        """
        Initialize the AcceleratedNLPTrainer.
        
        Args:
            args_dir (str): Directory containing configuration arguments.
            model: The model to be trained.
            tokenizer: The tokenizer for text processing.
            data_collator: The data collator for batching.
            train_dataset: The training dataset.
            eval_dataset: The evaluation dataset.
            raw_dataset: The raw dataset for processing.
            task_type (str): The type of task (e.g., classification, generation).
            optimizer: The optimizer for training.
            chosen_metric (str): The metric for evaluation.
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
            
            self.optimizer = self._setup_optimizer(optimizer)
            
            self.prepare_with_accelerator(train_dataset, eval_dataset)
            self.prepare_scheduler()

            self.progress_bar = tqdm(range(self.num_training_steps))
            self.task_type = task_type
            self.losses = []
            self.metric = evaluate.load(chosen_metric)
            
            self._setup_gradient_accumulation()
            
            Logger.info("AcceleratedNLPTrainer initialized successfully.")
        except Exception as e:
            ErrorHandler.handle_error(e, "AcceleratedNLPTrainer initialization")

    def setup_configuration(self):
        """
        Set up configuration using ConfigManager.
        """
        config_manager = ConfigManager()
        self.specs = vars(config_manager.parse_args())

    def _check_is_peft_model(self, model):
        """Check if the model is a PEFT model and get its type"""
        try:
            from peft import PeftModel
            is_peft = isinstance(model, PeftModel)
            if is_peft:
                peft_type = getattr(model.peft_config, "peft_type", "unknown")
                Logger.info(f"Detected PEFT model of type: {peft_type}")
                self._log_peft_params(model)
            return is_peft
        except ImportError:
            Logger.info("PEFT not installed, continuing with standard training")
            return False

    def _log_peft_params(self, model):
        """Log PEFT-specific parameter information"""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        Logger.info(f"PEFT model has {trainable_params} trainable parameters out of {total_params} total parameters")
        
        # Log adapter-specific information
        if hasattr(model, "peft_config"):
            config = model.peft_config
            if hasattr(config, "r"):  # LoRA
                Logger.info(f"LoRA rank: {config.r}")
            elif hasattr(config, "num_virtual_tokens"):  # Prefix Tuning
                Logger.info(f"Number of prefix tokens: {config.num_virtual_tokens}")

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

    def _setup_optimizer(self, optimizer):
        """Setup optimizer with PEFT-specific configurations"""
        if optimizer is not None:
            return optimizer
            
        if self.is_peft_model:
            # Only optimize trainable parameters for PEFT
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.specs.get('learning_rate', 2e-5),
                weight_decay=self.specs.get('weight_decay', 0.01),
            )
            Logger.info(f"Created PEFT-aware optimizer with {len(trainable_params)} trainable parameters")
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.specs.get('learning_rate', 2e-5),
                weight_decay=self.specs.get('weight_decay', 0.01),
            )
        return optimizer

    def prepare_with_accelerator(self, train_dataset, eval_dataset):
        """Prepare with Accelerator, including PEFT and mixed precision support"""
        try:
            # Configure mixed precision based on PEFT and model type
            mixed_precision = 'fp16' if self.specs.get('fp16', True) else 'no'
            if self.is_peft_model and hasattr(self.model, 'quantization_config'):
                if self.model.quantization_config.get('quantization_type') in ['4bit', '8bit']:
                    Logger.info("Detected quantized PEFT model, adjusting mixed precision settings")
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
                
            Logger.info(f"Model prepared with Accelerator using {mixed_precision} precision")
        except Exception as e:
            Logger.error(f"Error in Accelerator preparation: {e}")
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
        #Prepare scheduler specs
        self.scheduler_strategy = self.specs['scheduler_strategy']
        self.num_train_epochs = self.specs['num_train_epochs']
        self.num_warmup_steps = self.specs.get('num_warmup_steps', 0)
        
        self.calculate_training_steps()

        self.lr_scheduler = get_scheduler(
            self.scheduler_strategy,
            optimizer=self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
    
    def calculate_training_steps(self):
        """
        Calculate the number of training steps.
        """
        num_update_steps_per_epoch = len(self.train_dataloader)
        self.num_training_steps = self.num_train_epochs * num_update_steps_per_epoch

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
                Logger.info(f"Saving PEFT adapter state to {self.specs['output_dir']}")
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
                Logger.error(f"Error saving PEFT model: {e}")
        
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

    def train(self):
        """
        Train the model with mixed precision and gradient accumulation.
        """
        for epoch in range(self.num_train_epochs):
            logging.info(f"Starting epoch {epoch}")

            self.model.train()
            for step, batch in enumerate(self.train_dataloader):
                self._train_step(batch, step)
            
            self.model.eval()
            for step, batch in enumerate(self.eval_dataloader):
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
        print("I am an AcceleratedNLPTrainer!")


class AcceleratedNLPSeq2SeqTrainer(AcceleratedNLPTrainer):
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, raw_dataset,
                task_type, optimizer, chosen_metric):
        super().__init__(args_dir, model, tokenizer, 
                        data_collator, train_dataset, eval_dataset, raw_dataset,
                        task_type, optimizer, chosen_metric)
        
        # Additional Seq2Seq specific setup
        self.setup_generation_config()
        
    def setup_generation_config(self):
        """Setup generation configuration with PEFT awareness"""
        from transformers import GenerationConfig
        
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
                
    def _train_step(self, batch, step):
        """Specialized training step for Seq2Seq models with PEFT support"""
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                outputs = self.model(**batch)
                loss = outputs.loss
                
                if self.specs.get('gradient_accumulation_steps', 1) > 1:
                    loss = loss / self.specs['gradient_accumulation_steps']
                
            self.accelerator.backward(loss)
            
            if self.is_peft_model:
                # Memory optimization for PEFT sequence models
                if hasattr(self.model, "active_adapter"):
                    # Free memory of inactive adapters during backward pass
                    current_adapter = self.model.active_adapter
                    for adapter in self.model.peft_config:
                        if adapter != current_adapter:
                            self.model.disable_adapter_layers(adapter)
                
                # Apply gradient clipping based on PEFT type
                clip_norm = self.specs.get('max_grad_norm', 1.0)
                if "PREFIX" in self.peft_config.peft_type:
                    clip_norm = clip_norm / 2  # More conservative for prefix tuning
                
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    clip_norm
                )
            
            if not self.accelerator.step_was_skipped:
                self.optimizer.step()
                self.lr_scheduler.step()
            
            self.optimizer.zero_grad()
            
        return loss.item()
        
    def generate(self, batch):
        """Generate sequences with PEFT-aware handling"""
        if self.is_peft_model:
            # Special handling for PEFT models during generation
            try:
                generation_kwargs = self.generation_config.to_dict()
                
                # Adjust settings based on PEFT type
                if self.peft_config.peft_type == "LORA":
                    # LoRA-specific generation optimizations
                    generation_kwargs["max_new_tokens"] = min(
                        generation_kwargs.get("max_length", 128),
                        self.specs.get("max_new_tokens", 64)
                    )
                elif "PREFIX" in self.peft_config.peft_type:
                    # Prefix tuning specific adjustments
                    generation_kwargs["length_penalty"] = 0.8
                    
                return self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **generation_kwargs
                )
            except Exception as e:
                Logger.warning(f"Error in PEFT generation, falling back to default: {e}")
                
        # Default generation for non-PEFT models
        return self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=self.specs.get("max_length", 128)
        )
        
    def prepare_with_accelerator(self, train_dataset, eval_dataset):
        """Enhanced accelerator preparation for Seq2Seq PEFT models"""
        super().prepare_with_accelerator(train_dataset, eval_dataset)
        
        if self.is_peft_model:
            # Additional memory optimizations for Seq2Seq PEFT models
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                Logger.info("Enabled gradient checkpointing for PEFT Seq2Seq model")
                
            # Configure model for efficient generation
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False  # Better memory efficiency
                
            # Set up memory efficient attention if available
            try:
                from transformers.utils import is_flash_attn_available
                if is_flash_attn_available():
                    self.model.enable_flash_attention()
                    Logger.info("Enabled flash attention for PEFT Seq2Seq model")
            except ImportError:
                pass

    def postprocess(self, predictions, labels):
        """Seq2Seq specific postprocessing of predictions and labels"""
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

    def handle_predictions_and_metric(self, outputs, batch, metric):
        """Seq2Seq specific prediction and metric handling"""
        predictions = self.generate(batch)
        labels = batch["labels"]
        
        # Handle distributed training
        predictions = self.accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
        predictions = self.accelerator.gather(predictions)
        labels = self.accelerator.gather(labels)
        
        # Process predictions and labels
        true_predictions, true_labels = self.postprocess(predictions, labels)
        metric.add_batch(predictions=true_predictions, references=true_labels)

    def handle_outputs(self, outputs, batch, batch_size, losses, metric):
        """Seq2Seq specific output handling"""
        loss = outputs.loss
        self.accelerator.log({"loss": loss.item()})
        losses.append(self.accelerator.gather(loss.repeat(batch_size)))
        self.handle_predictions_and_metric(outputs, batch, metric)

    def print_trainer_type(self):
        print("I am an AcceleratedNLPSeq2SeqTrainer!")
