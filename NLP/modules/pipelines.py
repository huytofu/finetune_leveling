# Standard library imports
import collections
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any

# Third-party imports
import evaluate
import mlflow
import nltk
import numpy as np
import psutil
import torch
from torch.optim import AdamW
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    DataCollatorForTokenClassification,
    default_data_collator,
    pipeline
)
import time
import mlflow
from datetime import datetime

# Local imports
currdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currdir)
sys.path.append(currdir)
sys.path.append(parentdir)

from adapter_manager import MultiAdapterManager
from classes.accelerated_trainers import (
    AcceleratedNLPTrainer,
    AcceleratedNLPSeq2SeqTrainer
)
from classes.accelerated_trainers_with_lightning import (
    AcceleratedNLPTrainer as AcceleratedNLPTrainerWithLightning,
    AcceleratedNLPSeq2SeqTrainer as AcceleratedNLPSeq2SeqTrainerWithLightning
)
from classes.checkpoint_manager import CheckpointManager
from classes.peft_callbacks import (
    PeftAdapterMonitorCallback,
    PeftEarlyPruningCallback
)
from classes.quantization_manager import QuantizationManager
from classes.trainers import (
    NLPTrainer,
    NLPSeq2SeqTrainer
)
from classes.trainers_with_lightning import (
    NLPTrainer as NLPTrainerWithLightning,
    NLPSeq2SeqTrainer as NLPSeq2SeqTrainerWithLightning
)
from classes.type_utils import TypeUtils
from configs.default_config import DEFAULT_SPECS
from dataset_modules import DatasetModules
from model_modules import ModelModules
from tokenizer_modules import TokenizerModules

class MLflowCallback:
    """Callback for tracking training metrics in MLflow."""
    
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.epoch_start_time = None
        self.step_start_time = None
        
    def on_train_begin(self, args, state, control):
        """Called when training begins."""
        self.epoch_start_time = time.time()
        mlflow.log_param("train_batch_size", args.per_device_train_batch_size)
        mlflow.log_param("eval_batch_size", args.per_device_eval_batch_size)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("num_train_epochs", args.num_train_epochs)
        mlflow.log_param("warmup_steps", args.warmup_steps)
        mlflow.log_param("weight_decay", args.weight_decay)
        mlflow.log_param("gradient_accumulation_steps", args.gradient_accumulation_steps)
        mlflow.log_param("max_grad_norm", args.max_grad_norm)
        
    def on_epoch_begin(self, args, state, control):
        """Called when an epoch begins."""
        self.epoch_start_time = time.time()
        mlflow.log_param(f"epoch_{state.epoch}", state.epoch)
        
    def on_epoch_end(self, args, state, control):
        """Called when an epoch ends."""
        epoch_duration = time.time() - self.epoch_start_time
        mlflow.log_metric(f"epoch_{state.epoch}_duration", epoch_duration)
        
        # Log resource usage at epoch end
        mlflow.log_metric(f"epoch_{state.epoch}_cpu_percent", psutil.cpu_percent())
        mlflow.log_metric(f"epoch_{state.epoch}_memory_percent", psutil.virtual_memory().percent)
        if torch.cuda.is_available():
            mlflow.log_metric(f"epoch_{state.epoch}_gpu_percent", torch.cuda.utilization())
            mlflow.log_metric(f"epoch_{state.epoch}_gpu_memory_percent", 
                            torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100)
        
    def on_step_begin(self, args, state, control):
        """Called when a training step begins."""
        self.step_start_time = time.time()
        
    def on_step_end(self, args, state, control):
        """Called when a training step ends."""
        if self.step_start_time:
            step_duration = time.time() - self.step_start_time
            mlflow.log_metric(f"step_{state.global_step}_duration", step_duration)
            
            # Log step metrics
            if state.log_history:
                for log in state.log_history:
                    if isinstance(log, dict):
                        for key, value in log.items():
                            if isinstance(value, (int, float)):
                                mlflow.log_metric(f"step_{state.global_step}_{key}", value)
        
    def on_evaluate(self, args, state, control, metrics=None):
        """Called after evaluation."""
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"eval_{key}", value)
                    
    def on_save(self, args, state, control):
        """Called when a checkpoint is saved."""
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(checkpoint_path):
            mlflow.log_artifact(checkpoint_path, f"checkpoint_{state.global_step}")

class InferencePipeLine():
    def __init__(self, task_type, checkpoint, adapter_config=None):
        """
        Initialize the inference pipeline.
        
        Args:
            task_type: Type of task (text_generation, summarization, etc.)
            checkpoint: Path to the model checkpoint
            adapter_config: Configuration for adapter support (optional)
                {
                    'cache_dir': str,          # Directory to cache adapters
                    'max_adapters': int,       # Maximum number of adapters in memory
                    'aws_credentials': dict,   # AWS credentials
                    'gcp_credentials': str,    # Path to GCP credentials file
                    'preload_adapters': list   # List of adapter configurations to preload
                }
        """
        self.task_type = task_type.replace("_","-")
        self.specs = {**DEFAULT_SPECS}
        
        # Create the base pipeline without adapters
        if task_type == "token_classification":
            self.loaded_pipeline = pipeline(self.task_type, model=checkpoint, aggregation_strategy="simple")     
        elif task_type == "masked_language_modeling":
            self.loaded_pipeline = pipeline(self.task_type, model=checkpoint)
        elif task_type == "translation":
            self.loaded_pipeline = pipeline(self.task_type, model=checkpoint)
        elif task_type == "summarization":
            self.loaded_pipeline = pipeline(self.task_type, model=checkpoint)
        elif task_type == "question_answering":
            self.loaded_pipeline = pipeline(self.task_type, model=checkpoint)
        elif task_type == "text_generation":
            self.loaded_pipeline = pipeline(self.task_type, model=checkpoint)
        else: 
            raise ValueError(f"Unsupported task type: {task_type}")
        
        # Initialize adapter support if configured
        self.adapter_manager = None
        self.has_adapter_support = False
        
        if adapter_config is not None:
            self._initialize_adapter_support(checkpoint, adapter_config)
    
    def _initialize_adapter_support(self, checkpoint, adapter_config):
        """Initialize adapter support with the given configuration"""
        # Extract configuration
        cache_dir = adapter_config.get('cache_dir')
        max_adapters = adapter_config.get('max_adapters', 5)
        aws_credentials = adapter_config.get('aws_credentials')
        gcp_credentials = adapter_config.get('gcp_credentials')
        preload_adapters = adapter_config.get('preload_adapters', [])
        
        # Initialize MultiAdapterManager
        self.adapter_manager = MultiAdapterManager(
            base_model=self.loaded_pipeline.model,
            cache_dir=cache_dir,
            max_adapters_in_memory=max_adapters,
            aws_credentials=aws_credentials,
            gcp_credentials=gcp_credentials
        )
        
        # Preload adapters if specified
        for adapter_conf in preload_adapters:
            source = adapter_conf.get('source', 'local')
            adapter_id = adapter_conf.get('adapter_id')
            
            if not adapter_id:
                continue
                
            try:
                if source == 'local':
                    path = adapter_conf.get('path')
                    if path:
                        self.adapter_manager.register_adapter_from_local(
                            adapter_id=adapter_id,
                            adapter_path=path,
                            adapter_name=adapter_conf.get('adapter_name'),
                            adapter_type=adapter_conf.get('adapter_type', 'lora'),
                            metadata=adapter_conf.get('metadata')
                        )
                
                elif source == 'huggingface':
                    repo_id = adapter_conf.get('repo_id')
                    if repo_id:
                        self.adapter_manager.register_adapter_from_huggingface(
                            adapter_id=adapter_id,
                            repo_id=repo_id,
                            adapter_name=adapter_conf.get('adapter_name'),
                            adapter_type=adapter_conf.get('adapter_type', 'lora'),
                            metadata=adapter_conf.get('metadata'),
                            revision=adapter_conf.get('revision', 'main'),
                            use_auth_token=adapter_conf.get('use_auth_token')
                        )
                
                elif source == 'aws':
                    bucket = adapter_conf.get('bucket')
                    prefix = adapter_conf.get('prefix')
                    if bucket and prefix:
                        self.adapter_manager.register_adapter_from_aws(
                            adapter_id=adapter_id,
                            bucket=bucket,
                            prefix=prefix,
                            adapter_name=adapter_conf.get('adapter_name'),
                            adapter_type=adapter_conf.get('adapter_type', 'lora'),
                            metadata=adapter_conf.get('metadata')
                        )
                
                elif source == 'gcp':
                    bucket = adapter_conf.get('bucket')
                    prefix = adapter_conf.get('prefix')
                    if bucket and prefix:
                        self.adapter_manager.register_adapter_from_gcp(
                            adapter_id=adapter_id,
                            bucket=bucket,
                            prefix=prefix,
                            adapter_name=adapter_conf.get('adapter_name'),
                            adapter_type=adapter_conf.get('adapter_type', 'lora'),
                            metadata=adapter_conf.get('metadata')
                        )
            except Exception as e:
                logging.error(f"Failed to preload adapter {adapter_id}: {str(e)}")
        
        self.has_adapter_support = True
    
    def list_adapters(self):
        """
        List all registered adapters
        
        Returns:
            list: Information about registered adapters
        """
        if not self.has_adapter_support:
            return []
        
        adapter_list = self.adapter_manager.list_adapters()
        return [
            {
                'id': a.adapter_id,
                'name': a.adapter_name,
                'type': a.adapter_type,
                'source': a.source,
                'is_loaded': a.is_loaded
            }
            for a in adapter_list
        ]
    
    def run(self, input_text, adapter_id=None, auto_adapter=False, select_adapter_with_llm=False, **kwargs):
        """
        Run inference with the model
        
        Args:
            input_text: Input text for the model
            adapter_id: Optional adapter ID to use for inference
            auto_adapter: If True, let the system decide which adapter to use
            select_adapter_with_llm: If True, use LLM to decide the best adapter
            **kwargs: Additional arguments for the model
            
        Returns:
            Model output
        """
        # Determine adapter if auto_adapter is enabled
        if auto_adapter:
            if select_adapter_with_llm:
                adapter_id = self._select_best_adapter_with_llm(input_text)
            else:
                adapter_id = self._select_best_adapter(input_text)
            print(f"System selected adapter: {adapter_id}")

        # Use adapter if specified and adapter support is enabled
        if adapter_id and self.has_adapter_support:
            try:
                # Load the specified adapter
                adapter_model = self.adapter_manager.load_adapter(adapter_id)
                
                # Create a temporary pipeline with the adapter model
                temp_pipeline = None
                
                if self.task_type == "translation":
                    temp_pipeline = pipeline(self.task_type, model=adapter_model)
                    result = temp_pipeline(input_text, max_length=kwargs.get("max_length", self.specs["max_length"]), return_text=True)[0]["translation_text"]
                
                elif self.task_type == "summarization":
                    temp_pipeline = pipeline(self.task_type, model=adapter_model)
                    result = temp_pipeline(input_text, max_length=kwargs.get("max_length", self.specs["max_length"]), return_text=True)[0]["summary_text"]
                
                elif self.task_type == "question-answering":
                    temp_pipeline = pipeline(self.task_type, model=adapter_model)
                    result = temp_pipeline(input_text, max_length=kwargs.get("max_length", self.specs["max_length"]), return_text=True)[0]["answer"]
                
                elif self.task_type == "text-generation":
                    temp_pipeline = pipeline(self.task_type, model=adapter_model)
                    result = temp_pipeline(
                        input_text, 
                        max_length=kwargs.get("max_length", self.specs["max_length"]), 
                        do_sample=kwargs.get("do_sample", True),
                        temperature=kwargs.get("temperature", 0.7),
                        top_p=kwargs.get("top_p", 0.9),
                        top_k=kwargs.get("top_k", 50),
                        num_return_sequences=kwargs.get("num_return_sequences", 1),
                        return_text=True
                    )[0]["generated_text"]
                
                else:
                    temp_pipeline = pipeline(self.task_type, model=adapter_model)
                    result = temp_pipeline(input_text)
                
                return result
            
            except Exception as e:
                logging.error(f"Error using adapter {adapter_id}: {str(e)}")
                logging.info(f"Falling back to base model")
                # Fall back to base model
        
        # Use base model
        if self.task_type == "translation":
            return self.loaded_pipeline(input_text, max_length=kwargs.get("max_length", self.specs["max_length"]), return_text=True)[0]["translation_text"]
        elif self.task_type == "summarization":
            return self.loaded_pipeline(input_text, max_length=kwargs.get("max_length", self.specs["max_length"]), return_text=True)[0]["summary_text"]
        elif self.task_type == "question-answering":
            return self.loaded_pipeline(input_text, max_length=kwargs.get("max_length", self.specs["max_length"]), return_text=True)[0]["answer"]
        elif self.task_type == "text-generation":
            return self.loaded_pipeline(
                input_text, 
                max_length=kwargs.get("max_length", self.specs["max_length"]), 
                do_sample=kwargs.get("do_sample", True),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                top_k=kwargs.get("top_k", 50),
                num_return_sequences=kwargs.get("num_return_sequences", 1),
                return_text=True
            )[0]["generated_text"]
        return self.loaded_pipeline(input_text)

    def _select_best_adapter(self, input_text):
        """
        Select the most suitable adapter based on the input text or task type.
        """
        # Placeholder logic for selecting the best adapter
        # This can be replaced with more sophisticated logic based on task type, input text, etc.
        adapters = self.adapter_manager.list_adapters()
        if not adapters:
            return None
        
        # Example: Select the first available adapter
        return adapters[0]['id']

    def _select_best_adapter_with_llm(self, input_text):
        """
        Use the base model to decide the best adapter based on adapter names.
        """
        # Construct a prompt for the model
        prompt = """
        You are a smart AI model. Based on the following task and available adapters, choose the most suitable adapter:
        Task: {task_type}
        Available Adapters: {adapters}
        Choose the best adapter name:
        """.format(
            task_type=self.task_type,
            adapters=', '.join([a['name'] for a in self.adapter_manager.list_adapters()])
        )

        # Use the base model to generate a response
        response = self.loaded_pipeline(prompt, max_length=50, return_text=True)[0]['generated_text']

        # Extract the adapter name from the response
        for adapter in self.adapter_manager.list_adapters():
            if adapter['name'] in response:
                return adapter['id']

        # Fallback to the first adapter if no match is found
        return self.adapter_manager.list_adapters()[0]['id']

class FineTunePipeLine():
    def __init__(self, args_dir: str, specs: Dict[str, Any] = None):
        """Initialize the finetuning pipeline with MLflow tracking."""
        self.specs = {**DEFAULT_SPECS, **(specs or {})}
        self.args_dir = args_dir
        
        # Initialize MLflow tracking
        self.experiment_name = self.specs.get("experiment_name", "default_experiment")
        self.tracking_uri = self.specs.get("tracking_uri", "sqlite:///mlflow.db")
        
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        
        # Start MLflow run
        self.run = mlflow.start_run(run_name=f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.run_id = self.run.info.run_id
        
        # Log initial metadata
        mlflow.log_param("start_time", datetime.now().isoformat())
        mlflow.log_param("task_type", self.specs.get("task_type", "unknown"))
        mlflow.log_param("model_name", self.specs.get("model_name", "unknown"))
        mlflow.log_param("data_source", self.specs.get("data_source", "unknown"))
        
        # Initialize components
        self.model_modules = ModelModules(self.specs)
        self.tokenizer_modules = TokenizerModules(self.specs)
        self.dataset_modules = DatasetModules(self.specs.get("dataset_dir"), self.tokenizer_modules.tokenizer, self.specs)
        
        # Load configurations
        self._load_configs()
        
    def _load_configs(self):
        """Load configurations from args directory."""
        try:
            # Load training arguments
            with open(os.path.join(self.args_dir, "training_args.json"), "r") as f:
                self.training_args = json.load(f)
                
            # Load model arguments
            with open(os.path.join(self.args_dir, "model_args.json"), "r") as f:
                self.model_args = json.load(f)
                
            # Log configurations to MLflow
            mlflow.log_dict(self.training_args, "training_args.json")
            mlflow.log_dict(self.model_args, "model_args.json")
            
        except Exception as e:
            logging.error(f"Failed to load configurations: {e}")
            raise
            
    def prepare_optimizer(self):
        """Prepare optimizer with MLflow tracking."""
        try:
            optimizer = AdamW(
                self.model_modules.model.parameters(),
                lr=self.training_args.get("learning_rate", 2e-5),
                weight_decay=self.training_args.get("weight_decay", 0.01)
            )
            
            # Log optimizer configuration
            mlflow.log_param("optimizer_type", "AdamW")
            mlflow.log_param("learning_rate", self.training_args.get("learning_rate", 2e-5))
            mlflow.log_param("weight_decay", self.training_args.get("weight_decay", 0.01))
            
            return optimizer
            
        except Exception as e:
            logging.error(f"Failed to prepare optimizer: {e}")
            raise
            
    def prepare_data_collator(self):
        """Prepare data collator with MLflow tracking."""
        try:
            task_type = self.specs.get("task_type", "unknown")
            
            if task_type == "masked_language_modeling":
                collator = DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer_modules.tokenizer,
                    mlm=True,
                    mlm_probability=0.15
                )
            elif task_type == "token_classification":
                collator = DataCollatorForTokenClassification(
                    tokenizer=self.tokenizer_modules.tokenizer
                )
            elif task_type in ["translation", "summarization"]:
                collator = DataCollatorForSeq2Seq(
                    tokenizer=self.tokenizer_modules.tokenizer
                )
            else:
                collator = default_data_collator
                
            # Log collator configuration
            mlflow.log_param("data_collator_type", collator.__class__.__name__)
            
            return collator
            
        except Exception as e:
            logging.error(f"Failed to prepare data collator: {e}")
            raise
            
    def run(self):
        """Run the finetuning pipeline with MLflow tracking."""
        try:
            start_time = time.time()
            
            # Prepare model
            logging.info("Preparing model...")
            self.model_modules.prepare_model()
            
            # Prepare optimizer
            logging.info("Preparing optimizer...")
            optimizer = self.prepare_optimizer()
            
            # Prepare data collator
            logging.info("Preparing data collator...")
            data_collator = self.prepare_data_collator()
            
            # Prepare dataset
            logging.info("Preparing dataset...")
            dataset = self.dataset_modules.prepare_dataset_from_dir(self.specs.get("task_type"))
            
            # Initialize trainer with MLflow callback
            logging.info("Initializing trainer...")
            trainer_class = self._get_trainer_class()
            trainer = trainer_class(
                model=self.model_modules.model,
                args=self.training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["eval"],
                data_collator=data_collator,
                optimizers=(optimizer, None),
                callbacks=[MLflowCallback(self.run_id)]
            )
            
            # Train the model
            logging.info("Starting training...")
            trainer.train()
            
            # Save the model
            logging.info("Saving model...")
            trainer.save_model()
            
            # Log final metrics
            total_duration = time.time() - start_time
            mlflow.log_metric("total_training_duration", total_duration)
            mlflow.log_param("end_time", datetime.now().isoformat())
            
            # Log model artifacts
            model_path = os.path.join(self.training_args.get("output_dir", "output"), "final_model")
            if os.path.exists(model_path):
                mlflow.log_artifact(model_path, "final_model")
                
            # End MLflow run
            mlflow.end_run()
            
        except Exception as e:
            logging.error(f"Error in pipeline execution: {e}")
            mlflow.end_run(status="FAILED")
            raise
            
    def _get_trainer_class(self):
        """Get appropriate trainer class based on task type."""
        task_type = self.specs.get("task_type", "unknown")
        use_lightning = self.specs.get("use_lightning", False)
        
        if task_type in ["translation", "summarization"]:
            if use_lightning:
                return AcceleratedNLPSeq2SeqTrainerWithLightning
            return AcceleratedNLPSeq2SeqTrainer
        else:
            if use_lightning:
                return AcceleratedNLPTrainerWithLightning
            return AcceleratedNLPTrainer

    def _prepare_fallback_optimizer(self, model):
        """
        Prepare a fallback optimizer in case the trainer doesn't have its own.
        
        Args:
            model: The model to optimize
        
        Returns:
            The prepared optimizer
        """
        return AdamW(
            model.parameters(),
            lr=self.specs['learning_rate'],
            weight_decay=self.specs['weight_decay']
        )

    def _prepare_fallback_data_collator(self, tokenizer, model=None):
        """
        Prepare a fallback data collator in case the trainer doesn't have its own.
        
        Args:
            tokenizer: The tokenizer to use
            model: The model (required for some collators)
        
        Returns:
            The prepared data collator
        """
        if self.task_type == "masked_language_modeling":
            return DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm_probability=self.specs['mlm_probability']
            )
        elif self.task_type == "token_classification":
            return DataCollatorForTokenClassification(tokenizer=tokenizer)
        elif self.task_type in ["translation", "summarization", "text_generation"]:
            return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        elif self.task_type == "question_answering":
            return default_data_collator
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def load_specs(self):
        """Load specifications from the arguments file."""
        import json
        from configs.default_config import DEFAULT_SPECS
        try:
            specs = json.load(open(self.args_dir, 'r'))
            self.specs = {**DEFAULT_SPECS, **specs}
        except Exception as e:
            print(f"Error loading specs: {e}")
            self.specs = DEFAULT_SPECS

    def _get_trainer_class(self):
        """Get the appropriate trainer class based on configuration."""
        if self.use_lightning:
            if self.specs.get("task_type") == "seq2seq":
                return AcceleratedNLPSeq2SeqTrainerWithLightning
            return AcceleratedNLPTrainerWithLightning
        elif self.use_accelerate:
            if self.specs.get("task_type") == "seq2seq":
                return AcceleratedNLPSeq2SeqTrainer
            return AcceleratedNLPTrainer
        else:
            if self.specs.get("task_type") == "seq2seq":
                return NLPSeq2SeqTrainer
            return NLPTrainer

    def run(self):
        """Run the fine-tuning pipeline."""
        # Load model and tokenizer with appropriate quantization
        model_kwargs = {}
        if self.quantization:
            model_kwargs = self.quantization_manager.prepare_model_for_quantization(
                model_name_or_path=self.specs.get("model_name_or_path", ""),
                quant_type=self.quantization,
                custom_quantization_config=self.specs.get("quantization_config", None)
            )
            
        model = self.model_modules.load_model(self.task_type, model_kwargs)
        tokenizer = self.tokenizer_modules.load_tokenizer()
        
        # Optimize memory layout
        model = self.type_utils.optimize_memory_layout(model, self.peft_method)
        
        # Apply PEFT if specified
        if self.peft_method:
            # Check compatibility with quantization
            if self.quantization:
                is_compatible, reason = self.quantization_manager.check_peft_compatible(
                    model, self.peft_method
                )
                if not is_compatible:
                    print(f"Warning: {reason}")
                
                # Optimize for PEFT with quantization
                model = self.quantization_manager.optimize_model_for_peft(
                    model, self.quantization, self.peft_method
                )
                
            model, tokenizer = self.model_modules.apply_peft(
                model=model,
                tokenizer=tokenizer,
                peft_method=self.peft_method,
                peft_config=self.peft_config,
                task_type=self.task_type
            )
            
            # Validate parameter types
            self.type_utils.check_parameter_types(model)
        
        # Load dataset
        raw_dataset = self.dataset_modules.load_dataset()
        
        # Preprocess dataset based on task type
        if self.task_type == "masked_language_modeling":
            train_dataset, eval_dataset = self.dataset_modules.prepare_mlm(raw_dataset, tokenizer, self.specs)
        elif self.task_type == "token_classification":
            train_dataset, eval_dataset = self.dataset_modules.prepare_token_classification(raw_dataset, tokenizer, self.specs)
        elif self.task_type == "translation":
            train_dataset, eval_dataset = self.dataset_modules.prepare_translation(raw_dataset, tokenizer, self.specs)
        elif self.task_type == "summarization":
            train_dataset, eval_dataset = self.dataset_modules.prepare_summarization(raw_dataset, tokenizer, self.specs)
        elif self.task_type == "text_generation":
            train_dataset, eval_dataset = self.dataset_modules.prepare_text_generation(raw_dataset, tokenizer, self.specs)
        elif self.task_type == "question_answering":
            train_dataset, eval_dataset = self.dataset_modules.prepare_question_answering(raw_dataset, tokenizer, self.specs)
        else:
            train_dataset, eval_dataset = None, None
        
        # Setup additional callbacks for Lightning
        callbacks = []
        if self.use_lightning and self.peft_method:
            # Add PEFT-specific callbacks
            peft_monitor = PeftAdapterMonitorCallback(
                monitor_gradients=True,
                monitor_weights=True,
                log_every_n_steps=100,
                save_path=os.path.join(self.specs["output_dir"], "peft_monitoring")
            )
            callbacks.append(peft_monitor)
            
            # Only add pruning if enabled in specs
            if self.specs.get("peft_pruning_enabled", False):
                peft_pruning = PeftEarlyPruningCallback(
                    start_pruning_epoch=1,
                    pruning_threshold=self.specs.get("peft_pruning_threshold", 0.01),
                    final_sparsity=self.specs.get("peft_pruning_sparsity", 0.3)
                )
                callbacks.append(peft_pruning)
        
        # Prepare fallback optimizer and data collator
        fallback_optimizer = self._prepare_fallback_optimizer(model)
        fallback_data_collator = self._prepare_fallback_data_collator(tokenizer, model)
        
        # Initialize the trainer
        trainer_kwargs = {
            "args_dir": self.args_dir,
            "model": model,
            "tokenizer": tokenizer,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "raw_dataset": raw_dataset,
            "task_type": self.task_type,
            "chosen_metric": self.chosen_metric,
            "specs": self.specs,
            "fallback_optimizer": fallback_optimizer,
            "fallback_data_collator": fallback_data_collator
        }
        
        # Add callbacks for Lightning trainers
        if self.use_lightning and callbacks:
            trainer_kwargs["callbacks"] = callbacks
        
        trainer = self.trainer_class(**trainer_kwargs)
        
        # Run the training process
        if self.use_lightning:
            trainer.fit()
        else:
            trainer.train()
        
        # Save model and checkpoint
        self.save_model_and_checkpoint(model, tokenizer, trainer)
        
        return model, tokenizer

    def save_model_and_checkpoint(self, model, tokenizer, trainer):
        """Save the model, tokenizer, and create a checkpoint."""
        output_dir = self.specs["output_dir"]
        
        # For PEFT models, use different saving approach
        if self.peft_method:
            # Save using the checkpoint manager
            self.checkpoint_manager.save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                epoch=self.specs.get("num_train_epochs", 3),
                is_best=True
            )
        else:
            # For regular models, we can use the trainer to save
            if hasattr(trainer, "save_model"):
                trainer.save_model(output_dir)
            elif hasattr(model, "save_pretrained"):
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
            else:
                # Fallback for other model types
                torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                tokenizer.save_pretrained(output_dir)
    
    def load_checkpoint(self, checkpoint_dir, convert_precision=None):
        """
        Load a checkpoint using the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory containing the checkpoint
            convert_precision: Optional precision to convert to (fp16, bf16, fp32)
            
        Returns:
            Tuple of (loaded_model, loaded_tokenizer)
        """
        # Load specs for model initialization
        self.load_specs()
        
        # Load base model
        model = self.model_modules.load_model(self.task_type)
        tokenizer = self.tokenizer_modules.load_tokenizer()
        
        # Load checkpoint
        model, tokenizer, meta = self.checkpoint_manager.load_checkpoint(
            model=model,
            tokenizer=tokenizer,
            checkpoint_dir=checkpoint_dir,
            target_precision=convert_precision
        )
        
        return model, tokenizer, meta

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
    
    def prepare_dataset(self, dataset=None, huggingface_dataset=None):
        dataset_modules = DatasetModules(self.dataset_dir, self.tokenizer, self.specs)
        
        if dataset is not None:
            self.raw_dataset = dataset
            self.dataset = dataset_modules.prepare_dataset(self.raw_dataset)
        elif self.dataset_dir is not None:
            self.raw_dataset = dataset_modules.prepare_dataset_from_dir(self.task_type)
            self.dataset = dataset_modules.prepare_dataset(self.raw_dataset)
        else:
            self.raw_dataset = dataset_modules.load_dataset(huggingface_dataset)
            self.dataset = dataset_modules.prepare_dataset(self.raw_dataset)

    def get_compute_metrics(self, task_type, chosen_metric):
        self.metric = evaluate.load_metric(chosen_metric)
        if task_type == "token_classification":
            def compute_metrics(eval_pred):
                # TO ADD LATER
                pass
        elif task_type == "masked_language_modeling":
            def compute_metrics(eval_pred):
                # TO ADD LATER
                pass
        elif task_type == "translation":
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                true_labels, true_predictions = self.postprocess(predictions, labels)
                return self.metric.compute(predictions=true_predictions, references=true_labels)
        elif task_type == "summarization":
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                true_labels, true_predictions = self.postprocess(predictions, labels)
                if chosen_metric == "rouge":
                    true_predictions = ["\n".join(nltk.sent_tokenize(pred)) for pred in true_predictions]
                    true_labels = ["\n".join(nltk.sent_tokenize(label)) for label in true_labels]
                    result = self.metric.compute(predictions=true_predictions, references=true_labels, use_stemmer=True)
                    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
                    return result
                else:
                    return self.metric.compute(predictions=true_predictions, references=true_labels)
        elif task_type == "text_generation":
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                true_labels, true_predictions = self.postprocess(predictions, labels)
                return self.metric.compute(predictions=true_predictions, references=true_labels)
        return compute_metrics

    def compute_qna_metrics(self, start_logits, end_logits, eval_dataset, raw_eval_dataset, chosen_metric, max_answer_length):
        self.metric = evaluate.load_metric(chosen_metric)
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(eval_dataset):
            example_to_features[feature["example_id"]].append(idx)

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





    