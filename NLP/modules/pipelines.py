# Standard library imports
import collections
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union

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
    pipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments
)
import time
import mlflow
from datetime import datetime
from dataclasses import dataclass, field

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
from distributed_manager import DistributedManager, DistributedConfig, DistributedBackend
from .error_handler import ErrorHandler
from .training_optimizations import OptimizationConfig, TrainingOptimizer
from .distributed_training import DistributedTrainer
from .curriculum_learning import CurriculumConfig, CurriculumManager
from .few_shot_adaptation import FewShotConfig, FewShotAdapter
from .distillation import DistillationConfig, DistillationManager
from .mlflow_tracking import MLflowTracker

# Set up logging
logger = logging.getLogger(__name__)

class MLflowCallback:
    """
    Callback for tracking training metrics in MLflow.
    
    This callback integrates with the Hugging Face Transformers trainer to track
    metrics, parameters, and resource usage throughout the training process.
    """
    
    def __init__(self, run_id: str):
        """
        Initialize the MLflow callback.
        
        Args:
            run_id (str): The MLflow run ID to log metrics to
        """
        self.run_id = run_id
        self.epoch_start_time = None
        self.step_start_time = None
        
    def on_train_begin(self, args, state, control):
        """
        Called when training begins.
        
        Logs initial training parameters to MLflow.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control
        """
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

class FineTuneConfig:
    """Configuration for fine-tuning pipeline.
    
    Attributes:
        task_type: Type of fine-tuning task
        model_name: Name or path of the model
        optimization: Training optimization configuration
        distributed: Distributed training configuration
        curriculum: Curriculum learning configuration
        few_shot: Few-shot adaptation configuration
        distillation: Knowledge distillation configuration
        mlflow_tracking: Whether to track metrics with MLflow
    """
    task_type: str
    model_name: str
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    distributed: Optional[DistributedConfig] = None
    curriculum: Optional[CurriculumConfig] = None
    few_shot: Optional[FewShotConfig] = None
    distillation: Optional[DistillationConfig] = None
    mlflow_tracking: bool = True

class FineTunePipeline:
    """Advanced fine-tuning pipeline with optimizations."""
    
    def __init__(
        self,
        config: FineTuneConfig,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        try:
            self.config = config
            self.model = model
            self.tokenizer = tokenizer
            
            # Validate configuration
            self._validate_config()
            
            # Initialize MLflow tracking
            self.mlflow_tracker = None
            if config.mlflow_tracking:
                try:
                    self.mlflow_tracker = MLflowTracker("fine_tuning")
                except Exception as e:
                    logger.warning(f"Failed to initialize MLflow tracking: {str(e)}")
            
            # Get appropriate trainer class
            self.trainer_class = self._get_trainer_class()
            
            # Initialize components with error handling
            self._initialize_components()
            
        except Exception as e:
            logger.error(f"Error initializing FineTunePipeline: {str(e)}")
            raise

    def _validate_config(self):
        """Validate the configuration settings."""
        if not self.config.task_type:
            raise ValueError("task_type must be specified in config")
        if not self.config.model_name:
            raise ValueError("model_name must be specified in config")
            
        # Validate task type
        valid_task_types = [
            "classification", "token-classification", "summarization",
            "translation", "question-answering", "language-modeling"
        ]
        if self.config.task_type not in valid_task_types:
            raise ValueError(f"Invalid task_type: {self.config.task_type}. Must be one of {valid_task_types}")
            
        # Validate optimization settings
        if self.config.optimization:
            if self.config.optimization.use_lightning and not self._is_lightning_available():
                logger.warning("PyTorch Lightning not available, falling back to standard training")
                self.config.optimization.use_lightning = False
                
            if self.config.optimization.use_accelerate and not self._is_accelerate_available():
                logger.warning("Accelerate not available, falling back to standard training")
                self.config.optimization.use_accelerate = False

    def _initialize_components(self):
        """Initialize all pipeline components with error handling."""
        try:
            # Initialize optimization components
            self.training_optimizer = TrainingOptimizer(
                self.model,
                self.config.optimization
            )
            
            # Initialize distributed training if configured
            self.distributed_trainer = None
            if self.config.distributed and torch.cuda.device_count() > 1:
                try:
                    self.distributed_trainer = DistributedTrainer(
                        self.model,
                        self.config.distributed,
                        self.config.optimization
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize distributed training: {str(e)}")
                    
            # Initialize curriculum learning if configured
            self.curriculum_manager = None
            if self.config.curriculum:
                try:
                    self.curriculum_manager = CurriculumManager(
                        self.model,
                        self.tokenizer,
                        self.config.curriculum
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize curriculum learning: {str(e)}")
                    
            # Initialize few-shot adaptation if configured
            self.few_shot_adapter = None
            if self.config.few_shot:
                try:
                    self.few_shot_adapter = FewShotAdapter(
                        self.model,
                        self.tokenizer,
                        self.config.few_shot
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize few-shot adaptation: {str(e)}")
                    
            # Initialize knowledge distillation if configured
            self.distillation_manager = None
            if self.config.distillation:
                try:
                    self.distillation_manager = DistillationManager(
                        self.model,
                        None,
                        self.tokenizer,
                        self.config.distillation
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize knowledge distillation: {str(e)}")
            
            # Initialize checkpoint manager
            try:
                self.checkpoint_manager = CheckpointManager(
                    base_path=os.path.join("checkpoints", self.config.task_type),
                    model_name=self.config.model_name
                )
            except Exception as e:
                logger.error(f"Failed to initialize checkpoint manager: {str(e)}")
                raise
            
            # Initialize metrics
            self._initialize_metrics()
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise

    def _initialize_metrics(self):
        """Initialize evaluation metrics based on task type."""
        try:
            self.metric = None
            if self.config.task_type in ["classification", "token-classification"]:
                self.metric = evaluate.load("accuracy")
            elif self.config.task_type in ["summarization", "translation"]:
                self.metric = evaluate.load("rouge")
            elif self.config.task_type == "question-answering":
                self.metric = evaluate.load("squad")
        except Exception as e:
            logger.warning(f"Failed to initialize metrics: {str(e)}")

    @staticmethod
    def _is_lightning_available():
        """Check if PyTorch Lightning is available."""
        try:
            import pytorch_lightning
            return True
        except ImportError:
            return False

    @staticmethod
    def _is_accelerate_available():
        """Check if Accelerate is available."""
        try:
            import accelerate
            return True
        except ImportError:
            return False

    def _get_trainer_class(self):
        """Get the appropriate trainer class based on task type and configuration."""
        is_seq2seq = self.config.task_type in ["summarization", "translation"]
        use_lightning = getattr(self.config.optimization, "use_lightning", False)
        use_accelerate = getattr(self.config.optimization, "use_accelerate", False)
        
        if use_lightning and use_accelerate:
            logger.info("Using Lightning with Accelerate integration")
            return (
                AcceleratedNLPSeq2SeqTrainerWithLightning
                if is_seq2seq
                else AcceleratedNLPTrainerWithLightning
            )
        elif use_lightning:
            logger.info("Using Lightning trainer")
            return (
                NLPSeq2SeqTrainerWithLightning
                if is_seq2seq
                else NLPTrainerWithLightning
            )
        elif use_accelerate:
            logger.info("Using Accelerate trainer")
            return (
                AcceleratedNLPSeq2SeqTrainer
                if is_seq2seq
                else AcceleratedNLPTrainer
            )
        else:
            logger.info("Using standard trainer")
            return NLPSeq2SeqTrainer if is_seq2seq else NLPTrainer

    def _initialize_trainer(self, train_dataset, eval_dataset, **kwargs):
        """Initialize the appropriate trainer with all necessary components."""
        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join("checkpoints", self.config.task_type),
            **self.config.optimization.__dict__
        )
        
        # Get data collator based on task type
        data_collator = self._get_data_collator()
        
        # Initialize trainer
        trainer = self.trainer_class(
            args_dir=training_args,
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            task_type=self.config.task_type,
            compute_metrics=self.compute_metrics,
            callbacks=[MLflowCallback(self.mlflow_tracker.run_id)] if self.mlflow_tracker else None,
            **kwargs
        )
        
        return trainer

    def _get_data_collator(self):
        """Get appropriate data collator based on task type."""
        if self.config.task_type == "language-modeling":
            return DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=True,
                mlm_probability=0.15
            )
        elif self.config.task_type in ["summarization", "translation"]:
            return DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                padding=True
            )
        elif self.config.task_type == "token-classification":
            return DataCollatorForTokenClassification(
                tokenizer=self.tokenizer
                )
        else:
            return default_data_collator

    def prepare_training_data(
        self,
        train_dataset: List[Dict],
        eval_dataset: Optional[List[Dict]] = None,
        batch_size: int = 32
    ) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
        """Prepare training and evaluation data with optimizations."""
        if self.curriculum_manager:
            # Apply curriculum learning
            train_dataset = self.curriculum_manager.prepare_curriculum(train_dataset)
            if eval_dataset:
                eval_dataset = self.curriculum_manager.prepare_curriculum(eval_dataset)
                
        if self.few_shot_adapter:
            # Apply few-shot adaptation
            train_dataset = self.few_shot_adapter.prepare_dataset(train_dataset)
            if eval_dataset:
                eval_dataset = self.few_shot_adapter.prepare_dataset(eval_dataset)
                
        # Prepare optimized dataloaders
        train_loader = self.training_optimizer.prepare_dataloader(
            train_dataset,
            batch_size=batch_size
        )
        
        eval_loader = None
        if eval_dataset:
            eval_loader = self.training_optimizer.prepare_dataloader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False
            )
            
        return train_loader, eval_loader
    
    def _train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Perform one training step with all optimizations."""
        def forward_func(batch_data):
            outputs = self.model(**batch_data)
            return outputs.loss
            
        metrics = self.training_optimizer.optimizer_step(
            batch,
            optimizer,
            forward_func
        )
        
        if self.curriculum_manager:
            self.curriculum_manager.update_curriculum(metrics)
            
        if self.distributed_trainer:
            metrics = self.distributed_trainer.reduce_metrics(metrics)
            
        return metrics
    
    def train(
        self,
        train_dataset: List[Dict],
        eval_dataset: Optional[List[Dict]] = None,
        num_epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 5e-5,
        **kwargs
    ) -> PreTrainedModel:
        """Run fine-tuning with all configured optimizations."""
        if self.mlflow_tracker:
            self.mlflow_tracker.start_run()
            self.mlflow_tracker.log_params({
                "task_type": self.config.task_type,
                "model_name": self.config.model_name,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "trainer_type": self.trainer_class.__name__
            })
        
        try:
            # Initialize trainer
            trainer = self._initialize_trainer(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                learning_rate=learning_rate,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                **kwargs
            )
            
            # Apply curriculum learning if configured
            if self.curriculum_manager:
                train_dataset = self.curriculum_manager.prepare_curriculum(train_dataset)
                if eval_dataset:
                    eval_dataset = self.curriculum_manager.prepare_curriculum(eval_dataset)
                    
            # Apply few-shot adaptation if configured
            if self.few_shot_adapter:
                train_dataset = self.few_shot_adapter.prepare_dataset(train_dataset)
                if eval_dataset:
                    eval_dataset = self.few_shot_adapter.prepare_dataset(eval_dataset)
            
            # Set up distributed training if configured
            if self.distributed_trainer:
                trainer = self.distributed_trainer.wrap_trainer(trainer)
            
            # Set up knowledge distillation if configured
            if self.distillation_manager:
                trainer = self.distillation_manager.wrap_trainer(trainer)
            
            # Train the model
            logger.info(f"Starting fine-tuning for task type: {self.config.task_type}")
            train_result = trainer.train()
            
            # Log final metrics
            metrics = train_result.metrics
            if eval_dataset:
                eval_metrics = trainer.evaluate()
                metrics.update(eval_metrics)
                
            if self.mlflow_tracker:
                self.mlflow_tracker.log_metrics(metrics)
                
            # Apply model pruning if enabled
            self.training_optimizer.apply_pruning()
            
            # Save the final model
            if trainer.is_world_process_zero():
                trainer.save_model()
                if self.mlflow_tracker:
                    self.mlflow_tracker.log_artifact(trainer.args.output_dir)
            
            # Clean up
            self.training_optimizer.cleanup()
            if self.distributed_trainer:
                self.distributed_trainer.cleanup()
                
            return self.model
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
            
        finally:
            if self.mlflow_tracker:
                self.mlflow_tracker.end_run()
    
    def _evaluate(
        self,
        eval_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate model with optimizations."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1
                
        metrics = {"eval_loss": total_loss / num_batches}
        
        if self.distributed_trainer:
            metrics = self.distributed_trainer.reduce_metrics(metrics)
            
        return metrics
    
    def save_model(self, path: str):
        """Save the fine-tuned model."""
        if hasattr(self.model, "module"):  # Unwrap DDP
            self.model.module.save_pretrained(path)
        else:
            self.model.save_pretrained(path)
            
        if self.mlflow_tracker:
            self.mlflow_tracker.log_artifact(path)
    
    @classmethod
    def load_model(
        cls,
        path: str,
        config: FineTuneConfig
    ) -> "FineTunePipeline":
        """Load a fine-tuned model."""
        model = PreTrainedModel.from_pretrained(path)
        tokenizer = PreTrainedTokenizer.from_pretrained(path)
        return cls(config, model, tokenizer)

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
        
        if self.mlflow_tracker:
            self.mlflow_tracker.log_artifact(checkpoint_path)
            
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
        
    def compute_metrics(
        self,
        eval_pred: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        """Compute task-specific evaluation metrics."""
        predictions, labels = eval_pred
        if self.config.task_type in ["classification", "token-classification"]:
            predictions = np.argmax(predictions, axis=-1)
            return self.metric.compute(predictions=predictions, references=labels)
            
        elif self.config.task_type in ["summarization", "translation"]:
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
            
        elif self.config.task_type == "question-answering":
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
        
    def post_process_predictions(
        self,
        predictions: np.ndarray,
        task_type: str
    ) -> Union[List[str], List[Dict]]:
        """Post-process model predictions based on task type."""
        if task_type in ["classification", "token-classification"]:
            return np.argmax(predictions, axis=-1).tolist()
            
        elif task_type in ["summarization", "translation", "text-generation"]:
            return self.tokenizer.batch_decode(
                predictions,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
        elif task_type == "question-answering":
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





    