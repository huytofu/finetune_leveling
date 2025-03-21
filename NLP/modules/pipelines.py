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
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    AutoModel,
    AutoModelForCausalLM
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
from .advanced_training import AdvancedTrainingConfig, AdvancedTrainingManager
from .model_merging import ModelMergeConfig, ModelMerger
from .monitoring import UnifiedMonitor, MetricConfig

# Set up logging
logger = logging.getLogger(__name__)

class MLflowCallback:
    """Callback for tracking training metrics in MLflow."""
    
    def __init__(self, monitor: UnifiedMonitor):
        """
        Initialize the MLflow callback.
        
        Args:
            monitor: UnifiedMonitor instance for tracking
        """
        self.monitor = monitor
    
    def on_init(self, args, state, control, **kwargs):
        """Log initial training parameters."""
        self.monitor.log_metrics({
            "train_batch_size": args.per_device_train_batch_size,
            "eval_batch_size": args.per_device_eval_batch_size,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_grad_norm": args.max_grad_norm
        })
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Log epoch start."""
        self.monitor.log_metric(f"epoch_{state.epoch}", state.epoch)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Log epoch metrics."""
        epoch_duration = time.time() - state.epoch_start_time
        self.monitor.log_metric(f"epoch_{state.epoch}_duration", epoch_duration)
        
        # Log hardware metrics
        self.monitor.log_hardware_metrics(state.global_step)
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Log step start time."""
        state.step_start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        """Log step metrics."""
        step_duration = time.time() - state.step_start_time
        self.monitor.log_metric(f"step_{state.global_step}_duration", step_duration)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation metrics."""
        if metrics:
            self.monitor.log_metrics({f"eval_{k}": v for k, v in metrics.items()})
    
    def on_save(self, args, state, control, **kwargs):
        """Log model checkpoint."""
        if state.best_model_checkpoint:
            self.monitor.log_artifact(state.best_model_checkpoint, f"checkpoint_{state.global_step}")

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

@dataclass
class FineTuneConfig:
    """Configuration for fine-tuning pipeline."""
    model_name: str
    task_type: str
    
    # Basic training parameters
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    max_length: int = 512
    
    # Framework options
    use_lightning: bool = False
    use_accelerate: bool = False
    
    # Advanced optimization techniques
    use_mixed_precision: bool = False
    use_dynamic_batching: bool = False
    use_curriculum_learning: bool = False
    use_few_shot: bool = False
    
    # Large model optimization (for 10B-100B models)
    use_gradient_checkpointing: bool = False
    use_flash_attention: bool = False
    use_deepspeed: bool = False
    deepspeed_stage: int = 2
    deepspeed_offload_optimizer: bool = False
    deepspeed_offload_parameters: bool = False
    
    # Quantization options
    use_quantization: bool = False
    quantization_bits: int = 8  # 4 or 8 bits supported
    quantization_method: str = "bitsandbytes"  # "bitsandbytes", "auto_gptq" or "awq"
    awq_zero_point: bool = True  # AWQ-specific setting
    awq_group_size: int = 128    # AWQ-specific setting
    
    # Advanced training features
    use_advanced_training: bool = False
    
    # Multi-node training (Axolotl based)
    use_multi_node: bool = False
    num_nodes: int = 1
    node_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "29500"
    
    # Hardware optimization (Unsloth based)
    use_unsloth: bool = False
    unsloth_max_seq_length: int = 2048
    
    # RLHF (TRL based)
    use_rlhf: bool = False
    rlhf_method: str = "ppo"  # Options: "ppo", "dpo", "orpo"
    reward_model_name: Optional[str] = None
    num_ppo_epochs: int = 1
    kl_penalty_coefficient: float = 0.1
    beta: float = 0.1  # DPO specific
    
    # PEFT configuration
    use_peft: bool = False
    peft_method: str = "lora"  # Primary: "lora", "prefix", "prompt", "adapter"
    secondary_peft_method: str = None  # For mixing methods (optional)
    
    # PEFT parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    prefix_length: int = 30
    adapter_size: int = 64
    
    # Pruning config
    use_pruning: bool = False
    pruning_sparsity: float = 0.3
    pruning_method: str = "magnitude"  # "magnitude", "structured", or "movement"
    
    # Distillation config
    use_distillation: bool = False
    teacher_model_name: str = None
    distillation_alpha: float = 0.5
    distillation_temperature: float = 2.0
    
    # Model merging config
    use_model_merging: bool = False
    merge_method: str = "weighted_average"  # "weighted_average", "selective", "frankenstein"
    merge_models: List[str] = field(default_factory=list)  # List of model names to merge
    merge_weights: Dict[str, float] = field(default_factory=dict)  # Weights for weighted average
    merge_layer_mapping: Dict[str, Union[List[str], str]] = field(default_factory=dict)  # For selective/frankenstein
    verify_merged_model: bool = True
    
    # LoRA specific settings
    lora_r: int = 8  # rank of LoRA matrices
    lora_alpha: int = 16  # scaling factor
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None
    
    # Prefix-tuning settings
    prefix_projection: bool = False
    
    # Prompt-tuning settings
    prompt_initialization: str = "random"  # Options: "random", "text", "embedding"
    
    # Adapter settings
    adapter_dropout: float = 0.1
    adapter_scaling: float = 1.0
    
    # Mixed precision settings
    mixed_precision_dtype: str = "float16"
    mixed_precision_loss_scale: str = "dynamic"
    
    # Dynamic batching settings
    dynamic_batch_size_range: Tuple[int, int] = (16, 128)
    dynamic_batch_growth_factor: float = 1.5
    dynamic_batch_memory_threshold: float = 0.8
    
    # Curriculum learning settings
    curriculum_difficulty_metric: str = "length"
    curriculum_steps: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    curriculum_scoring_function: str = "linear"
    
    # Few-shot settings
    few_shot_examples: int = 5
    few_shot_metric: str = "similarity"
    few_shot_selection: str = "kmeans"
    
    # MLflow tracking
    enable_mlflow: bool = True
    mlflow_experiment_name: Optional[str] = None
    
    def get_peft_config(self) -> Dict[str, Any]:
        """Get PEFT configuration based on selected method."""
        if not self.use_peft:
            return None
            
        if self.peft_method == "lora":
            return {
                "r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "target_modules": self.lora_target_modules,
                "bias": "none",
                "task_type": self.task_type
            }
        elif self.peft_method == "prefix":
            return {
                "num_virtual_tokens": self.prefix_length,
                "projection": self.prefix_projection,
                "task_type": self.task_type
            }
        elif self.peft_method == "prompt":
            return {
                "num_virtual_tokens": self.prefix_length,
                "prompt_initialization": self.prompt_initialization,
                "task_type": self.task_type
            }
        elif self.peft_method == "adapter":
            return {
                "adapter_size": self.adapter_size,
                "adapter_dropout": self.adapter_dropout,
                "adapter_scaling": self.adapter_scaling,
                "task_type": self.task_type
            }
        else:
            raise ValueError(f"Unsupported PEFT method: {self.peft_method}")

    def _get_trainer_class(self):
        """Get the appropriate trainer class based on task type and configuration."""
        is_seq2seq = self.task_type in ["summarization", "translation"]
        
        if self.use_lightning and self.use_accelerate:
            logger.info("Using Lightning with Accelerate integration")
            if is_seq2seq:
                return AcceleratedNLPSeq2SeqTrainerWithLightning
            else:
                return AcceleratedNLPTrainerWithLightning
        elif self.use_lightning:
            logger.info("Using Lightning trainer")
            if is_seq2seq:
                return NLPSeq2SeqTrainerWithLightning
            else:
                return NLPTrainerWithLightning
        elif self.use_accelerate:
            logger.info("Using Accelerate trainer")
            if is_seq2seq:
                return AcceleratedNLPSeq2SeqTrainer
            else:
                return AcceleratedNLPTrainer
        else:
            logger.info("Using standard trainer")
            if is_seq2seq:
                return NLPSeq2SeqTrainer
            else:
                return NLPTrainer
            
    def _validate_config(self):
        """Validate configuration."""
        # Validate framework options
        if self.use_lightning and not LIGHTNING_AVAILABLE:
            logger.warning("PyTorch Lightning not available. Lightning training disabled.")
            self.use_lightning = False
        
        if self.use_accelerate and not ACCELERATE_AVAILABLE:
            logger.warning("Accelerate not available. Accelerate training disabled.")
            self.use_accelerate = False
            
        # Validate RLHF compatibility
        if self.config.use_rlhf:
            # RLHF is not compatible with Lightning or Accelerate
            if self.use_lightning:
                logger.warning("RLHF is not compatible with PyTorch Lightning. Disabling Lightning.")
                self.use_lightning = False
            if self.use_accelerate:
                logger.warning("RLHF is not compatible with Accelerate. Disabling Accelerate.")
                self.use_accelerate = False
        
        # Validate Unsloth compatibility
        if hasattr(self.config, 'use_unsloth') and self.config.use_unsloth:
            # Check if model is supported by Unsloth
            from .advanced_training import UnslothIntegration
            if not UnslothIntegration.is_model_supported(self.config.model_name):
                logger.warning(f"Model {self.config.model_name} is not supported by Unsloth. Disabling Unsloth.")
                self.config.use_unsloth = False
        
        # Ensure advanced training config is validated
        if hasattr(self.config, 'use_advanced_training') and self.config.use_advanced_training:
            from .advanced_training import validate_advanced_config
            validate_advanced_config(self.config)

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
            if config.enable_mlflow:
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

    def _initialize_components(self):
        """Initialize all pipeline components with error handling."""
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Initialize model
            self.model = self._initialize_model(self.config)
            
            # Initialize data collator
            self.data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding='longest'
            )
            
            # Initialize metrics
            self.metric = self._get_metrics_for_task(self.config.task_type)
            
            # Initialize training optimizer
            self.training_optimizer = TrainingOptimizer(
                self.model,
                self.config
            )
            
            # Initialize distributed training if configured
            self.distributed_trainer = None
            if self.config.use_dynamic_batching and torch.cuda.device_count() > 1:
                try:
                    self.distributed_trainer = DistributedTrainer(
                        self.model,
                        self.config,
                        self.config
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize distributed training: {str(e)}")
            
            # Initialize curriculum learning if configured
            self.curriculum_manager = None
            if self.config.use_curriculum_learning:
                try:
                    self.curriculum_manager = CurriculumManager(
                        self.model,
                        self.tokenizer,
                        self.config
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize curriculum learning: {str(e)}")
            
            # Initialize few-shot adaptation if configured
            self.few_shot_adapter = None
            if self.config.use_few_shot:
                try:
                    self.few_shot_adapter = FewShotAdapter(
                        self.model,
                        self.tokenizer,
                        self.config
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize few-shot adaptation: {str(e)}")
            
            # Initialize knowledge distillation if configured
            self.distillation_manager = None
            if self.config.use_distillation:
                try:
                    # Load teacher model
                    teacher_model = self._initialize_distillation()
                    
                    self.distillation_manager = DistillationManager(
                        self.model,
                        teacher_model,
                        self.tokenizer,
                        self.config
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize knowledge distillation: {str(e)}")
            
            # Initialize MLflow tracking
            self.mlflow_tracker = None
            if self.config.enable_mlflow:
                try:
                    self.mlflow_tracker = MLflowTracker("fine_tuning")
                except Exception as e:
                    logger.warning(f"Failed to initialize MLflow tracking: {str(e)}")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
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

    def _initialize_trainer(self, model, train_dataset, eval_dataset=None):
        """Initialize trainer with optimizations for large models."""
        logger.info(f"Initializing {self.task_type} trainer with optimizations")
        
        # Handle advanced training features first (multi-node, RLHF)
        if hasattr(self.config, 'use_advanced_training') and self.config.use_advanced_training:
            try:
                from .advanced_training import AdvancedTrainingManager
                
                # Create advanced training manager
                advanced_config = self._create_advanced_training_config()
                self.advanced_manager = AdvancedTrainingManager(advanced_config)
                
                # Set up distributed environment if using multi-node
                if self.config.use_multi_node:
                    logger.info("Setting up distributed training environment")
                    success = self.advanced_manager.setup_distributed()
                    if not success:
                        logger.warning("Failed to set up distributed environment. Falling back to single-node training.")
                        self.config.use_multi_node = False
                    else:
                        # Prepare model for distributed training
                        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
                        from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
                        
                        # Try to identify the right transformer layer class
                        transformer_layer_cls = None
                        if "llama" in self.config.model_name.lower():
                            transformer_layer_cls = LlamaDecoderLayer
                        elif "mistral" in self.config.model_name.lower():
                            transformer_layer_cls = MistralDecoderLayer
                        
                        # Prepare model for distributed training (DDP or FSDP)
                        model = self.advanced_manager.prepare_model_for_distributed(model, transformer_layer_cls)
                        
                        # Get distributed sampler for dataset
                        train_sampler = self.advanced_manager.get_data_sampler(train_dataset)
                        eval_sampler = self.advanced_manager.get_data_sampler(eval_dataset) if eval_dataset else None
                
                # Apply RLHF if enabled
                if self.config.use_rlhf:
                    logger.info(f"Using RLHF with method: {self.config.rlhf_method}")
                    
                    # We need to ensure we're using the right datasets format for RLHF
                    # For DPO: we need preference pairs
                    # For PPO: we need a reward model and prompts
                    rlhf_datasets = self._prepare_rlhf_datasets(train_dataset, eval_dataset)
                    
                    # Train with RLHF
                    model = self.advanced_manager.train_with_rlhf(
                        model, 
                        self.tokenizer, 
                        rlhf_datasets,
                        reward_model=self.reward_model if hasattr(self, "reward_model") else None
                    )
                    
                    # Return the RLHF-trained model directly
                    # We don't need a standard trainer when using RLHF
                    return model
                    
            except ImportError as e:
                logger.warning(f"Could not initialize advanced training features: {str(e)}")
                logger.warning("Falling back to standard training")
        
        # Standard training (Lightning, Accelerate, or default)
        # Prepare Training Arguments
        training_args = self._create_training_arguments()
        
        # Get trainer class based on framework and task
        trainer_cls = self._get_trainer_class()
        
        # Prepare compute metrics function for evaluation
        compute_metrics_fn = self.compute_metrics if self.metric else None
        
        # Initialize trainer
        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics_fn,
            data_collator=self.data_collator,
        )
        
        return trainer

    def _prepare_rlhf_datasets(self, train_dataset, eval_dataset=None):
        """Prepare datasets in the right format for RLHF training."""
        # This is a utility function to ensure datasets are in the right format for RLHF
        
        rlhf_datasets = {}
        
        if self.config.rlhf_method == "ppo":
            # For PPO, we need reward data and prompt data
            
            # Check if we already have the right format
            if isinstance(train_dataset, dict) and "reward" in train_dataset and "prompt" in train_dataset:
                return train_dataset
            
            # Try to convert HF dataset to required format
            if hasattr(train_dataset, "features") and "input" in train_dataset.features and "output" in train_dataset.features:
                # Extract reward data
                reward_data = train_dataset.select_columns(["input", "output", "score"]) if "score" in train_dataset.features else None
                
                # Extract prompt data
                prompt_data = train_dataset.select_columns(["input"])
                
                rlhf_datasets["reward"] = reward_data
                rlhf_datasets["prompt"] = prompt_data
            else:
                logger.warning("Dataset format not suitable for PPO training. Please provide datasets with 'input', 'output', and 'score' columns.")
                
        elif self.config.rlhf_method == "dpo":
            # For DPO, we need preference data
            
            # Check if we already have the right format
            if isinstance(train_dataset, dict) and "preference" in train_dataset:
                return train_dataset
            
            # Try to convert HF dataset to required format
            if hasattr(train_dataset, "features") and "input" in train_dataset.features and "chosen" in train_dataset.features and "rejected" in train_dataset.features:
                # Dataset already has preference format
                rlhf_datasets["preference"] = train_dataset
            else:
                logger.warning("Dataset format not suitable for DPO training. Please provide datasets with 'input', 'chosen', and 'rejected' columns.")
        
        return rlhf_datasets

    def _create_advanced_training_config(self):
        """Create config for advanced training features."""
        from .advanced_training import AdvancedTrainingConfig
        
        # Extract relevant settings from main config
        config = AdvancedTrainingConfig(
            # Multi-node settings
            use_multi_node=self.config.use_multi_node if hasattr(self.config, "use_multi_node") else False,
            use_fsdp=self.config.use_fsdp if hasattr(self.config, "use_fsdp") else False,
            num_nodes=self.config.num_nodes if hasattr(self.config, "num_nodes") else 1,
            node_rank=self.config.node_rank if hasattr(self.config, "node_rank") else 0,
            local_rank=self.config.local_rank if hasattr(self.config, "local_rank") else 0,
            master_addr=self.config.master_addr if hasattr(self.config, "master_addr") else "localhost",
            master_port=self.config.master_port if hasattr(self.config, "master_port") else "29500",
            fsdp_sharding_strategy=self.config.fsdp_sharding_strategy if hasattr(self.config, "fsdp_sharding_strategy") else "full",
            fsdp_cpu_offload=self.config.fsdp_cpu_offload if hasattr(self.config, "fsdp_cpu_offload") else False,
            
            # Unsloth settings
            use_unsloth=self.config.use_unsloth if hasattr(self.config, "use_unsloth") else False,
            unsloth_max_seq_length=self.config.unsloth_max_seq_length if hasattr(self.config, "unsloth_max_seq_length") else 2048,
            
            # RLHF settings
            use_rlhf=self.config.use_rlhf if hasattr(self.config, "use_rlhf") else False,
            rlhf_method=self.config.rlhf_method if hasattr(self.config, "rlhf_method") else "ppo",
            reward_model_name=self.config.reward_model_name if hasattr(self.config, "reward_model_name") else None,
            num_ppo_epochs=self.config.num_ppo_epochs if hasattr(self.config, "num_ppo_epochs") else 1,
            kl_penalty_coefficient=self.config.kl_penalty_coefficient if hasattr(self.config, "kl_penalty_coefficient") else 0.1,
            beta=self.config.beta if hasattr(self.config, "beta") else 0.1,
        )
        
        return config

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
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset
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

    def _initialize_model(self, config: FineTuneConfig):
        """Initialize model with optimizations for large models."""
        try:
            # Apply model merging if enabled
            if config.use_model_merging and len(config.merge_models) > 0:
                logger.info(f"Performing model merging using {config.merge_method} method")
                try:
                    # Create merge configuration
                    merge_config = ModelMergeConfig(
                        base_model_name=config.model_name,
                        secondary_model_names=config.merge_models,
                        merge_method=config.merge_method,
                        layer_weights=config.merge_weights if config.merge_weights else None,
                        verify_compatibility=config.verify_merged_model
                    )
                    
                    # Set up additional merge configs based on method
                    if config.merge_method == "selective":
                        merge_config.selective_layers = config.merge_layer_mapping
                    elif config.merge_method == "frankenstein":
                        merge_config.frankenstein_mapping = config.merge_layer_mapping
                    
                    # Perform merging
                    merger = ModelMerger(merge_config)
                    merged_model = merger.merge()
                    
                    logger.info("Model merging completed successfully")
                    
                    # Use merged model as our base model
                    # We'll still apply other optimizations to it
                    base_model = merged_model
                except ImportError:
                    logger.warning("Model merging module not available")
                    # Continue with standard initialization
                except Exception as e:
                    logger.warning(f"Model merging failed: {str(e)}")
                    logger.warning("Continuing with standard model initialization")
            
            # If advanced training is enabled, try to use the optimized model loading
            if config.use_advanced_training and (config.use_unsloth or config.use_multi_node):
                try:
                    from .advanced_training import AdvancedTrainingConfig, AdvancedTrainingManager
                    
                    # Create advanced training config from our config
                    advanced_config = AdvancedTrainingConfig(
                        # Multi-node settings
                        use_multi_node=config.use_multi_node,
                        num_nodes=config.num_nodes,
                        node_rank=config.node_rank,
                        master_addr=config.master_addr,
                        master_port=config.master_port,
                        
                        # Unsloth settings
                        use_unsloth=config.use_unsloth,
                        unsloth_max_seq_length=config.unsloth_max_seq_length,
                        
                        # RLHF settings (if enabled)
                        use_rlhf=config.use_rlhf,
                        rlhf_method=config.rlhf_method,
                        reward_model_name=config.reward_model_name,
                        num_ppo_epochs=config.num_ppo_epochs,
                        kl_penalty_coefficient=config.kl_penalty_coefficient,
                        beta=config.beta
                    )
                    
                    # Create advanced training manager
                    advanced_manager = AdvancedTrainingManager(advanced_config)
                    
                    # Try to get optimized model
                    if config.use_unsloth:
                        # Create PEFT config for Unsloth if needed
                        peft_config = None
                        if config.use_peft:
                            peft_config = config  # Just pass our config
                        
                        # Try to get Unsloth optimized model
                        result = advanced_manager.get_optimized_model(config.model_name, peft_config)
                        if result:
                            logger.info("Successfully loaded model with Unsloth optimizations")
                            model, tokenizer = result
                            self.tokenizer = tokenizer  # Update tokenizer
                            
                            # Store advanced training manager for later use
                            self.advanced_manager = advanced_manager
                            
                            return model
                except ImportError:
                    logger.warning("Advanced training modules not available. Install with: pip install unsloth trl axolotl")
                except Exception as e:
                    logger.warning(f"Failed to initialize advanced training: {str(e)}")
            
            # Standard model loading if advanced optimizations failed or not enabled
            # Determine model loading kwargs
            model_kwargs = {}
            
            # Enable Flash Attention if requested
            if config.use_flash_attention:
                logger.info("Enabling Flash Attention 2.0")
                model_kwargs["attn_implementation"] = "flash_attention_2"
                # Flash Attention works best with BF16
                model_kwargs["torch_dtype"] = torch.bfloat16
            
            # Enable quantization if requested
            if config.use_quantization:
                logger.info(f"Enabling {config.quantization_bits}-bit quantization with {config.quantization_method}")
                if config.quantization_method == "bitsandbytes":
                    # BitsAndBytes quantization
                    from transformers import BitsAndBytesConfig
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=config.quantization_bits == 4,
                        load_in_8bit=config.quantization_bits == 8,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                    model_kwargs["quantization_config"] = quantization_config
                elif config.quantization_method == "auto_gptq":
                    # GPTQ quantization
                    model_kwargs["device_map"] = "auto"
                    model_kwargs["quantization_config"] = {
                        "bits": config.quantization_bits,
                        "group_size": 128,
                        "desc_act": True
                    }
                elif config.quantization_method == "awq":
                    # AWQ quantization
                    logger.info("Using AWQ quantization")
                    try:
                        from awq import AutoAWQForCausalLM
                        
                        # AWQ requires different loading mechanism
                        if config.task_type in ["text-generation", "causal-lm"]:
                            # Save original model name for later loading
                            model_name = config.model_name
                            
                            # AWQ is loaded differently and we'll return it directly
                            model = AutoAWQForCausalLM.from_quantized(
                                model_name,
                                device_map="auto",
                                use_exllama=True,  # Use exllama backend for better performance
                                safetensors=True,
                                fuse_layers=True,
                                zero_point=config.awq_zero_point,
                                group_size=config.awq_group_size,
                                trust_remote_code=True
                            )
                            
                            # Skip normal model loading
                            logger.info(f"Loaded AWQ quantized model: {model_name}")
                            return model
                        else:
                            logger.warning(f"AWQ only supports causal language models. Falling back to BitsAndBytes for {config.task_type}")
                            # Fall back to BitsAndBytes for non-CLM models
                            from transformers import BitsAndBytesConfig
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_quant_type="nf4"
                            )
                            model_kwargs["quantization_config"] = quantization_config
                    except ImportError:
                        logger.warning("AWQ package not found. Please install it with: pip install awq")
                        logger.warning("Falling back to BitsAndBytes quantization")
                        # Fall back to BitsAndBytes
                        from transformers import BitsAndBytesConfig
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        model_kwargs["quantization_config"] = quantization_config
            
            # Check if we have a merged model to use instead of loading from scratch
            if config.use_model_merging and 'base_model' in locals():
                logger.info("Using merged model as base model")
                model = base_model
            else:
                # Load base model normally
                logger.info(f"Loading model: {config.model_name}")
                if config.task_type == "text_classification":
                    model = AutoModelForSequenceClassification.from_pretrained(
                        config.model_name,
                        num_labels=self.num_labels,
                        **model_kwargs
                    )
                elif config.task_type == "token-classification":
                    model = AutoModelForTokenClassification.from_pretrained(
                        config.model_name,
                        num_labels=self.num_labels,
                        **model_kwargs
                    )
                elif config.task_type in ["summarization", "translation"]:
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        config.model_name,
                        **model_kwargs
                    )
                elif config.task_type == "question-answering":
                    model = AutoModelForQuestionAnswering.from_pretrained(
                        config.model_name,
                        **model_kwargs
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        config.model_name,
                        **model_kwargs
                    )
            
            # Enable Gradient Checkpointing if requested
            if config.use_gradient_checkpointing:
                logger.info("Enabling Gradient Checkpointing")
                model.gradient_checkpointing_enable()
            
            # Apply PEFT if enabled
            if config.use_peft:
                logger.info(f"Applying PEFT method: {config.peft_method}")
                
                # Import PEFT modules
                from peft import (
                    get_peft_model, 
                    LoraConfig, 
                    PrefixTuningConfig, 
                    PromptTuningConfig,
                    AdapterConfig,
                    TaskType,
                    MultitaskPromptTuningConfig,
                    PromptEncoderConfig
                )
                
                task_type = TaskType.CAUSAL_LM
                if config.task_type == "text_classification":
                    task_type = TaskType.SEQ_CLS
                elif config.task_type == "token-classification":
                    task_type = TaskType.TOKEN_CLS
                elif config.task_type in ["summarization", "translation"]:
                    task_type = TaskType.SEQ_2_SEQ_LM
                
                # Configure primary PEFT method
                peft_configs = []
                
                if config.peft_method == "lora":
                    peft_configs.append(LoraConfig(
                        r=config.lora_r,
                        lora_alpha=config.lora_alpha,
                        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        lora_dropout=config.lora_dropout,
                        bias="none",
                        task_type=task_type
                    ))
                elif config.peft_method == "prefix":
                    peft_configs.append(PrefixTuningConfig(
                        task_type=task_type,
                        prefix_length=config.prefix_length,
                        num_virtual_tokens=config.prefix_length
                    ))
                elif config.peft_method == "prompt":
                    peft_configs.append(PromptTuningConfig(
                        task_type=task_type,
                        num_virtual_tokens=config.prefix_length
                    ))
                elif config.peft_method == "adapter":
                    peft_configs.append(AdapterConfig(
                        adapter_size=config.adapter_size
                    ))
                
                # Configure secondary PEFT method if specified
                if config.secondary_peft_method:
                    logger.info(f"Applying secondary PEFT method: {config.secondary_peft_method}")
                    
                    if config.secondary_peft_method == "lora":
                        peft_configs.append(LoraConfig(
                            r=config.lora_r,
                            lora_alpha=config.lora_alpha,
                            target_modules=["q_proj", "v_proj"],  # Using fewer target modules for secondary
                            lora_dropout=config.lora_dropout,
                            bias="none",
                            task_type=task_type
                        ))
                    elif config.secondary_peft_method == "prefix":
                        peft_configs.append(PrefixTuningConfig(
                            task_type=task_type,
                            prefix_length=config.prefix_length // 2,  # Using shorter prefix for secondary
                            num_virtual_tokens=config.prefix_length // 2
                        ))
                    elif config.secondary_peft_method == "adapter":
                        peft_configs.append(AdapterConfig(
                            adapter_size=config.adapter_size // 2  # Smaller adapter for secondary
                        ))
                
                # Apply PEFT configuration(s)
                if len(peft_configs) == 1:
                    # Single PEFT method
                    model = get_peft_model(model, peft_configs[0])
                else:
                    # Multiple PEFT methods (experimental)
                    from peft import PeftModel
                    
                    # Apply first method
                    model = get_peft_model(model, peft_configs[0])
                    
                    # Apply second method with a different adapter name
                    model = PeftModel.from_pretrained(
                        model,
                        model.base_model.model_name_or_path,
                        adapter_name="secondary_adapter",
                        peft_config=peft_configs[1]
                    )
                    
                    # Activate both adapters
                    model.active_adapters = ["default", "secondary_adapter"]
                
                model.print_trainable_parameters()
            
            # Apply pruning if enabled (with minimal code)
            if config.use_pruning and not config.use_peft:
                # Only apply pruning if not using PEFT (as a simple approach)
                logger.info(f"Applying {config.pruning_method} pruning with {config.pruning_sparsity*100}% sparsity")
                try:
                    # Simple magnitude pruning with torch.nn.utils.prune
                    import torch.nn.utils.prune as prune
                    
                    # Apply pruning to linear layers only
                    for name, module in model.named_modules():
                        if isinstance(module, torch.nn.Linear):
                            prune.l1_unstructured(module, name='weight', amount=config.pruning_sparsity)
                            
                    logger.info("Pruning applied successfully")
                except Exception as e:
                    logger.warning(f"Pruning could not be applied: {str(e)}")
            
            return model
                
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def _initialize_distillation(self):
        """Initialize distillation components if enabled."""
        if self.config.use_distillation and self.config.teacher_model_name:
            logger.info(f"Initializing knowledge distillation with teacher model: {self.config.teacher_model_name}")
            try:
                # Load teacher model with the same configuration but without PEFT/quantization
                teacher_kwargs = {}
                
                # Teacher should be loaded in evaluation mode and with low precision if possible
                teacher_kwargs["torch_dtype"] = torch.float16
                
                if self.config.task_type == "text_classification":
                    teacher_model = AutoModelForSequenceClassification.from_pretrained(
                        self.config.teacher_model_name,
                        num_labels=self.num_labels,
                        **teacher_kwargs
                    )
                elif self.config.task_type == "token-classification":
                    teacher_model = AutoModelForTokenClassification.from_pretrained(
                        self.config.teacher_model_name,
                        num_labels=self.num_labels,
                        **teacher_kwargs
                    )
                elif self.config.task_type in ["summarization", "translation"]:
                    teacher_model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.config.teacher_model_name,
                        **teacher_kwargs
                    )
                elif self.config.task_type == "question-answering":
                    teacher_model = AutoModelForQuestionAnswering.from_pretrained(
                        self.config.teacher_model_name,
                        **teacher_kwargs
                )
                else:
                    teacher_model = AutoModelForCausalLM.from_pretrained(
                        self.config.teacher_model_name,
                        **teacher_kwargs
                    )
                
                # Set teacher model to evaluation mode
                teacher_model.eval()
                for param in teacher_model.parameters():
                    param.requires_grad = False
                    
                return teacher_model
            except Exception as e:
                logger.error(f"Error loading teacher model: {str(e)}")
                ErrorHandler.handle_error(e, "distillation_initialization")
                return None
        return None

    def fine_tune(self, train_dataset, eval_dataset=None):
        """Fine-tune the model on the provided dataset."""
        # Rest of existing code...
        
        # Apply RLHF training if enabled
        if self.config.use_rlhf and hasattr(self, 'advanced_manager'):
            logger.info(f"Applying RLHF training with {self.config.rlhf_method} method")
            
            try:
                # For RLHF we need preference datasets
                reward_dataset = self.prepare_rlhf_preference_dataset(train_dataset)
                
                # For PPO we also need a prompt dataset
                prompt_dataset = None
                if self.config.rlhf_method == "ppo":
                    prompt_dataset = self.prepare_rlhf_prompt_dataset(train_dataset)
                
                # Apply RLHF
                self.model = self.advanced_manager.apply_rlhf(
                    self.model, 
                    self.tokenizer, 
                    reward_dataset, 
                    prompt_dataset
                )
                
                logger.info("RLHF training completed successfully")
            except Exception as e:
                logger.error(f"RLHF training failed: {str(e)}")
        
        # Rest of existing code...

    def prepare_rlhf_preference_dataset(self, dataset):
        """Prepare a preference dataset for RLHF training."""
        # Simplified implementation - in a real application this would 
        # convert your dataset to the format expected by TRL:
        # {
        #   "prompt": "User query", 
        #   "chosen": "Good response", 
        #   "rejected": "Bad response"
        # }
        logger.info("Preparing preference dataset for RLHF")
        
        # For demonstration purposes only - would need to be implemented based on data format
        preference_data = []
        for item in dataset:
            # Example conversion - you'd need to adapt this to your actual data format
            if "prompt" in item and "responses" in item and len(item["responses"]) >= 2:
                preference_data.append({
                    "prompt": item["prompt"],
                    "chosen": item["responses"][0],  # Assuming first response is preferred
                    "rejected": item["responses"][1]  # Assuming second response is less preferred
                })
        
        return preference_data

    def prepare_rlhf_prompt_dataset(self, dataset):
        """Prepare a prompt dataset for PPO training."""
        # Simplified implementation - in a real application this would
        # extract prompts from your dataset for PPO generation
        logger.info("Preparing prompt dataset for PPO")
        
        # For demonstration purposes only
        prompt_data = []
        for item in dataset:
            if "prompt" in item:
                prompt_data.append({"prompt": item["prompt"]})
        
        # Group prompts into batches
        batch_size = min(8, len(prompt_data))
        prompt_batches = []
        for i in range(0, len(prompt_data), batch_size):
            batch = prompt_data[i:i+batch_size]
            prompt_batches.append(batch)
        
        return prompt_batches





    