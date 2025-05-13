"""Pipeline modules for fine-tuning and training management."""

import logging
import os
import torch
from typing import Dict, Any, Optional, List, Tuple, Union
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments
)

from .config.training_config import FineTuneConfig
from .config.mlflow_config import MLflowConfig, MLflowTracker
from .monitoring import UnifiedMonitor
from .model_modules import ModelModules
from .tokenizer_modules import TokenizerModules
from .dataset_modules import DatasetModules
from .advanced_training import AdvancedTrainingManager
from .curriculum_learning import CurriculumManager
from .few_shot_adaptation import FewShotAdapter
from .distillation import DistillationManager
from .model_merging import ModelMerger

logger = logging.getLogger(__name__)

class ModelInitializer:
    """Handles model initialization and configuration."""
    
    @staticmethod
    def initialize_model(config: FineTuneConfig) -> PreTrainedModel:
        """Initialize and configure the model based on the config."""
        model_modules = ModelModules()
        model = model_modules.load_pretrained_model(
            model_name=config.model_name,
            task_type=config.task_type
        )
        
        # Configure model optimizations
        if config.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            
        if config.use_flash_attention:
            model = model_modules.enable_flash_attention(model)
            
        # Handle quantization
        if config.use_quantization:
            model = model_modules.quantize_model(
                model,
                bits=config.quantization_bits,
                method=config.quantization_method
            )
            
        return model

class DatasetPreparator:
    """Handles dataset preparation and processing."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, config: FineTuneConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.dataset_modules = DatasetModules()
        
    def prepare_datasets(
        self,
        train_dataset: List[Dict],
        eval_dataset: Optional[List[Dict]] = None
    ) -> Tuple[torch.utils.data.Dataset, Optional[torch.utils.data.Dataset]]:
        """Prepare datasets for training."""
        # Process datasets
        processed_train = self.dataset_modules.process_dataset(
            train_dataset,
            self.tokenizer,
            max_length=self.config.max_length,
            task_type=self.config.task_type
        )
        
        processed_eval = None
        if eval_dataset:
            processed_eval = self.dataset_modules.process_dataset(
                eval_dataset,
                self.tokenizer,
                max_length=self.config.max_length,
                task_type=self.config.task_type
            )
            
        # Apply curriculum learning if enabled
        if self.config.use_curriculum_learning:
            curriculum_manager = CurriculumManager(
                difficulty_metric=self.config.curriculum_difficulty_metric,
                steps=self.config.curriculum_steps,
                scoring_function=self.config.curriculum_scoring_function
            )
            processed_train = curriculum_manager.prepare_curriculum(processed_train)
            
        # Apply few-shot learning if enabled
        if self.config.use_few_shot:
            few_shot_adapter = FewShotAdapter(
                num_examples=self.config.few_shot_examples,
                metric=self.config.few_shot_metric,
                selection=self.config.few_shot_selection
            )
            processed_train = few_shot_adapter.prepare_few_shot(processed_train)
            
        return processed_train, processed_eval

class TrainingOptimizer:
    """Handles training optimizations and configurations."""
    
    @staticmethod
    def configure_training_args(config: FineTuneConfig) -> TrainingArguments:
        """Configure training arguments based on the config."""
        training_args = TrainingArguments(
            output_dir=os.path.join("outputs", config.model_name),
            per_device_train_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_epochs,
            fp16=config.use_mixed_precision,
            gradient_checkpointing=config.use_gradient_checkpointing
        )
        
        # Configure DeepSpeed if enabled
        if config.use_deepspeed:
            training_args.deepspeed = {
                "zero_optimization": {
                    "stage": config.deepspeed_stage,
                    "offload_optimizer": config.deepspeed_offload_optimizer,
                    "offload_param": config.deepspeed_offload_parameters
                }
            }
            
        return training_args

class TrainingManager:
    """Manages the training process and advanced features."""
    
    def __init__(
        self,
        config: FineTuneConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        mlflow_tracker: Optional[MLflowTracker] = None
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.mlflow_tracker = mlflow_tracker
        
    def setup_advanced_features(self):
        """Set up advanced training features."""
        if self.config.use_advanced_training:
            advanced_manager = AdvancedTrainingManager(self.config)
            self.model = advanced_manager.prepare_model(self.model)
            
        if self.config.use_distillation:
            distillation_manager = DistillationManager(
                teacher_model_name=self.config.teacher_model_name,
                alpha=self.config.distillation_alpha,
                temperature=self.config.distillation_temperature
            )
            self.model = distillation_manager.setup_distillation(self.model)
            
        if self.config.use_model_merging:
            model_merger = ModelMerger(
                method=self.config.merge_method,
                models=self.config.merge_models,
                weights=self.config.merge_weights,
                layer_mapping=self.config.merge_layer_mapping
            )
            self.model = model_merger.merge_models(self.model)
            
    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        training_args: Optional[TrainingArguments] = None
    ) -> PreTrainedModel:
        """Execute the training process."""
        # Configure training arguments if not provided
        if training_args is None:
            training_args = TrainingOptimizer.configure_training_args(self.config)
            
        # Set up MLflow tracking if enabled
        if self.config.enable_mlflow and self.mlflow_tracker:
            callback = self.mlflow_tracker.start_run()
            training_args.callbacks = [callback]
            
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        # Train the model
        try:
            trainer.train()
            
            # Log final metrics if MLflow is enabled
            if self.config.enable_mlflow and self.mlflow_tracker:
                eval_results = trainer.evaluate()
                self.mlflow_tracker.log_metrics(eval_results)
                self.mlflow_tracker.log_model(self.model, "model")
                
        finally:
            if self.config.enable_mlflow and self.mlflow_tracker:
                self.mlflow_tracker.end_run()
                
        return self.model

class PipelineOrchestrator:
    """Orchestrates the entire fine-tuning pipeline."""
    
    def __init__(
        self,
        config: FineTuneConfig,
        mlflow_config: Optional[MLflowConfig] = None
    ):
        self.config = config
        self.mlflow_tracker = MLflowTracker(mlflow_config) if mlflow_config else None
        
    def run(
        self,
        train_dataset: List[Dict],
        eval_dataset: Optional[List[Dict]] = None
    ) -> PreTrainedModel:
        """Run the complete fine-tuning pipeline."""
        # Initialize model
        model = ModelInitializer.initialize_model(self.config)
        
        # Initialize tokenizer
        tokenizer_modules = TokenizerModules()
        tokenizer = tokenizer_modules.load_pretrained_tokenizer(self.config.model_name)
        
        # Prepare datasets
        dataset_preparator = DatasetPreparator(tokenizer, self.config)
        processed_train, processed_eval = dataset_preparator.prepare_datasets(
            train_dataset,
            eval_dataset
        )
        
        # Set up training manager
        training_manager = TrainingManager(
            config=self.config,
            model=model,
            tokenizer=tokenizer,
            mlflow_tracker=self.mlflow_tracker
        )
        
        # Set up advanced features
        training_manager.setup_advanced_features()
        
        # Execute training
        trained_model = training_manager.train(
            train_dataset=processed_train,
            eval_dataset=processed_eval
        )
        
        return trained_model 