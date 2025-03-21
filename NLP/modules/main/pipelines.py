"""Pipeline implementations for model training and inference."""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional, List, Union
from transformers import (
    pipeline,
    PreTrainedModel,
    PreTrainedTokenizer
)

# Add to Python path
currdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(currdir)

# Local imports
from configs.default_config import DEFAULT_SPECS
from modules.managers.adapter_manager import MultiAdapterManager
from modules.config.training_config import FineTuneConfig
from modules.config.mlflow_config import MLflowConfig
from modules.orchestration.pipeline_modules import PipelineOrchestrator

logger = logging.getLogger(__name__)

class InferencePipeline:
    """Pipeline for model inference with adapter support."""
    
    def __init__(self, task_type: str, checkpoint: str, adapter_config: Optional[Dict] = None):
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
        self.task_type = task_type.replace("_", "-")
        self.specs = {**DEFAULT_SPECS}
        
        # Create the base pipeline without adapters
        self.loaded_pipeline = self._create_pipeline(checkpoint)
        
        # Initialize adapter support
        self.adapter_manager = None
        self.has_adapter_support = False
        
        if adapter_config is not None:
            self._initialize_adapter_support(checkpoint, adapter_config)
            
    def _create_pipeline(self, checkpoint: str):
        """Create the appropriate pipeline based on task type."""
        supported_tasks = {
            "token_classification": {"aggregation_strategy": "simple"},
            "masked_language_modeling": {},
            "translation": {},
            "summarization": {},
            "question_answering": {},
            "text_generation": {}
        }
        
        if self.task_type not in supported_tasks:
            raise ValueError(f"Unsupported task type: {self.task_type}")
            
        return pipeline(
            self.task_type,
            model=checkpoint,
            **supported_tasks[self.task_type]
        )
    
    def _initialize_adapter_support(self, checkpoint: str, adapter_config: Dict):
        """Initialize adapter support with the given configuration."""
        try:
            # Initialize MultiAdapterManager
            self.adapter_manager = MultiAdapterManager(
                base_model=self.loaded_pipeline.model,
                cache_dir=adapter_config.get('cache_dir'),
                max_adapters_in_memory=adapter_config.get('max_adapters', 5),
                aws_credentials=adapter_config.get('aws_credentials'),
                gcp_credentials=adapter_config.get('gcp_credentials')
            )
            
            # Preload adapters if specified
            if preload_adapters := adapter_config.get('preload_adapters', []):
                self._preload_adapters(preload_adapters)
                
        except Exception as e:
            logger.error(f"Failed to initialize adapter support: {str(e)}")
            self.adapter_manager = None
            
    def _preload_adapters(self, preload_adapters: List[Dict]):
        """Preload specified adapters."""
        for adapter_conf in preload_adapters:
            source = adapter_conf.get('source', 'local')
            adapter_id = adapter_conf.get('adapter_id')
            
            if not adapter_id:
                continue
                
            try:
                self._load_adapter(source, adapter_id, adapter_conf)
                self.has_adapter_support = True
                logger.info(f"Successfully registered adapter {adapter_id} from {source}")
                
            except Exception as e:
                logger.error(f"Failed to register adapter {adapter_id} from {source}: {str(e)}")
                
    def _load_adapter(self, source: str, adapter_id: str, config: Dict):
        """Load an adapter from the specified source."""
        if source == 'local' and (path := config.get('path')):
            self.adapter_manager.register_adapter_from_local(
                adapter_id=adapter_id,
                adapter_path=path,
                adapter_name=config.get('adapter_name'),
                adapter_type=config.get('adapter_type', 'lora'),
                metadata=config.get('metadata')
            )
        elif source == 'huggingface' and (repo_id := config.get('repo_id')):
            self.adapter_manager.register_adapter_from_huggingface(
                adapter_id=adapter_id,
                repo_id=repo_id,
                adapter_name=config.get('adapter_name'),
                adapter_type=config.get('adapter_type', 'lora'),
                metadata=config.get('metadata'),
                revision=config.get('revision', 'main'),
                use_auth_token=config.get('use_auth_token')
            )
        elif source in ['aws', 'gcp'] and (bucket := config.get('bucket')) and (prefix := config.get('prefix')):
            register_method = getattr(
                self.adapter_manager,
                f'register_adapter_from_{source}'
            )
            register_method(
                adapter_id=adapter_id,
                bucket=bucket,
                prefix=prefix,
                adapter_name=config.get('adapter_name'),
                adapter_type=config.get('adapter_type', 'lora'),
                metadata=config.get('metadata')
            )
                
    def list_adapters(self) -> List[Dict]:
        """List all registered adapters with their metadata."""
        if not self.has_adapter_support:
            return []
            
        return [{
            'id': adapter_id,
            'metadata': self.adapter_manager.get_adapter_metadata(adapter_id)
        } for adapter_id in self.adapter_manager.list_adapters()]
        
    def run(
        self,
        input_text: str,
        adapter_id: Optional[str] = None,
        auto_adapter: bool = False,
        select_adapter_with_llm: bool = False,
        **kwargs
    ) -> Any:
        """
        Run inference with optional adapter support.
        
        Args:
            input_text: Input text for inference
            adapter_id: ID of the adapter to use (optional)
            auto_adapter: Whether to automatically select the best adapter
            select_adapter_with_llm: Whether to use LLM for adapter selection
            **kwargs: Additional arguments for the pipeline
            
        Returns:
            Pipeline output
        """
        if not self.has_adapter_support:
            return self.loaded_pipeline(input_text, **kwargs)
            
        # Handle adapter selection
        selected_adapter = adapter_id
        if not selected_adapter and auto_adapter:
            selected_adapter = (
                self._select_best_adapter_with_llm(input_text)
                if select_adapter_with_llm
                else self._select_best_adapter(input_text)
            )
                
        if selected_adapter:
            try:
                with self._use_adapter(selected_adapter):
                    return self.loaded_pipeline(input_text, **kwargs)
            except Exception as e:
                logger.error(f"Error using adapter {selected_adapter}: {str(e)}")
                
        return self.loaded_pipeline(input_text, **kwargs)
            
    def _select_best_adapter(self, input_text: str) -> Optional[str]:
        """Select the best adapter based on metadata matching."""
        if not self.has_adapter_support:
            return None
            
        return max(
            ((adapter_id, self._compute_adapter_score(input_text, self.adapter_manager.get_adapter_metadata(adapter_id)))
             for adapter_id in self.adapter_manager.list_adapters()),
            key=lambda x: x[1],
            default=(None, -1)
        )[0]
        
    def _select_best_adapter_with_llm(self, input_text: str) -> Optional[str]:
        """Select the best adapter using LLM-based analysis."""
        if not self.has_adapter_support:
            return None
            
        try:
            adapters_info = self.list_adapters()
            prompt = self._create_adapter_selection_prompt(input_text, adapters_info)
            selected_adapter = self._call_llm(prompt).strip()
            
            if selected_adapter in self.adapter_manager.list_adapters():
                return selected_adapter
                
        except Exception as e:
            logger.error(f"Error in LLM-based adapter selection: {str(e)}")
            
        return None
        
    def _create_adapter_selection_prompt(self, input_text: str, adapters_info: List[Dict]) -> str:
        """Create prompt for LLM-based adapter selection."""
        return f"""Given the input text: '{input_text}'
        And the following adapters with their metadata:
        {json.dumps(adapters_info, indent=2)}
        
        Which adapter would be most suitable for processing this input?
        Consider the task type, domain, and any specific capabilities of each adapter.
        Return only the adapter ID of the most suitable adapter."""
        
    def _compute_adapter_score(self, input_text: str, metadata: Dict) -> float:
        """
        Compute compatibility score between input text and adapter metadata.
        Override this method to implement custom scoring logic.
        """
        return 0.0
        
    def _call_llm(self, prompt: str) -> str:
        """
        Placeholder for LLM call - override this method to implement
        custom LLM-based adapter selection.
        """
        raise NotImplementedError("LLM-based adapter selection not implemented")
        
    def _use_adapter(self, adapter_id: str):
        """Context manager for using an adapter."""
        class AdapterContext:
            def __init__(self, manager, adapter):
                self.manager = manager
                self.adapter = adapter
                
            def __enter__(self):
                self.manager.activate_adapter(self.adapter)
                logger.info(f"Activated adapter: {self.adapter}")
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.manager.deactivate_adapter(self.adapter)
                
        return AdapterContext(self.adapter_manager, adapter_id)

class FineTunePipeline:
    """Pipeline for fine-tuning models with comprehensive configuration options."""
    
    def __init__(
        self,
        config: FineTuneConfig,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        mlflow_config: Optional[MLflowConfig] = None
    ):
        """Initialize the fine-tuning pipeline."""
        self.orchestrator = PipelineOrchestrator(config, mlflow_config)
        self.model = model
        self.tokenizer = tokenizer
        
    def fine_tune(
        self,
        train_dataset: List[Dict],
        eval_dataset: Optional[List[Dict]] = None
    ) -> PreTrainedModel:
        """Execute the fine-tuning process."""
        return self.orchestrator.run(train_dataset, eval_dataset)





    