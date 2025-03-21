# Standard library imports
import hashlib
import json
import logging
import os
import pickle
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import polars as pl
import psutil
import ray
import torch
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset as TorchDataset, DistributedSampler
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    DataCollatorForTokenClassification,
    default_data_collator
)

# Local imports
from .mlflow_tracking import MLflowTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ParallelMetrics:
    """Metrics for parallel processing performance."""
    total_processing_time: float = 0.0
    chunks_processed: int = 0
    avg_chunk_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    io_wait_time: float = 0.0
    thread_count: int = 0
    failed_chunks: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    data_throughput: float = 0.0  # MB/s

@dataclass
class DatasetConfig:
    """Enhanced configuration for dataset processing."""
    max_length: int
    chunk_size: int
    stride: int
    batch_size: int = 1000
    cache_dir: str = ".cache"
    use_streaming: bool = False
    memory_limit_gb: float = 32.0
    distributed: bool = False
    world_size: int = 1
    local_rank: int = 0
    cache_strategy: str = 'memory'
    # New parallel processing configurations
    max_workers: int = os.cpu_count()
    chunk_queue_size: int = 100
    ray_object_store_memory: int = 10 * 1024 * 1024 * 1024  # 10GB
    adaptive_batching: bool = True
    min_chunk_size: int = 100
    max_chunk_size: int = 10000
    monitoring_interval: float = 1.0  # seconds

@dataclass
class DataQualityMetrics:
    """Metrics for data quality assessment."""
    null_count: int = 0
    duplicate_count: int = 0
    invalid_format_count: int = 0
    out_of_range_count: int = 0
    total_samples: int = 0
    validation_errors: List[str] = None

    def __post_init__(self):
        self.validation_errors = []

    @property
    def quality_score(self) -> float:
        """Calculate overall quality score."""
        if self.total_samples == 0:
            return 0.0
        error_rate = (self.null_count + self.duplicate_count + 
                     self.invalid_format_count + self.out_of_range_count) / self.total_samples
        return max(0.0, 1.0 - error_rate)

class ParallelProcessingMonitor:
    """Monitor parallel processing performance."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.metrics = defaultdict(ParallelMetrics)
        self.start_time = time.time()
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()
        self.lock = threading.Lock()
        
    def start_monitoring(self):
        """Start monitoring thread."""
        self._monitor_thread = threading.Thread(target=self._monitor_resources)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring thread."""
        if self._monitor_thread:
            self._stop_monitoring.set()
            self._monitor_thread.join()
            
    def _monitor_resources(self):
        """Monitor system resources."""
        while not self._stop_monitoring.is_set():
            with self.lock:
                for task_type in self.metrics:
                    self.metrics[task_type].memory_usage = psutil.Process().memory_percent()
                    self.metrics[task_type].cpu_usage = psutil.cpu_percent()
                    self.metrics[task_type].thread_count = threading.active_count()
            time.sleep(self.config.monitoring_interval)
            
    def update_metrics(self, task_type: str, chunk_time: float, chunk_size: int, success: bool = True):
        """Update processing metrics."""
        with self.lock:
            metrics = self.metrics[task_type]
            metrics.total_processing_time += chunk_time
            metrics.chunks_processed += 1
            metrics.avg_chunk_time = metrics.total_processing_time / metrics.chunks_processed
            if not success:
                metrics.failed_chunks += 1
            metrics.data_throughput = (chunk_size * 8) / (chunk_time * 1024 * 1024)  # MB/s
            
    def get_metrics(self, task_type: str) -> Dict[str, float]:
        """Get current metrics."""
        with self.lock:
            return asdict(self.metrics[task_type])

class AdaptiveBatchSizer:
    """Dynamically adjust batch sizes based on system performance."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.current_size = config.batch_size
        self.performance_history = []
        self.adjustment_threshold = 3  # Number of samples before adjustment
        
    def update(self, processing_time: float, memory_usage: float) -> int:
        """Update batch size based on performance metrics."""
        self.performance_history.append((processing_time, memory_usage))
        
        if len(self.performance_history) >= self.adjustment_threshold:
            avg_time = np.mean([t for t, _ in self.performance_history])
            avg_memory = np.mean([m for _, m in self.performance_history])
            
            # Adjust batch size based on performance
            if avg_memory < 70 and avg_time < 1.0:  # Less than 70% memory usage and fast processing
                self.current_size = min(
                    self.current_size * 1.2,
                    self.config.max_chunk_size
                )
            elif avg_memory > 85 or avg_time > 2.0:  # High memory usage or slow processing
                self.current_size = max(
                    self.current_size * 0.8,
                    self.config.min_chunk_size
                )
            
            self.performance_history = []
            
        return int(self.current_size)

class DataValidator:
    """Validate dataset quality and schema."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.metrics = defaultdict(DataQualityMetrics)
        
    def validate_schema(self, data: Dict[str, List], expected_schema: Dict[str, type]) -> bool:
        """Validate data schema."""
        try:
            for key, expected_type in expected_schema.items():
                if key not in data:
                    raise ValueError(f"Missing required field: {key}")
                if not all(isinstance(item, expected_type) for item in data[key]):
                    raise ValueError(f"Invalid type for field {key}")
            return True
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False
            
    def validate_data_quality(self, data: Dict[str, List], task_type: str) -> DataQualityMetrics:
        """Validate data quality."""
        metrics = DataQualityMetrics()
        metrics.total_samples = len(next(iter(data.values())))
        
        try:
            # Check for nulls
            for key, values in data.items():
                null_count = sum(1 for v in values if v is None or (isinstance(v, str) and not v.strip()))
                metrics.null_count += null_count
                
            # Check for duplicates
            if 'inputs' in data:
                seen = set()
                for input_text in data['inputs']:
                    if input_text in seen:
                        metrics.duplicate_count += 1
                    seen.add(input_text)
                    
            # Task-specific validation
            if task_type == "token_classification":
                self._validate_token_classification(data, metrics)
            elif task_type == "question_answering":
                self._validate_qa(data, metrics)
                
            self.metrics[task_type] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Data quality validation failed: {e}")
            metrics.validation_errors.append(str(e))
            return metrics
            
    def _validate_token_classification(self, data: Dict[str, List], metrics: DataQualityMetrics):
        """Validate token classification data."""
        if 'labels' in data:
            for labels in data['labels']:
                if not all(isinstance(l, int) for l in labels):
                    metrics.invalid_format_count += 1
                    
    def _validate_qa(self, data: Dict[str, List], metrics: DataQualityMetrics):
        """Validate question answering data."""
        if 'answers' in data:
            for answer in data['answers']:
                if not isinstance(answer, dict) or 'text' not in answer:
                    metrics.invalid_format_count += 1

class CheckpointManager:
    """Manage checkpoints for long-running processes."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.checkpoint_dir = Path(config.cache_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, data: Any, task_type: str, step: int):
        """Save processing checkpoint."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = self.checkpoint_dir / f"{task_type}_{step}_{timestamp}.pkl"
            
            # Calculate data hash for integrity check
            data_hash = hashlib.md5(pickle.dumps(data)).hexdigest()
            
            checkpoint_data = {
                'data': data,
                'hash': data_hash,
                'step': step,
                'timestamp': timestamp
            }
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
                
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None
            
    def load_checkpoint(self, task_type: str, step: Optional[int] = None) -> Optional[Dict]:
        """Load latest checkpoint or specific step."""
        try:
            pattern = f"{task_type}_*.pkl" if step is None else f"{task_type}_{step}_*.pkl"
            checkpoints = list(self.checkpoint_dir.glob(pattern))
            
            if not checkpoints:
                return None
                
            latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
            
            with open(latest_checkpoint, 'rb') as f:
                checkpoint_data = pickle.load(f)
                
            # Verify data integrity
            loaded_hash = checkpoint_data['hash']
            computed_hash = hashlib.md5(pickle.dumps(checkpoint_data['data'])).hexdigest()
            
            if loaded_hash != computed_hash:
                raise ValueError("Checkpoint data corruption detected")
                
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

class DatasetModules:
    def __init__(self, dataset_dir: str, tokenizer: Any, specs: Dict[str, Any]):
        """Initialize with enhanced features."""
        try:
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.specs = specs

            # Initialize configurations
            self.config = DatasetConfig(**{
                **specs,
                'max_workers': specs.get('max_workers', os.cpu_count()),
                'chunk_queue_size': specs.get('chunk_queue_size', 100),
                'adaptive_batching': specs.get('adaptive_batching', True)
            })
            
            # Initialize MLflow tracker
            self.mlflow_tracker = MLflowTracker(
                experiment_name=specs.get("experiment_name", "default_experiment"),
                tracking_uri=specs.get("tracking_uri", "sqlite:///mlflow.db")
            )
            self.mlflow_tracker.start_run()
            
            # Log initial experiment metadata
            self.mlflow_tracker.log_experiment_metadata({
                "task_type": specs.get("task_type", "unknown"),
                "data_source": dataset_dir,
                "model_name": specs.get("model_name", "unknown"),
                "max_length": self.config.max_length,
                "batch_size": self.config.batch_size,
                "chunk_size": self.config.chunk_size,
                "stride": self.config.stride,
                "distributed": self.config.distributed,
                "world_size": self.config.world_size,
                "local_rank": self.config.local_rank
            })
            
            # Initialize monitoring and optimization components
            self.monitor = ParallelProcessingMonitor(self.config)
            self.batch_sizer = AdaptiveBatchSizer(self.config)
            
            # Initialize Ray for distributed computing
            if not ray.is_initialized():
                ray.init(
                    object_store_memory=self.config.ray_object_store_memory,
                    ignore_reinit_error=True
                )
            
            # Setup other components
            self._setup_polars_config()
            if self.config.distributed:
                self._setup_distributed_cache()
            
            # Start monitoring
            self.monitor.start_monitoring()
            
            # Initialize new components
            self.validator = DataValidator(self.config)
            self.checkpoint_manager = CheckpointManager(self.config)
            
            # Define retry parameters
            self.max_retries = specs.get('max_retries', 3)
            self.retry_delay = specs.get('retry_delay', 1.0)
            
        except Exception as e:
            logger.error(f"Failed to initialize DatasetModules: {e}")
            raise

    def _setup_polars_config(self):
        """Configure Polars for optimal performance."""
        try:
            # Set Polars configuration for large datasets
            pl.Config.set_fmt_str_lengths(self.config.max_length)
            pl.Config.set_tbl_rows(self.config.batch_size)
            pl.Config.set_tbl_cols(100)
            
            # Enable parallel processing in Polars
            if self.config.distributed:
                pl.Config.set_n_threads(max(1, os.cpu_count() // self.config.world_size))
            else:
                pl.Config.set_n_threads(os.cpu_count())
                
        except Exception as e:
            logger.error(f"Failed to setup Polars configuration: {e}")
            raise

    def _setup_distributed_cache(self):
        """Setup cache directory for distributed training."""
        try:
            cache_dir = Path(self.config.cache_dir) / f"rank_{self.config.local_rank}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.local_cache_dir = cache_dir
        except Exception as e:
            logger.error(f"Failed to setup distributed cache: {e}")
            raise

    def _process_large_dataset(self, examples: Dict[str, List], processor_func: callable) -> Dict[str, List]:
        """Process large datasets with enhanced features."""
        try:
            start_time = time.time()
            
            # Validate input data
            if not self.validator.validate_schema(examples, self._get_expected_schema()):
                raise ValueError("Invalid input data schema")
            
            quality_metrics = self.validator.validate_data_quality(examples, self.task_type)
            if quality_metrics.quality_score < 0.8:  # Configurable threshold
                logger.warning(f"Low data quality score: {quality_metrics.quality_score}")
            
            # Log dataset metrics
            self.mlflow_tracker.log_dataset_metrics({
                "size": len(next(iter(examples.values()))),
                "quality_score": quality_metrics.quality_score,
                "null_count": quality_metrics.null_count,
                "duplicate_count": quality_metrics.duplicate_count,
                "invalid_format_count": quality_metrics.invalid_format_count,
                "out_of_range_count": quality_metrics.out_of_range_count
            })
            
            # Process in chunks with checkpointing
            df = pl.DataFrame(examples)
            results = []
            current_pos = 0
            
            while current_pos < len(df):
                chunk_start_time = time.time()
                
                # Load checkpoint if exists
                checkpoint = self.checkpoint_manager.load_checkpoint(self.task_type, current_pos)
                if checkpoint:
                    results.extend(checkpoint['data'])
                    current_pos = checkpoint['step'] + self.config.batch_size
                    continue
                
                # Process chunk with retry mechanism
                chunk = df.slice(current_pos, self.config.batch_size)
                processed_chunk = self._process_with_retry(processor_func, chunk.to_dict())
                results.append(processed_chunk)
                
                # Save checkpoint
                checkpoint_path = self.checkpoint_manager.save_checkpoint(results, self.task_type, current_pos)
                if checkpoint_path:
                    self.mlflow_tracker.log_artifact(str(checkpoint_path), "checkpoint")
                
                # Log timing and resource metrics
                chunk_duration = time.time() - chunk_start_time
                self.mlflow_tracker.log_timing_metric("chunk_processing", chunk_duration)
                self.mlflow_tracker.log_resource_metrics()
                
                current_pos += self.config.batch_size
            
            # Combine and validate results
            combined = self._combine_results(results)
            if not self.validator.validate_schema(combined, self._get_expected_schema()):
                raise ValueError("Invalid output data schema")
            
            # Log total processing time
            total_duration = time.time() - start_time
            self.mlflow_tracker.log_timing_metric("total_processing", total_duration)
            
            return combined
            
        except Exception as e:
            logger.error(f"Error in large dataset processing: {e}")
            raise

    def _get_expected_schema(self) -> Dict[str, type]:
        """Get expected schema based on task type."""
        base_schema = {
            'inputs': str,
            'attention_mask': list
        }
        
        task_schemas = {
            'token_classification': {'labels': list},
            'question_answering': {
                'question': str,
                'context': str,
                'answers': dict
            }
        }
        
        return {**base_schema, **(task_schemas.get(self.task_type, {}))}

    def _combine_results(self, results: List[Dict[str, List]]) -> Dict[str, List]:
        """Combine results with validation."""
        try:
            combined = {}
            for key in results[0].keys():
                combined[key] = []
                for result in results:
                    if key not in result:
                        raise KeyError(f"Missing key {key} in result")
                    combined[key].extend(result[key])
            return combined
        except Exception as e:
            logger.error(f"Failed to combine results: {e}")
            raise

    def _process_with_retry(self, processor_func: Callable, data: Any, max_retries: int = None) -> Any:
        """Process data with retry mechanism."""
        retries = 0
        max_retries = max_retries or self.max_retries
        
        while retries < max_retries:
            try:
                result = processor_func(data)
                return result
            except Exception as e:
                retries += 1
                if retries == max_retries:
                    logger.error(f"Processing failed after {max_retries} retries: {e}")
                    raise
                logger.warning(f"Retry {retries}/{max_retries} after error: {e}")
                time.sleep(self.retry_delay * retries)

    def __del__(self):
        """Cleanup monitoring resources."""
        self.monitor.stop_monitoring()
        if ray.is_initialized():
            ray.shutdown()
        if hasattr(self, 'mlflow_tracker'):
            self.mlflow_tracker.end_run()

    def _cache_processed_data(self, data: Dict[str, List], cache_key: str):
        """Cache processed data based on strategy."""
        try:
            if self.config.cache_strategy == 'none':
                return
                
            if self.config.cache_strategy == 'memory':
                self._memory_cache[cache_key] = data
            else:  # disk cache
                cache_file = self.local_cache_dir / f"{cache_key}.json"
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to cache data: {e}")
            logger.warning("Continuing without caching")

    def _get_cached_data(self, cache_key: str) -> Optional[Dict[str, List]]:
        """Retrieve cached data based on strategy."""
        try:
            if self.config.cache_strategy == 'none':
                return None
                
            if self.config.cache_strategy == 'memory':
                return self._memory_cache.get(cache_key)
            else:  # disk cache
                cache_file = self.local_cache_dir / f"{cache_key}.json"
                if cache_file.exists():
                    with open(cache_file, 'r') as f:
                        return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve cached data: {e}")
            return None

    def _process_token_classification(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Process token classification data with Polars optimization."""
        try:
            # Convert to Polars DataFrame for efficient processing
            df = pl.DataFrame(examples)
            
            # Process tokens in parallel using Polars
            tokenized = self.tokenizer(
                df['inputs'].to_list(),
                is_split_into_words=True,
                truncation=True,
                return_offsets_mapping=True,
                padding=True,
                max_length=self.config.max_length
            )
            
            # Process labels efficiently using Polars
            labels_df = pl.DataFrame({'labels': examples['labels']})
            new_labels = labels_df.select([
                pl.col('labels').map_elements(lambda x: 
                    self.align_labels_with_tokens(tuple(x), tuple(tokenized.word_ids(i))))
                for i in range(len(examples['labels']))
            ]).to_dict()['labels']
            
            tokenized['labels'] = new_labels
            return tokenized
        except Exception as e:
            logger.error(f"Error in token classification processing: {e}")
            raise

    def _process_mlm(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Process data for masked language modeling task."""
        try:
            tokenized = self.tokenizer(
                examples['inputs'],
                is_split_into_words=True,
                truncation=True,
                return_offsets_mapping=True,
                padding=True,
                max_length=self.config.max_length
            )
            tokenized['word_ids'] = tokenized.word_ids()
            return tokenized
        except Exception as e:
            logger.error(f"Error in MLM processing: {e}")
            raise

    def _process_translation(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Process data for translation task."""
        try:
            return self.tokenizer(
                examples['inputs'],
                text_target=examples['targets'],
                max_length=self.config.max_length,
                truncation=True,
                padding=True
            )
        except Exception as e:
            logger.error(f"Error in translation processing: {e}")
            raise

    def _process_summarization(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Process data for summarization task."""
        try:
            tokenized = self.tokenizer(
                examples['inputs'],
                max_length=self.config.max_length,
                truncation=True,
                padding=True
            )
            
            labels = self.tokenizer(
                examples['targets'],
                max_length=self.config.max_length // 4,
                truncation=True,
                padding=True
            )
            
            tokenized['labels'] = labels['input_ids']
            return tokenized
        except Exception as e:
            logger.error(f"Error in summarization processing: {e}")
            raise

    def _process_qa_train(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Process data for question answering task - training mode."""
        try:
            questions = [q.strip() for q in examples['question']]
            inputs = self.tokenizer(
                questions,
                examples['context'],
                max_length=self.config.max_length,
                truncation="only_second",
                stride=self.config.stride,
                padding="max_length",
                return_overflowing_tokens=True,
                return_offsets_mapping=True
            )
            
            offset_mapping = inputs.pop("offset_mapping")
            sample_map = inputs.pop("overflow_to_sample_mapping")
            
            # Process answer positions for training
            start_positions = []
            end_positions = []

            for i, offset in enumerate(offset_mapping):
                sample_idx = sample_map[i]
                answer = examples["answers"][sample_idx]
                start_char = answer["answer_start"][0]
                end_char = start_char + len(answer["text"][0])
                
                sequence_ids = inputs.sequence_ids(i)
                context_start = next(idx for idx, val in enumerate(sequence_ids) if val == 1)
                context_end = next(idx for idx, val in reversed(list(enumerate(sequence_ids))) if val == 1)
                
                if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    idx_start = context_start
                    while idx_start <= context_end and offset[idx_start][0] <= start_char:
                        idx_start += 1
                    start_positions.append(idx_start - 1)
                    
                    idx_end = context_end
                    while idx_end >= context_start and offset[idx_end][1] >= end_char:
                        idx_end -= 1
                    end_positions.append(idx_end + 1)
            
            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions
            return inputs
        except Exception as e:
            logger.error(f"Error in QA training processing: {e}")
            raise

    def _process_qa_validation(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Process data for question answering task - validation mode."""
        try:
            questions = [q.strip() for q in examples['question']]
            inputs = self.tokenizer(
                questions,
                examples['context'],
                max_length=self.config.max_length,
                truncation="only_second",
                stride=self.config.stride,
                padding="max_length",
                return_overflowing_tokens=True,
                return_offsets_mapping=True
            )
            
            offset_mapping = inputs.pop("offset_mapping")
            sample_map = inputs.pop("overflow_to_sample_mapping")
            example_ids = []

            # Process validation specific data
            for i in range(len(inputs["input_ids"])):
                sample_idx = sample_map[i]
                example_ids.append(examples["id"][sample_idx])

                sequence_ids = inputs.sequence_ids(i)
                # Set to None for non-context tokens
                inputs["offset_mapping"][i] = [
                    o if sequence_ids[k] == 1 else None for k, o in enumerate(offset_mapping[i])
                ]

            inputs["example_id"] = example_ids
            return inputs
        except Exception as e:
            logger.error(f"Error in QA validation processing: {e}")
            raise

    def tokenize_dataset(self, examples: Dict[str, List], is_validation: bool = False) -> Dict[str, List]:
        """Tokenize dataset examples based on task type.
        
        Args:
            examples: Dataset examples to tokenize
            is_validation: Whether this is for validation dataset
        
        Returns:
            Tokenized dataset
            
        Raises:
            ValueError: If task_type is not supported
        """
        processors = {
            "token_classification": self._process_token_classification,
            "masked_language_modeling": self._process_mlm,
            "translation": self._process_translation,
            "summarization": self._process_summarization,
            "question_answering": self._process_qa_validation if is_validation else self._process_qa_train
        }
        
        try:
            processor = processors.get(self.task_type)
            if not processor:
                raise ValueError(f"Unsupported task type: {self.task_type}")
            
            return processor(examples)
        except Exception as e:
            logger.error(f"Error in tokenize_dataset: {e}")
            raise

    def group_texts(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Group texts for masked language modeling.
        
        Args:
            examples: Examples to group
            
        Returns:
            Grouped examples
        """
        try:
            # Use Polars for efficient concatenation
            df = pl.DataFrame(examples)
            
        # Concatenate all texts
            concatenated = {
                k: df[k].explode().to_list()
                for k in examples.keys()
            }
            
            total_length = len(concatenated[list(examples.keys())[0]])
            chunk_size = self.config.chunk_size
        total_length = (total_length // chunk_size) * chunk_size
            
            # Split by chunks efficiently
        results = {
                k: [
                    t[i:i + chunk_size]
                    for i in range(0, total_length, chunk_size)
                ]
                for k, t in concatenated.items()
            }
            
        results["labels"] = results["input_ids"].copy()
        return results
        except Exception as e:
            logger.error(f"Error in group_texts: {e}")
            raise

    def prepare_dataset(self, dataset: Any, task_type: str) -> Dict[str, Any]:
        """Prepare dataset with distributed training support."""
        try:
            start_time = time.time()
        self.task_type = task_type
        self.raw_dataset = dataset

            # Split dataset for distributed training
            if self.config.distributed:
        if 'eval' not in self.raw_dataset.keys():
                    self.raw_dataset = self.raw_dataset["train"].train_test_split(
                        train_size=0.9,
                        seed=42 + self.config.local_rank  # Different seed per rank
                    )
                
                # Create distributed samplers
                train_sampler = DistributedSampler(
                    self.raw_dataset['train'],
                    num_replicas=self.config.world_size,
                    rank=self.config.local_rank
                )
                eval_sampler = DistributedSampler(
                    self.raw_dataset['eval'],
                    num_replicas=self.config.world_size,
                    rank=self.config.local_rank
                )
            else:
                if 'eval' not in self.raw_dataset.keys():
                    self.raw_dataset = self.raw_dataset["train"].train_test_split(
                        train_size=0.9,
                        seed=42
                    )
                train_sampler = eval_sampler = None
            
            # Process datasets with Polars optimization
            logger.info(f"Processing training dataset (rank {self.config.local_rank})...")
            train_dataset = self._process_large_dataset(
                self.raw_dataset['train'],
                lambda x: self.tokenize_dataset(x, is_validation=False)
            )
            
            logger.info(f"Processing evaluation dataset (rank {self.config.local_rank})...")
            eval_dataset = self._process_large_dataset(
                self.raw_dataset['eval'],
                lambda x: self.tokenize_dataset(x, is_validation=True)
            )
            
            if self.task_type == "masked_language_modeling":
                logger.info(f"Grouping texts for MLM (rank {self.config.local_rank})...")
                train_dataset = self._process_large_dataset(
                    train_dataset,
                    self.group_texts
                )
                eval_dataset = self._process_large_dataset(
                    eval_dataset,
                    self.group_texts
                )
            
            # Log dataset preparation metrics
            total_duration = time.time() - start_time
            self.mlflow_tracker.log_timing_metric("dataset_preparation", total_duration)
            
        return {
            "train": train_dataset,
                "eval": eval_dataset,
                "train_sampler": train_sampler,
                "eval_sampler": eval_sampler
            }
            
        except Exception as e:
            logger.error(f"Error in prepare_dataset: {e}")
            raise

    def prepare_dataset_from_dir(self, task_type: str) -> Dict[str, Any]:
        """Prepare dataset from directory with distributed support."""
        try:
            logger.info(f"Loading dataset from {self.dataset_dir} (rank {self.config.local_rank})...")
            
            # Use streaming for large datasets
            if self.config.use_streaming:
                self.raw_dataset = load_dataset(
                    self.dataset_dir,
                    streaming=True,
                    split=f'train[{self.config.local_rank}::{self.config.world_size}]' 
                    if self.config.distributed else 'train'
                )
            else:
        self.raw_dataset = load_dataset(self.dataset_dir)

            return self.prepare_dataset(self.raw_dataset, task_type)
            
        except Exception as e:
            logger.error(f"Error in prepare_dataset_from_dir: {e}")
            raise

    def load_dataset(self, name):
        self.raw_dataset = load_dataset(name)
        pass

    def save_dataset(self):
        pass

    @staticmethod
    def get_distributed_config() -> Tuple[bool, int, int]:
        """Get distributed training configuration."""
        try:
            is_distributed = dist.is_available() and dist.is_initialized()
            world_size = dist.get_world_size() if is_distributed else 1
            local_rank = dist.get_rank() if is_distributed else 0
            return is_distributed, world_size, local_rank
        except Exception as e:
            logger.error(f"Error getting distributed config: {e}")
            return False, 1, 0