from datasets import load_dataset
import polars as pl
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DistributedSampler
import os
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
from dataclasses import asdict
import threading
from queue import Queue
import ray
from collections import defaultdict

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

class DatasetModules:
    def __init__(self, dataset_dir: str, tokenizer: Any, specs: Dict[str, Any]):
        """Initialize with enhanced parallel processing support."""
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
        """Process large datasets with enhanced parallel processing and monitoring."""
        try:
            start_time = time.time()
            df = pl.DataFrame(examples)
            
            # Initialize processing queue and results
            chunk_queue = Queue(maxsize=self.config.chunk_queue_size)
            results = []
            
            def process_chunk(chunk_data):
                """Process a single chunk with monitoring."""
                chunk_start = time.time()
                try:
                    processed = processor_func(chunk_data.to_dict())
                    chunk_time = time.time() - chunk_start
                    self.monitor.update_metrics(
                        self.task_type,
                        chunk_time,
                        len(chunk_data),
                        success=True
                    )
                    return processed
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
                    self.monitor.update_metrics(
                        self.task_type,
                        time.time() - chunk_start,
                        len(chunk_data),
                        success=False
                    )
                    return None

            # Process chunks in parallel
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                current_pos = 0
                
                while current_pos < len(df):
                    # Get adaptive batch size
                    metrics = self.monitor.get_metrics(self.task_type)
                    chunk_size = self.batch_sizer.update(
                        metrics['avg_chunk_time'],
                        metrics['memory_usage']
                    )
                    
                    # Process chunk
                    chunk = df.slice(current_pos, chunk_size)
                    future = executor.submit(process_chunk, chunk)
                    futures.append(future)
                    current_pos += chunk_size
                    
                    # Log progress
                    if current_pos % (chunk_size * 10) == 0:
                        logger.info(f"Processing progress: {current_pos}/{len(df)} examples")
                        logger.info(f"Current metrics: {metrics}")
                
                # Collect results
                results = []
                for future in tqdm(futures, desc="Collecting results"):
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to get result: {e}")
            
            # Combine results
            combined = {}
            for key in results[0].keys():
                combined[key] = [item for chunk in results for item in chunk[key]]
            
            # Update final metrics
            total_time = time.time() - start_time
            logger.info(f"Total processing time: {total_time:.2f}s")
            logger.info(f"Final metrics: {self.monitor.get_metrics(self.task_type)}")
            
            return combined
            
        except Exception as e:
            logger.error(f"Error in large dataset processing: {e}")
            raise
        
    def __del__(self):
        """Cleanup monitoring resources."""
        self.monitor.stop_monitoring()
        if ray.is_initialized():
            ray.shutdown()

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