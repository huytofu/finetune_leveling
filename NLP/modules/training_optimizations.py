"""Training Optimizations Module for Enhanced Efficiency

This module implements advanced training optimization techniques for language models,
enabling faster training, reduced memory usage, and improved hardware utilization.
Key features include:
- Dynamic batch sizing based on GPU memory
- Gradient checkpointing for memory efficiency
- Mixed precision training (FP16/BF16)
- Smart data prefetching
- Model pruning capabilities
- Memory monitoring and optimization
"""

import logging
import gc
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple, Callable
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
import psutil
import GPUtil
from functools import partial
from torch.utils.data.sampler import BatchSampler
from collections import deque

from .mlflow_tracking import MLflowTracker

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for training optimizations.
    
    Attributes:
        enable_mixed_precision: Whether to use mixed precision training
        mixed_precision_dtype: Type of mixed precision (fp16 or bf16)
        enable_gradient_checkpointing: Whether to use gradient checkpointing
        enable_dynamic_batching: Whether to use dynamic batch sizing
        enable_smart_prefetching: Whether to use smart data prefetching
        enable_pruning: Whether to apply model pruning
        target_batch_size: Target batch size for dynamic batching
        min_batch_size: Minimum allowed batch size
        max_batch_size: Maximum allowed batch size
        memory_threshold: GPU memory threshold for batch size adjustment
        pruning_method: Method for model pruning
        pruning_amount: Amount of weights to prune
        prefetch_factor: Number of batches to prefetch
        mlflow_tracking: Whether to track metrics with MLflow
    """
    enable_mixed_precision: bool = True
    mixed_precision_dtype: str = "fp16"  # or "bf16"
    enable_gradient_checkpointing: bool = True
    enable_dynamic_batching: bool = True
    enable_smart_prefetching: bool = True
    enable_pruning: bool = False
    target_batch_size: int = 32
    min_batch_size: int = 4
    max_batch_size: int = 128
    memory_threshold: float = 0.85
    pruning_method: str = "magnitude"  # or "structured", "movement"
    pruning_amount: float = 0.3
    prefetch_factor: int = 2
    mlflow_tracking: bool = True

class MemoryTracker:
    """Tracks GPU and CPU memory usage."""
    
    @staticmethod
    def get_gpu_memory_info() -> Dict[str, float]:
        """Get GPU memory usage information."""
        try:
            gpu = GPUtil.getGPUs()[0]  # Assuming first GPU
            return {
                "gpu_memory_used": gpu.memoryUsed,
                "gpu_memory_total": gpu.memoryTotal,
                "gpu_memory_util": gpu.memoryUtil
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
            return {}

    @staticmethod
    def get_cpu_memory_info() -> Dict[str, float]:
        """Get CPU memory usage information."""
        process = psutil.Process()
        return {
            "cpu_memory_used": process.memory_info().rss / 1024**2,  # MB
            "cpu_memory_percent": process.memory_percent()
        }

class DynamicBatchSampler(BatchSampler):
    """Implements dynamic batch sizing based on memory usage."""
    
    def __init__(
        self,
        sampler,
        config: OptimizationConfig,
        memory_tracker: MemoryTracker
    ):
        self.sampler = sampler
        self.config = config
        self.memory_tracker = memory_tracker
        self.current_batch_size = config.target_batch_size
        
    def _adjust_batch_size(self):
        """Adjust batch size based on current memory usage."""
        gpu_info = self.memory_tracker.get_gpu_memory_info()
        if not gpu_info:
            return
            
        memory_util = gpu_info["gpu_memory_util"]
        
        if memory_util > self.config.memory_threshold:
            # Decrease batch size
            self.current_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * 0.8)
            )
        elif memory_util < self.config.memory_threshold * 0.7:
            # Increase batch size
            self.current_batch_size = min(
                self.config.max_batch_size,
                int(self.current_batch_size * 1.2)
            )
            
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) >= self.current_batch_size:
                yield batch
                batch = []
                self._adjust_batch_size()
        if batch:
            yield batch

class SmartPrefetcher:
    """Implements intelligent data prefetching."""
    
    def __init__(
        self,
        dataloader: DataLoader,
        config: OptimizationConfig
    ):
        self.dataloader = dataloader
        self.config = config
        self.prefetch_queue = deque(maxlen=config.prefetch_factor)
        self.stream = torch.cuda.Stream()
        
    def prefetch(self):
        """Prefetch next batch in separate CUDA stream."""
        try:
            next_batch = next(self.dataloader_iter)
        except StopIteration:
            self.prefetch_queue.append(None)
            return
            
        with torch.cuda.stream(self.stream):
            self.prefetch_queue.append({
                k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v
                for k, v in next_batch.items()
            })
            
    def __iter__(self):
        self.dataloader_iter = iter(self.dataloader)
        self.prefetch_queue.clear()
        
        # Initial prefetch
        for _ in range(self.config.prefetch_factor):
            self.prefetch()
            
        while self.prefetch_queue:
            if self.prefetch_queue[0] is None:
                break
                
            batch = self.prefetch_queue.popleft()
            self.prefetch()  # Prefetch next batch
            
            yield batch

class ModelPruner:
    """Implements various model pruning strategies."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        config: OptimizationConfig
    ):
        self.model = model
        self.config = config
        self.original_weights = {}
        self._store_original_weights()
        
    def _store_original_weights(self):
        """Store original model weights for reference."""
        for name, param in self.model.named_parameters():
            if "weight" in name:
                self.original_weights[name] = param.data.clone()
                
    def _magnitude_pruning(self, tensor: torch.Tensor) -> torch.Tensor:
        """Prune weights based on absolute magnitude."""
        threshold = torch.quantile(
            tensor.abs(),
            self.config.pruning_amount
        )
        mask = tensor.abs() > threshold
        return tensor * mask
        
    def _structured_pruning(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply structured pruning (e.g., entire neurons/filters)."""
        if len(tensor.shape) < 2:
            return tensor
            
        norms = torch.norm(tensor, dim=1)
        threshold = torch.quantile(norms, self.config.pruning_amount)
        mask = (norms > threshold).unsqueeze(1).expand_as(tensor)
        return tensor * mask
        
    def _movement_pruning(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Prune weights based on their movement from initialization."""
        if name not in self.original_weights:
            return tensor
            
        movement = (tensor - self.original_weights[name]).abs()
        threshold = torch.quantile(movement, self.config.pruning_amount)
        mask = movement > threshold
        return tensor * mask
        
    def prune(self):
        """Apply pruning to model weights."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "weight" in name:
                    if self.config.pruning_method == "magnitude":
                        param.data = self._magnitude_pruning(param.data)
                    elif self.config.pruning_method == "structured":
                        param.data = self._structured_pruning(param.data)
                    elif self.config.pruning_method == "movement":
                        param.data = self._movement_pruning(param.data, name)

class TrainingOptimizer:
    """Main class for managing training optimizations."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        config: OptimizationConfig
    ):
        self.model = model
        self.config = config
        self.memory_tracker = MemoryTracker()
        self.mlflow_tracker = MLflowTracker("training_optimizations") if config.mlflow_tracking else None
        
        if config.enable_mixed_precision:
            self.scaler = amp.GradScaler()
            
        if config.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        if config.enable_pruning:
            self.pruner = ModelPruner(model, config)
            
    def prepare_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True
    ) -> DataLoader:
        """Prepare optimized DataLoader."""
        sampler = torch.utils.data.RandomSampler(dataset) if shuffle else None
        
        if self.config.enable_dynamic_batching:
            batch_sampler = DynamicBatchSampler(
                sampler or range(len(dataset)),
                self.config,
                self.memory_tracker
            )
            dataloader = DataLoader(dataset, batch_sampler=batch_sampler)
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler
            )
            
        if self.config.enable_smart_prefetching:
            return SmartPrefetcher(dataloader, self.config)
        return dataloader
        
    def optimizer_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        forward_func: Callable
    ) -> Dict[str, float]:
        """Perform one optimization step with all enabled optimizations."""
        optimizer.zero_grad()
        
        if self.config.enable_mixed_precision:
            with amp.autocast(dtype=torch.float16 if self.config.mixed_precision_dtype == "fp16" else torch.bfloat16):
                loss = forward_func(batch)
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss = forward_func(batch)
            loss.backward()
            optimizer.step()
            
        metrics = {"loss": loss.item()}
        
        # Track memory usage
        metrics.update(self.memory_tracker.get_gpu_memory_info())
        metrics.update(self.memory_tracker.get_cpu_memory_info())
        
        if self.mlflow_tracker:
            self.mlflow_tracker.log_metrics(metrics)
            
        return metrics
        
    def apply_pruning(self):
        """Apply model pruning if enabled."""
        if self.config.enable_pruning:
            self.pruner.prune()
            
    def cleanup(self):
        """Perform cleanup operations."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        stats = {}
        stats.update(self.memory_tracker.get_gpu_memory_info())
        stats.update(self.memory_tracker.get_cpu_memory_info())
        return stats 