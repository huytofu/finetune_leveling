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
- Hardware-aware automatic optimization selection
"""

import logging
import gc
import time
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
from datetime import datetime
import platform

from .mlflow_tracking import MLflowTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_optimizations.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class HardwareSpecs:
    """Hardware specifications for optimization decisions."""
    gpu_name: Optional[str] = None
    gpu_memory: Optional[int] = None  # in GB
    gpu_count: int = 0
    cpu_count: int = 0
    total_ram: int = 0  # in GB
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    compute_capability: Optional[float] = None
    is_ampere_or_newer: bool = False
    is_ampere: bool = False
    is_hopper: bool = False
    is_ada_lovelace: bool = False

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
        hardware_specs: Hardware specifications for optimization decisions
        auto_optimize: Whether to automatically select optimizations based on hardware
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
    hardware_specs: Optional[HardwareSpecs] = None
    auto_optimize: bool = True

    def __post_init__(self):
        if self.auto_optimize:
            self._auto_configure_optimizations()

    def _auto_configure_optimizations(self):
        """Automatically configure optimizations based on hardware specs."""
        if not self.hardware_specs:
            self.hardware_specs = self._detect_hardware()
        
        specs = self.hardware_specs
        
        # Memory Efficiency Optimizations
        if specs.gpu_memory and specs.gpu_memory < 32:
            self.enable_gradient_checkpointing = True
            self.enable_dynamic_batching = True
            self.target_batch_size = min(16, self.target_batch_size)
            self.max_batch_size = min(64, self.max_batch_size)
        
        # Parallelism Optimizations
        if specs.gpu_count > 1:
            self.enable_smart_prefetching = True
            self.prefetch_factor = max(2, specs.gpu_count)
        
        # Data Optimization
        if specs.total_ram < 64:
            self.enable_dynamic_batching = True
            self.memory_threshold = 0.75  # More conservative threshold
        
        # Computation Optimization
        if specs.is_ampere_or_newer:
            self.enable_mixed_precision = True
            self.mixed_precision_dtype = "bf16" if specs.is_hopper else "fp16"
    
    @staticmethod
    def _detect_hardware() -> HardwareSpecs:
        """Detect hardware specifications."""
        specs = HardwareSpecs()
        
        # CPU and RAM info
        specs.cpu_count = psutil.cpu_count()
        specs.total_ram = psutil.virtual_memory().total // (1024**3)  # Convert to GB
        
        # GPU info
        specs.cuda_available = torch.cuda.is_available()
        if specs.cuda_available:
            specs.gpu_count = torch.cuda.device_count()
            specs.gpu_name = torch.cuda.get_device_name(0)
            specs.gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)  # Convert to GB
            specs.cuda_version = torch.version.cuda
            specs.compute_capability = torch.cuda.get_device_capability(0)[0] + torch.cuda.get_device_capability(0)[1] / 10
            
            # Check GPU architecture
            specs.is_ampere_or_newer = specs.compute_capability >= 8.0
            specs.is_ampere = 8.0 <= specs.compute_capability < 9.0
            specs.is_hopper = 9.0 <= specs.compute_capability < 10.0
            specs.is_ada_lovelace = specs.compute_capability >= 10.0
        
        return specs

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

class OptimizationMetrics:
    """Metrics for tracking optimization effectiveness."""
    def __init__(self):
        self.gpu_memory_usage: float = 0.0
        self.cpu_memory_usage: float = 0.0
        self.training_speed: float = 0.0
        self.batch_processing_time: float = 0.0
        self.optimization_effectiveness: float = 0.0
        self.timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            'gpu_memory_usage': self.gpu_memory_usage,
            'cpu_memory_usage': self.cpu_memory_usage,
            'training_speed': self.training_speed,
            'batch_processing_time': self.batch_processing_time,
            'optimization_effectiveness': self.optimization_effectiveness,
            'timestamp': self.timestamp
        }

class MetricsTracker:
    """Tracks and logs optimization metrics."""
    def __init__(self):
        self.metrics_history: List[OptimizationMetrics] = []
        self.start_time = time.time()
        self.last_batch_time = time.time()

    def update_metrics(self, metrics: OptimizationMetrics):
        """Update and log metrics."""
        self.metrics_history.append(metrics)
        logger.info(f"Optimization Metrics: {metrics.to_dict()}")
        
        # Log significant changes
        if len(self.metrics_history) > 1:
            prev_metrics = self.metrics_history[-2]
            if abs(metrics.gpu_memory_usage - prev_metrics.gpu_memory_usage) > 0.1:
                logger.warning(f"Significant GPU memory change: {prev_metrics.gpu_memory_usage:.2f} -> {metrics.gpu_memory_usage:.2f}")

    def get_metrics_summary(self) -> Dict:
        """Get summary of metrics history."""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        return {
            'current_metrics': latest.to_dict(),
            'history_length': len(self.metrics_history),
            'training_duration': time.time() - self.start_time
        }

    def log_batch_completion(self, batch_size: int):
        """Log batch processing completion."""
        current_time = time.time()
        batch_time = current_time - self.last_batch_time
        self.last_batch_time = current_time
        
        logger.info(f"Batch completed: size={batch_size}, time={batch_time:.2f}s")

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
        self.metrics_tracker = MetricsTracker()
        
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

    def get_optimization_metrics(self) -> OptimizationMetrics:
        """Get current optimization metrics."""
        metrics = OptimizationMetrics(
            gpu_memory_usage=self._get_gpu_memory_usage(),
            cpu_memory_usage=self._get_cpu_memory_usage(),
            training_speed=self._calculate_training_speed(),
            batch_processing_time=self._get_batch_processing_time(),
            optimization_effectiveness=self._calculate_optimization_effectiveness()
        )
        self.metrics_tracker.update_metrics(metrics)
        return metrics

    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage percentage."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
        return 0.0

    def _get_cpu_memory_usage(self) -> float:
        """Get current CPU memory usage percentage."""
        return psutil.Process().memory_percent()

    def _calculate_training_speed(self) -> float:
        """Calculate current training speed in samples per second."""
        if not self.metrics_tracker.metrics_history:
            return 0.0
        return self.metrics_tracker.metrics_history[-1].training_speed

    def _get_batch_processing_time(self) -> float:
        """Get average batch processing time."""
        if not self.metrics_tracker.metrics_history:
            return 0.0
        return self.metrics_tracker.metrics_history[-1].batch_processing_time

    def _calculate_optimization_effectiveness(self) -> float:
        """Calculate optimization effectiveness score."""
        # Implementation depends on specific metrics
        return 0.0  # Placeholder

class HardwareAwareOptimizer(TrainingOptimizer):
    """Enhanced training optimizer with hardware-aware optimizations."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        config: OptimizationConfig
    ):
        super().__init__(model, config)
        self.hardware_specs = config.hardware_specs or OptimizationConfig._detect_hardware()
        self._apply_hardware_optimizations()
        logger.info("HardwareAwareOptimizer initialized with optimizations")
    
    def _apply_hardware_optimizations(self):
        """Apply hardware-specific optimizations."""
        specs = self.hardware_specs
        
        # Memory Efficiency
        if specs.gpu_memory and specs.gpu_memory < 32:
            self._apply_memory_efficiency_optimizations()
        
        # Parallelism
        if specs.gpu_count > 1:
            self._apply_parallelism_optimizations()
        
        # Data Optimization
        if specs.total_ram < 64:
            self._apply_data_optimizations()
        
        # Computation Optimization
        if specs.is_ampere_or_newer:
            self._apply_computation_optimizations()
    
    def _apply_memory_efficiency_optimizations(self):
        """Apply memory efficiency optimizations."""
        specs = self.hardware_specs
        if specs.gpu_memory and specs.gpu_memory < 32:
            # Enable gradient checkpointing
            self.model.gradient_checkpointing_enable()
            
            # Adjust batch sizes
            self.config.target_batch_size = min(16, self.config.target_batch_size)
            self.config.max_batch_size = min(64, self.config.max_batch_size)
            
            # Enable dynamic batching
            self.config.enable_dynamic_batching = True
    
    def _apply_parallelism_optimizations(self):
        """Apply parallelism optimizations."""
        specs = self.hardware_specs
        if specs.gpu_count > 1:
            # Enable smart prefetching
            self.config.enable_smart_prefetching = True
            self.config.prefetch_factor = max(2, specs.gpu_count)
            
            # Enable data parallel
            self.model = nn.DataParallel(self.model)
    
    def _apply_data_optimizations(self):
        """Apply data optimization techniques."""
        specs = self.hardware_specs
        if specs.total_ram < 64:
            # Enable dynamic batching
            self.config.enable_dynamic_batching = True
            self.config.memory_threshold = 0.75
            
            # Enable memory mapping
            self.config.enable_smart_prefetching = True
    
    def _apply_computation_optimizations(self):
        """Apply computation optimizations."""
        specs = self.hardware_specs
        if specs.is_ampere_or_newer:
            # Enable mixed precision
            self.config.enable_mixed_precision = True
            self.config.mixed_precision_dtype = "bf16" if specs.is_hopper else "fp16"
            
            # Enable flash attention if available
            if hasattr(self.model, "enable_flash_attention"):
                self.model.enable_flash_attention()
    
    def get_optimization_summary(self) -> str:
        """Get a human-readable summary of applied optimizations."""
        specs = self.hardware_specs
        summary = [
            "Hardware-Aware Optimization Summary",
            "=" * 40,
            f"Hardware Specifications:",
            f"- GPU: {specs.gpu_name}",
            f"- Memory: {specs.gpu_memory}GB",
            f"- CUDA: {specs.cuda_version}",
            "",
            "Applied Optimizations:",
            f"- Memory Efficiency: {self._get_memory_efficiency_summary()}",
            f"- Parallelism: {self._get_parallelism_summary()}",
            f"- Data Optimization: {self._get_data_optimization_summary()}",
            f"- Computation: {self._get_computation_summary()}",
            "",
            "Current Metrics:",
            f"- GPU Memory Usage: {self._get_gpu_memory_usage():.2f}%",
            f"- Training Speed: {self._calculate_training_speed():.2f} samples/s",
            f"- Optimization Effectiveness: {self._calculate_optimization_effectiveness():.2f}%"
        ]
        return "\n".join(summary)

    def _get_memory_efficiency_summary(self) -> str:
        """Get memory efficiency optimization summary."""
        return f"Enabled (Threshold: {self.config.memory_threshold})"

    def _get_parallelism_summary(self) -> str:
        """Get parallelism optimization summary."""
        return f"Enabled (GPUs: {self.config.hardware_specs.gpu_count})"

    def _get_data_optimization_summary(self) -> str:
        """Get data optimization summary."""
        return f"Enabled (Batch Size: {self.config.target_batch_size})"

    def _get_computation_summary(self) -> str:
        """Get computation optimization summary."""
        return f"Enabled (Mixed Precision: {self.config.mixed_precision_dtype})" 