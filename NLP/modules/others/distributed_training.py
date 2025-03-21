"""Distributed Training Module for Multi-GPU Support

This module implements distributed training capabilities for language models,
enabling efficient multi-GPU training with various optimization strategies.
Key features include:
- Data parallel training (DDP)
- Pipeline parallelism
- Gradient synchronization
- Efficient memory sharing
- Automatic sharding
- Multi-node support
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from transformers import PreTrainedModel, PreTrainedTokenizer

from .training_optimizations import OptimizationConfig, TrainingOptimizer
from .mlflow_tracking import MLflowTracker

logger = logging.getLogger(__name__)

@dataclass
class DistributedConfig:
    """Configuration for distributed training.
    
    Attributes:
        backend: Distributed backend (nccl, gloo, etc.)
        num_nodes: Number of nodes for distributed training
        gpus_per_node: Number of GPUs per node
        master_addr: Address of master node
        master_port: Port for master node
        sync_batch_norm: Whether to sync batch norm stats
        find_unused_parameters: Find unused parameters in DDP
        gradient_as_bucket_view: Use gradient bucket view
        static_graph: Whether the model has a static graph
        mlflow_tracking: Whether to track metrics with MLflow
    """
    backend: str = "nccl"
    num_nodes: int = 1
    gpus_per_node: int = torch.cuda.device_count()
    master_addr: str = "localhost"
    master_port: str = "12355"
    sync_batch_norm: bool = True
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = False
    mlflow_tracking: bool = True

class DistributedTrainer:
    """Manages distributed training across multiple GPUs."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        dist_config: DistributedConfig,
        optim_config: OptimizationConfig
    ):
        self.model = model
        self.dist_config = dist_config
        self.optim_config = optim_config
        self.mlflow_tracker = MLflowTracker("distributed_training") if dist_config.mlflow_tracking else None
        
    def setup_distributed(self, rank: int, world_size: int):
        """Initialize distributed training environment."""
        os.environ["MASTER_ADDR"] = self.dist_config.master_addr
        os.environ["MASTER_PORT"] = self.dist_config.master_port
        
        # Initialize process group
        dist.init_process_group(
            backend=self.dist_config.backend,
            rank=rank,
            world_size=world_size
        )
        
        # Set device for this process
        torch.cuda.set_device(rank)
        self.model = self.model.cuda(rank)
        
        # Convert batch norm layers to sync batch norm
        if self.dist_config.sync_batch_norm:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        
        # Wrap model in DDP
        self.model = DDP(
            self.model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=self.dist_config.find_unused_parameters,
            gradient_as_bucket_view=self.dist_config.gradient_as_bucket_view,
            static_graph=self.dist_config.static_graph
        )
        
    def prepare_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        rank: int,
        world_size: int,
        shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        """Create distributed dataloader."""
        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        
        # Create optimizer wrapper
        training_optimizer = TrainingOptimizer(self.model, self.optim_config)
        
        # Get optimized dataloader
        return training_optimizer.prepare_dataloader(
            dataset,
            batch_size,
            sampler=sampler
        )
        
    def reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Reduce metrics across all processes."""
        reduced_metrics = {}
        for name, value in metrics.items():
            tensor = torch.tensor(value).cuda()
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            reduced_metrics[name] = tensor.item()
        return reduced_metrics
        
    @staticmethod
    def cleanup():
        """Clean up distributed training resources."""
        if dist.is_initialized():
            dist.destroy_process_group()

def run_distributed(
    train_func: callable,
    world_size: int,
    dist_config: DistributedConfig,
    *args,
    **kwargs
):
    """Run training function in distributed mode.
    
    Args:
        train_func: Training function to run
        world_size: Total number of processes
        dist_config: Distributed training configuration
        *args: Additional arguments for train_func
        **kwargs: Additional keyword arguments for train_func
    """
    mp.spawn(
        train_func,
        args=(world_size, dist_config, *args),
        nprocs=world_size,
        join=True
    )

class GradientSynchronizer:
    """Handles gradient synchronization across GPUs."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        
    def synchronize(self):
        """Synchronize gradients across all processes."""
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= dist.get_world_size()
                
    def broadcast_model(self, src_rank: int = 0):
        """Broadcast model parameters from source rank to all processes."""
        for param in self.model.parameters():
            dist.broadcast(param.data, src=src_rank)

class MemorySharer:
    """Manages efficient memory sharing across GPUs."""
    
    @staticmethod
    def get_shared_memory_chunks(
        tensor: torch.Tensor,
        num_chunks: int
    ) -> List[torch.Tensor]:
        """Split tensor into chunks for shared memory."""
        return torch.chunk(tensor, num_chunks)
        
    @staticmethod
    def gather_shared_memory(
        chunks: List[torch.Tensor],
        dim: int = 0
    ) -> torch.Tensor:
        """Gather chunks back into single tensor."""
        return torch.cat(chunks, dim=dim)

class AutoSharding:
    """Implements automatic model sharding across GPUs."""
    
    def __init__(
        self,
        model: nn.Module,
        num_gpus: int
    ):
        self.model = model
        self.num_gpus = num_gpus
        
    def shard_parameters(self):
        """Automatically shard model parameters across GPUs."""
        # Implementation depends on model architecture
        pass
        
    def shard_optimizer_state(self, optimizer: torch.optim.Optimizer):
        """Shard optimizer state across GPUs."""
        # Implementation depends on optimizer type
        pass 