from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import os
import logging
from enum import Enum

class DistributedBackend(Enum):
    DDP = "ddp"
    DEEPSPEED = "deepspeed"
    FSDP = "fsdp"

@dataclass
class DistributedConfig:
    backend: DistributedBackend = DistributedBackend.DDP
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    timeout: int = 1800
    init_method: str = "env://"
    gradient_sync_device: str = "cuda"
    gradient_sync_bucket_size: int = 25 * 1024 * 1024  # 25MB

class DistributedManager:
    """Manages distributed training setup and configuration."""
    
    def __init__(
        self,
        config: Optional[DistributedConfig] = None,
        specs: Optional[Dict[str, Any]] = None
    ):
        self.config = config or DistributedConfig()
        self.specs = specs or {}
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for distributed training."""
        if self.config.rank == 0:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)
            
    def initialize(self) -> None:
        """Initialize distributed training environment."""
        try:
            if self.config.world_size > 1:
                self._setup_distributed_environment()
                self._setup_process_groups()
                self._setup_device()
                self.logger.info(f"Distributed training initialized on rank {self.config.rank}")
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed training: {str(e)}")
            raise
            
    def _setup_distributed_environment(self) -> None:
        """Setup distributed training environment variables."""
        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = self.config.master_port
        os.environ["WORLD_SIZE"] = str(self.config.world_size)
        os.environ["RANK"] = str(self.config.rank)
        os.environ["LOCAL_RANK"] = str(self.config.local_rank)
        
    def _setup_process_groups(self) -> None:
        """Setup process groups for distributed training."""
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.config.backend.value,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank,
                timeout=torch.timedelta(seconds=self.config.timeout)
            )
            
    def _setup_device(self) -> None:
        """Setup device for distributed training."""
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.local_rank)
            self.device = torch.device(f"cuda:{self.config.local_rank}")
        else:
            self.device = torch.device("cpu")
            
    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model for distributed training."""
        if self.config.world_size > 1:
            if self.config.backend == DistributedBackend.DDP:
                model = DDP(
                    model,
                    device_ids=[self.config.local_rank],
                    output_device=self.config.local_rank,
                    find_unused_parameters=True,
                    gradient_sync_device=self.config.gradient_sync_device,
                    gradient_sync_bucket_size=self.config.gradient_sync_bucket_size
                )
            elif self.config.backend == DistributedBackend.DEEPSPEED:
                # DeepSpeed integration would go here
                pass
            elif self.config.backend == DistributedBackend.FSDP:
                # FSDP integration would go here
                pass
        return model
        
    def get_sampler(self, dataset: torch.utils.data.Dataset) -> DistributedSampler:
        """Get distributed sampler for dataset."""
        if self.config.world_size > 1:
            return DistributedSampler(
                dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=True
            )
        return None
        
    def get_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        **kwargs
    ) -> DataLoader:
        """Get distributed dataloader."""
        sampler = self.get_sampler(dataset)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            **kwargs
        )
        
    def cleanup(self) -> None:
        """Cleanup distributed training environment."""
        if dist.is_initialized():
            dist.destroy_process_group()
            
    def is_main_process(self) -> bool:
        """Check if current process is main process."""
        return self.config.rank == 0
        
    def barrier(self) -> None:
        """Synchronize all processes."""
        if dist.is_initialized():
            dist.barrier()
            
    def broadcast(self, tensor: torch.Tensor) -> None:
        """Broadcast tensor to all processes."""
        if dist.is_initialized():
            dist.broadcast(tensor, src=0)
            
    def all_reduce(self, tensor: torch.Tensor) -> None:
        """All-reduce operation across all processes."""
        if dist.is_initialized():
            dist.all_reduce(tensor) 