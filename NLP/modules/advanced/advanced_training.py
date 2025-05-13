"""
Advanced training integrations for multi-node training, hardware optimizations, and RLHF.

This module provides integrations with:
1. Custom FSDP Implementation - for multi-node distributed training
2. Unsloth - for hardware-specific kernel optimizations
3. TRL - for Reinforcement Learning from Human Feedback (RLHF)
"""

import os
import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple, Callable
import warnings

logger = logging.getLogger(__name__)

# Attempt to import optional dependencies
try:
    import unsloth
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False
    logger.info("Unsloth not installed. To enable hardware optimizations: pip install unsloth")

try:
    import trl
    from trl import PPOConfig, PPOTrainer, DPOTrainer, RewardConfig, RewardTrainer
    HAS_TRL = True
except ImportError:
    HAS_TRL = False
    logger.info("TRL not installed. To enable RLHF: pip install trl")

@dataclass
class AdvancedTrainingConfig:
    """Configuration for advanced training features."""
    
    # Multi-node training (Custom FSDP)
    use_multi_node: bool = False
    use_fsdp: bool = False
    num_nodes: int = 1
    node_rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "29500"
    fsdp_sharding_strategy: str = "full"  # Options: "full", "hybrid", "zero2"
    fsdp_cpu_offload: bool = False
    fsdp_backward_prefetch: bool = True
    fsdp_min_module_size: int = 1e6  # Minimum module size for auto-wrapping
    
    # Hardware optimization (Unsloth)
    use_unsloth: bool = False
    unsloth_max_seq_length: int = 2048
    unsloth_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # RLHF (TRL)
    use_rlhf: bool = False
    rlhf_method: str = "ppo"  # Options: "ppo", "dpo", "orpo"
    reward_model_name: Optional[str] = None
    num_ppo_epochs: int = 1
    kl_penalty_coefficient: float = 0.1
    beta: float = 0.1  # DPO specific
    prompt_max_length: int = 512
    reward_model_ckpt: Optional[str] = None


class MultiNodeManager:
    """Custom implementation of multi-node distributed training."""
    
    def __init__(self, config: AdvancedTrainingConfig):
        self.config = config
        self.is_distributed = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.device = None
        
    def setup_distributed_environment(self):
        """Configure distributed environment for multi-node training."""
        if not self.config.use_multi_node:
            logger.info("Multi-node training not enabled")
            return False
        
        # Initialize environment variables for distributed training
        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = self.config.master_port
        os.environ["RANK"] = str(self.config.node_rank)
        os.environ["WORLD_SIZE"] = str(self.config.num_nodes)
        os.environ["LOCAL_RANK"] = str(self.config.local_rank)
        
        # Initialize PyTorch distributed
        try:
            # If CUDA is available, use NCCL backend, otherwise use gloo
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            
            # Initialize the process group
            dist.init_process_group(
                backend=backend,
                init_method=f"tcp://{self.config.master_addr}:{self.config.master_port}",
                world_size=self.config.num_nodes,
                rank=self.config.node_rank,
            )
            
            # Set the device
            if torch.cuda.is_available():
                self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
            else:
                self.device = torch.device("cpu")
            
            # Update attributes
            self.is_distributed = True
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            
            logger.info(f"Initialized PyTorch distributed process group "
                       f"(rank {self.rank}/{self.world_size}, "
                       f"local_rank: {self.local_rank}, "
                       f"device: {self.device})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed environment: {str(e)}")
            return False
    
    def get_sampler(self, dataset):
        """Get appropriate sampler for distributed training."""
        if not self.is_distributed:
            return None
        
        from torch.utils.data import DistributedSampler
        return DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True
        )
    
    def prepare_model_for_ddp(self, model):
        """Prepare model for DDP training."""
        if not self.is_distributed:
            return model
        
        # Ensure model is on correct device
        model = model.to(self.device)
        
        # Wrap with DDP
        ddp_model = DDP(
            model,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            output_device=self.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False
        )
        
        logger.info("Model prepared for DDP training")
        return ddp_model
    
    def prepare_model_for_fsdp(self, model, transformer_layer_cls=None):
        """Prepare model for FSDP training."""
        if not self.is_distributed or not self.config.use_fsdp:
            return model
        
        # Ensure model is on correct device
        model = model.to(self.device)
        
        # Define wrapping policy
        if transformer_layer_cls is not None:
            # Create an auto wrap policy for transformer layers
            auto_wrap_policy = transformer_auto_wrap_policy(
                transformer_layer_cls=transformer_layer_cls
            )
        else:
            # Use size-based policy
            auto_wrap_policy = size_based_auto_wrap_policy(
                min_num_params=self.config.fsdp_min_module_size
            )
        
        # Configure FSDP settings
        sharding_strategy_map = {
            "full": ShardingStrategy.FULL_SHARD,
            "hybrid": ShardingStrategy.HYBRID_SHARD,
            "zero2": ShardingStrategy.SHARD_GRAD_OP
        }
        sharding_strategy = sharding_strategy_map.get(
            self.config.fsdp_sharding_strategy.lower(), 
            ShardingStrategy.FULL_SHARD
        )
        
        # Configure mixed precision
        mp_policy = None
        if hasattr(torch, "bfloat16") and torch.cuda.is_available():
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        
        # Configure CPU offload if enabled
        cpu_offload = None
        if self.config.fsdp_cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)
        
        # Configure backward prefetch
        backward_prefetch = None
        if self.config.fsdp_backward_prefetch:
            backward_prefetch = BackwardPrefetch.BACKWARD_PRE
        
        # Wrap model with FSDP
        fsdp_model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            cpu_offload=cpu_offload,
            backward_prefetch=backward_prefetch,
            device_id=self.device,
        )
        
        logger.info(f"Model prepared for FSDP training with strategy: {self.config.fsdp_sharding_strategy}")
        return fsdp_model
    
    def cleanup(self):
        """Clean up distributed environment."""
        if self.is_distributed:
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed")


class UnslothIntegration:
    """Integration with Unsloth for hardware-specific kernel optimizations."""
    
    def __init__(self, config: AdvancedTrainingConfig):
        self.config = config
        if not HAS_UNSLOTH:
            raise ImportError("Unsloth is required for hardware optimizations. Install with: pip install unsloth")
    
    def optimize_model(self, model_name, peft_config=None):
        """Load and optimize model using Unsloth's optimizations."""
        if not self.config.use_unsloth:
            return None
        
        logger.info(f"Loading model with Unsloth optimizations: {model_name}")
        
        # Configure Unsloth model loading
        model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.config.unsloth_max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=peft_config.quantization_bits == 4 if hasattr(peft_config, "quantization_bits") else False,
            load_in_8bit=peft_config.quantization_bits == 8 if hasattr(peft_config, "quantization_bits") else False,
        )
        
        # Add LoRA if PEFT is requested
        if peft_config and hasattr(peft_config, "use_peft") and peft_config.use_peft:
            logger.info("Applying PEFT configuration with Unsloth")
            
            # Check what PEFT method is being used
            if hasattr(peft_config, "peft_method") and peft_config.peft_method.lower() == "lora":
                model = unsloth.FastLanguageModel.get_peft_model(
                    model,
                    r=peft_config.lora_r if hasattr(peft_config, "lora_r") else 8,
                    target_modules=self.config.unsloth_target_modules,
                    lora_alpha=peft_config.lora_alpha if hasattr(peft_config, "lora_alpha") else 16,
                    lora_dropout=peft_config.lora_dropout if hasattr(peft_config, "lora_dropout") else 0.05,
                )
                logger.info(f"Applied LoRA with r={peft_config.lora_r if hasattr(peft_config, 'lora_r') else 8}")
            else:
                logger.warning("Unsloth only supports LoRA PEFT method. Other methods will be ignored.")
        
        return model, tokenizer
    
    @staticmethod
    def is_model_supported(model_name):
        """Check if the model is supported by Unsloth optimizations."""
        supported_models = ["llama", "mistral", "phi", "falcon", "bloom"]
        return any(model in model_name.lower() for model in supported_models)


class RLHFIntegration:
    """Integration with TRL for Reinforcement Learning from Human Feedback (RLHF)."""
    
    def __init__(self, config: AdvancedTrainingConfig):
        self.config = config
        if not HAS_TRL:
            raise ImportError("TRL is required for RLHF. Install with: pip install trl")
    
    def train_reward_model(self, model, tokenizer, dataset):
        """Train a reward model using TRL's RewardTrainer."""
        if not self.config.use_rlhf:
            return model
        
        logger.info("Initializing reward model training with TRL")
        
        # Configure reward training
        reward_config = RewardConfig(
            output_dir=f"reward_model_{self.config.reward_model_name.split('/')[-1] if self.config.reward_model_name else 'custom'}",
            learning_rate=1e-5,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            max_length=self.config.prompt_max_length,
            num_train_epochs=1,
            gradient_checkpointing=True,
        )
        
        # Create reward model (value head)
        from trl import AutoModelForCausalLMWithValueHead
        reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.reward_model_name or model.config._name_or_path,
            device_map="auto",
        )
        
        # Initialize reward trainer
        trainer = RewardTrainer(
            model=reward_model,
            args=reward_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        
        # Train reward model
        trainer.train()
        
        # Save trained reward model
        if self.config.reward_model_ckpt:
            trainer.model.save_pretrained(self.config.reward_model_ckpt)
            logger.info(f"Saved reward model to {self.config.reward_model_ckpt}")
        
        return trainer.model
    
    def train_with_ppo(self, model, tokenizer, reward_model, prompt_dataset):
        """Train model with PPO using TRL's PPOTrainer."""
        if not self.config.use_rlhf or self.config.rlhf_method != "ppo":
            return model
        
        logger.info("Initializing PPO training with TRL")
        
        # Configure PPO
        ppo_config = PPOConfig(
            learning_rate=1e-5,
            batch_size=8,
            mini_batch_size=1,
            gradient_accumulation_steps=4,
            optimize_cuda_cache=True,
            early_stopping=True,
            target_kl=self.config.kl_penalty_coefficient,
            kl_penalty="kl",
            seed=42,
            use_score_scaling=True,
            use_score_norm=True,
        )
        
        # Initialize PPO trainer
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            ref_model=None,  # Use default reference model
            tokenizer=tokenizer,
            reward_model=reward_model,
        )
        
        # Train with PPO
        for epoch in range(self.config.num_ppo_epochs):
            logger.info(f"Starting PPO epoch {epoch+1}/{self.config.num_ppo_epochs}")
            for batch in prompt_dataset:
                # Generate responses
                query_tensors = [tokenizer.encode(prompt) for prompt in batch["prompt"]]
                response_tensors = ppo_trainer.generate(query_tensors)
                
                # Compute rewards
                rewards = ppo_trainer.compute_rewards(query_tensors, response_tensors)
                
                # Run PPO step
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                
                # Log stats
                logger.info(f"PPO step stats: {stats}")
        
        return ppo_trainer.model
    
    def train_with_dpo(self, model, tokenizer, dataset):
        """Train model with DPO using TRL's DPOTrainer."""
        if not self.config.use_rlhf or self.config.rlhf_method != "dpo":
            return model
        
        logger.info("Initializing DPO training with TRL")
        
        # Configure DPO training
        from transformers import TrainingArguments
        
        training_args = TrainingArguments(
            output_dir=f"dpo_model_{model.config._name_or_path.split('/')[-1]}",
            learning_rate=5e-6,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            max_steps=1000,
            gradient_checkpointing=True,
            fp16=torch.cuda.is_available(),
            save_total_limit=3,
            logging_steps=10,
            optim="adamw_torch",
        )
        
        # Initialize DPO trainer
        dpo_trainer = DPOTrainer(
            model=model,
            args=training_args,
            beta=self.config.beta,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        
        # Train with DPO
        dpo_trainer.train()
        
        # Save trained model
        dpo_trainer.save_model()
        
        return dpo_trainer.model


class AdvancedTrainingManager:
    """Manager for advanced training features."""
    
    def __init__(self, config: AdvancedTrainingConfig):
        self.config = config
        
        # Initialize managers for different features
        if config.use_multi_node:
            self.multi_node = MultiNodeManager(config)
        else:
            self.multi_node = None
        
        if config.use_unsloth:
            try:
                self.unsloth = UnslothIntegration(config)
            except ImportError:
                logger.warning("Unsloth not available. Hardware optimizations disabled.")
                self.unsloth = None
                config.use_unsloth = False
        else:
            self.unsloth = None
        
        if config.use_rlhf:
            try:
                self.rlhf = RLHFIntegration(config)
            except ImportError:
                logger.warning("TRL not available. RLHF disabled.")
                self.rlhf = None
                config.use_rlhf = False
        else:
            self.rlhf = None
    
    def setup_distributed(self):
        """Set up distributed training environment."""
        if self.multi_node:
            return self.multi_node.setup_distributed_environment()
        return False
    
    def get_optimized_model(self, model_name, peft_config=None):
        """Get model with hardware-specific optimizations."""
        if self.unsloth and UnslothIntegration.is_model_supported(model_name):
            try:
                return self.unsloth.optimize_model(model_name, peft_config)
            except Exception as e:
                logger.warning(f"Failed to apply Unsloth optimizations: {str(e)}")
                logger.warning("Falling back to standard model loading")
        return None
    
    def prepare_model_for_distributed(self, model, transformer_layer_cls=None):
        """Prepare model for distributed training."""
        if not self.multi_node:
            return model
        
        if self.config.use_fsdp:
            return self.multi_node.prepare_model_for_fsdp(model, transformer_layer_cls)
        else:
            return self.multi_node.prepare_model_for_ddp(model)
    
    def get_data_sampler(self, dataset):
        """Get data sampler for distributed training."""
        if self.multi_node:
            return self.multi_node.get_sampler(dataset)
        return None
    
    def train_with_rlhf(self, model, tokenizer, train_dataset, eval_dataset=None, reward_model=None):
        """Train model with RLHF."""
        if not self.rlhf:
            return model
        
        # Train or load reward model if needed
        if self.config.rlhf_method == "ppo" and reward_model is None:
            logger.info("Training reward model for PPO")
            # We need paired data for reward model training
            if isinstance(train_dataset, dict) and "reward" in train_dataset:
                reward_dataset = train_dataset["reward"]
                reward_model = self.rlhf.train_reward_model(model, tokenizer, reward_dataset)
            else:
                logger.warning("No reward dataset provided. Cannot train reward model for PPO.")
                return model
        
        # Train with the appropriate RLHF method
        if self.config.rlhf_method == "ppo":
            # We need prompts for PPO
            if isinstance(train_dataset, dict) and "prompt" in train_dataset:
                prompt_dataset = train_dataset["prompt"]
                return self.rlhf.train_with_ppo(model, tokenizer, reward_model, prompt_dataset)
            else:
                logger.warning("No prompt dataset provided. Cannot train with PPO.")
                return model
        elif self.config.rlhf_method == "dpo":
            # We need preference pairs for DPO
            if isinstance(train_dataset, dict) and "preference" in train_dataset:
                preference_dataset = train_dataset["preference"]
                return self.rlhf.train_with_dpo(model, tokenizer, preference_dataset)
            else:
                logger.warning("No preference dataset provided. Cannot train with DPO.")
                return model
        
        return model
    
    def cleanup(self):
        """Clean up resources."""
        if self.multi_node:
            self.multi_node.cleanup()


def validate_advanced_config(config):
    """Validate advanced training configuration for compatibility."""
    if not hasattr(config, "use_rlhf"):
        return True
    
    # Check for RLHF + Lightning/Accelerate incompatibility
    if config.use_rlhf:
        if hasattr(config, "use_lightning") and config.use_lightning:
            logger.warning("RLHF is not compatible with PyTorch Lightning. Disabling RLHF.")
            config.use_rlhf = False
            return False
        
        if hasattr(config, "use_accelerate") and config.use_accelerate:
            logger.warning("RLHF is not compatible with Accelerate. Disabling RLHF.")
            config.use_rlhf = False
            return False
    
    # Check for Unsloth compatibility
    if hasattr(config, "use_unsloth") and config.use_unsloth:
        if not UnslothIntegration.is_model_supported(config.model_name):
            logger.warning(f"Model {config.model_name} is not supported by Unsloth. Disabling Unsloth optimizations.")
            config.use_unsloth = False
    
    return True 