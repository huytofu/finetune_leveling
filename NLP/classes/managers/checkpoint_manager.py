import os
import sys
import json
import torch
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer

# Add to Python path
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parentdir)

# Local imports
from modules.managers.config_manager import ConfigManager
from modules.monitoring_and_tracking.mlflow_tracking import MLflowTracker
from modules.customizations_and_optimizations.training_optimizations import TrainingOptimizer

logger = logging.getLogger(__name__)

class CheckpointManager:
    """
    Manages checkpoint saving, loading, and transferring for PEFT and non-PEFT models.
    This class provides a consistent interface for handling checkpoints across different trainer types.
    """
    
    def __init__(self):
        """Initialize the CheckpointManager."""
        self.supported_precision_types = ["fp16", "bf16", "fp32"]
    
    def save_checkpoint(
        self, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        optimizer: torch.optim.Optimizer = None,
        scheduler: Any = None,
        output_dir: str = "output",
        epoch: int = 0,
        metrics: Dict[str, float] = None,
        is_best: bool = False,
        precision: str = "fp32",
        save_optimizer_state: bool = True
    ) -> str:
        """
        Save model checkpoint with specialized handling for PEFT models.
        
        Args:
            model: The model to save
            tokenizer: The tokenizer to save
            optimizer: Optional optimizer to save state
            scheduler: Optional scheduler to save state
            output_dir: Directory to save checkpoint
            epoch: Current training epoch
            metrics: Optional dict of metrics to save with checkpoint
            is_best: Whether this is the best checkpoint so far
            precision: Precision format ("fp16", "bf16", or "fp32")
            save_optimizer_state: Whether to save optimizer and scheduler state
            
        Returns:
            Path to the saved checkpoint
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine if this is a PEFT model
        is_peft_model = self._is_peft_model(model)
        
        # Checkpoint metadata
        checkpoint_meta = {
            "epoch": epoch,
            "metrics": metrics or {},
            "is_peft": is_peft_model,
            "precision": precision,
            "model_type": getattr(model.config, "model_type", "unknown"),
            "timestamp": self._get_timestamp()
        }
        
        # Create checkpoint directory
        checkpoint_name = f"checkpoint-epoch-{epoch}"
        if is_best:
            checkpoint_name = "checkpoint-best"
        
        checkpoint_dir = os.path.join(output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        if is_peft_model:
            return self._save_peft_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_dir=checkpoint_dir,
                checkpoint_meta=checkpoint_meta,
                save_optimizer_state=save_optimizer_state
            )
        else:
            return self._save_standard_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_dir=checkpoint_dir,
                checkpoint_meta=checkpoint_meta,
                save_optimizer_state=save_optimizer_state
            )
    
    def _save_peft_checkpoint(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        checkpoint_dir: str,
        checkpoint_meta: Dict[str, Any],
        save_optimizer_state: bool
    ) -> str:
        """Save a PEFT model checkpoint with all required states."""
        try:
            from peft import PeftModel, get_peft_model_state_dict
            
            logger.info(f"Saving PEFT model to {checkpoint_dir}")
            
            # Save adapter
            model.save_pretrained(checkpoint_dir)
            
            # Save tokenizer
            tokenizer.save_pretrained(checkpoint_dir)
            
            # Get adapter config and save it separately for better compatibility
            if hasattr(model, "peft_config"):
                with open(os.path.join(checkpoint_dir, "adapter_config.json"), "w") as f:
                    # Convert to dict if it's not already
                    if not isinstance(model.peft_config, dict):
                        # If it's a list of configs (for multi-adapter)
                        if isinstance(model.peft_config, list):
                            configs = {}
                            for i, config in enumerate(model.peft_config):
                                configs[f"adapter_{i}"] = config.to_dict()
                            json.dump(configs, f, indent=2)
                        else:
                            # Single adapter config
                            json.dump(model.peft_config.to_dict(), f, indent=2)
                    else:
                        # Already a dict
                        json.dump(model.peft_config, f, indent=2)
            
            # Save optimizer and scheduler state if requested
            if save_optimizer_state and optimizer is not None:
                # Get optimizer state
                optimizer_state = optimizer.state_dict()
                torch.save(optimizer_state, os.path.join(checkpoint_dir, "optimizer.pt"))
                
                # Save scheduler if provided
                if scheduler is not None:
                    scheduler_state = scheduler.state_dict()
                    torch.save(scheduler_state, os.path.join(checkpoint_dir, "scheduler.pt"))
            
            # Save checkpoint metadata
            with open(os.path.join(checkpoint_dir, "checkpoint_meta.json"), "w") as f:
                json.dump(checkpoint_meta, f, indent=2)
            
            return checkpoint_dir
            
        except (ImportError, AttributeError) as e:
            logger.warning(f"Error saving PEFT checkpoint: {e}")
            # Fallback to standard checkpoint saving
            return self._save_standard_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_dir=checkpoint_dir,
                checkpoint_meta=checkpoint_meta,
                save_optimizer_state=save_optimizer_state
            )
    
    def _save_standard_checkpoint(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        checkpoint_dir: str,
        checkpoint_meta: Dict[str, Any],
        save_optimizer_state: bool
    ) -> str:
        """Save a standard model checkpoint with all required states."""
        logger.info(f"Saving standard model to {checkpoint_dir}")
        
        # Save model
        try:
            model.save_pretrained(checkpoint_dir)
        except (AttributeError, TypeError):
            # Fallback for models without save_pretrained
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
            if hasattr(model, "config"):
                model.config.save_pretrained(checkpoint_dir)
        
        # Save tokenizer
        tokenizer.save_pretrained(checkpoint_dir)
        
        # Save optimizer and scheduler state if requested
        if save_optimizer_state and optimizer is not None:
            # Get optimizer state
            optimizer_state = optimizer.state_dict()
            torch.save(optimizer_state, os.path.join(checkpoint_dir, "optimizer.pt"))
            
            # Save scheduler if provided
            if scheduler is not None:
                scheduler_state = scheduler.state_dict()
                torch.save(scheduler_state, os.path.join(checkpoint_dir, "scheduler.pt"))
        
        # Save checkpoint metadata
        with open(os.path.join(checkpoint_dir, "checkpoint_meta.json"), "w") as f:
            json.dump(checkpoint_meta, f, indent=2)
        
        return checkpoint_dir
    
    def load_checkpoint(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        checkpoint_dir: str,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Any = None,
        strict: bool = True,
        load_optimizer_state: bool = True,
        target_precision: str = None
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Dict[str, Any]]:
        """
        Load model and states from checkpoint with specialized handling for PEFT models.
        
        Args:
            model: The model to load checkpoint into
            tokenizer: The tokenizer to use
            checkpoint_dir: Directory containing checkpoint
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            strict: Whether to strictly enforce that the keys in state_dict match
            load_optimizer_state: Whether to load optimizer and scheduler state
            target_precision: Optional target precision to convert to (fp16, bf16, fp32)
            
        Returns:
            Tuple of (loaded_model, loaded_tokenizer, checkpoint_meta)
        """
        # Verify checkpoint exists
        if not os.path.exists(checkpoint_dir):
            raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Load checkpoint metadata
        meta_path = os.path.join(checkpoint_dir, "checkpoint_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                checkpoint_meta = json.load(f)
        else:
            checkpoint_meta = {"is_peft": self._is_peft_model(model)}
        
        # Determine if this is a PEFT checkpoint
        is_peft_checkpoint = checkpoint_meta.get("is_peft", False)
        
        # Load appropriate checkpoint type
        if is_peft_checkpoint:
            model, tokenizer = self._load_peft_checkpoint(
                model=model,
                tokenizer=tokenizer,
                checkpoint_dir=checkpoint_dir,
                optimizer=optimizer,
                scheduler=scheduler,
                strict=strict,
                load_optimizer_state=load_optimizer_state,
                target_precision=target_precision
            )
        else:
            model, tokenizer = self._load_standard_checkpoint(
                model=model,
                tokenizer=tokenizer,
                checkpoint_dir=checkpoint_dir,
                optimizer=optimizer,
                scheduler=scheduler,
                strict=strict,
                load_optimizer_state=load_optimizer_state,
                target_precision=target_precision
            )
        
        return model, tokenizer, checkpoint_meta
    
    def _load_peft_checkpoint(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        checkpoint_dir: str,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        strict: bool,
        load_optimizer_state: bool,
        target_precision: str
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load a PEFT model checkpoint with specialized handling."""
        try:
            from peft import PeftModel, PeftConfig
            
            logger.info(f"Loading PEFT model from {checkpoint_dir}")
            
            # Determine if model is already a PeftModel
            is_already_peft = self._is_peft_model(model)
            
            if is_already_peft:
                # If already a PEFT model, load just the adapter weights
                model.load_adapter(checkpoint_dir, adapter_name="default")
            else:
                # If not a PEFT model yet, load the adapter config and create a PEFT model
                peft_config_path = os.path.join(checkpoint_dir, "adapter_config.json")
                if os.path.exists(peft_config_path):
                    # Create a PEFT model using the saved config
                    peft_config = PeftConfig.from_pretrained(checkpoint_dir)
                    
                    # Import get_peft_model dynamically as we need it only in this branch
                    from peft import get_peft_model
                    model = get_peft_model(model, peft_config)
                    
                    # Load adapter weights
                    adapter_weights = os.path.join(checkpoint_dir, "adapter_model.bin")
                    if os.path.exists(adapter_weights):
                        adapter_state_dict = torch.load(adapter_weights, map_location="cpu")
                        model.load_state_dict(adapter_state_dict, strict=strict)
                else:
                    # Try to infer config from saved model
                    model = PeftModel.from_pretrained(model, checkpoint_dir)
            
            # Apply precision conversion if needed
            if target_precision and target_precision in self.supported_precision_types:
                model = self._convert_precision(model, target_precision)
            
            # Load optimizer and scheduler if requested
            if load_optimizer_state:
                self._load_optimizer_and_scheduler(
                    optimizer=optimizer,
                    scheduler=scheduler,
                    checkpoint_dir=checkpoint_dir
                )
            
            # Load tokenizer
            tokenizer_path = os.path.join(checkpoint_dir, "tokenizer")
            if os.path.exists(tokenizer_path):
                tokenizer = type(tokenizer).from_pretrained(tokenizer_path)
            
            return model, tokenizer
            
        except (ImportError, AttributeError, ValueError) as e:
            logger.warning(f"Error loading PEFT checkpoint: {e}. Falling back to standard loading.")
            return self._load_standard_checkpoint(
                model=model,
                tokenizer=tokenizer,
                checkpoint_dir=checkpoint_dir,
                optimizer=optimizer,
                scheduler=scheduler,
                strict=strict,
                load_optimizer_state=load_optimizer_state,
                target_precision=target_precision
            )
    
    def _load_standard_checkpoint(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        checkpoint_dir: str,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        strict: bool,
        load_optimizer_state: bool,
        target_precision: str
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load a standard model checkpoint."""
        logger.info(f"Loading standard model from {checkpoint_dir}")
        
        # Load model weights
        model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=strict)
        else:
            # Try to use from_pretrained
            try:
                model = type(model).from_pretrained(checkpoint_dir)
            except Exception as e:
                logger.error(f"Failed to load model weights: {e}")
        
        # Apply precision conversion if needed
        if target_precision and target_precision in self.supported_precision_types:
            model = self._convert_precision(model, target_precision)
        
        # Load optimizer and scheduler if requested
        if load_optimizer_state:
            self._load_optimizer_and_scheduler(
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_dir=checkpoint_dir
            )
        
        # Load tokenizer
        tokenizer_path = os.path.join(checkpoint_dir, "tokenizer")
        if os.path.exists(tokenizer_path):
            tokenizer = type(tokenizer).from_pretrained(tokenizer_path)
        
        return model, tokenizer
    
    def _load_optimizer_and_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        checkpoint_dir: str
    ) -> None:
        """Load optimizer and scheduler states from checkpoint."""
        if optimizer is not None:
            optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
            if os.path.exists(optimizer_path):
                try:
                    optimizer_state = torch.load(optimizer_path, map_location="cpu")
                    optimizer.load_state_dict(optimizer_state)
                    logger.info("Loaded optimizer state from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load optimizer state: {e}")
        
        if scheduler is not None:
            scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
            if os.path.exists(scheduler_path):
                try:
                    scheduler_state = torch.load(scheduler_path, map_location="cpu")
                    scheduler.load_state_dict(scheduler_state)
                    logger.info("Loaded scheduler state from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load scheduler state: {e}")
    
    def _is_peft_model(self, model: PreTrainedModel) -> bool:
        """Determine if a model is a PEFT model."""
        try:
            from peft import PeftModel
            return isinstance(model, PeftModel)
        except ImportError:
            # If peft is not installed, it can't be a PEFT model
            return False
    
    def _get_timestamp(self) -> str:
        """Get a timestamp string for checkpoint naming."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    def _convert_precision(
        self, 
        model: PreTrainedModel, 
        target_precision: str
    ) -> PreTrainedModel:
        """Convert model to target precision."""
        if target_precision not in self.supported_precision_types:
            raise ValueError(f"Unsupported precision type: {target_precision}. "
                            f"Supported types: {self.supported_precision_types}")
        
        if target_precision == "fp16":
            model = model.half()
        elif target_precision == "bf16":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                model = model.to(torch.bfloat16)
            else:
                logger.warning("BF16 not supported on this device, falling back to FP32")
        elif target_precision == "fp32":
            model = model.float()
        
        return model
            
    def transfer_adapter(
        self, 
        source_model: PreTrainedModel,
        target_model: PreTrainedModel,
        adapter_name: str = "default",
        target_adapter_name: str = None
    ) -> PreTrainedModel:
        """
        Transfer a PEFT adapter from one model to another.
        
        Args:
            source_model: The source model containing the adapter
            target_model: The target model to transfer the adapter to
            adapter_name: The name of the adapter in the source model
            target_adapter_name: Optional new name for the adapter in the target model
            
        Returns:
            The target model with the transferred adapter
        """
        try:
            from peft import PeftModel
            
            if not self._is_peft_model(source_model):
                raise ValueError("Source model is not a PEFT model")
            
            target_adapter_name = target_adapter_name or adapter_name
            
            # Get adapter config and state from source model
            if hasattr(source_model, "peft_config"):
                if isinstance(source_model.peft_config, dict):
                    adapter_config = source_model.peft_config.get(adapter_name)
                else:
                    adapter_config = source_model.peft_config
                
                # Get adapter state dict
                adapter_state = source_model.get_adapter_state_dict(adapter_name)
                
                # Check if target is already a PEFT model
                if self._is_peft_model(target_model):
                    # Add adapter to existing PEFT model
                    target_model.add_adapter(adapter_name=target_adapter_name, peft_config=adapter_config)
                    target_model.set_adapter_state_dict(adapter_state, adapter_name=target_adapter_name)
                else:
                    # Create new PEFT model
                    from peft import get_peft_model
                    target_model = get_peft_model(target_model, adapter_config)
                    target_model.load_state_dict(adapter_state, strict=False)
                
                logger.info(f"Transferred adapter '{adapter_name}' to target model as '{target_adapter_name}'")
                return target_model
            else:
                raise AttributeError("Source model does not have peft_config attribute")
                
        except (ImportError, AttributeError, ValueError) as e:
            logger.error(f"Failed to transfer adapter: {e}")
            return target_model
            
    def merge_and_unload(
        self, 
        model: PreTrainedModel, 
        adapter_name: str = "default",
        safe_merge: bool = True
    ) -> PreTrainedModel:
        """
        Merge PEFT adapter weights into base model and unload the adapter.
        
        Args:
            model: The PEFT model
            adapter_name: The name of the adapter to merge
            safe_merge: Whether to use safe merging (with validation)
            
        Returns:
            The model with merged weights
        """
        try:
            from peft import PeftModel
            
            if not self._is_peft_model(model):
                logger.warning("Model is not a PEFT model, returning as is")
                return model
            
            logger.info(f"Merging adapter '{adapter_name}' into base model")
            
            # Check if model has the required methods
            if hasattr(model, "merge_and_unload"):
                # New PEFT versions have this method
                return model.merge_and_unload(safe_merge=safe_merge)
            elif hasattr(model, "merge_adapter"):
                # Older PEFT versions
                model.merge_adapter(adapter_name)
                return model.base_model
            else:
                logger.warning("Model does not support merging adapters, returning as is")
                return model
                
        except Exception as e:
            logger.error(f"Failed to merge adapter: {e}")
            return model 