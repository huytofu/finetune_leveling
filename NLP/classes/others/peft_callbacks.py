import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Union
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
import logging

logger = logging.getLogger(__name__)

class PeftAdapterMonitorCallback(Callback):
    """
    A PyTorch Lightning callback that monitors the training of PEFT adapters.
    
    This callback tracks adapter parameters, gradients, and training dynamics
    to provide insights into parameter-efficient fine-tuning.
    """
    
    def __init__(
        self,
        monitor_gradients: bool = True,
        monitor_weights: bool = True,
        log_every_n_steps: int = 100,
        save_path: Optional[str] = None,
        peft_type_monitoring: Dict[str, bool] = None
    ):
        """
        Initialize the PEFT adapter monitoring callback.
        
        Args:
            monitor_gradients: Whether to monitor gradients of PEFT parameters
            monitor_weights: Whether to monitor weight values of PEFT parameters
            log_every_n_steps: How often to log monitored values
            save_path: Directory to save visualizations
            peft_type_monitoring: Dictionary of PEFT types to enable specific monitoring for
        """
        super().__init__()
        self.monitor_gradients = monitor_gradients
        self.monitor_weights = monitor_weights
        self.log_every_n_steps = log_every_n_steps
        self.save_path = save_path
        
        # Default monitoring configurations for different PEFT types
        self.peft_type_monitoring = peft_type_monitoring or {
            "LORA": True,
            "PREFIX": True,
            "PROMPT": True,
            "P_TUNING": True,
            "IA3": True
        }
        
        # Initialize trackers
        self.gradient_norms = {}
        self.weight_norms = {}
        self.adapter_params_count = {}
        self.steps = []
        self.peft_type = None
    
    def setup(self, trainer, pl_module, stage=None):
        """
        Setup the callback when training begins.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: The LightningModule being trained
            stage: The stage of training
        """
        # Try to identify if this is a PEFT model
        try:
            from peft import PeftModel
            if not isinstance(pl_module.model, PeftModel):
                logger.info("Model is not a PEFT model, PeftAdapterMonitorCallback will have limited functionality")
                return
                
            # Determine PEFT type
            self.peft_type = getattr(pl_module.model.peft_config, "peft_type", "UNKNOWN")
            logger.info(f"Detected PEFT type: {self.peft_type}")
            
            # Create save path if specified
            if self.save_path:
                os.makedirs(self.save_path, exist_ok=True)
                
            # Log trainable parameters info
            trainable_params = [p for p in pl_module.model.parameters() if p.requires_grad]
            total_trainable_params = sum(p.numel() for p in trainable_params)
            logger.info(f"Monitoring {len(trainable_params)} trainable adapter parameters "
                       f"({total_trainable_params} total elements)")
                       
            # Setup parameter groups for monitoring based on PEFT type
            self._setup_parameter_groups(pl_module.model)
            
        except (ImportError, AttributeError) as e:
            logger.warning(f"Error setting up PeftAdapterMonitorCallback: {e}")
    
    def _setup_parameter_groups(self, model):
        """
        Setup parameter groups for monitoring based on PEFT type.
        
        Args:
            model: The PEFT model
        """
        # Group parameters by layer and type for more structured monitoring
        if "LORA" in str(self.peft_type):
            lora_a_params = [n for n, _ in model.named_parameters() if 'lora_A' in n]
            lora_b_params = [n for n, _ in model.named_parameters() if 'lora_B' in n]
            self.param_groups = {
                "lora_a": lora_a_params,
                "lora_b": lora_b_params
            }
            # Count parameters by group
            self.adapter_params_count = {
                "lora_a": sum(model.get_parameter(n).numel() for n in lora_a_params),
                "lora_b": sum(model.get_parameter(n).numel() for n in lora_b_params)
            }
            
        elif "PREFIX" in str(self.peft_type):
            prefix_params = [n for n, _ in model.named_parameters() if any(x in n for x in ['prefix', 'embedding'])]
            self.param_groups = {"prefix": prefix_params}
            self.adapter_params_count = {
                "prefix": sum(model.get_parameter(n).numel() for n in prefix_params)
            }
            
        elif "PROMPT" in str(self.peft_type) or "P_TUNING" in str(self.peft_type):
            prompt_params = [n for n, _ in model.named_parameters() if 'prompt' in n.lower()]
            self.param_groups = {"prompt": prompt_params}
            self.adapter_params_count = {
                "prompt": sum(model.get_parameter(n).numel() for n in prompt_params)
            }
            
        else:
            # For other PEFT types, monitor all trainable parameters
            trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
            self.param_groups = {"trainable": trainable_params}
            self.adapter_params_count = {
                "trainable": sum(model.get_parameter(n).numel() for n in trainable_params)
            }
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Monitor PEFT parameters after each training batch.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: The LightningModule being trained
            outputs: Training step outputs
            batch: The current batch of data
            batch_idx: The index of the current batch
        """
        if batch_idx % self.log_every_n_steps != 0:
            return
            
        try:
            from peft import PeftModel
            if not isinstance(pl_module.model, PeftModel):
                return
                
            # Record current step
            step = trainer.global_step
            self.steps.append(step)
            
            # Monitor gradients if enabled
            if self.monitor_gradients:
                self._monitor_gradients(pl_module.model, step)
                
            # Monitor weights if enabled
            if self.monitor_weights:
                self._monitor_weights(pl_module.model, step)
                
            # Create visualization every 10 log intervals
            if len(self.steps) % 10 == 0 and self.save_path:
                self._visualize_monitoring()
                
        except (ImportError, AttributeError) as e:
            pass
    
    @rank_zero_only
    def _monitor_gradients(self, model, step):
        """
        Monitor gradients of PEFT parameters.
        
        Args:
            model: The PEFT model
            step: Current training step
        """
        for group_name, param_names in self.param_groups.items():
            if group_name not in self.gradient_norms:
                self.gradient_norms[group_name] = []
                
            # Calculate average gradient norm for this parameter group
            grad_norms = []
            for param_name in param_names:
                param = model.get_parameter(param_name)
                if param.grad is not None:
                    grad_norms.append(param.grad.norm().item())
                    
            # Record average gradient norm
            if grad_norms:
                avg_grad_norm = sum(grad_norms) / len(grad_norms)
                self.gradient_norms[group_name].append(avg_grad_norm)
                # Log to Lightning logs
                model.log(f"peft/grad_norm/{group_name}", avg_grad_norm, prog_bar=False)
    
    @rank_zero_only
    def _monitor_weights(self, model, step):
        """
        Monitor weight values of PEFT parameters.
        
        Args:
            model: The PEFT model
            step: Current training step
        """
        for group_name, param_names in self.param_groups.items():
            if group_name not in self.weight_norms:
                self.weight_norms[group_name] = []
                
            # Calculate average weight norm for this parameter group
            weight_norms = []
            for param_name in param_names:
                param = model.get_parameter(param_name)
                weight_norms.append(param.norm().item())
                    
            # Record average weight norm
            if weight_norms:
                avg_weight_norm = sum(weight_norms) / len(weight_norms)
                self.weight_norms[group_name].append(avg_weight_norm)
                # Log to Lightning logs
                model.log(f"peft/weight_norm/{group_name}", avg_weight_norm, prog_bar=False)
    
    @rank_zero_only
    def _visualize_monitoring(self):
        """
        Create visualizations of monitored PEFT parameters.
        """
        if not self.steps:
            return
            
        # Create gradient norms plot
        if self.gradient_norms:
            plt.figure(figsize=(10, 6))
            for group_name, norms in self.gradient_norms.items():
                if len(norms) == len(self.steps):
                    plt.plot(self.steps, norms, label=f"{group_name} (params: {self.adapter_params_count.get(group_name, '?')})")
            plt.xlabel("Training Step")
            plt.ylabel("Gradient Norm")
            plt.title(f"PEFT Adapter Gradient Norms - {self.peft_type}")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, "peft_grad_norms.png"))
            plt.close()
            
        # Create weight norms plot
        if self.weight_norms:
            plt.figure(figsize=(10, 6))
            for group_name, norms in self.weight_norms.items():
                if len(norms) == len(self.steps):
                    plt.plot(self.steps, norms, label=f"{group_name} (params: {self.adapter_params_count.get(group_name, '?')})")
            plt.xlabel("Training Step")
            plt.ylabel("Weight Norm")
            plt.title(f"PEFT Adapter Weight Norms - {self.peft_type}")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, "peft_weight_norms.png"))
            plt.close()
    
    def on_train_end(self, trainer, pl_module):
        """
        Create final visualizations when training ends.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: The LightningModule being trained
        """
        # Create comprehensive visualizations at the end of training
        if self.save_path:
            self._visualize_monitoring()
            self._create_training_summary(pl_module.model)
    
    @rank_zero_only
    def _create_training_summary(self, model):
        """
        Create a summary of the PEFT adapter training.
        
        Args:
            model: The PEFT model
        """
        # Create a summary text file
        summary_path = os.path.join(self.save_path, "peft_training_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"PEFT Adapter Training Summary\n")
            f.write(f"===========================\n\n")
            f.write(f"PEFT Type: {self.peft_type}\n\n")
            
            f.write(f"Parameter Groups:\n")
            for group_name, param_count in self.adapter_params_count.items():
                f.write(f"  - {group_name}: {param_count} parameters\n")
            
            f.write(f"\nTotal Trainable Parameters: {sum(self.adapter_params_count.values())}\n")
            
            if hasattr(model, 'peft_config'):
                f.write(f"\nPEFT Configuration:\n")
                for key, value in vars(model.peft_config).items():
                    if not key.startswith('_'):
                        f.write(f"  - {key}: {value}\n")


class PeftEarlyPruningCallback(Callback):
    """
    A PyTorch Lightning callback that prunes PEFT adapters during training.
    
    This callback monitors PEFT adapter parameters and prunes (sets to zero)
    parameters with small magnitudes to improve efficiency and reduce overfitting.
    """
    
    def __init__(
        self,
        prune_on_epoch_end: bool = True,
        start_pruning_epoch: int = 1,
        pruning_threshold: float = 0.01,
        pruning_schedule: str = "linear",
        final_sparsity: float = 0.5,
        peft_modules_to_prune: Optional[List[str]] = None,
        protected_modules: Optional[List[str]] = None,
        restore_pruned_on_fit_end: bool = False
    ):
        """
        Initialize the PEFT adapter pruning callback.
        
        Args:
            prune_on_epoch_end: Whether to prune at the end of each epoch
            start_pruning_epoch: Epoch to start pruning
            pruning_threshold: Threshold below which to prune parameters
            pruning_schedule: Schedule for increasing pruning threshold ("linear", "exponential")
            final_sparsity: Target sparsity by the end of training
            peft_modules_to_prune: List of specific module types to prune
            protected_modules: List of modules to protect from pruning
            restore_pruned_on_fit_end: Whether to restore pruned parameters at the end of training
        """
        super().__init__()
        self.prune_on_epoch_end = prune_on_epoch_end
        self.start_pruning_epoch = start_pruning_epoch
        self.pruning_threshold = pruning_threshold
        self.pruning_schedule = pruning_schedule
        self.final_sparsity = final_sparsity
        self.peft_modules_to_prune = peft_modules_to_prune
        self.protected_modules = protected_modules or []
        self.restore_pruned_on_fit_end = restore_pruned_on_fit_end
        
        # Initialize storage for original values
        self.original_values = {}
        self.pruning_masks = {}
        self.current_sparsity = 0.0
    
    def setup(self, trainer, pl_module, stage=None):
        """
        Setup the callback when training begins.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: The LightningModule being trained
            stage: The stage of training
        """
        # Try to identify if this is a PEFT model
        try:
            from peft import PeftModel
            if not isinstance(pl_module.model, PeftModel):
                logger.info("Model is not a PEFT model, PeftEarlyPruningCallback will be disabled")
                self.enabled = False
                return
            
            self.enabled = True
            self.peft_type = getattr(pl_module.model.peft_config, "peft_type", "UNKNOWN")
            
            # Set default modules to prune based on PEFT type if not specified
            if not self.peft_modules_to_prune:
                if "LORA" in str(self.peft_type):
                    self.peft_modules_to_prune = ["lora_A", "lora_B"]
                elif "PREFIX" in str(self.peft_type):
                    self.peft_modules_to_prune = ["prefix_encoder"]
                else:
                    # For other types, use all trainable parameters
                    self.peft_modules_to_prune = []
            
            # Get trainable parameters to prune
            self.prunable_params = [
                n for n, p in pl_module.model.named_parameters()
                if p.requires_grad and self._is_prunable(n)
            ]
            
            logger.info(f"PeftEarlyPruningCallback identified {len(self.prunable_params)} prunable parameters")
            
        except (ImportError, AttributeError) as e:
            logger.warning(f"Error setting up PeftEarlyPruningCallback: {e}")
            self.enabled = False
    
    def _is_prunable(self, param_name):
        """
        Determine if a parameter should be pruned.
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            Boolean indicating if the parameter should be pruned
        """
        # Check if parameter is protected
        if any(protected in param_name for protected in self.protected_modules):
            return False
            
        # If specific modules are specified, check if parameter is in those modules
        if self.peft_modules_to_prune:
            return any(module in param_name for module in self.peft_modules_to_prune)
        
        # By default, consider all trainable parameters prunable
        return True
    
    def _get_current_threshold(self, current_epoch, max_epochs):
        """
        Get the current pruning threshold based on the schedule.
        
        Args:
            current_epoch: The current epoch
            max_epochs: Maximum number of epochs
            
        Returns:
            Current pruning threshold
        """
        if current_epoch < self.start_pruning_epoch:
            return 0.0
            
        progress = (current_epoch - self.start_pruning_epoch) / (max_epochs - self.start_pruning_epoch)
        progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
        
        if self.pruning_schedule == "linear":
            current_sparsity = progress * self.final_sparsity
        elif self.pruning_schedule == "exponential":
            # Exponential schedule: sparsity increases more rapidly at the beginning
            current_sparsity = self.final_sparsity * (1 - np.exp(-5 * progress))
        else:
            # Default to linear
            current_sparsity = progress * self.final_sparsity
            
        self.current_sparsity = current_sparsity
        return self.pruning_threshold * (1 + progress)
    
    def on_train_epoch_end(self, trainer, pl_module):
        """
        Apply pruning at the end of each training epoch.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: The LightningModule being trained
        """
        if not self.enabled or not self.prune_on_epoch_end:
            return
            
        current_epoch = trainer.current_epoch
        max_epochs = trainer.max_epochs or 10  # Default to 10 if not specified
        
        # Skip if before start_pruning_epoch
        if current_epoch < self.start_pruning_epoch:
            logger.info(f"Epoch {current_epoch}: Pruning not started yet (starts at epoch {self.start_pruning_epoch})")
            return
            
        # Calculate current threshold
        current_threshold = self._get_current_threshold(current_epoch, max_epochs)
        
        # Apply pruning
        self._apply_pruning(pl_module.model, current_threshold)
        
        # Log pruning statistics
        logger.info(f"Epoch {current_epoch}: Applied pruning with threshold {current_threshold:.4f}, "
                   f"current sparsity: {self.current_sparsity:.2%}")
        pl_module.log("peft/pruning/sparsity", self.current_sparsity, prog_bar=False)
        pl_module.log("peft/pruning/threshold", current_threshold, prog_bar=False)
    
    @rank_zero_only
    def _apply_pruning(self, model, threshold):
        """
        Apply pruning to PEFT adapter parameters.
        
        Args:
            model: The PEFT model
            threshold: Pruning threshold
        """
        # Count total and pruned parameters
        total_params = 0
        pruned_params = 0
        
        for param_name in self.prunable_params:
            param = model.get_parameter(param_name)
            
            # Store original values if not already stored
            if param_name not in self.original_values:
                self.original_values[param_name] = param.data.clone()
            
            # Create mask based on absolute value threshold
            mask = (param.abs() >= threshold).float()
            
            # Store mask for restoration if needed
            self.pruning_masks[param_name] = mask
            
            # Apply mask - zero out values below threshold
            param.data = param.data * mask
            
            # Update counts
            total_params += param.numel()
            pruned_params += param.numel() - mask.sum().item()
            
        # Update current sparsity
        if total_params > 0:
            self.current_sparsity = pruned_params / total_params
    
    def on_fit_end(self, trainer, pl_module):
        """
        Restore pruned parameters if requested when training ends.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: The LightningModule being trained
        """
        if not self.enabled:
            return
            
        if self.restore_pruned_on_fit_end:
            logger.info("Restoring pruned parameters to original values")
            self._restore_pruned_parameters(pl_module.model)
    
    @rank_zero_only
    def _restore_pruned_parameters(self, model):
        """
        Restore pruned parameters to their original values.
        
        Args:
            model: The PEFT model
        """
        for param_name, original_value in self.original_values.items():
            param = model.get_parameter(param_name)
            param.data = original_value


class PeftAdapterFusionCallback(Callback):
    """
    A PyTorch Lightning callback that handles adapter fusion for multi-adapter PEFT models.
    
    This callback supports training with multiple adapters and optionally fusing them
    at the end of training.
    """
    
    def __init__(
        self,
        adapter_list: List[str],
        fusion_strategy: str = "average",  # "average", "weighted", "learned"
        weights: Optional[List[float]] = None,
        fuse_on_fit_end: bool = True,
        save_fused_adapter: bool = True,
        fused_adapter_name: str = "fused_adapter"
    ):
        """
        Initialize the PEFT adapter fusion callback.
        
        Args:
            adapter_list: List of adapter names to fuse
            fusion_strategy: Strategy for fusing adapters
            weights: Weights for weighted fusion strategy
            fuse_on_fit_end: Whether to fuse adapters at the end of training
            save_fused_adapter: Whether to save the fused adapter
            fused_adapter_name: Name for the fused adapter
        """
        super().__init__()
        self.adapter_list = adapter_list
        self.fusion_strategy = fusion_strategy
        self.weights = weights if weights is not None else [1.0] * len(adapter_list)
        self.fuse_on_fit_end = fuse_on_fit_end
        self.save_fused_adapter = save_fused_adapter
        self.fused_adapter_name = fused_adapter_name
        
        # Normalize weights
        if self.weights:
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
    
    def setup(self, trainer, pl_module, stage=None):
        """
        Setup the callback when training begins.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: The LightningModule being trained
            stage: The stage of training
        """
        # Try to identify if this is a PEFT model
        try:
            from peft import PeftModel
            if not isinstance(pl_module.model, PeftModel):
                logger.info("Model is not a PEFT model, PeftAdapterFusionCallback will be disabled")
                self.enabled = False
                return
                
            # Check if the model has multiple adapters
            if not hasattr(pl_module.model, "active_adapters") or not hasattr(pl_module.model, "load_adapter"):
                logger.info("Model does not support multiple adapters, PeftAdapterFusionCallback will be disabled")
                self.enabled = False
                return
                
            self.enabled = True
            
            # Verify adapters exist
            available_adapters = getattr(pl_module.model, "adapters", {})
            if not all(adapter in available_adapters for adapter in self.adapter_list):
                logger.warning(f"Not all specified adapters {self.adapter_list} are available in the model. "
                              f"Available adapters: {list(available_adapters.keys())}")
            
        except (ImportError, AttributeError) as e:
            logger.warning(f"Error setting up PeftAdapterFusionCallback: {e}")
            self.enabled = False
    
    def on_fit_end(self, trainer, pl_module):
        """
        Fuse adapters at the end of training if requested.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: The LightningModule being trained
        """
        if not self.enabled or not self.fuse_on_fit_end:
            return
            
        # Perform adapter fusion
        try:
            self._fuse_adapters(pl_module.model)
            
            # Save fused adapter if requested
            if self.save_fused_adapter:
                output_dir = getattr(pl_module, "output_dir", None) or trainer.default_root_dir
                save_path = os.path.join(output_dir, self.fused_adapter_name)
                os.makedirs(save_path, exist_ok=True)
                pl_module.model.save_pretrained(save_path)
                logger.info(f"Saved fused adapter to {save_path}")
                
        except Exception as e:
            logger.error(f"Error during adapter fusion: {e}")
    
    @rank_zero_only
    def _fuse_adapters(self, model):
        """
        Fuse multiple adapters using the specified strategy.
        
        Args:
            model: The PEFT model
        """
        logger.info(f"Fusing adapters {self.adapter_list} using {self.fusion_strategy} strategy")
        
        # Get adapter state dicts
        adapter_states = {}
        for adapter_name in self.adapter_list:
            try:
                adapter_states[adapter_name] = model.get_adapter_state_dict(adapter_name)
            except Exception as e:
                logger.warning(f"Error getting state dict for adapter {adapter_name}: {e}")
                continue
        
        if not adapter_states:
            logger.warning("No adapter states could be loaded, fusion aborted")
            return
            
        # Create fused adapter state
        fused_state = {}
        
        if self.fusion_strategy == "average":
            # Simple averaging of parameter values
            for param_name in adapter_states[list(adapter_states.keys())[0]]:
                tensors = [states[param_name] for adapter_name, states in adapter_states.items() 
                          if param_name in states]
                if tensors:
                    fused_state[param_name] = sum(tensors) / len(tensors)
                    
        elif self.fusion_strategy == "weighted":
            # Weighted averaging using specified weights
            weights_dict = {adapter: weight for adapter, weight in zip(self.adapter_list, self.weights) 
                           if adapter in adapter_states}
                           
            for param_name in adapter_states[list(adapter_states.keys())[0]]:
                weighted_sum = None
                total_weight = 0
                
                for adapter_name, states in adapter_states.items():
                    if param_name in states and adapter_name in weights_dict:
                        weight = weights_dict[adapter_name]
                        if weighted_sum is None:
                            weighted_sum = states[param_name] * weight
                        else:
                            weighted_sum += states[param_name] * weight
                        total_weight += weight
                
                if weighted_sum is not None and total_weight > 0:
                    fused_state[param_name] = weighted_sum / total_weight
        
        else:
            logger.warning(f"Unsupported fusion strategy: {self.fusion_strategy}, using first adapter")
            fused_state = adapter_states[list(adapter_states.keys())[0]]
        
        # Create or update fused adapter
        if hasattr(model, "add_adapter"):
            # If adapter doesn't exist, create it based on the first adapter's config
            if self.fused_adapter_name not in getattr(model, "adapters", {}):
                first_adapter = list(adapter_states.keys())[0]
                # Get config for the first adapter
                if hasattr(model, "peft_config") and isinstance(model.peft_config, dict):
                    config = model.peft_config.get(first_adapter)
                    if config:
                        model.add_adapter(self.fused_adapter_name, config)
            
            # Set adapter weights
            model.set_adapter_state_dict(fused_state, self.fused_adapter_name)
            
            # Set as active adapter
            model.set_adapter(self.fused_adapter_name)
        
        logger.info(f"Successfully fused {len(adapter_states)} adapters into {self.fused_adapter_name}") 