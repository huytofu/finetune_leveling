"""
Model merging utilities for combining models or frankensteining different architectures.

This module provides functions to:
1. Average weights of models with the same architecture
2. Selectively merge parts of models with different architectures
3. Merge models at different layers ("frankensteining")
"""

import os
import logging
import torch
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Any, Callable
from transformers import PreTrainedModel, AutoConfig

logger = logging.getLogger(__name__)

@dataclass
class ModelMergeConfig:
    """Configuration for model merging operations."""
    
    # Source models
    base_model_name: str = None  # Primary model to use as a base
    secondary_model_names: List[str] = field(default_factory=list)  # Secondary models to merge
    
    # Merging options
    merge_method: str = "weighted_average"  # "weighted_average", "selective", "frankenstein"
    layer_weights: Optional[Dict[str, float]] = None  # Weights for each model in weighted average
    
    # Selective merging options (for merge_method="selective")
    # Format: {"source_model_name": ["layer1", "layer2", ...]}
    selective_layers: Optional[Dict[str, List[str]]] = None
    
    # Frankenstein options (for merge_method="frankenstein")
    # Format: {"target_layer": "source_model_name"}
    frankenstein_mapping: Optional[Dict[str, str]] = None
    
    # Advanced options
    tie_weights: bool = True  # Whether to tie weights in the resulting model
    output_model_name: str = "merged_model"  # Name for the merged model
    save_path: Optional[str] = None  # Path to save the merged model
    device: str = "auto"  # Device to perform merging on
    
    # Compatibility verification
    verify_compatibility: bool = True  # Run a forward pass to verify the model works
    test_input: Optional[Dict[str, torch.Tensor]] = None  # Test input for verification
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.base_model_name:
            raise ValueError("base_model_name is required")
        
        if self.merge_method not in ["weighted_average", "selective", "frankenstein"]:
            raise ValueError(f"Unsupported merge method: {self.merge_method}")
        
        if self.merge_method == "weighted_average" and not self.layer_weights:
            # Default to equal weights if not specified
            models = [self.base_model_name] + self.secondary_model_names
            self.layer_weights = {model: 1.0 / len(models) for model in models}
        
        if self.merge_method == "selective" and not self.selective_layers:
            raise ValueError("selective_layers must be specified for selective merging")
        
        if self.merge_method == "frankenstein" and not self.frankenstein_mapping:
            raise ValueError("frankenstein_mapping must be specified for frankenstein merging")


class ModelMerger:
    """Utility for merging transformer models with different strategies."""
    
    def __init__(self, config: ModelMergeConfig):
        """Initialize model merger with configuration."""
        self.config = config
        self.device = self._resolve_device(config.device)
        self.base_model = None
        self.secondary_models = {}
        self.merged_model = None
        
        # Model configs
        self.base_config = None
        self.secondary_configs = {}
        
    def _resolve_device(self, device: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def load_models(self):
        """Load all models specified in the configuration."""
        logger.info(f"Loading base model: {self.config.base_model_name}")
        
        try:
            # Load base model config first
            self.base_config = AutoConfig.from_pretrained(self.config.base_model_name)
            
            # Load base model
            self.base_model = PreTrainedModel.from_pretrained(
                self.config.base_model_name,
                config=self.base_config
            )
            
            # Load secondary models if needed
            for model_name in self.config.secondary_model_names:
                logger.info(f"Loading secondary model: {model_name}")
                
                # Load secondary model config
                self.secondary_configs[model_name] = AutoConfig.from_pretrained(model_name)
                
                # Load secondary model
                self.secondary_models[model_name] = PreTrainedModel.from_pretrained(
                    model_name,
                    config=self.secondary_configs[model_name]
                )
            
            logger.info("All models loaded successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def _get_state_dict_structure(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
        """Extract structure information from a state dict."""
        structure = {}
        
        for key, tensor in state_dict.items():
            # Get layer or component name (first part of the key)
            parts = key.split('.')
            layer_name = parts[0]
            
            if layer_name not in structure:
                structure[layer_name] = {}
            
            # Add tensor shape and dtype
            structure[layer_name][key] = {
                "shape": tensor.shape,
                "dtype": str(tensor.dtype),
                "requires_grad": tensor.requires_grad
            }
        
        return structure
    
    def _model_structures_compatible(self, src_model, tgt_model) -> Tuple[bool, str]:
        """Check if model structures are compatible for merging."""
        src_dict = src_model.state_dict()
        tgt_dict = tgt_model.state_dict()
        
        # Check if all keys in the target model exist in the source model
        missing_keys = [k for k in tgt_dict.keys() if k not in src_dict]
        if missing_keys:
            return False, f"Keys missing in source model: {missing_keys[:5]}..."
        
        # Check if tensor shapes match for all keys
        for key in tgt_dict:
            if key in src_dict:
                if src_dict[key].shape != tgt_dict[key].shape:
                    return False, f"Shape mismatch for key {key}: {src_dict[key].shape} vs {tgt_dict[key].shape}"
        
        return True, "Models are compatible"
    
    def merge_weighted_average(self) -> PreTrainedModel:
        """Merge models using weighted average of parameters."""
        logger.info("Performing weighted average merge")
        
        # Initialize merged model with base model's state dict
        merged_state_dict = {k: v.clone() for k, v in self.base_model.state_dict().items()}
        
        # Apply weights to base model parameters
        base_weight = self.config.layer_weights.get(self.config.base_model_name, 0.0)
        for key in merged_state_dict:
            merged_state_dict[key] *= base_weight
        
        # Add weighted parameters from secondary models
        for model_name, model in self.secondary_models.items():
            weight = self.config.layer_weights.get(model_name, 0.0)
            if weight <= 0.0:
                continue
                
            # Check compatibility
            compatible, message = self._model_structures_compatible(model, self.base_model)
            if not compatible:
                logger.warning(f"Model {model_name} is not compatible with base model: {message}")
                continue
            
            # Add weighted parameters
            for key, param in model.state_dict().items():
                if key in merged_state_dict:
                    merged_state_dict[key] += param.to(merged_state_dict[key].device) * weight
        
        # Create new model from merged state dict
        merged_model = type(self.base_model).from_pretrained(
            None,
            config=self.base_config,
            state_dict=merged_state_dict
        )
        
        self.merged_model = merged_model
        return merged_model
    
    def merge_selective(self) -> PreTrainedModel:
        """Merge models by selectively taking layers from different models."""
        logger.info("Performing selective merge")
        
        # Start with base model's state dict
        merged_state_dict = {k: v.clone() for k, v in self.base_model.state_dict().items()}
        
        # Selectively replace parameters based on configuration
        for model_name, layer_patterns in self.config.selective_layers.items():
            # Skip if model is not loaded
            if model_name != self.config.base_model_name and model_name not in self.secondary_models:
                logger.warning(f"Model {model_name} not loaded, skipping its layers")
                continue
            
            # Get source model
            source_model = self.base_model if model_name == self.config.base_model_name else self.secondary_models[model_name]
            source_state_dict = source_model.state_dict()
            
            # Replace specified layers
            for pattern in layer_patterns:
                matching_keys = [k for k in source_state_dict.keys() if pattern in k]
                
                for key in matching_keys:
                    if key in merged_state_dict:
                        # Check if shapes match
                        if merged_state_dict[key].shape == source_state_dict[key].shape:
                            merged_state_dict[key] = source_state_dict[key].clone()
                        else:
                            logger.warning(f"Shape mismatch for {key}, skipping: {merged_state_dict[key].shape} vs {source_state_dict[key].shape}")
        
        # Create new model from merged state dict
        merged_model = type(self.base_model).from_pretrained(
            None,
            config=self.base_config,
            state_dict=merged_state_dict
        )
        
        self.merged_model = merged_model
        return merged_model
    
    def merge_frankenstein(self) -> PreTrainedModel:
        """Merge models by mapping layers from different architectures."""
        logger.info("Performing frankenstein merge")
        
        # Start with base model's state dict and config
        merged_state_dict = {k: v.clone() for k, v in self.base_model.state_dict().items()}
        
        # Apply frankenstein mappings
        for target_layer, source_spec in self.config.frankenstein_mapping.items():
            # Parse source spec (format: "model_name:layer_name")
            model_name, source_layer = source_spec.split(":")
            
            # Skip if model is not loaded
            if model_name != self.config.base_model_name and model_name not in self.secondary_models:
                logger.warning(f"Model {model_name} not loaded, skipping mapping for {target_layer}")
                continue
            
            # Get source model
            source_model = self.base_model if model_name == self.config.base_model_name else self.secondary_models[model_name]
            source_state_dict = source_model.state_dict()
            
            # Find matching keys
            target_keys = [k for k in merged_state_dict.keys() if k.startswith(f"{target_layer}.")]
            source_keys = [k for k in source_state_dict.keys() if k.startswith(f"{source_layer}.")]
            
            # Create mapping between source and target keys
            key_mapping = self._create_key_mapping(target_keys, source_keys)
            
            # Apply mapping
            for target_key, source_key in key_mapping.items():
                source_param = source_state_dict[source_key]
                target_param = merged_state_dict[target_key]
                
                # Check dimensions and reshape if necessary
                if source_param.shape == target_param.shape:
                    # Direct replacement
                    merged_state_dict[target_key] = source_param.clone()
                else:
                    # Try to reshape if possible
                    reshaped_param = self._try_reshape_tensor(source_param, target_param.shape)
                    if reshaped_param is not None:
                        merged_state_dict[target_key] = reshaped_param
                    else:
                        logger.warning(f"Cannot reshape {source_key} {source_param.shape} to match {target_key} {target_param.shape}")
        
        # Create new model from merged state dict
        merged_model = type(self.base_model).from_pretrained(
            None,
            config=self.base_config,
            state_dict=merged_state_dict
        )
        
        self.merged_model = merged_model
        return merged_model
    
    def _create_key_mapping(self, target_keys: List[str], source_keys: List[str]) -> Dict[str, str]:
        """Create mapping between source and target keys based on structural similarity."""
        mapping = {}
        
        # Extract suffixes (parts after the layer name)
        target_suffixes = [k.split(".", 1)[1] if "." in k else "" for k in target_keys]
        source_suffixes = [k.split(".", 1)[1] if "." in k else "" for k in source_keys]
        
        # Map based on matching suffixes
        for i, target_key in enumerate(target_keys):
            target_suffix = target_suffixes[i]
            
            # Direct match - same suffix
            if target_suffix in source_suffixes:
                source_idx = source_suffixes.index(target_suffix)
                mapping[target_key] = source_keys[source_idx]
            else:
                # Try to find a similar suffix
                best_match = self._find_closest_suffix(target_suffix, source_suffixes)
                if best_match >= 0:
                    mapping[target_key] = source_keys[best_match]
        
        return mapping
    
    def _find_closest_suffix(self, target: str, candidates: List[str]) -> int:
        """Find the closest matching suffix from candidates."""
        # Simple matching for now - can be enhanced with more sophisticated metrics
        best_score = -1
        best_idx = -1
        
        for i, candidate in enumerate(candidates):
            # Count common substrings
            score = sum(a == b for a, b in zip(target.split("."), candidate.split(".")))
            if score > best_score:
                best_score = score
                best_idx = i
        
        return best_idx if best_score > 0 else -1
    
    def _try_reshape_tensor(self, tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> Optional[torch.Tensor]:
        """Try to reshape a tensor to match the target shape."""
        # Check if total elements match
        if tensor.numel() != torch.Size(target_shape).numel():
            # Special case: handle different embedding sizes
            if len(tensor.shape) == 2 and len(target_shape) == 2:
                # For embeddings, we can try to crop or pad
                src_rows, src_cols = tensor.shape
                tgt_rows, tgt_cols = target_shape
                
                if src_rows == tgt_rows:
                    # Same number of embeddings, different dimensions
                    if src_cols < tgt_cols:
                        # Pad with zeros
                        result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
                        result[:, :src_cols] = tensor
                        return result
                    else:
                        # Crop
                        return tensor[:, :tgt_cols].clone()
            
            return None
        
        try:
            # Attempt to reshape
            return tensor.reshape(target_shape)
        except RuntimeError:
            return None
    
    def verify_model(self, model: PreTrainedModel) -> bool:
        """Verify the merged model works by running a test forward pass."""
        if not self.config.verify_compatibility:
            return True
        
        logger.info("Verifying merged model compatibility")
        
        try:
            # Create a simple test input if not provided
            test_input = self.config.test_input
            if test_input is None:
                # Default test input with batch size 1 and sequence length 8
                test_input = {
                    "input_ids": torch.randint(0, 1000, (1, 8), device=self.device),
                    "attention_mask": torch.ones((1, 8), device=self.device)
                }
            
            # Run a forward pass
            model.to(self.device)
            model.eval()
            
            with torch.no_grad():
                outputs = model(**test_input)
            
            logger.info(f"Model verification successful. Output shape: {outputs.logits.shape if hasattr(outputs, 'logits') else 'unknown'}")
            return True
            
        except Exception as e:
            logger.error(f"Model verification failed: {str(e)}")
            return False
    
    def merge(self) -> PreTrainedModel:
        """Merge models according to the specified strategy."""
        # Load models if not already loaded
        if self.base_model is None:
            success = self.load_models()
            if not success:
                raise RuntimeError("Failed to load models for merging")
        
        # Perform merging based on the specified method
        if self.config.merge_method == "weighted_average":
            merged_model = self.merge_weighted_average()
        elif self.config.merge_method == "selective":
            merged_model = self.merge_selective()
        elif self.config.merge_method == "frankenstein":
            merged_model = self.merge_frankenstein()
        else:
            raise ValueError(f"Unsupported merge method: {self.config.merge_method}")
        
        # Verify the merged model
        if self.config.verify_compatibility:
            is_valid = self.verify_model(merged_model)
            if not is_valid:
                logger.warning("Merged model verification failed")
        
        # Save the merged model if requested
        if self.config.save_path:
            save_path = os.path.join(self.config.save_path, self.config.output_model_name)
            logger.info(f"Saving merged model to {save_path}")
            merged_model.save_pretrained(save_path)
        
        return merged_model


def weighted_average_merge(
    models: List[PreTrainedModel],
    weights: Optional[List[float]] = None,
    verify: bool = True
) -> PreTrainedModel:
    """Convenience function for weighted average model merging."""
    if weights is None:
        # Equal weights
        weights = [1.0 / len(models)] * len(models)
    
    if len(models) != len(weights):
        raise ValueError("Number of models must match number of weights")
    
    # Create config
    config = ModelMergeConfig(
        base_model_name="<in-memory>",
        secondary_model_names=["<in-memory>"] * (len(models) - 1),
        merge_method="weighted_average",
        layer_weights={f"<in-memory-{i}>": w for i, w in enumerate(weights)},
        verify_compatibility=verify
    )
    
    # Initialize merger
    merger = ModelMerger(config)
    
    # Set models directly instead of loading
    merger.base_model = models[0]
    merger.secondary_models = {f"<in-memory-{i+1}>": models[i+1] for i in range(len(models) - 1)}
    
    # Perform merge
    return merger.merge_weighted_average()


def selective_merge(
    base_model: PreTrainedModel,
    secondary_models: Dict[str, PreTrainedModel],
    layer_mapping: Dict[str, List[str]],
    verify: bool = True
) -> PreTrainedModel:
    """Convenience function for selective model merging."""
    # Create config
    config = ModelMergeConfig(
        base_model_name="<in-memory-base>",
        secondary_model_names=[f"<in-memory-{name}>" for name in secondary_models],
        merge_method="selective",
        selective_layers={f"<in-memory-{name}>": patterns for name, patterns in layer_mapping.items()},
        verify_compatibility=verify
    )
    
    # Initialize merger
    merger = ModelMerger(config)
    
    # Set models directly instead of loading
    merger.base_model = base_model
    merger.secondary_models = {f"<in-memory-{name}>": model for name, model in secondary_models.items()}
    
    # Perform merge
    return merger.merge_selective()


def frankenstein_merge(
    base_model: PreTrainedModel,
    secondary_models: Dict[str, PreTrainedModel],
    layer_mapping: Dict[str, str],
    verify: bool = True
) -> PreTrainedModel:
    """Convenience function for frankenstein model merging."""
    # Create config
    config = ModelMergeConfig(
        base_model_name="<in-memory-base>",
        secondary_model_names=[f"<in-memory-{name}>" for name in secondary_models],
        merge_method="frankenstein",
        frankenstein_mapping={target: f"<in-memory-{source.split(':')[0]}>:{source.split(':')[1]}" 
                             for target, source in layer_mapping.items()},
        verify_compatibility=verify
    )
    
    # Initialize merger
    merger = ModelMerger(config)
    
    # Set models directly instead of loading
    merger.base_model = base_model
    merger.secondary_models = {f"<in-memory-{name}>": model for name, model in secondary_models.items()}
    
    # Perform merge
    return merger.merge_frankenstein() 