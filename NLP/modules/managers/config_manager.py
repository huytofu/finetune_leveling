import json
import os
import sys
from typing import Dict, Any, Optional

# Import default configuration
currdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currdir)
sys.path.append(parentdir)
from configs.default_config import DEFAULT_SPECS

class ConfigurationManager:
    """
    Manages all configuration settings with validation and sensible defaults.
    
    This class centralizes configuration handling, providing validation,
    sensible defaults, and helpful error messages.
    """
    
    def __init__(self, config_path: Optional[str] = None, defaults: Optional[Dict[str, Any]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to a JSON configuration file
            defaults: Default configuration values to override DEFAULT_SPECS
        """
        self.config = DEFAULT_SPECS.copy()
        if defaults:
            self.config.update(defaults)
        if config_path:
            self._load_config(config_path)
        self._validate_config()
    
    def _load_config(self, config_path: str) -> None:
        """
        Load configuration from file with friendly error messages.
        
        Args:
            config_path: Path to a JSON configuration file
        """
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            self.config.update(user_config)
        except FileNotFoundError:
            print(f"Configuration file not found: {config_path}")
            print("Using default configuration")
        except json.JSONDecodeError:
            print(f"Invalid JSON in configuration file: {config_path}")
            print("Using default configuration")
    
    def _validate_config(self) -> None:
        """Validate configuration values with helpful messages."""
        # Validate required fields
        required_fields = ["output_dir"]
        for field in required_fields:
            if not self.config.get(field):
                print(f"Warning: '{field}' is not set in configuration")
        
        # Validate compatible settings
        if self.config.get("use_peft") and self.config.get("peft_method") not in ["lora", "qlora", "prefix_tuning", "prompt_tuning", "p_tuning"]:
            print(f"Warning: Invalid PEFT method '{self.config.get('peft_method')}', defaulting to 'lora'")
            self.config["peft_method"] = "lora"
        
        # Validate quantization settings
        if self.config.get("use_quantization") and self.config.get("quantization_type") not in ["4bit", "8bit"]:
            print(f"Warning: Invalid quantization type '{self.config.get('quantization_type')}', defaulting to '8bit'")
            self.config["quantization_type"] = "8bit"
        
        # Set derived defaults
        if self.config.get("use_deepspeed") and not self.config.get("deepspeed_config_path"):
            self.config["deepspeed_config_path"] = os.path.join(os.path.dirname(__file__), "../configs/deepspeed_config.json")
    
    def get_config(self) -> Dict[str, Any]:
        """Get the full configuration."""
        return self.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key doesn't exist
            
        Returns:
            The configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a specific configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
        self._validate_config()  # Revalidate after changes 