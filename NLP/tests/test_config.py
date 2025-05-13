import os
import sys
import pytest
from typing import Dict, Any
import torch

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from modules.config.training_config import FineTuneConfig
from modules.config.mlflow_config import MLflowConfig, MLflowCallback, MLflowTracker

class TestFineTuneConfig:
    """Test suite for FineTuneConfig class."""
    
    def test_basic_initialization(self):
        """Test basic initialization of FineTuneConfig."""
        config = FineTuneConfig(
            model_name="test-model",
            task_type="test-task"
        )
        assert config.model_name == "test-model"
        assert config.task_type == "test-task"
        assert config.batch_size == 32  # default value
        assert config.learning_rate == 2e-5  # default value

    def test_peft_config_lora(self):
        """Test PEFT configuration for LoRA."""
        config = FineTuneConfig(
            model_name="test-model",
            task_type="test-task",
            use_peft=True,
            peft_method="lora",
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1
        )
        peft_config = config.get_peft_config()
        assert peft_config is not None
        assert peft_config["r"] == 16
        assert peft_config["lora_alpha"] == 32
        assert peft_config["lora_dropout"] == 0.1

    def test_peft_config_prefix(self):
        """Test PEFT configuration for prefix tuning."""
        config = FineTuneConfig(
            model_name="test-model",
            task_type="test-task",
            use_peft=True,
            peft_method="prefix",
            prefix_length=20,
            prefix_projection=True
        )
        peft_config = config.get_peft_config()
        assert peft_config is not None
        assert peft_config["num_virtual_tokens"] == 20
        assert peft_config["projection"] is True

    def test_framework_validation(self):
        """Test framework validation logic."""
        config = FineTuneConfig(
            model_name="test-model",
            task_type="test-task",
            use_lightning=True,
            use_accelerate=True,
            use_rlhf=True
        )
        config._validate_config()
        # RLHF should disable Lightning and Accelerate
        assert not config.use_lightning
        assert not config.use_accelerate

    @pytest.mark.parametrize("quantization_method", ["bitsandbytes", "auto_gptq", "awq"])
    def test_quantization_config(self, quantization_method):
        """Test quantization configuration."""
        config = FineTuneConfig(
            model_name="test-model",
            task_type="test-task",
            use_quantization=True,
            quantization_method=quantization_method,
            quantization_bits=8
        )
        assert config.use_quantization
        assert config.quantization_method == quantization_method
        assert config.quantization_bits == 8

class TestMLflowConfig:
    """Test suite for MLflow configuration classes."""
    
    def test_mlflow_config_initialization(self):
        """Test basic initialization of MLflowConfig."""
        config = MLflowConfig(
            experiment_name="test-experiment",
            run_name="test-run"
        )
        assert config.experiment_name == "test-experiment"
        assert config.run_name == "test-run"
        assert config.log_artifacts is True  # default value
        assert config.log_system_metrics is True  # default value

    def test_mlflow_callback(self, mocker):
        """Test MLflowCallback functionality."""
        # Mock UnifiedMonitor
        mock_monitor = mocker.Mock()
        callback = MLflowCallback(monitor=mock_monitor)
        
        # Mock training arguments and state
        mock_args = mocker.Mock(
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=2e-5,
            num_train_epochs=3,
            warmup_steps=100,
            weight_decay=0.01,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0
        )
        mock_state = mocker.Mock(epoch=1, global_step=100)
        
        # Test callback methods
        callback.on_init(mock_args, mock_state, None)
        mock_monitor.log_metrics.assert_called_once()
        
        callback.on_epoch_begin(mock_args, mock_state, None)
        mock_monitor.log_metric.assert_called_with("epoch_1", 1)

    def test_mlflow_tracker(self, mocker):
        """Test MLflowTracker functionality."""
        # Mock MLflow
        mocker.patch('mlflow.set_tracking_uri')
        mocker.patch('mlflow.get_experiment_by_name', return_value=None)
        mocker.patch('mlflow.create_experiment', return_value="test-id")
        mocker.patch('mlflow.start_run')
        mocker.patch('mlflow.active_run')
        
        config = MLflowConfig(
            experiment_name="test-experiment",
            tracking_uri="test-uri"
        )
        tracker = MLflowTracker(config)
        
        # Test tracker methods
        tracker.start_run("test-run")
        mlflow.set_tracking_uri.assert_called_once_with("test-uri")
        mlflow.create_experiment.assert_called_once()
        mlflow.start_run.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__]) 