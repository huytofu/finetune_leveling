import os
import sys
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from unittest.mock import Mock, patch
import tempfile
import json

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from modules.config.training_config import FineTuneConfig
from modules.config.mlflow_config import MLflowConfig
from modules.main.pipelines import InferencePipeline, FineTunePipeline
from modules.pipeline_modules import PipelineOrchestrator

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock(spec=AutoModelForCausalLM)
    model.config = Mock()
    model.config.model_type = "gpt2"
    return model

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock(spec=AutoTokenizer)
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    return tokenizer

@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    return {
        "train": Mock(),
        "validation": Mock()
    }

@pytest.fixture
def config_file():
    """Create a temporary config file for testing."""
    config = {
        "model_name": "sshleifer/tiny-gpt2",
        "task_type": "causal-lm",
        "batch_size": 2,
        "learning_rate": 1e-4,
        "num_epochs": 1
    }
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(config, f)
        return f.name

class TestInferencePipeline:
    """Test suite for InferencePipeline."""
    
    def test_initialization(self, mock_model, mock_tokenizer):
        """Test pipeline initialization."""
        pipeline = InferencePipeline(
            model=mock_model,
            tokenizer=mock_tokenizer,
            task_type="text-generation"
        )
        assert pipeline.model == mock_model
        assert pipeline.tokenizer == mock_tokenizer
        assert pipeline.task_type == "text-generation"
    
    def test_adapter_support(self, mock_model, mock_tokenizer):
        """Test adapter support functionality."""
        pipeline = InferencePipeline(
            model=mock_model,
            tokenizer=mock_tokenizer,
            task_type="text-generation",
            adapter_name="test-adapter"
        )
        assert pipeline.adapter_name == "test-adapter"
        mock_model.set_adapter.assert_called_once_with("test-adapter")
    
    @patch("torch.cuda.is_available", return_value=True)
    def test_device_placement(self, mock_cuda, mock_model, mock_tokenizer):
        """Test model device placement."""
        pipeline = InferencePipeline(
            model=mock_model,
            tokenizer=mock_tokenizer,
            task_type="text-generation"
        )
        mock_model.to.assert_called_once_with("cuda")

class TestFineTunePipeline:
    """Test suite for FineTunePipeline."""
    
    def test_initialization_from_config(self, config_file):
        """Test pipeline initialization from config file."""
        pipeline = FineTunePipeline.from_config(config_file)
        assert isinstance(pipeline.config, FineTuneConfig)
        assert pipeline.config.model_name == "sshleifer/tiny-gpt2"
    
    def test_initialization_with_mlflow(self, mock_model, mock_tokenizer, mock_dataset):
        """Test pipeline initialization with MLflow tracking."""
        config = FineTuneConfig(
            model_name="test-model",
            task_type="causal-lm"
        )
        mlflow_config = MLflowConfig(
            experiment_name="test-experiment",
            run_name="test-run"
        )
        
        pipeline = FineTunePipeline(
            model=mock_model,
            tokenizer=mock_tokenizer,
            datasets=mock_dataset,
            config=config,
            mlflow_config=mlflow_config
        )
        assert pipeline.mlflow_config == mlflow_config
    
    @patch("modules.pipeline_modules.PipelineOrchestrator")
    def test_fine_tune_execution(self, mock_orchestrator, mock_model, mock_tokenizer, mock_dataset):
        """Test fine-tuning execution."""
        config = FineTuneConfig(
            model_name="test-model",
            task_type="causal-lm",
            batch_size=2,
            num_epochs=1
        )
        
        pipeline = FineTunePipeline(
            model=mock_model,
            tokenizer=mock_tokenizer,
            datasets=mock_dataset,
            config=config
        )
        
        pipeline.fine_tune()
        mock_orchestrator.return_value.run_training.assert_called_once()

class TestPipelineOrchestrator:
    """Test suite for PipelineOrchestrator."""
    
    def test_initialization(self, mock_model, mock_tokenizer, mock_dataset):
        """Test orchestrator initialization."""
        config = FineTuneConfig(
            model_name="test-model",
            task_type="causal-lm"
        )
        
        orchestrator = PipelineOrchestrator(
            model=mock_model,
            tokenizer=mock_tokenizer,
            datasets=mock_dataset,
            config=config
        )
        assert orchestrator.model == mock_model
        assert orchestrator.config == config
    
    def test_trainer_selection(self, mock_model, mock_tokenizer, mock_dataset):
        """Test trainer selection based on configuration."""
        # Test Lightning trainer selection
        config = FineTuneConfig(
            model_name="test-model",
            task_type="causal-lm",
            use_lightning=True
        )
        orchestrator = PipelineOrchestrator(
            model=mock_model,
            tokenizer=mock_tokenizer,
            datasets=mock_dataset,
            config=config
        )
        trainer = orchestrator._select_trainer()
        assert "Lightning" in trainer.__class__.__name__
        
        # Test Accelerate trainer selection
        config.use_lightning = False
        config.use_accelerate = True
        trainer = orchestrator._select_trainer()
        assert "Accelerate" in trainer.__class__.__name__
    
    @patch("torch.cuda.is_available", return_value=True)
    def test_device_setup(self, mock_cuda, mock_model, mock_tokenizer, mock_dataset):
        """Test device setup."""
        config = FineTuneConfig(
            model_name="test-model",
            task_type="causal-lm"
        )
        orchestrator = PipelineOrchestrator(
            model=mock_model,
            tokenizer=mock_tokenizer,
            datasets=mock_dataset,
            config=config
        )
        orchestrator._setup_device()
        mock_model.to.assert_called_once_with("cuda")

if __name__ == "__main__":
    pytest.main([__file__]) 