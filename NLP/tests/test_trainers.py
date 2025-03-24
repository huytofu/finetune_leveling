import os
import sys
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from unittest.mock import Mock, patch

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from modules.config.training_config import FineTuneConfig
from modules.config.mlflow_config import MLflowConfig
from classes.trainers.trainers import NLPTrainer, NLPSeq2SeqTrainer
from classes.trainers.trainers_with_lightning import AcceleratedNLPTrainer as LightningTrainer
from classes.trainers.accelerated_trainers import AcceleratedNLPTrainer as AccelerateTrainer

class TestNLPTrainer:
    """Test suite for base NLPTrainer class."""
    
    @pytest.fixture
    def mock_model(self):
        return Mock(spec=AutoModelForCausalLM)
    
    @pytest.fixture
    def mock_tokenizer(self):
        return Mock(spec=AutoTokenizer)
    
    @pytest.fixture
    def mock_dataset(self):
        return {
            "train": Mock(),
            "validation": Mock()
        }
    
    @pytest.fixture
    def config(self):
        return FineTuneConfig(
            model_name="test-model",
            task_type="causal-lm",
            batch_size=4,
            learning_rate=1e-4,
            num_epochs=2
        )
    
    def test_initialization(self, mock_model, mock_tokenizer, mock_dataset, config):
        """Test basic trainer initialization."""
        trainer = NLPTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            datasets=mock_dataset,
            config=config
        )
        assert trainer.model == mock_model
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.datasets == mock_dataset
        assert trainer.config == config
    
    def test_optimizer_setup(self, mock_model, mock_tokenizer, mock_dataset, config):
        """Test optimizer configuration."""
        trainer = NLPTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            datasets=mock_dataset,
            config=config
        )
        optimizer = trainer.configure_optimizer()
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == config.learning_rate
    
    @patch("torch.cuda.is_available", return_value=True)
    def test_memory_optimization(self, mock_cuda, mock_model, mock_tokenizer, mock_dataset, config):
        """Test memory optimization settings."""
        trainer = NLPTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            datasets=mock_dataset,
            config=config
        )
        trainer.optimize_memory_settings()
        mock_model.gradient_checkpointing_enable.assert_called_once()

class TestNLPSeq2SeqTrainer:
    """Test suite for NLPSeq2SeqTrainer class."""
    
    @pytest.fixture
    def seq2seq_config(self):
        return FineTuneConfig(
            model_name="test-model",
            task_type="seq2seq",
            batch_size=4,
            learning_rate=1e-4,
            num_epochs=2,
            max_source_length=128,
            max_target_length=64
        )
    
    def test_seq2seq_initialization(self, mock_model, mock_tokenizer, mock_dataset, seq2seq_config):
        """Test sequence-to-sequence trainer initialization."""
        trainer = NLPSeq2SeqTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            datasets=mock_dataset,
            config=seq2seq_config
        )
        assert trainer.max_source_length == 128
        assert trainer.max_target_length == 64

class TestLightningTrainer:
    """Test suite for Lightning-based trainer."""
    
    @pytest.fixture
    def lightning_config(self):
        return FineTuneConfig(
            model_name="test-model",
            task_type="causal-lm",
            batch_size=4,
            learning_rate=1e-4,
            num_epochs=2,
            use_lightning=True
        )
    
    def test_lightning_initialization(self, mock_model, mock_tokenizer, mock_dataset, lightning_config):
        """Test Lightning trainer initialization."""
        trainer = LightningTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            datasets=mock_dataset,
            config=lightning_config
        )
        assert trainer.automatic_optimization
        assert trainer.learning_rate == lightning_config.learning_rate
    
    @patch("pytorch_lightning.Trainer")
    def test_lightning_training_setup(self, mock_pl_trainer, mock_model, mock_tokenizer, mock_dataset, lightning_config):
        """Test Lightning training configuration."""
        trainer = LightningTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            datasets=mock_dataset,
            config=lightning_config
        )
        trainer.configure_optimizers()
        assert isinstance(trainer.optimizer, torch.optim.AdamW)

class TestAccelerateTrainer:
    """Test suite for Accelerate-based trainer."""
    
    @pytest.fixture
    def accelerate_config(self):
        return FineTuneConfig(
            model_name="test-model",
            task_type="causal-lm",
            batch_size=4,
            learning_rate=1e-4,
            num_epochs=2,
            use_accelerate=True
        )
    
    @patch("accelerate.Accelerator")
    def test_accelerate_initialization(self, mock_accelerator, mock_model, mock_tokenizer, mock_dataset, accelerate_config):
        """Test Accelerate trainer initialization."""
        trainer = AccelerateTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            datasets=mock_dataset,
            config=accelerate_config
        )
        assert trainer.config.use_accelerate
        assert trainer.learning_rate == accelerate_config.learning_rate
    
    @patch("accelerate.Accelerator")
    def test_accelerate_training_setup(self, mock_accelerator, mock_model, mock_tokenizer, mock_dataset, accelerate_config):
        """Test Accelerate training configuration."""
        trainer = AccelerateTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            datasets=mock_dataset,
            config=accelerate_config
        )
        optimizer = trainer.configure_optimizer()
        assert isinstance(optimizer, torch.optim.AdamW)

if __name__ == "__main__":
    pytest.main([__file__]) 