import os
import sys
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from unittest.mock import Mock, patch
import tempfile

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from modules.utils.model_utils import ModelOptimizer, GradientHandler
from modules.utils.memory_utils import MemoryManager
from modules.utils.checkpoint_utils import CheckpointManager

class TestModelOptimizer:
    """Test suite for model optimization utilities."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock(spec=AutoModelForCausalLM)
        model.parameters = lambda: [torch.nn.Parameter(torch.randn(10, 10))]
        return model
    
    def test_optimizer_creation(self, mock_model):
        """Test optimizer creation with different configurations."""
        optimizer = ModelOptimizer(mock_model)
        
        # Test AdamW creation
        opt = optimizer.create_optimizer(
            optimizer_type="adamw",
            learning_rate=1e-4,
            weight_decay=0.01
        )
        assert isinstance(opt, torch.optim.AdamW)
        assert opt.defaults["lr"] == 1e-4
        assert opt.defaults["weight_decay"] == 0.01
        
        # Test Adafactor creation
        opt = optimizer.create_optimizer(
            optimizer_type="adafactor",
            learning_rate=1e-3
        )
        assert "Adafactor" in opt.__class__.__name__
    
    def test_learning_rate_scheduling(self, mock_model):
        """Test learning rate scheduler creation."""
        optimizer = ModelOptimizer(mock_model)
        opt = optimizer.create_optimizer(optimizer_type="adamw", learning_rate=1e-4)
        
        # Test linear scheduler
        scheduler = optimizer.create_scheduler(
            optimizer=opt,
            scheduler_type="linear",
            num_training_steps=1000,
            num_warmup_steps=100
        )
        assert hasattr(scheduler, "step")
        
        # Test cosine scheduler
        scheduler = optimizer.create_scheduler(
            optimizer=opt,
            scheduler_type="cosine",
            num_training_steps=1000,
            num_warmup_steps=100
        )
        assert hasattr(scheduler, "step")
    
    def test_weight_initialization(self, mock_model):
        """Test weight initialization methods."""
        optimizer = ModelOptimizer(mock_model)
        
        # Test normal initialization
        optimizer.initialize_weights(
            method="normal",
            mean=0.0,
            std=0.02
        )
        mock_model.apply.assert_called_once()
        
        # Test xavier initialization
        optimizer.initialize_weights(method="xavier")
        assert mock_model.apply.call_count == 2

class TestGradientHandler:
    """Test suite for gradient handling utilities."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model with parameters."""
        model = Mock()
        param = torch.nn.Parameter(torch.randn(10, 10))
        param.grad = torch.randn(10, 10)
        model.parameters = lambda: [param]
        return model
    
    def test_gradient_clipping(self, mock_model):
        """Test gradient clipping functionality."""
        handler = GradientHandler(mock_model)
        
        # Test norm clipping
        handler.clip_gradients(max_grad_norm=1.0)
        for param in mock_model.parameters():
            grad_norm = torch.norm(param.grad)
            assert grad_norm <= 1.0
    
    def test_gradient_accumulation(self, mock_model):
        """Test gradient accumulation."""
        handler = GradientHandler(mock_model)
        
        # Test accumulation over steps
        for _ in range(4):
            handler.accumulate_gradients(accumulation_steps=4)
        
        # Check if gradients are properly scaled
        for param in mock_model.parameters():
            assert torch.allclose(param.grad, param.grad / 4)
    
    def test_gradient_checkpointing(self, mock_model):
        """Test gradient checkpointing setup."""
        handler = GradientHandler(mock_model)
        handler.enable_gradient_checkpointing()
        mock_model.gradient_checkpointing_enable.assert_called_once()

class TestMemoryManager:
    """Test suite for memory management utilities."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        return Mock(spec=AutoModelForCausalLM)
    
    @patch("torch.cuda.empty_cache")
    def test_memory_cleanup(self, mock_empty_cache, mock_model):
        """Test memory cleanup operations."""
        manager = MemoryManager()
        manager.cleanup()
        mock_empty_cache.assert_called_once()
    
    def test_memory_efficient_loading(self, mock_model):
        """Test memory-efficient model loading."""
        manager = MemoryManager()
        manager.load_in_8bit = True
        
        with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_load:
            manager.load_model("test-model")
            mock_load.assert_called_once_with(
                "test-model",
                load_in_8bit=True,
                device_map="auto"
            )
    
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    def test_memory_monitoring(self, mock_reserved, mock_allocated, mock_model):
        """Test memory monitoring functionality."""
        mock_allocated.return_value = 1000
        mock_reserved.return_value = 2000
        
        manager = MemoryManager()
        stats = manager.get_memory_stats()
        
        assert "allocated" in stats
        assert "reserved" in stats
        assert stats["allocated"] == 1000
        assert stats["reserved"] == 2000

class TestCheckpointManager:
    """Test suite for checkpoint management utilities."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        return Mock(spec=AutoModelForCausalLM)
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        return Mock(spec=AutoTokenizer)
    
    def test_checkpoint_saving(self, mock_model, mock_tokenizer):
        """Test checkpoint saving functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(save_dir=tmp_dir)
            
            # Test saving checkpoint
            checkpoint_path = manager.save_checkpoint(
                model=mock_model,
                tokenizer=mock_tokenizer,
                step=1000,
                metrics={"loss": 0.5}
            )
            
            assert os.path.exists(checkpoint_path)
            mock_model.save_pretrained.assert_called_once()
            mock_tokenizer.save_pretrained.assert_called_once()
    
    def test_checkpoint_loading(self, mock_model, mock_tokenizer):
        """Test checkpoint loading functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(save_dir=tmp_dir)
            
            with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_load_model:
                with patch("transformers.AutoTokenizer.from_pretrained") as mock_load_tokenizer:
                    manager.load_checkpoint(
                        checkpoint_path=tmp_dir,
                        model_class=AutoModelForCausalLM,
                        tokenizer_class=AutoTokenizer
                    )
                    
                    mock_load_model.assert_called_once_with(tmp_dir)
                    mock_load_tokenizer.assert_called_once_with(tmp_dir)
    
    def test_checkpoint_cleanup(self, mock_model, mock_tokenizer):
        """Test checkpoint cleanup functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(
                save_dir=tmp_dir,
                max_checkpoints=2
            )
            
            # Create multiple checkpoints
            for i in range(3):
                manager.save_checkpoint(
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    step=i * 1000,
                    metrics={"loss": 0.5 - i * 0.1}
                )
            
            # Check if only the best 2 checkpoints are kept
            checkpoints = os.listdir(tmp_dir)
            assert len(checkpoints) == 2

if __name__ == "__main__":
    pytest.main([__file__]) 