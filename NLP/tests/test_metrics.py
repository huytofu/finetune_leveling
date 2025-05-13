import os
import sys
import pytest
import numpy as np
from typing import List, Dict
from datasets import Dataset
from unittest.mock import Mock, patch

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from modules.utils.metrics import MetricsCalculator
from modules.utils.evaluation import ModelEvaluator
from modules.utils.validation import ValidationManager

@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing."""
    return {
        "generated_text": [
            "This is a test prediction",
            "Another test prediction",
            "Third test prediction"
        ]
    }

@pytest.fixture
def sample_references():
    """Create sample references for testing."""
    return {
        "target_text": [
            "This is a reference",
            "Another reference",
            "Third reference"
        ]
    }

@pytest.fixture
def sample_logits():
    """Create sample logits for testing."""
    return np.random.randn(3, 100, 32000)  # batch_size=3, seq_len=100, vocab_size=32000

class TestMetricsCalculator:
    """Test suite for metrics calculation utilities."""
    
    def test_text_generation_metrics(self, sample_predictions, sample_references):
        """Test text generation metrics calculation."""
        calculator = MetricsCalculator()
        metrics = calculator.calculate_generation_metrics(
            predictions=sample_predictions["generated_text"],
            references=sample_references["target_text"]
        )
        
        # Check if all required metrics are present
        assert "bleu" in metrics
        assert "rouge" in metrics
        assert "meteor" in metrics
        
        # Check metric values are in valid ranges
        assert 0 <= metrics["bleu"] <= 100
        assert all(0 <= score <= 1 for score in metrics["rouge"].values())
    
    def test_classification_metrics(self):
        """Test classification metrics calculation."""
        calculator = MetricsCalculator()
        
        # Test with binary classification
        y_true = [0, 1, 0, 1, 1]
        y_pred = [0, 1, 0, 0, 1]
        
        metrics = calculator.calculate_classification_metrics(
            y_true=y_true,
            y_pred=y_pred
        )
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert all(0 <= score <= 1 for score in metrics.values())
    
    def test_language_modeling_metrics(self, sample_logits):
        """Test language modeling metrics calculation."""
        calculator = MetricsCalculator()
        
        # Create target tokens
        target_tokens = np.random.randint(0, 32000, size=(3, 100))
        
        metrics = calculator.calculate_language_modeling_metrics(
            logits=sample_logits,
            labels=target_tokens
        )
        
        assert "perplexity" in metrics
        assert "loss" in metrics
        assert metrics["perplexity"] > 0
        assert metrics["loss"] > 0

class TestModelEvaluator:
    """Test suite for model evaluation utilities."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
        model.forward = Mock(return_value=Mock(logits=torch.randn(1, 10, 100)))
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.batch_decode = Mock(return_value=["Test output"])
        return tokenizer
    
    def test_generation_evaluation(self, mock_model, mock_tokenizer, sample_references):
        """Test generation model evaluation."""
        evaluator = ModelEvaluator(
            model=mock_model,
            tokenizer=mock_tokenizer
        )
        
        results = evaluator.evaluate_generation(
            test_data=sample_references["target_text"],
            max_length=50
        )
        
        assert "bleu" in results
        assert "rouge" in results
        mock_model.generate.assert_called()
        mock_tokenizer.batch_decode.assert_called()
    
    def test_classification_evaluation(self, mock_model):
        """Test classification model evaluation."""
        evaluator = ModelEvaluator(model=mock_model)
        
        # Create test data
        test_data = Dataset.from_dict({
            "input_text": ["Test input"],
            "label": [1]
        })
        
        results = evaluator.evaluate_classification(test_data)
        
        assert "accuracy" in results
        assert "f1" in results
        mock_model.forward.assert_called()
    
    @patch("torch.no_grad")
    def test_language_modeling_evaluation(self, mock_no_grad, mock_model, mock_tokenizer):
        """Test language modeling evaluation."""
        evaluator = ModelEvaluator(
            model=mock_model,
            tokenizer=mock_tokenizer
        )
        
        # Create test data
        test_data = ["Test sentence 1", "Test sentence 2"]
        
        results = evaluator.evaluate_language_modeling(test_data)
        
        assert "perplexity" in results
        assert "loss" in results
        mock_model.forward.assert_called()

class TestValidationManager:
    """Test suite for validation utilities."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        return Mock()
    
    @pytest.fixture
    def validation_data(self):
        """Create validation dataset."""
        return Dataset.from_dict({
            "input_text": ["Test input 1", "Test input 2"],
            "target_text": ["Test target 1", "Test target 2"]
        })
    
    def test_validation_loop(self, mock_model, validation_data):
        """Test validation loop functionality."""
        manager = ValidationManager(
            model=mock_model,
            validation_data=validation_data,
            batch_size=1
        )
        
        # Mock forward pass
        mock_model.forward.return_value = Mock(
            loss=torch.tensor(0.5),
            logits=torch.randn(1, 10, 100)
        )
        
        results = manager.run_validation()
        
        assert "validation_loss" in results
        assert "validation_perplexity" in results
        assert mock_model.forward.call_count == 2  # Called once per example
    
    def test_early_stopping(self, mock_model, validation_data):
        """Test early stopping functionality."""
        manager = ValidationManager(
            model=mock_model,
            validation_data=validation_data,
            patience=2
        )
        
        # Simulate validation steps with increasing loss
        losses = [0.5, 0.6, 0.7, 0.8]
        for loss in losses:
            should_stop = manager.check_early_stopping({"validation_loss": loss})
            if len(losses) - losses.index(loss) <= 2:
                assert should_stop
            else:
                assert not should_stop
    
    def test_best_model_tracking(self, mock_model, validation_data):
        """Test best model tracking functionality."""
        manager = ValidationManager(
            model=mock_model,
            validation_data=validation_data
        )
        
        # Simulate validation steps
        metrics_history = [
            {"validation_loss": 0.5},
            {"validation_loss": 0.4},  # Better
            {"validation_loss": 0.6},
            {"validation_loss": 0.3}   # Best
        ]
        
        for metrics in metrics_history:
            is_best = manager.is_best_model(metrics)
            if metrics["validation_loss"] == 0.3:
                assert is_best
            else:
                assert not is_best

if __name__ == "__main__":
    pytest.main([__file__]) 