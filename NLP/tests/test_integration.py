import os
import sys
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from modules.config.training_config import FineTuneConfig
from modules.config.mlflow_config import MLflowConfig
from classes.trainers.trainers import NLPTrainer, NLPSeq2SeqTrainer
from classes.trainers.trainers_with_lightning import AcceleratedNLPTrainer as LightningTrainer
from classes.trainers.accelerated_trainers import AcceleratedNLPTrainer as AccelerateTrainer

@pytest.fixture(scope="module")
def tiny_gpt2():
    """Load a tiny GPT2 model for testing."""
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    return model, tokenizer

@pytest.fixture(scope="module")
def tiny_t5():
    """Load a tiny T5 model for testing."""
    model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/t5-small")
    return model, tokenizer

@pytest.fixture
def dummy_dataset():
    """Create a small dummy dataset for testing."""
    train_data = {
        "input_text": ["This is a test", "Another test example"],
        "target_text": ["Test output", "Example output"]
    }
    val_data = {
        "input_text": ["Validation test"],
        "target_text": ["Val output"]
    }
    return {
        "train": Dataset.from_dict(train_data),
        "validation": Dataset.from_dict(val_data)
    }

class TestPEFTIntegration:
    """Integration tests for PEFT methods."""
    
    @pytest.mark.parametrize("peft_method", ["lora", "prefix", "prompt"])
    def test_peft_with_base_trainer(self, tiny_gpt2, dummy_dataset, peft_method):
        """Test different PEFT methods with base trainer."""
        model, tokenizer = tiny_gpt2
        config = FineTuneConfig(
            model_name="sshleifer/tiny-gpt2",
            task_type="causal-lm",
            batch_size=2,
            learning_rate=1e-4,
            num_epochs=1,
            use_peft=True,
            peft_method=peft_method
        )
        
        trainer = NLPTrainer(
            model=model,
            tokenizer=tokenizer,
            datasets=dummy_dataset,
            config=config
        )
        
        # Run a single training step
        trainer.train(max_steps=1)
        assert hasattr(trainer.model, "peft_config")
    
    @pytest.mark.parametrize("peft_method", ["lora", "prefix"])
    def test_peft_with_seq2seq(self, tiny_t5, dummy_dataset, peft_method):
        """Test PEFT methods with sequence-to-sequence models."""
        model, tokenizer = tiny_t5
        config = FineTuneConfig(
            model_name="google/t5-small",
            task_type="seq2seq",
            batch_size=2,
            learning_rate=1e-4,
            num_epochs=1,
            use_peft=True,
            peft_method=peft_method,
            max_source_length=32,
            max_target_length=32
        )
        
        trainer = NLPSeq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            datasets=dummy_dataset,
            config=config
        )
        
        # Run a single training step
        trainer.train(max_steps=1)
        assert hasattr(trainer.model, "peft_config")

class TestFrameworkIntegration:
    """Integration tests for different training frameworks."""
    
    def test_lightning_training(self, tiny_gpt2, dummy_dataset):
        """Test training with PyTorch Lightning."""
        model, tokenizer = tiny_gpt2
        config = FineTuneConfig(
            model_name="sshleifer/tiny-gpt2",
            task_type="causal-lm",
            batch_size=2,
            learning_rate=1e-4,
            num_epochs=1,
            use_lightning=True
        )
        
        trainer = LightningTrainer(
            model=model,
            tokenizer=tokenizer,
            datasets=dummy_dataset,
            config=config
        )
        
        # Run a single training step
        trainer.train(max_steps=1)
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
    
    def test_accelerate_training(self, tiny_gpt2, dummy_dataset):
        """Test training with Hugging Face Accelerate."""
        model, tokenizer = tiny_gpt2
        config = FineTuneConfig(
            model_name="sshleifer/tiny-gpt2",
            task_type="causal-lm",
            batch_size=2,
            learning_rate=1e-4,
            num_epochs=1,
            use_accelerate=True
        )
        
        trainer = AccelerateTrainer(
            model=model,
            tokenizer=tokenizer,
            datasets=dummy_dataset,
            config=config
        )
        
        # Run a single training step
        trainer.train(max_steps=1)
        assert isinstance(trainer.optimizer, torch.optim.AdamW)

class TestCombinedFeatures:
    """Integration tests for combined features."""
    
    @pytest.mark.parametrize("framework", ["lightning", "accelerate"])
    @pytest.mark.parametrize("peft_method", ["lora", "prefix"])
    def test_peft_with_frameworks(self, tiny_gpt2, dummy_dataset, framework, peft_method):
        """Test PEFT methods with different training frameworks."""
        model, tokenizer = tiny_gpt2
        config = FineTuneConfig(
            model_name="sshleifer/tiny-gpt2",
            task_type="causal-lm",
            batch_size=2,
            learning_rate=1e-4,
            num_epochs=1,
            use_peft=True,
            peft_method=peft_method,
            use_lightning=(framework == "lightning"),
            use_accelerate=(framework == "accelerate")
        )
        
        trainer_cls = LightningTrainer if framework == "lightning" else AccelerateTrainer
        trainer = trainer_cls(
            model=model,
            tokenizer=tokenizer,
            datasets=dummy_dataset,
            config=config
        )
        
        # Run a single training step
        trainer.train(max_steps=1)
        assert hasattr(trainer.model, "peft_config")
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
    
    def test_quantization_with_peft(self, tiny_gpt2, dummy_dataset):
        """Test quantization with PEFT."""
        model, tokenizer = tiny_gpt2
        config = FineTuneConfig(
            model_name="sshleifer/tiny-gpt2",
            task_type="causal-lm",
            batch_size=2,
            learning_rate=1e-4,
            num_epochs=1,
            use_peft=True,
            peft_method="lora",
            use_quantization=True,
            quantization_method="bitsandbytes",
            quantization_bits=8
        )
        
        trainer = NLPTrainer(
            model=model,
            tokenizer=tokenizer,
            datasets=dummy_dataset,
            config=config
        )
        
        # Run a single training step
        trainer.train(max_steps=1)
        assert hasattr(trainer.model, "peft_config")
        # Check if model parameters are quantized
        for param in trainer.model.parameters():
            if param.requires_grad:
                assert param.dtype in [torch.int8, torch.uint8, torch.float16]

if __name__ == "__main__":
    pytest.main([__file__]) 