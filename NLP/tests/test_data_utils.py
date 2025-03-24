import os
import sys
import pytest
from datasets import Dataset
import torch
from typing import Dict, List
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from modules.utils.data_processing import DataProcessor
from modules.utils.tokenization_utils import TokenizationHelper
from modules.utils.batch_utils import BatchProcessor

@pytest.fixture
def sample_texts():
    """Create sample texts for testing."""
    return {
        "input_text": [
            "This is a test input",
            "Another example input",
            "Third test case"
        ],
        "target_text": [
            "Test output",
            "Example output",
            "Third output"
        ]
    }

@pytest.fixture
def sample_dataset(sample_texts):
    """Create a sample dataset for testing."""
    return Dataset.from_dict(sample_texts)

class TestDataProcessor:
    """Test suite for data processing utilities."""
    
    def test_text_cleaning(self, sample_texts):
        """Test text cleaning functionality."""
        processor = DataProcessor()
        cleaned_texts = processor.clean_texts(sample_texts["input_text"])
        
        # Check basic cleaning operations
        assert all(text.strip() == text for text in cleaned_texts)
        assert all(not text.endswith("\n") for text in cleaned_texts)
        assert all(not "  " in text for text in cleaned_texts)  # no double spaces
    
    def test_sequence_length_validation(self, sample_texts):
        """Test sequence length validation."""
        processor = DataProcessor(max_length=20)
        valid_sequences = processor.validate_sequence_lengths(
            sample_texts["input_text"],
            max_length=20
        )
        assert all(len(text.split()) <= 20 for text in valid_sequences)
    
    def test_data_augmentation(self, sample_texts):
        """Test data augmentation methods."""
        processor = DataProcessor()
        augmented_texts = processor.augment_data(
            sample_texts["input_text"],
            methods=["synonym_replacement", "back_translation"]
        )
        
        # Check augmentation results
        assert len(augmented_texts) > len(sample_texts["input_text"])
        assert all(isinstance(text, str) for text in augmented_texts)

class TestTokenizationHelper:
    """Test suite for tokenization utilities."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create a mock tokenizer."""
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def test_batch_tokenization(self, sample_texts, tokenizer):
        """Test batch tokenization functionality."""
        helper = TokenizationHelper(tokenizer)
        encoded = helper.batch_tokenize(
            sample_texts["input_text"],
            max_length=32,
            padding=True,
            truncation=True
        )
        
        assert isinstance(encoded, dict)
        assert "input_ids" in encoded
        assert "attention_mask" in encoded
        assert all(len(ids) <= 32 for ids in encoded["input_ids"])
    
    def test_special_token_handling(self, sample_texts, tokenizer):
        """Test special token handling."""
        helper = TokenizationHelper(tokenizer)
        encoded = helper.batch_tokenize(
            sample_texts["input_text"],
            add_special_tokens=True
        )
        
        # Check if special tokens are properly added
        assert all(seq[0] == tokenizer.cls_token_id for seq in encoded["input_ids"])
        assert all(seq[-1] == tokenizer.sep_token_id for seq in encoded["input_ids"])
    
    def test_dynamic_padding(self, sample_texts, tokenizer):
        """Test dynamic padding functionality."""
        helper = TokenizationHelper(tokenizer)
        encoded = helper.batch_tokenize(
            sample_texts["input_text"],
            padding="longest"
        )
        
        # Check if all sequences in batch have same length
        seq_lengths = [len(seq) for seq in encoded["input_ids"]]
        assert all(length == seq_lengths[0] for length in seq_lengths)

class TestBatchProcessor:
    """Test suite for batch processing utilities."""
    
    def test_batch_creation(self, sample_dataset):
        """Test batch creation functionality."""
        processor = BatchProcessor(batch_size=2)
        batches = list(processor.create_batches(sample_dataset))
        
        # Check batch properties
        assert len(batches) == 2  # 3 samples with batch_size=2 should give 2 batches
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1
    
    def test_dynamic_batching(self, sample_dataset):
        """Test dynamic batching based on sequence length."""
        processor = BatchProcessor(
            batch_size=2,
            max_tokens_per_batch=50
        )
        batches = list(processor.create_dynamic_batches(
            sample_dataset,
            length_func=lambda x: len(x.split())
        ))
        
        # Check if batches respect max_tokens_per_batch
        for batch in batches:
            total_tokens = sum(len(text.split()) for text in batch["input_text"])
            assert total_tokens <= 50
    
    def test_batch_collation(self, sample_dataset):
        """Test batch collation functionality."""
        processor = BatchProcessor(batch_size=2)
        
        def collate_fn(examples: List[Dict]) -> Dict:
            return {
                "input_text": [ex["input_text"] for ex in examples],
                "target_text": [ex["target_text"] for ex in examples]
            }
        
        batches = list(processor.create_batches(
            sample_dataset,
            collate_fn=collate_fn
        ))
        
        # Check collated batch structure
        assert all(isinstance(batch, dict) for batch in batches)
        assert all("input_text" in batch for batch in batches)
        assert all("target_text" in batch for batch in batches)

if __name__ == "__main__":
    pytest.main([__file__]) 