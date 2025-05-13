import unittest
import torch
from unittest.mock import MagicMock, patch

# Import the QuantizationManager class
from finetune_leveling.NLP.classes.quantization_manager import QuantizationManager


class TestQuantizationManager(unittest.TestCase):
    """Unit tests for the QuantizationManager class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.quant_manager = QuantizationManager()
        
        # Mock model
        self.mock_model = MagicMock()
        self.mock_model.config = MagicMock()
        self.mock_model.config.model_type = "llama"
        
    @patch("transformers.BitsAndBytesConfig")
    def test_prepare_model_for_quantization_4bit(self, mock_bnb_config):
        """Test preparing model loading configuration for 4-bit quantization."""
        # Set up mock BitsAndBytesConfig
        mock_bnb_config.return_value = MagicMock()
        
        # Call prepare_model_for_quantization for 4-bit quantization
        result = self.quant_manager.prepare_model_for_quantization(
            model_name_or_path="llama-7b",
            quant_type="4bit"
        )
        
        # Assertions
        self.assertIn("quantization_config", result)
        mock_bnb_config.assert_called_once()
        args, kwargs = mock_bnb_config.call_args
        self.assertTrue(kwargs.get("load_in_4bit", False))
        self.assertFalse(kwargs.get("load_in_8bit", False))
    
    @patch("transformers.BitsAndBytesConfig")
    def test_prepare_model_for_quantization_8bit(self, mock_bnb_config):
        """Test preparing model loading configuration for 8-bit quantization."""
        # Set up mock BitsAndBytesConfig
        mock_bnb_config.return_value = MagicMock()
        
        # Call prepare_model_for_quantization for 8-bit quantization
        result = self.quant_manager.prepare_model_for_quantization(
            model_name_or_path="llama-7b",
            quant_type="8bit"
        )
        
        # Assertions
        self.assertIn("quantization_config", result)
        mock_bnb_config.assert_called_once()
        args, kwargs = mock_bnb_config.call_args
        self.assertFalse(kwargs.get("load_in_4bit", False))
        self.assertTrue(kwargs.get("load_in_8bit", False))
    
    @patch("transformers.BitsAndBytesConfig")
    def test_prepare_model_for_quantization_nf4(self, mock_bnb_config):
        """Test preparing model loading configuration for NF4 quantization."""
        # Set up mock BitsAndBytesConfig
        mock_bnb_config.return_value = MagicMock()
        
        # Call prepare_model_for_quantization for NF4 quantization
        result = self.quant_manager.prepare_model_for_quantization(
            model_name_or_path="llama-7b",
            quant_type="nf4"
        )
        
        # Assertions
        self.assertIn("quantization_config", result)
        mock_bnb_config.assert_called_once()
        args, kwargs = mock_bnb_config.call_args
        self.assertTrue(kwargs.get("load_in_4bit", False))
        self.assertEqual(kwargs.get("bnb_4bit_quant_type", ""), "nf4")
    
    @patch("transformers.BitsAndBytesConfig")
    def test_prepare_model_for_quantization_with_custom_config(self, mock_bnb_config):
        """Test preparing model with custom quantization configuration."""
        # Set up mock BitsAndBytesConfig
        mock_bnb_config.return_value = MagicMock()
        
        # Custom quantization config
        custom_config = {
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True
        }
        
        # Call prepare_model_for_quantization with custom config
        result = self.quant_manager.prepare_model_for_quantization(
            model_name_or_path="llama-7b",
            quant_type="4bit",
            custom_quantization_config=custom_config
        )
        
        # Assertions
        self.assertIn("quantization_config", result)
        mock_bnb_config.assert_called_once()
        args, kwargs = mock_bnb_config.call_args
        self.assertTrue(kwargs.get("load_in_4bit", False))
        self.assertEqual(kwargs.get("bnb_4bit_compute_dtype"), torch.float16)
        self.assertTrue(kwargs.get("bnb_4bit_use_double_quant"))
    
    def test_optimize_model_for_peft_lora(self):
        """Test optimizing a quantized model for LoRA."""
        # Mock the necessary methods
        self.mock_model.get_input_embeddings = MagicMock(return_value=torch.nn.Embedding(100, 100))
        
        # Mock the prepare_model_for_kbit_training
        with patch("peft.utils.prepare_model_for_kbit_training") as mock_prepare:
            mock_prepare.return_value = self.mock_model
            
            # Call optimize_model_for_peft for LoRA
            result = self.quant_manager.optimize_model_for_peft(
                model=self.mock_model,
                quant_type="4bit",
                peft_method="lora"
            )
            
            # Assertions
            self.assertEqual(result, self.mock_model)
            mock_prepare.assert_called_once_with(self.mock_model)
    
    def test_check_peft_compatible(self):
        """Test checking PEFT compatibility with different model types."""
        # Test with compatible combinations
        is_compatible, _ = self.quant_manager.check_peft_compatible(
            model=self.mock_model,
            peft_method="lora"
        )
        self.assertTrue(is_compatible)
        
        # Test with incompatible combinations
        self.mock_model.config.model_type = "unsupported_model"
        is_compatible, reason = self.quant_manager.check_peft_compatible(
            model=self.mock_model,
            peft_method="prompt_tuning"
        )
        self.assertFalse(is_compatible)
        self.assertIsNotNone(reason)
    
    def test_auto_adjust_batch_size(self):
        """Test auto-adjusting batch size based on quantization and PEFT method."""
        # Test with different combinations
        # 4-bit quantization with LoRA should allow larger batch sizes
        adjusted_bs_4bit_lora = self.quant_manager.auto_adjust_batch_size(
            base_batch_size=32,
            quant_type="4bit",
            peft_method="lora",
            max_length=512
        )
        
        # 8-bit quantization with Prefix Tuning should be more conservative
        adjusted_bs_8bit_prefix = self.quant_manager.auto_adjust_batch_size(
            base_batch_size=32,
            quant_type="8bit",
            peft_method="prefix_tuning",
            max_length=512
        )
        
        # No quantization should be most conservative
        adjusted_bs_no_quant = self.quant_manager.auto_adjust_batch_size(
            base_batch_size=32,
            quant_type=None,
            peft_method="lora",
            max_length=512
        )
        
        # Assertions - relationships between batch sizes
        self.assertGreaterEqual(adjusted_bs_4bit_lora, adjusted_bs_8bit_prefix)
        self.assertGreaterEqual(adjusted_bs_8bit_prefix, adjusted_bs_no_quant)
    
    def test_get_quant_config_for_model_type(self):
        """Test getting quantization configuration for specific model types."""
        # Test with known model type
        config = self.quant_manager._get_quant_config_for_model_type(
            model_type="llama",
            quant_type="4bit"
        )
        self.assertIsNotNone(config)
        
        # Test with unknown model type
        config = self.quant_manager._get_quant_config_for_model_type(
            model_type="unknown_model",
            quant_type="4bit"
        )
        self.assertIsNotNone(config)  # Should return default config
    
    def test_is_eligible_for_quantization(self):
        """Test checking if a model is eligible for quantization."""
        # Test with eligible model
        self.mock_model.config.model_type = "llama"
        is_eligible, _ = self.quant_manager._is_eligible_for_quantization(
            model=self.mock_model,
            quant_type="4bit"
        )
        self.assertTrue(is_eligible)
        
        # Test with ineligible model
        self.mock_model.config.model_type = "bert"  # Example for testing
        is_eligible, reason = self.quant_manager._is_eligible_for_quantization(
            model=self.mock_model,
            quant_type="fp4"  # Assuming fp4 is not supported for bert
        )
        self.assertFalse(is_eligible)
        self.assertIsNotNone(reason)


if __name__ == "__main__":
    unittest.main() 