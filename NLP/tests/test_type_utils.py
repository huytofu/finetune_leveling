import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import transformers
from collections import Counter

# Import the TypeUtils class
from explore_llm.NLP.classes.type_utils import TypeUtils


class MockParameter:
    """Mock Parameter class for testing."""
    
    def __init__(self, data=None, dtype=None, device='cpu'):
        self.data = data if data is not None else torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
        self.device = device


class TestTypeUtils(unittest.TestCase):
    """Unit tests for the TypeUtils class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.type_utils = TypeUtils()
        
        # Create a simple model for testing
        self.test_model = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5)
        )
        
        # Create a mock PEFT model
        self.mock_peft_model = MagicMock()
        self.mock_peft_model.is_peft_model = True
        self.mock_peft_model.get_adapter_names = MagicMock(return_value=["default"])
        self.mock_peft_model.modules = {}
        
        # Mock model with different parameter types
        self.mixed_dtype_model = MagicMock()
        
        # Create a dictionary to hold parameters
        self.mixed_params = {}
        self.mixed_params['param1'] = MockParameter(dtype=torch.float32)
        self.mixed_params['param2'] = MockParameter(dtype=torch.float16)
        self.mixed_params['param3'] = MockParameter(dtype=torch.float32)
        
        # Set up model parameters
        self.mixed_dtype_model.named_parameters = MagicMock(return_value=self.mixed_params.items())

    def test_check_model_dtypes(self):
        """Test checking model data types."""
        # Test with a model that has consistent dtypes
        result = self.type_utils.check_model_dtypes(self.test_model)
        self.assertIsNotNone(result)
        self.assertIn('primary_dtype', result)
        self.assertIn('dtypes_count', result)
        self.assertIn('is_mixed_precision', result)
        
        # Test with a model that has mixed dtypes
        result = self.type_utils.check_model_dtypes(self.mixed_dtype_model)
        self.assertTrue(result['is_mixed_precision'])
        self.assertEqual(result['dtypes_count'][torch.float32], 2)
        self.assertEqual(result['dtypes_count'][torch.float16], 1)
    
    def test_convert_model_dtype(self):
        """Test converting model dtype."""
        # Test converting a regular model
        with patch('torch.nn.Module.to') as mock_to:
            mock_to.return_value = self.test_model
            result = self.type_utils.convert_model_dtype(
                model=self.test_model,
                target_dtype="fp16"
            )
            mock_to.assert_called_once()
            self.assertEqual(result, self.test_model)
        
        # Test converting a PEFT model
        with patch('torch.nn.Module.to') as mock_to:
            mock_to.return_value = self.mock_peft_model
            result = self.type_utils.convert_model_dtype(
                model=self.mock_peft_model,
                target_dtype="bf16",
                exclude_modules=["lm_head"]
            )
            self.assertEqual(result, self.mock_peft_model)
    
    def test_validate_device_compatibility(self):
        """Test validating device compatibility."""
        # Create a device info Counter
        device_counter = Counter({'cpu': 2, 'cuda:0': 1})
        
        # Mock _get_device_info to return our counter
        with patch.object(self.type_utils, '_get_device_info') as mock_get_device_info:
            mock_get_device_info.return_value = device_counter
            
            # Call validate_device_compatibility
            result = self.type_utils.validate_device_compatibility(self.test_model)
            
            # Assertions
            self.assertIn('devices', result)
            self.assertIn('primary_device', result)
            self.assertIn('inconsistent_devices', result)
            self.assertTrue(result['inconsistent_devices'])
            self.assertEqual(result['primary_device'], 'cpu')
    
    def test_check_parameter_types(self):
        """Test checking for problematic parameter types."""
        # Create model with problematic parameters
        problem_model = MagicMock()
        
        # Create parameters with NaN and Inf values
        nan_param = MockParameter(data=torch.tensor([float('nan'), 1.0, 2.0]))
        inf_param = MockParameter(data=torch.tensor([float('inf'), 1.0, 2.0]))
        normal_param = MockParameter(data=torch.tensor([1.0, 2.0, 3.0]))
        
        problem_params = {
            'nan_param': nan_param,
            'inf_param': inf_param,
            'normal_param': normal_param
        }
        
        problem_model.named_parameters = MagicMock(return_value=problem_params.items())
        
        # Call check_parameter_types
        result = self.type_utils.check_parameter_types(problem_model)
        
        # Assertions
        self.assertIn('nan_params', result)
        self.assertIn('inf_params', result)
        self.assertIn('total_params', result)
        self.assertEqual(len(result['nan_params']), 1)
        self.assertEqual(len(result['inf_params']), 1)
        self.assertEqual(result['total_params'], 3)
    
    def test_automatic_type_repair(self):
        """Test automatic repair of common type issues."""
        # Create model with problematic parameters
        problem_model = MagicMock()
        
        # Create parameters with NaN and Inf values
        nan_param = MockParameter(data=torch.tensor([float('nan'), 1.0, 2.0]))
        inf_param = MockParameter(data=torch.tensor([float('inf'), 1.0, 2.0]))
        
        problem_params = {
            'nan_param': nan_param,
            'inf_param': inf_param
        }
        
        problem_model.named_parameters = MagicMock(return_value=problem_params.items())
        
        # Mock _replace_nan_values and _replace_inf_values
        with patch.object(self.type_utils, '_replace_nan_values') as mock_replace_nan:
            with patch.object(self.type_utils, '_replace_inf_values') as mock_replace_inf:
                with patch.object(self.type_utils, 'convert_model_dtype') as mock_convert_dtype:
                    # Set return values
                    mock_replace_nan.return_value = problem_model
                    mock_replace_inf.return_value = problem_model
                    mock_convert_dtype.return_value = problem_model
                    
                    # Call automatic_type_repair
                    result = self.type_utils.automatic_type_repair(
                        model=problem_model,
                        repair_nan=True,
                        repair_inf=True,
                        consolidate_dtypes=True
                    )
                    
                    # Assertions
                    self.assertEqual(result, problem_model)
                    mock_replace_nan.assert_called_once()
                    mock_replace_inf.assert_called_once()
                    mock_convert_dtype.assert_called_once()
    
    def test_get_device_info(self):
        """Test getting device information from a model."""
        # Create a model with parameters on different devices
        multi_device_model = MagicMock()
        
        # Create parameters on different devices
        cpu_param = MockParameter(device='cpu')
        cuda_param = MockParameter(device='cuda:0')
        
        device_params = {
            'cpu_param': cpu_param,
            'cuda_param': cuda_param
        }
        
        multi_device_model.named_parameters = MagicMock(return_value=device_params.items())
        
        # Call _get_device_info
        result = self.type_utils._get_device_info(multi_device_model)
        
        # Assertions
        self.assertIsInstance(result, Counter)
        self.assertEqual(result['cpu'], 1)
        self.assertEqual(result['cuda:0'], 1)


if __name__ == "__main__":
    unittest.main() 