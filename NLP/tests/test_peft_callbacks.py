import unittest
import os
import shutil
import tempfile
import torch
from unittest.mock import MagicMock, patch
import pytorch_lightning as pl

# Import the PEFT callbacks
from explore_llm.NLP.classes.peft_callbacks import (
    PeftAdapterMonitorCallback,
    PeftEarlyPruningCallback,
    PeftAdapterFusionCallback
)


class TestPeftCallbacks(unittest.TestCase):
    """Unit tests for the PEFT callbacks."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Mock trainer, pl_module, and optimizer
        self.mock_trainer = MagicMock()
        self.mock_trainer.global_step = 10
        self.mock_trainer.current_epoch = 1
        
        self.mock_pl_module = MagicMock()
        
        # Mock PEFT model
        self.mock_model = MagicMock()
        self.mock_model.is_peft_model = True
        self.mock_model.get_adapter_names = MagicMock(return_value=["default"])
        
        # Set model attribute on the pl_module
        self.mock_pl_module.model = self.mock_model
        
        # Mock optimizer
        self.mock_optimizer = MagicMock()
        
        # Set optimizer attribute on the pl_module
        self.mock_pl_module.optimizers = MagicMock(return_value=self.mock_optimizer)
        
    def tearDown(self):
        """Clean up test environment after each test."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_adapter_monitor_callback_init(self):
        """Test initialization of PeftAdapterMonitorCallback."""
        # Create callback
        callback = PeftAdapterMonitorCallback(
            monitor_gradients=True, 
            monitor_weights=True,
            log_every_n_steps=100,
            save_path=os.path.join(self.test_dir, "monitor")
        )
        
        # Assertions
        self.assertTrue(callback.monitor_gradients)
        self.assertTrue(callback.monitor_weights)
        self.assertEqual(callback.log_every_n_steps, 100)
        self.assertEqual(callback.save_path, os.path.join(self.test_dir, "monitor"))
    
    def test_adapter_monitor_callback_on_train_batch_end(self):
        """Test on_train_batch_end method of PeftAdapterMonitorCallback."""
        # Create callback
        callback = PeftAdapterMonitorCallback(
            monitor_gradients=True, 
            monitor_weights=True,
            log_every_n_steps=10,  # Set to current global step for testing
            save_path=os.path.join(self.test_dir, "monitor")
        )
        
        # Mock log_weights and log_gradients methods
        callback.log_weights = MagicMock()
        callback.log_gradients = MagicMock()
        
        # Call on_train_batch_end
        callback.on_train_batch_end(
            self.mock_trainer, 
            self.mock_pl_module, 
            MagicMock(),  # outputs
            MagicMock(),  # batch
            0,            # batch_idx
        )
        
        # Assertions
        callback.log_weights.assert_called_once_with(self.mock_pl_module, self.mock_trainer)
        callback.log_gradients.assert_called_once_with(self.mock_pl_module, self.mock_trainer)
    
    def test_early_pruning_callback_init(self):
        """Test initialization of PeftEarlyPruningCallback."""
        # Create callback
        callback = PeftEarlyPruningCallback(
            prune_on_epoch_end=True,
            start_pruning_epoch=1,
            pruning_threshold=0.01,
            final_sparsity=0.5
        )
        
        # Assertions
        self.assertTrue(callback.prune_on_epoch_end)
        self.assertEqual(callback.start_pruning_epoch, 1)
        self.assertEqual(callback.pruning_threshold, 0.01)
        self.assertEqual(callback.final_sparsity, 0.5)
    
    def test_early_pruning_callback_on_epoch_end(self):
        """Test on_epoch_end method of PeftEarlyPruningCallback."""
        # Create callback
        callback = PeftEarlyPruningCallback(
            prune_on_epoch_end=True,
            start_pruning_epoch=1,  # Set to current epoch for testing
            pruning_threshold=0.01,
            final_sparsity=0.5
        )
        
        # Mock prune_adapters method
        callback.prune_adapters = MagicMock()
        
        # Call on_epoch_end
        callback.on_epoch_end(self.mock_trainer, self.mock_pl_module)
        
        # Assertions
        callback.prune_adapters.assert_called_once_with(
            pl_module=self.mock_pl_module,
            threshold=0.01,
            sparsity=0.5
        )
    
    def test_adapter_fusion_callback_init(self):
        """Test initialization of PeftAdapterFusionCallback."""
        # Create callback
        callback = PeftAdapterFusionCallback(
            adapter_list=["adapter1", "adapter2"],
            fusion_strategy="weighted",
            weights=[0.7, 0.3],
            fuse_on_fit_end=True
        )
        
        # Assertions
        self.assertEqual(callback.adapter_list, ["adapter1", "adapter2"])
        self.assertEqual(callback.fusion_strategy, "weighted")
        self.assertEqual(callback.weights, [0.7, 0.3])
        self.assertTrue(callback.fuse_on_fit_end)
    
    def test_adapter_fusion_callback_on_fit_end(self):
        """Test on_fit_end method of PeftAdapterFusionCallback."""
        # Create callback
        callback = PeftAdapterFusionCallback(
            adapter_list=["adapter1", "adapter2"],
            fusion_strategy="weighted",
            weights=[0.7, 0.3],
            fuse_on_fit_end=True
        )
        
        # Mock fuse_adapters method
        callback.fuse_adapters = MagicMock()
        
        # Call on_fit_end
        callback.on_fit_end(self.mock_trainer, self.mock_pl_module)
        
        # Assertions
        callback.fuse_adapters.assert_called_once_with(
            pl_module=self.mock_pl_module,
            adapter_list=["adapter1", "adapter2"],
            strategy="weighted",
            weights=[0.7, 0.3]
        )


if __name__ == "__main__":
    unittest.main() 