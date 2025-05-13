import pytest
import torch
from unittest.mock import Mock, patch
from ..modules.training_optimizations import (
    HardwareSpecs,
    OptimizationConfig,
    HardwareAwareOptimizer
)

@pytest.fixture
def mock_model():
    model = Mock()
    model.gradient_checkpointing_enable = Mock()
    model.enable_flash_attention = Mock()
    return model

@pytest.fixture
def mock_hardware_specs():
    return HardwareSpecs(
        gpu_name="NVIDIA A100-SXM4-40GB",
        gpu_memory=40,
        gpu_count=2,
        cpu_count=32,
        total_ram=128,
        cuda_available=True,
        cuda_version="11.8",
        compute_capability=8.0,
        is_ampere_or_newer=True,
        is_ampere=True,
        is_hopper=False,
        is_ada_lovelace=False
    )

def test_hardware_detection():
    """Test hardware detection functionality."""
    config = OptimizationConfig()
    assert config.hardware_specs is not None
    assert isinstance(config.hardware_specs, HardwareSpecs)

def test_auto_optimization_configuration(mock_hardware_specs):
    """Test automatic optimization configuration based on hardware."""
    config = OptimizationConfig(hardware_specs=mock_hardware_specs)
    
    # Test memory efficiency settings
    assert config.enable_gradient_checkpointing
    assert config.enable_dynamic_batching
    assert config.target_batch_size <= 16
    assert config.max_batch_size <= 64
    
    # Test parallelism settings
    assert config.enable_smart_prefetching
    assert config.prefetch_factor >= 2
    
    # Test computation settings
    assert config.enable_mixed_precision
    assert config.mixed_precision_dtype == "fp16"  # A100 uses FP16

def test_hardware_aware_optimizer(mock_model, mock_hardware_specs):
    """Test hardware-aware optimizer initialization and optimization application."""
    config = OptimizationConfig(hardware_specs=mock_hardware_specs)
    optimizer = HardwareAwareOptimizer(mock_model, config)
    
    # Verify hardware specs
    assert optimizer.hardware_specs == mock_hardware_specs
    
    # Verify optimization summary
    summary = optimizer.get_optimization_summary()
    assert "Hardware Specifications" in summary
    assert "Applied Optimizations" in summary
    assert "Memory Efficiency" in summary
    assert "Parallelism" in summary
    assert "Computation Optimization" in summary

def test_memory_efficiency_optimizations(mock_model):
    """Test memory efficiency optimizations for low-memory systems."""
    specs = HardwareSpecs(
        gpu_memory=16,
        gpu_count=1,
        total_ram=32
    )
    config = OptimizationConfig(hardware_specs=specs)
    optimizer = HardwareAwareOptimizer(mock_model, config)
    
    # Verify memory efficiency settings
    assert config.enable_gradient_checkpointing
    assert config.target_batch_size <= 16
    assert config.max_batch_size <= 64

def test_parallelism_optimizations(mock_model):
    """Test parallelism optimizations for multi-GPU systems."""
    specs = HardwareSpecs(
        gpu_count=4,
        gpu_memory=32
    )
    config = OptimizationConfig(hardware_specs=specs)
    optimizer = HardwareAwareOptimizer(mock_model, config)
    
    # Verify parallelism settings
    assert config.enable_smart_prefetching
    assert config.prefetch_factor >= 4
    assert isinstance(optimizer.model, torch.nn.DataParallel)

def test_computation_optimizations(mock_model):
    """Test computation optimizations for modern GPUs."""
    specs = HardwareSpecs(
        compute_capability=9.0,
        is_ampere_or_newer=True,
        is_hopper=True
    )
    config = OptimizationConfig(hardware_specs=specs)
    optimizer = HardwareAwareOptimizer(mock_model, config)
    
    # Verify computation settings
    assert config.enable_mixed_precision
    assert config.mixed_precision_dtype == "bf16"  # Hopper uses BF16
    mock_model.enable_flash_attention.assert_called_once()

def test_data_optimizations(mock_model):
    """Test data optimizations for systems with limited RAM."""
    specs = HardwareSpecs(
        total_ram=32,
        gpu_memory=16
    )
    config = OptimizationConfig(hardware_specs=specs)
    optimizer = HardwareAwareOptimizer(mock_model, config)
    
    # Verify data optimization settings
    assert config.enable_dynamic_batching
    assert config.memory_threshold == 0.75
    assert config.enable_smart_prefetching 