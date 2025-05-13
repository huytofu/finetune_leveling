# Fine-Tuning Pipeline Architecture

This document provides an in-depth overview of the fine-tuning pipeline architecture, explaining how different components interact to create a flexible and powerful fine-tuning framework.

## Overview

The fine-tuning pipeline is designed with a modular architecture that separates concerns into specialized components. This design allows for maximum flexibility and extensibility, while maintaining a clean and understandable codebase.

## Core Components

### 1. FineTunePipeLine

The `FineTunePipeLine` class serves as the main orchestrator of the fine-tuning process. It:

- Initializes and coordinates all other components
- Manages the flow of data between components
- Handles the setup of training configurations
- Coordinates distributed training when needed
- Integrates with tracking systems like MLflow
- Manages model checkpointing and saving

### 2. Trainers

The system has multiple trainer implementations to support different training frameworks:

#### Standard Trainers
- `NLPTrainer`: Base trainer for classification tasks
- `NLPSeq2SeqTrainer`: Specialized trainer for sequence-to-sequence tasks

#### Accelerate-based Trainers
- `AcceleratedNLPTrainer`: Uses Hugging Face Accelerate for hardware optimization
- `AcceleratedNLPSeq2SeqTrainer`: Accelerate-enabled trainer for sequence-to-sequence tasks

#### Lightning-based Trainers
- `NLPTrainerWithLightning`: Implements PyTorch Lightning for training
- `NLPSeq2SeqTrainerWithLightning`: Lightning-based trainer for sequence-to-sequence tasks

### 3. Dataset Modules

The `DatasetModules` class handles all aspects of dataset preparation:

- Loading datasets from different sources
- Preprocessing and tokenization
- Chunking for large datasets
- Streaming support for memory efficiency
- Distributed data sampling
- Data validation and quality checks

### 4. Model Modules

The `ModelModules` class manages model loading and configuration:

- Initializing models for different task types
- Applying PEFT methods (LoRA, QLoRA, etc.)
- Configuring model architecture
- Optimizing memory layout

### 5. Tokenizer Modules

The `TokenizerModules` class manages tokenizer loading and configuration:

- Loading the appropriate tokenizer for the model
- Configuring tokenization settings
- Handling special tokens

### 6. Support Modules

- `CheckpointManager`: Manages model checkpoints
- `TypeUtils`: Utilities for type handling and conversion
- `ErrorHandler`: Centralized error handling
- `QuantizationManager`: Handles model quantization

## Inference Pipeline

The inference pipeline uses the `InferencePipeLine` class to:

- Load trained models
- Support adapter management with `MultiAdapterManager`
- Process input data
- Generate predictions
- Post-process outputs

## Framework Coordination

### Lightning and Accelerate Integration

The system supports both PyTorch Lightning and Hugging Face Accelerate, and can use them individually or in combination:

#### Option 1: Accelerate Only
When `use_accelerate=True` and `use_lightning=False`:
- `AcceleratedNLPTrainer` or `AcceleratedNLPSeq2SeqTrainer` is used
- Accelerate manages device placement, distributed training, and mixed precision
- Memory optimization is handled by Accelerate
- The training loop is implemented directly within the trainer

#### Option 2: Lightning Only
When `use_accelerate=False` and `use_lightning=True`:
- `NLPTrainerWithLightning` or `NLPSeq2SeqTrainerWithLightning` is used
- Lightning manages the training loop through its lifecycle hooks
- Device placement and distributed training are handled by Lightning
- Lightning's built-in optimizations are leveraged

#### Option 3: Combined Approach
When `use_accelerate=True` and `use_lightning=True`:
- The Lightning-based trainer is used with Accelerate integration
- Lightning manages the overall training loop lifecycle
- Accelerate handles device placement and mixed precision
- Batch processing leverages Accelerate's optimizations
- The combined approach offers the best of both frameworks

## Data Flow

1. **Pipeline Initialization**: The `FineTunePipeLine` is initialized with task type and configuration
2. **Model and Tokenizer Loading**: `ModelModules` and `TokenizerModules` load the model and tokenizer
3. **Dataset Preparation**: `DatasetModules` prepares the dataset based on the task type
4. **Trainer Selection**: The appropriate trainer is selected based on `use_lightning` and `use_accelerate` flags
5. **Training Execution**: The trainer executes the training process
6. **Model Saving**: The trained model and checkpoint are saved
7. **Evaluation**: Model performance is evaluated against metrics

## Task Support

The system supports various NLP tasks:
- Masked Language Modeling
- Token Classification
- Sequence-to-Sequence (Translation, Summarization, Text Generation)
- Question Answering

Each task has specialized dataset preparation methods and evaluation metrics.

## Optimization Features

The pipeline includes several optimization features:
- Parameter-Efficient Fine-Tuning (PEFT) with methods like LoRA and QLoRA
- Quantization for reduced memory footprint
- Gradient checkpointing for memory efficiency
- Distributed training across multiple GPUs
- Streaming datasets for handling large data
- Efficient checkpointing

## Monitoring and Tracking

For experiment tracking and monitoring, the pipeline integrates with:
- MLflow for experiment tracking
- Custom monitoring for resource usage
- Performance metrics tracking
- Checkpointing for experiment reproducibility

## Error Handling

The system has a centralized error handling mechanism through the `ErrorHandler` class, which:
- Provides descriptive error messages
- Suggests solutions to common issues
- Logs errors for debugging
- Handles graceful exit when needed

## Customization

The pipeline is designed to be highly customizable:
- Configuration through `specs` dictionary
- Command-line interface via `cli.py`
- Pluggable components for custom dataset processing
- Custom evaluation metrics
- User-defined PEFT configurations 