# Knowledge Distillation Guide

## Overview
Knowledge distillation is a technique for model compression where a smaller model (student) is trained to mimic a larger model (teacher). This guide explains how to use the knowledge distillation capabilities in our NLP pipeline.

## Table of Contents
1. [Basic Concepts](#basic-concepts)
2. [Model Selection Guidelines](#model-selection-guidelines)
3. [Configuration Options](#configuration-options)
4. [Usage Examples](#usage-examples)
5. [Best Practices](#best-practices)
6. [Monitoring and Evaluation](#monitoring-and-evaluation)

## Basic Concepts

### What is Knowledge Distillation?
Knowledge distillation involves:
- A teacher model (larger, well-trained model)
- A student model (smaller model to be trained)
- Soft targets (teacher's softened probability distributions)
- Hard targets (actual labels)

### Minimum Requirements
- Teacher model: Pre-trained model with good performance
- Student model: Smaller architecture (typically 2-4x smaller)
- Training data: Same data used to train the teacher
- Validation data: For monitoring distillation quality

## Model Selection Guidelines

### Teacher Models
Choose based on task type:

```python
RECOMMENDED_TEACHER_MODELS = {
    "text_classification": {
        "default": "bert-large-uncased",
        "high_performance": "roberta-large",
        "multilingual": "xlm-roberta-large"
    },
    "summarization": {
        "default": "t5-large",
        "high_performance": "facebook/bart-large",
        "multilingual": "google/mt5-large"
    },
    "translation": {
        "default": "facebook/mbart-large-50",
        "high_performance": "google/mt5-large",
        "specialized": "Helsinki-NLP/opus-mt-en-ROMANCE"
    },
    "question_answering": {
        "default": "roberta-large",
        "high_performance": "deepset/deberta-v3-large-squad2",
        "efficient": "google/electra-large-discriminator"
    }
}
```

### Student Models
Choose based on deployment constraints:

```python
RECOMMENDED_STUDENT_MODELS = {
    "text_classification": {
        "balanced": "bert-base-uncased",
        "efficient": "distilbert-base-uncased",
        "tiny": "prajjwal1/bert-tiny"
    },
    "summarization": {
        "balanced": "t5-small",
        "efficient": "facebook/bart-base",
        "tiny": "sshleifer/tiny-mbart"
    },
    "translation": {
        "balanced": "facebook/mbart-large-50-small",
        "efficient": "Helsinki-NLP/opus-mt-en-fr",
        "tiny": "facebook/m2m100_418M"
    },
    "question_answering": {
        "balanced": "roberta-base",
        "efficient": "distilroberta-base",
        "tiny": "google/mobilebert-uncased"
    }
}
```

## Configuration Options

### Basic Configuration
```python
from explore_llm.NLP.modules.pipelines import FineTuneConfig

config = FineTuneConfig(
    model_name="distilbert-base-uncased",  # Student model
    task_type="text_classification",
    use_distillation=True,
    teacher_model_name="bert-large-uncased",
    distillation_alpha=0.5,  # Weight between soft and hard targets
    temperature=2.0  # Softening temperature
)
```

### Advanced Configuration
```python
config = FineTuneConfig(
    model_name="distilbert-base-uncased",
    task_type="text_classification",
    
    # Distillation settings
    use_distillation=True,
    teacher_model_name="bert-large-uncased",
    distillation_alpha=0.7,
    temperature=3.0,
    
    # What to distill
    distill_logits=True,
    distill_hidden_states=True,
    teacher_layers_to_distill=[0, 6, 11],
    
    # Training settings
    batch_size=32,
    learning_rate=2e-5,
    num_epochs=3
)
```

## Usage Examples

### Basic Text Classification
```python
from explore_llm.NLP.modules.pipelines import FineTunePipeline

# Configure distillation
config = FineTuneConfig(
    model_name="distilbert-base-uncased",
    task_type="text_classification",
    use_distillation=True,
    teacher_model_name="bert-large-uncased"
)

# Create and train pipeline
pipeline = FineTunePipeline(config)
pipeline.train(
    train_dataset=train_data,
    eval_dataset=eval_data
)
```

### Summarization with Layer Selection
```python
config = FineTuneConfig(
    model_name="t5-small",
    task_type="summarization",
    use_distillation=True,
    teacher_model_name="t5-large",
    distillation_alpha=0.6,
    temperature=2.5,
    distill_hidden_states=True,
    teacher_layers_to_distill=[0, 4, 8, 11]
)
```

### Question Answering with Custom Settings
```python
config = FineTuneConfig(
    model_name="distilroberta-base",
    task_type="question_answering",
    use_distillation=True,
    teacher_model_name="roberta-large",
    distillation_alpha=0.5,
    temperature=2.0,
    distill_logits=True,
    teacher_layers_to_distill=[0, 3, 5],
    
    # Additional training optimizations
    use_mixed_precision=True,
    use_dynamic_batching=True
)
```

## Best Practices

### Temperature Selection
- Simple classification: 1.5 - 2.0
- Complex tasks: 2.0 - 4.0
- Rule of thumb: Higher temperature = softer distributions

```python
# Task complexity based temperature
TEMPERATURE_GUIDE = {
    "binary_classification": 1.5,
    "multi_class_classification": 2.0,
    "token_classification": 2.5,
    "summarization": 3.0,
    "translation": 3.0,
    "question_answering": 2.5
}
```

### Layer Selection
Choose layers based on task:

```python
LAYER_SELECTION_GUIDE = {
    "classification": {
        "strategy": "top_layers",
        "layers": [-1, -2, -3]
    },
    "generation": {
        "strategy": "distributed",
        "layers": [0, 4, 8, 12]
    },
    "qa": {
        "strategy": "bookend",
        "layers": [0, 1, 10, 11]
    }
}
```

### Model Size Ratios
Recommended teacher-student size ratios:

```python
MODEL_SIZE_RATIOS = {
    "conservative": {
        "ratio": "2:1",
        "example": "bert-base → distilbert-base"
    },
    "standard": {
        "ratio": "3:1",
        "example": "bert-large → bert-base"
    },
    "aggressive": {
        "ratio": "4:1",
        "example": "t5-large → t5-small"
    }
}
```

## Monitoring and Evaluation

### Key Metrics to Monitor
1. Distillation Loss Components:
   - Student loss on hard targets
   - Distillation loss (KL divergence with teacher)
   - Combined loss

2. Performance Metrics:
   - Student vs Teacher accuracy
   - Inference speed comparison
   - Model size reduction

### Example Monitoring Setup
```python
from explore_llm.NLP.modules.monitoring import DistillationMonitor

monitor = DistillationMonitor(
    teacher_model=teacher,
    student_model=student,
    log_dir="distillation_logs"
)

# During training
monitor.log_metrics(
    step=step,
    student_loss=student_loss,
    distill_loss=distill_loss,
    teacher_predictions=teacher_preds,
    student_predictions=student_preds
)

# Generate visualization
monitor.plot_training_progress()
monitor.plot_prediction_comparison()
```

### Evaluation Criteria
- Student should achieve at least 90% of teacher's performance
- Inference speed should improve by at least 50%
- Model size should reduce by at least 40%

## Common Issues and Solutions

### Problem: High Distillation Loss
Solution:
- Increase temperature
- Adjust distillation_alpha
- Add more teacher layers to distill

### Problem: Poor Student Performance
Solution:
- Use larger student model
- Increase training epochs
- Add hidden state distillation

### Problem: Slow Training
Solution:
- Reduce number of teacher layers
- Use mixed precision training
- Implement gradient checkpointing

## Additional Resources
- [Original Knowledge Distillation Paper](https://arxiv.org/abs/1503.02531)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [TinyBERT Paper](https://arxiv.org/abs/1909.10351) 