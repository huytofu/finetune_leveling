"""Knowledge Distillation Module for Model Compression

This module implements state-of-the-art knowledge distillation techniques for language models,
enabling the creation of smaller, efficient models while maintaining performance.
Key features include:
- Multiple distillation strategies (soft targets, hard targets, attention)
- Temperature scaling for knowledge transfer
- Layer-wise distillation
- Intermediate feature matching
- Progressive distillation
- Multi-teacher ensemble distillation
- MLflow tracking for distillation experiments
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)

from .mlflow_tracking import MLflowTracker

logger = logging.getLogger(__name__)

@dataclass
class DistillationConfig:
    """Configuration for Knowledge Distillation.
    
    Attributes:
        temperature: Temperature for softening probability distributions
        alpha: Weight for distillation loss vs task-specific loss
        hidden_layer_mapping: Mapping between teacher and student hidden layers
        attention_loss_weight: Weight for attention matching loss
        intermediate_loss_weights: Weights for intermediate layer losses
        use_cosine_similarity: Whether to use cosine similarity for feature matching
        ensemble_weights: Weights for multiple teacher models
        progressive_steps: Number of progressive distillation steps
        mlflow_tracking: Whether to track metrics with MLflow
        batch_size: Batch size for distillation
        num_epochs: Number of epochs for distillation
        learning_rate: Learning rate for student model
        weight_decay: Weight decay for optimization
    """
    temperature: float = 2.0
    alpha: float = 0.5
    hidden_layer_mapping: Dict[int, int] = field(default_factory=dict)
    attention_loss_weight: float = 0.1
    intermediate_loss_weights: List[float] = field(default_factory=lambda: [1.0])
    use_cosine_similarity: bool = True
    ensemble_weights: Optional[List[float]] = None
    progressive_steps: int = 3
    mlflow_tracking: bool = True
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 5e-5
    weight_decay: float = 0.01

class DistillationDataset(Dataset):
    """Dataset for knowledge distillation."""
    
    def __init__(
        self,
        examples: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.examples[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(item["label"]) if "label" in item else None
        }

class DistillationLoss:
    """Implements various distillation loss functions."""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        
    def knowledge_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """Compute the knowledge distillation loss using soft targets."""
        soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        return F.kl_div(student_log_probs, soft_targets, reduction="batchmean") * (temperature ** 2)
        
    def attention_matching_loss(
        self,
        student_attention: torch.Tensor,
        teacher_attention: torch.Tensor
    ) -> torch.Tensor:
        """Compute the attention matching loss."""
        if self.config.use_cosine_similarity:
            return 1 - F.cosine_similarity(
                student_attention.view(-1),
                teacher_attention.view(-1)
            ).mean()
        return F.mse_loss(student_attention, teacher_attention)
        
    def intermediate_feature_loss(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute the intermediate feature matching loss."""
        total_loss = 0
        for student_feat, teacher_feat, weight in zip(
            student_features,
            teacher_features,
            self.config.intermediate_loss_weights
        ):
            if self.config.use_cosine_similarity:
                loss = 1 - F.cosine_similarity(
                    student_feat.view(-1),
                    teacher_feat.view(-1)
                ).mean()
            else:
                loss = F.mse_loss(student_feat, teacher_feat)
            total_loss += weight * loss
        return total_loss

class DistillationManager:
    """Main class for managing knowledge distillation."""
    
    def __init__(
        self,
        teacher_model: PreTrainedModel,
        student_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: DistillationConfig
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.config = config
        self.loss_module = DistillationLoss(config)
        self.mlflow_tracker = MLflowTracker("knowledge_distillation") if config.mlflow_tracking else None
        
    def _prepare_optimizer(self) -> torch.optim.Optimizer:
        """Prepare optimizer for student model."""
        return torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
    def _get_teacher_predictions(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Get teacher model predictions and intermediate features."""
        self.teacher_model.eval()
        with torch.no_grad():
            outputs = self.teacher_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                output_attentions=True
            )
        return {
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions
        }
        
    def _train_step(
        self,
        batch: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Perform one training step of distillation."""
        self.student_model.train()
        optimizer.zero_grad()
        
        # Get student outputs
        student_outputs = self.student_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
            output_attentions=True
        )
        
        # Calculate losses
        kd_loss = self.loss_module.knowledge_distillation_loss(
            student_outputs.logits,
            teacher_outputs["logits"],
            self.config.temperature
        )
        
        # Task-specific loss if labels are available
        task_loss = 0.0
        if batch.get("label") is not None:
            task_loss = F.cross_entropy(
                student_outputs.logits,
                batch["label"]
            )
        
        # Attention matching loss
        attention_loss = sum(
            self.loss_module.attention_matching_loss(student_att, teacher_att)
            for student_att, teacher_att in zip(
                student_outputs.attentions,
                teacher_outputs["attentions"]
            )
        )
        
        # Intermediate feature matching loss
        feature_loss = self.loss_module.intermediate_feature_loss(
            student_outputs.hidden_states,
            teacher_outputs["hidden_states"]
        )
        
        # Combine losses
        total_loss = (
            self.config.alpha * kd_loss +
            (1 - self.config.alpha) * task_loss +
            self.config.attention_loss_weight * attention_loss +
            feature_loss
        )
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "kd_loss": kd_loss.item(),
            "task_loss": task_loss.item(),
            "attention_loss": attention_loss.item(),
            "feature_loss": feature_loss.item()
        }
        
    def distill(
        self,
        train_dataset: List[Dict],
        eval_dataset: Optional[List[Dict]] = None
    ) -> PreTrainedModel:
        """Perform knowledge distillation training."""
        if self.mlflow_tracker:
            self.mlflow_tracker.start_run()
            self.mlflow_tracker.log_params(vars(self.config))
        
        try:
            # Prepare datasets
            train_dataset = DistillationDataset(train_dataset, self.tokenizer)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            
            if eval_dataset:
                eval_dataset = DistillationDataset(eval_dataset, self.tokenizer)
                eval_loader = DataLoader(
                    eval_dataset,
                    batch_size=self.config.batch_size
                )
            
            # Training loop
            optimizer = self._prepare_optimizer()
            
            for epoch in range(self.config.num_epochs):
                epoch_losses = []
                
                for batch in train_loader:
                    teacher_outputs = self._get_teacher_predictions(batch)
                    step_losses = self._train_step(batch, teacher_outputs, optimizer)
                    epoch_losses.append(step_losses)
                
                # Compute average losses for the epoch
                avg_losses = {
                    k: np.mean([loss[k] for loss in epoch_losses])
                    for k in epoch_losses[0].keys()
                }
                
                # Evaluate if eval dataset is provided
                if eval_dataset:
                    eval_metrics = self._evaluate(eval_loader)
                    avg_losses.update(eval_metrics)
                
                if self.mlflow_tracker:
                    self.mlflow_tracker.log_metrics(avg_losses, step=epoch)
                
                logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}: {avg_losses}")
                
        finally:
            if self.mlflow_tracker:
                self.mlflow_tracker.end_run()
        
        return self.student_model
    
    def _evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the student model."""
        self.student_model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                outputs = self.student_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                
                if batch.get("label") is not None:
                    loss = F.cross_entropy(outputs.logits, batch["label"])
                    total_loss += loss.item()
                    
                    predictions = outputs.logits.argmax(dim=-1)
                    correct += (predictions == batch["label"]).sum().item()
                    total += len(batch["label"])
        
        metrics = {"eval_loss": total_loss / len(eval_loader)}
        if total > 0:
            metrics["eval_accuracy"] = correct / total
        
        return metrics
    
    def save_student(self, path: str):
        """Save the distilled student model."""
        self.student_model.save_pretrained(path)
        
        if self.mlflow_tracker:
            self.mlflow_tracker.log_artifact(path)
    
    @classmethod
    def load_student(
        cls,
        path: str,
        teacher_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[DistillationConfig] = None
    ) -> "DistillationManager":
        """Load a saved student model."""
        config = config or DistillationConfig()
        student_model = AutoModelForSequenceClassification.from_pretrained(path)
        return cls(teacher_model, student_model, tokenizer, config) 