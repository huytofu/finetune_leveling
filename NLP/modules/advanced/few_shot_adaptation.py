"""Few-Shot Adaptation Module for Enhanced Fine-tuning

This module implements advanced few-shot adaptation techniques for language models,
enabling efficient learning from limited examples while maintaining model performance.
It includes support for:
- Dynamic prompt engineering
- Demonstration selection and ranking
- In-context learning optimization
- Meta-learning approaches
- MLflow tracking for few-shot experiments
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Tuple
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import mlflow
from transformers import PreTrainedModel, PreTrainedTokenizer

from .mlflow_tracking import MLflowTracker

logger = logging.getLogger(__name__)

@dataclass
class FewShotConfig:
    """Configuration for Few-Shot Adaptation.
    
    Attributes:
        n_shots: Number of examples per class for few-shot learning
        use_demonstrations: Whether to use demonstrations in prompts
        demo_selection_strategy: Strategy for selecting demonstrations
        prompt_template: Template for constructing few-shot prompts
        meta_learning_steps: Number of meta-learning steps
        similarity_metric: Metric for demonstration selection
        augmentation_strategy: Strategy for data augmentation
        mlflow_tracking: Whether to track metrics with MLflow
    """
    n_shots: int = 8
    use_demonstrations: bool = True
    demo_selection_strategy: str = "similarity"  # Options: similarity, random, clustered
    prompt_template: str = "{demonstration}\n{task_description}\n{input}"
    meta_learning_steps: int = 5
    similarity_metric: str = "cosine"  # Options: cosine, euclidean, semantic
    augmentation_strategy: Optional[str] = None
    mlflow_tracking: bool = True

class FewShotDataset(Dataset):
    """Dataset class for few-shot learning scenarios."""
    
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

class FewShotAdapter:
    """Main class for implementing few-shot adaptation techniques."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: FewShotConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.mlflow_tracker = MLflowTracker("few_shot_adaptation") if config.mlflow_tracking else None
        
    def _select_demonstrations(
        self,
        support_set: List[Dict],
        query: Dict,
        n_demos: int
    ) -> List[Dict]:
        """Select the most relevant demonstrations for a given query.
        
        Args:
            support_set: List of available demonstrations
            query: Current input requiring demonstrations
            n_demos: Number of demonstrations to select
            
        Returns:
            List of selected demonstrations
        """
        if self.config.demo_selection_strategy == "random":
            return random.sample(support_set, min(n_demos, len(support_set)))
            
        elif self.config.demo_selection_strategy == "similarity":
            # Compute embeddings for similarity-based selection
            query_embedding = self._compute_embedding(query["text"])
            demo_embeddings = [self._compute_embedding(demo["text"]) for demo in support_set]
            
            # Calculate similarities
            similarities = [
                self._compute_similarity(query_embedding, demo_embedding, self.config.similarity_metric)
                for demo_embedding in demo_embeddings
            ]
            
            # Select top-k most similar demonstrations
            top_indices = np.argsort(similarities)[-n_demos:]
            return [support_set[i] for i in top_indices]
            
        elif self.config.demo_selection_strategy == "clustered":
            # Implement clustering-based demonstration selection
            return self._cluster_based_selection(support_set, query, n_demos)
            
        else:
            raise ValueError(f"Unknown demonstration selection strategy: {self.config.demo_selection_strategy}")

    def _compute_embedding(self, text: str) -> torch.Tensor:
        """Compute embedding for a given text using the model."""
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze()

    def _compute_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        metric: str
    ) -> float:
        """Compute similarity between two embeddings."""
        if metric == "cosine":
            return torch.nn.functional.cosine_similarity(
                embedding1.unsqueeze(0),
                embedding2.unsqueeze(0)
            ).item()
        elif metric == "euclidean":
            return -torch.norm(embedding1 - embedding2).item()
        elif metric == "semantic":
            # Implement more sophisticated semantic similarity
            return self._semantic_similarity(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

    def _construct_prompt(
        self,
        demonstrations: List[Dict],
        query: Dict
    ) -> str:
        """Construct a prompt using selected demonstrations and query."""
        demo_texts = [demo["text"] for demo in demonstrations]
        demonstrations_str = "\n".join(demo_texts)
        
        return self.config.prompt_template.format(
            demonstration=demonstrations_str,
            task_description=query.get("task_description", ""),
            input=query["text"]
        )

    def adapt(
        self,
        support_set: List[Dict],
        query_set: List[Dict],
        **kwargs
    ) -> Tuple[PreTrainedModel, Dict[str, float]]:
        """Perform few-shot adaptation using the support set and evaluate on query set.
        
        Args:
            support_set: List of examples for few-shot learning
            query_set: List of examples to evaluate on
            **kwargs: Additional arguments for adaptation
            
        Returns:
            Tuple of (adapted model, metrics dictionary)
        """
        if self.mlflow_tracker:
            self.mlflow_tracker.start_run()
            self.mlflow_tracker.log_params({
                "n_shots": self.config.n_shots,
                "demo_selection_strategy": self.config.demo_selection_strategy,
                "meta_learning_steps": self.config.meta_learning_steps
            })

        metrics = {}
        
        try:
            # Prepare datasets
            support_dataset = FewShotDataset(support_set, self.tokenizer)
            query_dataset = FewShotDataset(query_set, self.tokenizer)
            
            # Perform meta-learning if configured
            if self.config.meta_learning_steps > 0:
                self._meta_learning(support_dataset)
            
            # Process each query example
            predictions = []
            for query in query_set:
                # Select demonstrations
                demos = self._select_demonstrations(
                    support_set,
                    query,
                    self.config.n_shots
                )
                
                # Construct prompt
                prompt = self._construct_prompt(demos, query)
                
                # Generate prediction
                prediction = self._generate_prediction(prompt)
                predictions.append(prediction)
            
            # Compute metrics
            metrics = self._compute_metrics(predictions, query_set)
            
            if self.mlflow_tracker:
                self.mlflow_tracker.log_metrics(metrics)
                
        finally:
            if self.mlflow_tracker:
                self.mlflow_tracker.end_run()
        
        return self.model, metrics

    def _meta_learning(self, support_dataset: FewShotDataset):
        """Implement meta-learning for few-shot adaptation."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        
        for step in range(self.config.meta_learning_steps):
            total_loss = 0
            self.model.train()
            
            # Meta-training loop
            for batch in DataLoader(support_dataset, batch_size=4, shuffle=True):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["label"]
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            
            if self.mlflow_tracker:
                self.mlflow_tracker.log_metrics({
                    "meta_learning_loss": total_loss / len(support_dataset)
                }, step=step)

    def _generate_prediction(self, prompt: str) -> str:
        """Generate prediction for a given prompt."""
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _compute_metrics(
        self,
        predictions: List[str],
        ground_truth: List[Dict]
    ) -> Dict[str, float]:
        """Compute evaluation metrics for the predictions."""
        # Implement relevant metrics calculation
        metrics = {
            "accuracy": 0.0,
            "f1_score": 0.0,
            # Add more metrics as needed
        }
        return metrics

    def save_adapter(self, path: str):
        """Save the adapted model and configuration."""
        self.model.save_pretrained(path)
        # Save additional adapter-specific configuration
        if self.mlflow_tracker:
            self.mlflow_tracker.log_artifact(path)

    @classmethod
    def load_adapter(
        cls,
        path: str,
        tokenizer: PreTrainedTokenizer,
        config: Optional[FewShotConfig] = None
    ) -> "FewShotAdapter":
        """Load a saved few-shot adapter."""
        model = PreTrainedModel.from_pretrained(path)
        config = config or FewShotConfig()
        return cls(model, tokenizer, config) 