import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

import json
import evaluate
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer
from configs.default_config import DEFAULT_SPECS

class NLPTrainer(Trainer):
    """
    A trainer class for NLP models using the Hugging Face Transformers library.
    
    Attributes:
        specs: Configuration specifications for training.
        args: Training arguments.
        datasets: A dictionary containing training and evaluation datasets.
        task_type: The type of task (e.g., classification, generation).
        losses: A list to store training losses.
    """
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, 
                task_type, optimizer=None, compute_metrics=None,
                model_init=None, callbacks=None, scheduler=None, **kwargs):
        """
        Initialize the NLPTrainer.
        
        Args:
            args_dir (str): Directory containing configuration arguments.
            model: The model to be trained.
            tokenizer: The tokenizer for text processing.
            data_collator: The data collator for batching.
            train_dataset: The training dataset.
            eval_dataset: The evaluation dataset.
            task_type (str): The type of task (e.g., classification, generation).
            optimizer: The optimizer for training.
            compute_metrics: Function to compute metrics during evaluation.
            model_init: Function to initialize the model.
            callbacks: List of callbacks to apply during training.
            scheduler: Learning rate scheduler.
            **kwargs: Additional keyword arguments.
        """
        # Load configuration specifications
        specs = json.load(open(args_dir, 'r'))
        self.specs = {**DEFAULT_SPECS, **specs}

        # Initialize training arguments
        self.args = TrainingArguments(
            **self.specs
        )

        # Store datasets and task type
        self.datasets = {"train": train_dataset, "eval": eval_dataset}
        self.task_type = task_type
        self.losses = []

        # Initialize the base Trainer class
        super().__init__(model, self.args, 
                        data_collator=data_collator, 
                        train_dataset=train_dataset, 
                        eval_dataset=eval_dataset,
                        tokenizer=tokenizer, 
                        optimizers=[optimizer],
                        model_init=model_init, 
                        compute_metrics=compute_metrics,
                        callbacks=callbacks,
                        scheduler=scheduler, **kwargs)

    def train(self):
        """
        Train the model using the base Trainer's train method.
        """
        super().train()

    def print_args(self):
        """
        Print the training arguments.
        """
        print(self.specs)

    def print_trainer_type(self):
        """
        Print the type of trainer.
        """
        print("I am an NLPTrainer!")

class NLPSeq2SeqTrainer(Seq2SeqTrainer):
    """
    A trainer class for sequence-to-sequence NLP models using the Hugging Face Transformers library.
    
    Attributes:
        specs: Configuration specifications for training.
        args: Training arguments.
        datasets: A dictionary containing training and evaluation datasets.
        task_type: The type of task (e.g., translation, summarization).
        losses: A list to store training losses.
    """
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, 
                task_type, optimizer=None, compute_metrics=None,
                model_init=None, callbacks=None, scheduler=None, **kwargs):
        """
        Initialize the NLPSeq2SeqTrainer.
        
        Args:
            args_dir (str): Directory containing configuration arguments.
            model: The model to be trained.
            tokenizer: The tokenizer for text processing.
            data_collator: The data collator for batching.
            train_dataset: The training dataset.
            eval_dataset: The evaluation dataset.
            task_type (str): The type of task (e.g., translation, summarization).
            optimizer: The optimizer for training.
            compute_metrics: Function to compute metrics during evaluation.
            model_init: Function to initialize the model.
            callbacks: List of callbacks to apply during training.
            scheduler: Learning rate scheduler.
            **kwargs: Additional keyword arguments.
        """
        # Load configuration specifications
        specs = json.load(open(args_dir, 'r'))
        self.specs = {**DEFAULT_SPECS, **specs}

        # Initialize training arguments
        self.args = TrainingArguments(
            **self.specs
        )

        # Store datasets and task type
        self.datasets = {"train": train_dataset, "eval": eval_dataset}
        self.task_type = task_type
        self.losses = []

        # Initialize the base Seq2SeqTrainer class
        super().__init__(model, self.args, 
                        data_collator=data_collator, 
                        train_dataset=train_dataset, 
                        eval_dataset=eval_dataset,
                        tokenizer=tokenizer, 
                        optimizers=[optimizer],
                        model_init=model_init, 
                        compute_metrics=compute_metrics,
                        callbacks=callbacks,
                        scheduler=scheduler, **kwargs)

    def train(self):
        """
        Train the model using the base Seq2SeqTrainer's train method.
        """
        super().train()

    def print_args(self):
        """
        Print the training arguments.
        """
        print(self.specs)

    def print_trainer_type(self):
        """
        Print the type of trainer.
        """
        print("I am an NLPSeq2SeqTrainer!")