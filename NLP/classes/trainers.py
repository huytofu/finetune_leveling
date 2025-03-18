import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

import json
import evaluate
import pytorch_lightning as pl
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer
from configs.default_config import DEFAULT_SPECS
from torch.utils.data import DataLoader

class NLPTrainer(pl.LightningModule):
    """
    A PyTorch Lightning trainer class for NLP models.
    
    Attributes:
        model: The model to be trained.
        specs: Configuration specifications for training.
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
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.task_type = task_type
        self.losses = []

        # Load configuration specifications
        specs = json.load(open(args_dir, 'r'))
        self.specs = {**DEFAULT_SPECS, **specs}

        # Store optimizer and scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, **inputs):
        """
        Forward pass through the model.
        
        Args:
            **inputs: Input data for the model.
        
        Returns:
            Model outputs.
        """
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        
        Args:
            batch: The batch of data for the current step.
            batch_idx: The index of the batch.
        
        Returns:
            Loss from the current training step.
        """
        outputs = self.forward(**batch)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.
        
        Args:
            batch: The batch of data for the current step.
            batch_idx: The index of the batch.
        
        Returns:
            Validation loss.
        """
        outputs = self.forward(**batch)
        val_loss = outputs.loss
        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.
        
        Returns:
            A dictionary containing the optimizer and scheduler.
        """
        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}

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

    def train(self, max_epochs=10, gpus=1):
        """
        Train the model using PyTorch Lightning's Trainer.
        
        Args:
            max_epochs (int): Maximum number of epochs to train.
            gpus (int): Number of GPUs to use for training.
        """
        trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus)
        trainer.fit(self, self.train_dataloader(), self.val_dataloader())

    def train_dataloader(self):
        """
        Create the training data loader.
        
        Returns:
            DataLoader: The training data loader.
        """
        return DataLoader(self.train_dataset, batch_size=self.specs['per_device_train_batch_size'], collate_fn=self.data_collator)

    def val_dataloader(self):
        """
        Create the validation data loader.
        
        Returns:
            DataLoader: The validation data loader.
        """
        return DataLoader(self.eval_dataset, batch_size=self.specs['per_device_eval_batch_size'], collate_fn=self.data_collator)

class NLPSeq2SeqTrainer(pl.LightningModule):
    """
    A PyTorch Lightning trainer class for sequence-to-sequence NLP models.
    
    Attributes:
        model: The model to be trained.
        specs: Configuration specifications for training.
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
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.task_type = task_type
        self.losses = []

        # Load configuration specifications
        specs = json.load(open(args_dir, 'r'))
        self.specs = {**DEFAULT_SPECS, **specs}

        # Store optimizer and scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Sequence generation parameters
        self.max_length = self.specs.get('max_length', 128)
        self.num_beams = self.specs.get('num_beams', 4)

    def forward(self, **inputs):
        """
        Forward pass through the model.
        
        Args:
            **inputs: Input data for the model.
        
        Returns:
            Model outputs.
        """
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        
        Args:
            batch: The batch of data for the current step.
            batch_idx: The index of the batch.
        
        Returns:
            Loss from the current training step.
        """
        outputs = self.forward(**batch)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step with sequence generation.
        
        Args:
            batch: The batch of data for the current step.
            batch_idx: The index of the batch.
        
        Returns:
            Validation loss and generated sequences.
        """
        outputs = self.forward(**batch)
        val_loss = outputs.loss
        self.log('val_loss', val_loss)

        # Generate sequences using beam search
        generated_tokens = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=self.max_length,
            num_beams=self.num_beams
        )

        # Decode generated tokens
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        # Compute metrics if provided
        if self.compute_metrics:
            self.compute_metrics(decoded_preds, decoded_labels)

        return val_loss

    def predict(self, batch):
        """
        Generate predictions for a batch of data.
        
        Args:
            batch: The batch of data for prediction.
        
        Returns:
            Generated sequences.
        """
        generated_tokens = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=self.max_length,
            num_beams=self.num_beams
        )
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.
        
        Returns:
            A dictionary containing the optimizer and scheduler.
        """
        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}

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

    def train(self, max_epochs=10, gpus=1):
        """
        Train the model using PyTorch Lightning's Trainer.
        
        Args:
            max_epochs (int): Maximum number of epochs to train.
            gpus (int): Number of GPUs to use for training.
        """
        trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus)
        trainer.fit(self, self.train_dataloader(), self.val_dataloader())

    def train_dataloader(self):
        """
        Create the training data loader.
        
        Returns:
            DataLoader: The training data loader.
        """
        return DataLoader(self.train_dataset, batch_size=self.specs['per_device_train_batch_size'], collate_fn=self.data_collator)

    def val_dataloader(self):
        """
        Create the validation data loader.
        
        Returns:
            DataLoader: The validation data loader.
        """
        return DataLoader(self.eval_dataset, batch_size=self.specs['per_device_eval_batch_size'], collate_fn=self.data_collator)