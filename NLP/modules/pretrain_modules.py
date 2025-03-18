import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

import json
from transformers import DataCollatorForLanguageModeling, DataCollatorForTokenClassification, DataCollatorForSeq2Seq
from transformers import default_data_collator
from torch.optim import AdamW
from configs.default_config import DEFAULT_SPECS

class PretrainModules():
    def __init__(self, args_dir, tokenizer, model):
        """
        Initialize the PretrainModules.
        
        Args:
            args_dir: Directory containing configuration arguments.
            tokenizer: The tokenizer to use for data processing.
            model: The model to be pre-trained.
        """
        specs = {}
        if args_dir is not None:
            specs = json.load(open(args_dir, 'r'))
        self.specs = {**DEFAULT_SPECS, **specs}
        self.tokenizer = tokenizer
        self.model = model

    def prepare_optimizer(self, type):
        """
        Prepare the optimizer for training.
        
        Args:
            type: The type of optimizer to use (e.g., 'adamw').
        
        Returns:
            The prepared optimizer.
        """
        if type == "adamw":
            optimizer = AdamW(
                self.model.parameters(), 
                lr=self.specs['learning_rate'], 
                weight_decay=self.specs['weight_decay']
            )
        else:
            optimizer = None
        return optimizer

    def prepare_data_collator(self, task_type):
        """
        Prepare the data collator for the specified task type.
        
        Args:
            task_type: The type of task (e.g., 'masked_language_modeling').
        
        Returns:
            The prepared data collator.
        """
        if task_type == "masked_language_modeling":
            # Data collator for masked language modeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm_probability=self.specs['mlm_probability']
            )
        elif task_type == "token_classification":
            # Data collator for token classification
            data_collator = DataCollatorForTokenClassification(
                tokenizer=self.tokenizer
            )
        elif task_type == "translation":
            data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
        elif task_type == "summarization":
            data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
        elif task_type == "text_generation":
            data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
        elif task_type == "question_answering":
            data_collator = default_data_collator
        else:
            data_collator = None

        return data_collator