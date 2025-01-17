import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

import json
import evaluate
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer
from configs.default_config import DEFAULT_SPECS

class NLPTrainer(Trainer):
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, 
                task_type, optimizer=None, compute_metrics=None,
                model_init=None, callbacks=None, scheduler=None, **kwargs):

        specs = json.load(open(args_dir, 'r'))
        self.specs = {**DEFAULT_SPECS, **specs}

        self.args = TrainingArguments(
            **self.specs
        )

        self.datasets = {"train": train_dataset, "eval": eval_dataset}
        self.task_type = task_type
        self.losses = []

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
        super().train()

    def print_args(self):
        print(self.args)

    def print_trainer_type(self):
        print("I am a NLPTrainer!")

class NLPSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, 
                task_type, optimizer=None, compute_metrics=None,
                model_init=None, callbacks=None, scheduler=None, **kwargs):

        specs = json.load(open(args_dir, 'r'))
        self.specs = {**DEFAULT_SPECS, **specs}

        self.args = TrainingArguments(
            **self.specs,
            predict_with_generate=True
        )

        self.datasets = {"train": train_dataset, "eval": eval_dataset}
        self.task_type = task_type
        self.losses = []

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
        super().train()

    def print_args(self):
        print(self.args)

    def print_trainer_type(self):
        print("I am a NLPSeq2SeqTrainer!")