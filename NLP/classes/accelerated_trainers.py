import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

import torch
import nltk
import json
import evaluate
import collections
import numpy as np

from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm
from configs.default_config import DEFAULT_SPECS

class AcceleratedNLPTrainer():
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, raw_dataset,
                task_type, optimizer, chosen_metric):
        self.model = model
        self.optimizer = optimizer
        self.data_collator = data_collator
        self.datasets = {"train": train_dataset, "eval": eval_dataset}
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        specs = json.load(open(args_dir, 'r'))
        self.specs = {**DEFAULT_SPECS, **specs}
        
        self.prepare_with_accelerator(train_dataset, eval_dataset)
        self.prepare_scheduler()

        self.progress_bar = tqdm(range(self.num_training_steps | 1))
        self.task_type = task_type
        self.losses = []
        self.metric = evaluate.load(chosen_metric)

    def prepare_with_accelerator(self, train_dataset, eval_dataset):
        #prepare data loader
        self.train_dataloader = self.prepare_data_loader("train", train_dataset)
        self.eval_dataloader = self.prepare_data_loader("eval", eval_dataset)

        self.accelerator = Accelerator(fp16=True)
        #Prepare the model, optimizer, train_dataloader, and eval_dataloader based on available devices/hardware
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader
        )

    def prepare_data_loader(self, slice_type, dataset):
        if slice_type == "train":
            shuffle = True
            batch_size = self.specs['per_device_train_batch_size']
        else:
            shuffle = False
            batch_size = self.specs['per_device_eval_batch_size']
        
        dataset.set_format('torch')
        self.dataloader = DataLoader(
            dataset, 
            shuffle=shuffle, 
            collate_fn=self.data_collator, 
            batch_size=batch_size
        )
        
        return self.dataloader
    
    def prepare_scheduler(self):
        #Prepare scheduler specs
        self.scheduler_strategy = self.specs['scheduler_strategy']
        self.num_train_epochs = self.specs['num_train_epochs']
        self.num_warmup_steps = self.specs.get('num_warmup_steps', 0)
        
        num_update_steps_per_epoch = len(self.train_dataloader)
        self.num_training_steps = self.num_train_epochs * num_update_steps_per_epoch

        self.lr_scheduler = get_scheduler(
            self.scheduler_strategy,
            optimizer=self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
    
    def save_and_upload(self, epoch):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(self.specs['output_dir'], save_function=self.accelerator.save)
        unwrapped_model.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(self.specs['output_dir'])
            self.tokenizer.push_to_hub(
                commit_message=f"Training in progress epoch {epoch}", blocking=False
            )

    def postprocess(self, predictions, labels):
        predictions = predictions.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        label_names = self.datasets["train"].features["labels"].feature.names

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return true_labels, true_predictions

    def compute_qna_metrics(self, start_logits, end_logits, eval_dataset, raw_eval_dataset, chosen_metric, max_answer_length):
        self.metric = evaluate.load_metric(chosen_metric)
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(eval_dataset):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in raw_eval_dataset:
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = eval_dataset[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1 : -5 - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -5 - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                            end_index < start_index or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in raw_eval_dataset]
        result = self.metric.compute(predictions=predicted_answers, references=theoretical_answers)
        return result
    
    def handle_predictions_and_metric(self, outputs, batch, metric):
        if self.task_type == "token_classification":
            predictions = outputs.logits.argmax(-1)
            labels = batch["labels"]
            predictions = self.accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            predictions = self.accelerator.gather(predictions)
            labels = self.accelerator.gather(labels)
            true_predictions, true_labels = self.postprocess(predictions, labels)
            if metric.name == "rouge":
                true_predictions = ["\n".join(nltk.sent_tokenize(pred)) for pred in true_predictions]
                true_labels = ["\n".join(nltk.sent_tokenize(label)) for label in true_labels]
            metric.add_batch(predictions=true_predictions, references=true_labels)
        elif self.task_type == "masked_language_modeling":
            # TO ADD LATER
            pass
        elif self.task_type in ["summarization", "translation", "text_generation"]:
            pass
        elif self.task_type == "question_answering":
            batch_start_logits = self.accelerator.gather(outputs.start_logits).cpu().numpy()
            batch_end_logits = self.accelerator.gather(outputs.end_logits).cpu().numpy()
            self.start_logits.append(batch_start_logits)
            self.end_logits.append(batch_end_logits)
        else: pass

    def handle_outputs(self, outputs, batch, batch_size, losses, metric):
        loss = outputs.loss
        # Log the loss
        self.accelerator.log({"loss": loss.item()})
        #Gather the losses
        losses.append(self.accelerator.gather(loss.repeat(batch_size)))

        #Gather the metrics
        if self.task_type == "token_classification":
            if batch["labels"] is not None:
                self.handle_predictions_and_metric(outputs, batch, metric)
        elif self.task_type == "masked_language_modeling":
            # TO ADD LATER
            pass

    def train(self):
        for epoch in range(self.num_train_epochs):
            print("Currently in epoch", epoch)

            self.model.train()
            #Training with forward & backward pass
            for batch in self.train_dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.progress_bar.update(1)
            
            self.model.eval()
            if self.task_type == "question_answering":
                self.start_logits = []
                self.end_logits = []

            for step, batch in enumerate(self.eval_dataloader):
                with torch.no_grad():
                    outputs = self.model(**batch)
                
                self.handle_outputs(outputs, batch, self.specs['per_device_eval_batch_size'], 
                                    self.losses, self.metric)
                print("Finished evaluating step", step)

            losses = torch.cat(self.losses)
            losses = losses[: len(self.datasets['eval'])]
            print("Losses:", losses)

            if self.task_type == "question_answering":
                self.start_logits = np.concatenate(self.start_logits)[: len(self.eval_dataset)]
                self.end_logits = np.concatenate(self.end_logits)[: len(self.eval_dataset)]
                self.compute_qna_metrics(self.start_logits, self.end_logits, self.eval_dataset, self.raw_dataset['eval'])
            else:
                self.metric.compute()
            
            self.save_and_upload(epoch)
            
    def print_args(self):
        print(self.specs)

    def print_trainer_type(self):
        print("I am an AcceleratedNLPTrainer!")


class AcceleratedNLPSeq2SeqTrainer():
    def __init__(self, args_dir, model, tokenizer, 
                data_collator, train_dataset, eval_dataset, raw_dataset,
                task_type, optimizer, chosen_metric):
        self.model = model
        self.optimizer = optimizer
        self.data_collator = data_collator
        self.datasets = {"train": train_dataset, "eval": eval_dataset}
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        specs = json.load(open(args_dir, 'r'))
        self.specs = {**DEFAULT_SPECS, **specs}
        
        self.prepare_with_accelerator(train_dataset, eval_dataset)
        self.prepare_scheduler()

        self.progress_bar = tqdm(range(self.num_training_steps | 1))
        self.task_type = task_type
        self.losses = []
        self.metric = evaluate.load(chosen_metric)

    def prepare_with_accelerator(self, train_dataset, eval_dataset):
        #prepare data loader
        self.train_dataloader = self.prepare_data_loader("train", train_dataset)
        self.eval_dataloader = self.prepare_data_loader("eval", eval_dataset)

        self.accelerator = Accelerator(fp16=True)
        #Prepare the model, optimizer, train_dataloader, and eval_dataloader based on available devices/hardware
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader
        )

    def prepare_data_loader(self, slice_type, dataset):
        if slice_type == "train":
            shuffle = True
            batch_size = self.specs['per_device_train_batch_size']
        else:
            shuffle = False
            batch_size = self.specs['per_device_eval_batch_size']
        
        dataset.set_format('torch')
        self.dataloader = DataLoader(
            dataset, 
            shuffle=shuffle, 
            collate_fn=self.data_collator, 
            batch_size=batch_size
        )
        
        return self.dataloader
    
    def prepare_scheduler(self):
        #Prepare scheduler specs
        self.scheduler_strategy = self.specs['scheduler_strategy']
        self.num_train_epochs = self.specs['num_train_epochs']
        self.num_warmup_steps = self.specs.get('num_warmup_steps', 0)
        
        num_update_steps_per_epoch = len(self.train_dataloader)
        self.num_training_steps = self.num_train_epochs * num_update_steps_per_epoch

        self.lr_scheduler = get_scheduler(
            self.scheduler_strategy,
            optimizer=self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
    
    def save_and_upload(self, epoch):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(self.specs['output_dir'], save_function=self.accelerator.save)
        unwrapped_model.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(self.specs['output_dir'])
            self.tokenizer.push_to_hub(
                commit_message=f"Training in progress epoch {epoch}", blocking=False
            )
    
    def postprocess(self, predictions, labels):
        predictions = predictions.detach().cpu().clone().numpy()
        labels = labels.detach().cpu().clone().numpy()
        
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Remove ignored index (special tokens) and convert to labels
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        true_predictions = [pred.strip() for pred in decoded_preds]
        true_labels = [label.strip() for label in decoded_labels]

        return true_labels, true_predictions

    def compute_qna_metrics(self, start_logits, end_logits, eval_dataset, raw_eval_dataset, chosen_metric, max_answer_length):
        self.metric = evaluate.load_metric(chosen_metric)
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(eval_dataset):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in raw_eval_dataset:
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = eval_dataset[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1 : -5 - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -5 - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                            end_index < start_index or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in raw_eval_dataset]
        result = self.metric.compute(predictions=predicted_answers, references=theoretical_answers)
        return result
    
    def handle_predictions_and_metric(self, outputs, batch, metric):
        predictions = self.accelerator.unwrap_model(self.model).generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=self.specs["max_length"],
        )
        labels = batch["labels"]
        predictions = self.accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
        predictions = self.accelerator.gather(predictions)
        labels = self.accelerator.gather(labels)
        true_predictions, true_labels = self.postprocess(predictions, labels)
        metric.add_batch(predictions=true_predictions, references=true_labels)

    def handle_outputs(self, outputs, batch, batch_size, losses, metric):
        loss = outputs.loss
        self.accelerator.log({"loss": loss.item()})
        #Gather the losses
        losses.append(self.accelerator.gather(loss.repeat(batch_size)))

        self.handle_predictions_and_metric(outputs, batch, metric)

    def train(self):
        for epoch in range(self.num_train_epochs):
            print("Currently in epoch", epoch)

            self.model.train()
            #Training with forward & backward pass
            for batch in self.train_dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.progress_bar.update(1)
            
            self.model.eval()
            if self.task_type == "question_answering":
                self.start_logits = []
                self.end_logits = []

            for step, batch in enumerate(self.eval_dataloader):
                with torch.no_grad():
                    outputs = self.model(**batch)
                    self.handle_outputs(outputs, batch, self.specs['per_device_eval_batch_size'], 
                                    self.losses, self.metric)
                print("Finished evaluating step", step)

            losses = torch.cat(self.losses)
            losses = losses[: len(self.datasets['eval'])]
            print("Losses:", losses)

            if self.task_type == "question_answering":
                self.start_logits = np.concatenate(self.start_logits)
                self.end_logits = np.concatenate(self.end_logits)
                self.compute_qna_metrics(self.start_logits, self.end_logits, self.eval_dataset, self.raw_dataset['eval'])
            else:
                self.metric.compute()
            
            self.save_and_upload(epoch)

    def print_args(self):
        print(self.specs)

    def print_trainer_type(self):
        print("I am an AcceleratedNLPSeq2SeqTrainer!")
