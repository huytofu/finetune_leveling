import numpy as np
import evaluate
import nltk
import collections
import json
import os
import sys
currdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currdir)
sys.path.append(currdir)
sys.path.append(parentdir)


from transformers import pipeline
from model_modules import ModelModules
from tokenizer_modules import TokenizerModules
from pretrain_modules import PretrainModules
from dataset_modules import DatasetModules
from classes.accelerated_trainers import AcceleratedNLPTrainer, AcceleratedNLPSeq2SeqTrainer
from classes.trainers import NLPTrainer, NLPSeq2SeqTrainer
from configs.default_config import DEFAULT_SPECS

class InferencePipeLine():
    def __init__(self, task_type, checkpoint):
        self.task_type = task_type.replace("_","-")
        self.specs = {**DEFAULT_SPECS}
        if task_type == "token_classification":
            self.loaded_pipeline = pipeline(self.task_type, model=checkpoint, aggregation_strategy="simple")     
        elif task_type == "masked_language_modeling":
            self.loaded_pipeline = pipeline(self.task_type, model=checkpoint)
        elif task_type == "translation":
            self.loaded_pipeline = pipeline(self.task_type, model=checkpoint)
        elif task_type == "summarization":
            self.loaded_pipeline = pipeline(self.task_type, model=checkpoint)
        elif task_type == "question_answering":
            self.loaded_pipeline = pipeline(self.task_type, model=checkpoint)
        elif task_type == "text_generation":
            self.loaded_pipeline = pipeline(self.task_type, model=checkpoint)
        else: pass
    
    def run(self, input_text):
        if self.task_type == "translation":
            return self.loaded_pipeline(input_text, max_length=self.specs["max_length"], return_text=True)[0]["translation_text"]
        elif self.task_type == "summarization":
            return self.loaded_pipeline(input_text, max_length=self.specs["max_length"], return_text=True)[0]["summary_text"]
        elif self.task_type == "question-answering":
            return self.loaded_pipeline(input_text, max_length=self.specs["max_length"], return_text=True)[0]["answer"]
        elif self.task_type == "text-generation":
            return self.loaded_pipeline(input_text, max_length=self.specs["max_length"], return_text=True)[0]["generated_text"]
        return self.loaded_pipeline(input_text)

class FineTunePipeLine():
    def __init__(self, task_type, checkpoint, chosen_metric, 
                args_dir=None, dataset_dir=None, accelerated=True):
        self.checkpoint = checkpoint
        self.task_type = task_type
        if args_dir is not None:
            self.args_dir = args_dir
            specs = json.load(open(self.args_dir, 'r'))
            self.specs = {**DEFAULT_SPECS, **specs}

        self.accelerated = accelerated
        self.dataset_dir = dataset_dir
        self.chosen_metric = chosen_metric

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
    
    def prepare_dataset(self, dataset=None, huggingface_dataset=None):
        dataset_modules = DatasetModules(self.dataset_dir, self.tokenizer, self.specs)
        
        if dataset is not None:
            self.raw_dataset = dataset
            self.dataset = dataset_modules.prepare_dataset(self.raw_dataset)
        elif self.dataset_dir is not None:
            self.raw_dataset = dataset_modules.prepare_dataset_from_dir(self.task_type)
            self.dataset = dataset_modules.prepare_dataset(self.raw_dataset)
        else:
            self.raw_dataset = dataset_modules.load_dataset(huggingface_dataset)
            self.dataset = dataset_modules.prepare_dataset(self.raw_dataset)

    def get_compute_metrics(self, task_type, chosen_metric):
        self.metric = evaluate.load_metric(chosen_metric)
        if task_type == "token_classification":
            def compute_metrics(eval_pred):
                # TO ADD LATER
                pass
        elif task_type == "masked_language_modeling":
            def compute_metrics(eval_pred):
                # TO ADD LATER
                pass
        elif task_type == "translation":
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                true_labels, true_predictions = self.postprocess(predictions, labels)
                return self.metric.compute(predictions=true_predictions, references=true_labels)
        elif task_type == "summarization":
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                true_labels, true_predictions = self.postprocess(predictions, labels)
                if chosen_metric == "rouge":
                    true_predictions = ["\n".join(nltk.sent_tokenize(pred)) for pred in true_predictions]
                    true_labels = ["\n".join(nltk.sent_tokenize(label)) for label in true_labels]
                    result = self.metric.compute(predictions=true_predictions, references=true_labels, use_stemmer=True)
                    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
                    return result
                else:
                    return self.metric.compute(predictions=true_predictions, references=true_labels)
        elif task_type == "text_generation":
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                true_labels, true_predictions = self.postprocess(predictions, labels)
                return self.metric.compute(predictions=true_predictions, references=true_labels)
        return compute_metrics

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

    def run(self):
        model_modules = ModelModules(self.checkpoint)
        self.model = model_modules.load_model(self.task_type)

        tokenizer_modules = TokenizerModules(self.checkpoint)
        self.tokenizer = tokenizer_modules.load_tokenizer()

        pretrain_modules = PretrainModules(self.checkpoint, self.args_dir, self.tokenizer, self.model)

        if self.accelerated:
            self.optimizer = pretrain_modules.prepare_optimizer("adamw")
            self.data_collator = pretrain_modules.prepare_data_collator(self.task_type)
            
            if self.task_type in ["translation", "summarization"]:
                accerelated_trainer = AcceleratedNLPSeq2SeqTrainer(
                    self.args_dir, self.model, self.tokenizer, 
                    self.data_collator, self.dataset['train'], self.dataset['eval'], self.raw_dataset,
                    self.task_type, optimizer=self.optimizer, chosen_metric=self.chosen_metric
                )
                accerelated_trainer.train()
            else:
                accerelated_trainer = AcceleratedNLPTrainer(
                    self.args_dir, self.model, self.tokenizer, 
                    self.data_collator, self.dataset['train'], self.dataset['eval'], self.raw_dataset,
                    self.task_type, optimizer=self.optimizer, chosen_metric=self.chosen_metric
                )
                accerelated_trainer.train()
        else:
            if self.task_type != "question_answering":
                compute_metrics = self.get_compute_metrics(self.task_type, self.chosen_metric)
                if self.task_type in ["translation", "summarization", "text_generation"]:
                    trainer = NLPSeq2SeqTrainer(
                        self.args_dir, self.model, self.tokenizer, 
                        self.data_collator, self.dataset['train'], self.dataset['eval'], 
                        self.task_type, optimizer=None, compute_metrics=compute_metrics
                    )
                    trainer.train()
                    trainer.evaluate()
                else:
                    trainer = NLPTrainer(
                        self.args_dir, self.model, self.tokenizer, 
                        self.data_collator, self.dataset['train'], self.dataset['eval'], 
                        self.task_type, optimizer=None, compute_metrics=compute_metrics
                    )
                    trainer.train()
                    trainer.evaluate()
            else:
                trainer = NLPTrainer(
                    self.args_dir, self.model, self.tokenizer, 
                    self.data_collator, self.dataset['train'], self.dataset['eval'], 
                    self.task_type, optimizer=None, compute_metrics=None
                )
                trainer.train()
                predictions, _, _ = trainer.predict(self.dataset['eval'])
                start_logits, end_logits = predictions
                max_answer_length = self.specs['max_length'] | 384
                self.compute_qna_metrics(start_logits, end_logits, self.dataset['eval'], self.raw_dataset['eval'], 
                                        self.chosen_metric, max_answer_length=max_answer_length)





    