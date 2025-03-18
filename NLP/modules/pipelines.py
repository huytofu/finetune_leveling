import numpy as np
import evaluate
import nltk
import collections
import json
import os
import sys
import torch
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
from classes.accelerated_trainers_with_lightning import AcceleratedNLPTrainer as AcceleratedNLPTrainerWithLightning, AcceleratedNLPSeq2SeqTrainer as AcceleratedNLPSeq2SeqTrainerWithLightning
from classes.trainers_with_lightning import NLPTrainer as NLPTrainerWithLightning, NLPSeq2SeqTrainer as NLPSeq2SeqTrainerWithLightning
from configs.default_config import DEFAULT_SPECS
from adapter_manager import MultiAdapterManager

class InferencePipeLine():
    def __init__(self, task_type, checkpoint, adapter_config=None):
        """
        Initialize the inference pipeline.
        
        Args:
            task_type: Type of task (text_generation, summarization, etc.)
            checkpoint: Path to the model checkpoint
            adapter_config: Configuration for adapter support (optional)
                {
                    'cache_dir': str,          # Directory to cache adapters
                    'max_adapters': int,       # Maximum number of adapters in memory
                    'aws_credentials': dict,   # AWS credentials
                    'gcp_credentials': str,    # Path to GCP credentials file
                    'preload_adapters': list   # List of adapter configurations to preload
                }
        """
        self.task_type = task_type.replace("_","-")
        self.specs = {**DEFAULT_SPECS}
        
        # Create the base pipeline without adapters
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
        else: 
            raise ValueError(f"Unsupported task type: {task_type}")
        
        # Initialize adapter support if configured
        self.adapter_manager = None
        self.has_adapter_support = False
        
        if adapter_config is not None:
            self._initialize_adapter_support(checkpoint, adapter_config)
    
    def _initialize_adapter_support(self, checkpoint, adapter_config):
        """Initialize adapter support with the given configuration"""
        # Extract configuration
        cache_dir = adapter_config.get('cache_dir')
        max_adapters = adapter_config.get('max_adapters', 5)
        aws_credentials = adapter_config.get('aws_credentials')
        gcp_credentials = adapter_config.get('gcp_credentials')
        preload_adapters = adapter_config.get('preload_adapters', [])
        
        # Initialize MultiAdapterManager
        self.adapter_manager = MultiAdapterManager(
            base_model=self.loaded_pipeline.model,
            cache_dir=cache_dir,
            max_adapters_in_memory=max_adapters,
            aws_credentials=aws_credentials,
            gcp_credentials=gcp_credentials
        )
        
        # Preload adapters if specified
        for adapter_conf in preload_adapters:
            source = adapter_conf.get('source', 'local')
            adapter_id = adapter_conf.get('adapter_id')
            
            if not adapter_id:
                continue
                
            try:
                if source == 'local':
                    path = adapter_conf.get('path')
                    if path:
                        self.adapter_manager.register_adapter_from_local(
                            adapter_id=adapter_id,
                            adapter_path=path,
                            adapter_name=adapter_conf.get('adapter_name'),
                            adapter_type=adapter_conf.get('adapter_type', 'lora'),
                            metadata=adapter_conf.get('metadata')
                        )
                
                elif source == 'huggingface':
                    repo_id = adapter_conf.get('repo_id')
                    if repo_id:
                        self.adapter_manager.register_adapter_from_huggingface(
                            adapter_id=adapter_id,
                            repo_id=repo_id,
                            adapter_name=adapter_conf.get('adapter_name'),
                            adapter_type=adapter_conf.get('adapter_type', 'lora'),
                            metadata=adapter_conf.get('metadata'),
                            revision=adapter_conf.get('revision', 'main'),
                            use_auth_token=adapter_conf.get('use_auth_token')
                        )
                
                elif source == 'aws':
                    bucket = adapter_conf.get('bucket')
                    prefix = adapter_conf.get('prefix')
                    if bucket and prefix:
                        self.adapter_manager.register_adapter_from_aws(
                            adapter_id=adapter_id,
                            bucket=bucket,
                            prefix=prefix,
                            adapter_name=adapter_conf.get('adapter_name'),
                            adapter_type=adapter_conf.get('adapter_type', 'lora'),
                            metadata=adapter_conf.get('metadata')
                        )
                
                elif source == 'gcp':
                    bucket = adapter_conf.get('bucket')
                    prefix = adapter_conf.get('prefix')
                    if bucket and prefix:
                        self.adapter_manager.register_adapter_from_gcp(
                            adapter_id=adapter_id,
                            bucket=bucket,
                            prefix=prefix,
                            adapter_name=adapter_conf.get('adapter_name'),
                            adapter_type=adapter_conf.get('adapter_type', 'lora'),
                            metadata=adapter_conf.get('metadata')
                        )
            except Exception as e:
                import logging
                logging.error(f"Failed to preload adapter {adapter_id}: {str(e)}")
        
        self.has_adapter_support = True
    
    def list_adapters(self):
        """
        List all registered adapters
        
        Returns:
            list: Information about registered adapters
        """
        if not self.has_adapter_support:
            return []
        
        adapter_list = self.adapter_manager.list_adapters()
        return [
            {
                'id': a.adapter_id,
                'name': a.adapter_name,
                'type': a.adapter_type,
                'source': a.source,
                'is_loaded': a.is_loaded
            }
            for a in adapter_list
        ]
    
    def run(self, input_text, adapter_id=None, auto_adapter=False, select_adapter_with_llm=False, **kwargs):
        """
        Run inference with the model
        
        Args:
            input_text: Input text for the model
            adapter_id: Optional adapter ID to use for inference
            auto_adapter: If True, let the system decide which adapter to use
            select_adapter_with_llm: If True, use LLM to decide the best adapter
            **kwargs: Additional arguments for the model
            
        Returns:
            Model output
        """
        # Determine adapter if auto_adapter is enabled
        if auto_adapter:
            if select_adapter_with_llm:
                adapter_id = self._select_best_adapter_with_llm(input_text)
            else:
                adapter_id = self._select_best_adapter(input_text)
            print(f"System selected adapter: {adapter_id}")

        # Use adapter if specified and adapter support is enabled
        if adapter_id and self.has_adapter_support:
            try:
                # Load the specified adapter
                adapter_model = self.adapter_manager.load_adapter(adapter_id)
                
                # Create a temporary pipeline with the adapter model
                temp_pipeline = None
                
                if self.task_type == "translation":
                    temp_pipeline = pipeline(self.task_type, model=adapter_model)
                    result = temp_pipeline(input_text, max_length=kwargs.get("max_length", self.specs["max_length"]), return_text=True)[0]["translation_text"]
                
                elif self.task_type == "summarization":
                    temp_pipeline = pipeline(self.task_type, model=adapter_model)
                    result = temp_pipeline(input_text, max_length=kwargs.get("max_length", self.specs["max_length"]), return_text=True)[0]["summary_text"]
                
                elif self.task_type == "question-answering":
                    temp_pipeline = pipeline(self.task_type, model=adapter_model)
                    result = temp_pipeline(input_text, max_length=kwargs.get("max_length", self.specs["max_length"]), return_text=True)[0]["answer"]
                
                elif self.task_type == "text-generation":
                    temp_pipeline = pipeline(self.task_type, model=adapter_model)
                    result = temp_pipeline(
                        input_text, 
                        max_length=kwargs.get("max_length", self.specs["max_length"]), 
                        do_sample=kwargs.get("do_sample", True),
                        temperature=kwargs.get("temperature", 0.7),
                        top_p=kwargs.get("top_p", 0.9),
                        top_k=kwargs.get("top_k", 50),
                        num_return_sequences=kwargs.get("num_return_sequences", 1),
                        return_text=True
                    )[0]["generated_text"]
                
                else:
                    temp_pipeline = pipeline(self.task_type, model=adapter_model)
                    result = temp_pipeline(input_text)
                
                return result
            
            except Exception as e:
                import logging
                logging.error(f"Error using adapter {adapter_id}: {str(e)}")
                logging.info(f"Falling back to base model")
                # Fall back to base model
        
        # Use base model
        if self.task_type == "translation":
            return self.loaded_pipeline(input_text, max_length=kwargs.get("max_length", self.specs["max_length"]), return_text=True)[0]["translation_text"]
        elif self.task_type == "summarization":
            return self.loaded_pipeline(input_text, max_length=kwargs.get("max_length", self.specs["max_length"]), return_text=True)[0]["summary_text"]
        elif self.task_type == "question-answering":
            return self.loaded_pipeline(input_text, max_length=kwargs.get("max_length", self.specs["max_length"]), return_text=True)[0]["answer"]
        elif self.task_type == "text-generation":
            return self.loaded_pipeline(
                input_text, 
                max_length=kwargs.get("max_length", self.specs["max_length"]), 
                do_sample=kwargs.get("do_sample", True),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                top_k=kwargs.get("top_k", 50),
                num_return_sequences=kwargs.get("num_return_sequences", 1),
                return_text=True
            )[0]["generated_text"]
        return self.loaded_pipeline(input_text)

    def _select_best_adapter(self, input_text):
        """
        Select the most suitable adapter based on the input text or task type.
        """
        # Placeholder logic for selecting the best adapter
        # This can be replaced with more sophisticated logic based on task type, input text, etc.
        adapters = self.adapter_manager.list_adapters()
        if not adapters:
            return None
        
        # Example: Select the first available adapter
        return adapters[0]['id']

    def _select_best_adapter_with_llm(self, input_text):
        """
        Use the base model to decide the best adapter based on adapter names.
        """
        # Construct a prompt for the model
        prompt = """
        You are a smart AI model. Based on the following task and available adapters, choose the most suitable adapter:
        Task: {task_type}
        Available Adapters: {adapters}
        Choose the best adapter name:
        """.format(
            task_type=self.task_type,
            adapters=', '.join([a['name'] for a in self.adapter_manager.list_adapters()])
        )

        # Use the base model to generate a response
        response = self.loaded_pipeline(prompt, max_length=50, return_text=True)[0]['generated_text']

        # Extract the adapter name from the response
        for adapter in self.adapter_manager.list_adapters():
            if adapter['name'] in response:
                return adapter['id']

        # Fallback to the first adapter if no match is found
        return self.adapter_manager.list_adapters()[0]['id']

class FineTunePipeLine():
    def __init__(self, args_dir, task_type, checkpoint, dataset_name, 
                dataset_config_name=None, text_column=None, summary_column=None, 
                use_bert=False, use_accelerate=False, chosen_metric="accuracy",
                peft_method=None, peft_config=None, quantization=None, use_lightning=False):
        """Initialize the fine-tuning pipeline.
        
        Args:
            args_dir: Path to the arguments directory
            task_type: Type of task to fine-tune for
            checkpoint: Model checkpoint to use
            dataset_name: Name of the dataset to use
            dataset_config_name: Configuration name for the dataset
            text_column: Name of the text column in the dataset
            summary_column: Name of the summary column in the dataset
            use_bert: Whether to use BERT models
            use_accelerate: Whether to use Accelerate for training
            chosen_metric: Metric to use for evaluation
            peft_method: Parameter-efficient fine-tuning method to use (None, "lora", "qlora", etc.)
            peft_config: Configuration for the PEFT method
            quantization: Quantization type to use (None, "4bit", "8bit")
            use_lightning: Whether to use Lightning for training
        """
        self.args_dir = args_dir
        self.task_type = task_type
        self.checkpoint = checkpoint
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.text_column = text_column
        self.summary_column = summary_column
        self.use_bert = use_bert
        self.use_accelerate = use_accelerate
        self.chosen_metric = chosen_metric
        self.peft_method = peft_method
        self.peft_config = peft_config or {}
        self.quantization = quantization
        self.use_lightning = use_lightning
        
        specs = json.load(open(args_dir, 'r'))
        self.specs = {**DEFAULT_SPECS, **specs}
        
        self.model_modules = ModelModules(checkpoint, use_bert)
        self.tokenizer_modules = TokenizerModules(checkpoint, use_bert)
        self.pretrain_modules = PretrainModules(checkpoint, use_bert)
        self.dataset_modules = DatasetModules(dataset_name, dataset_config_name, text_column, summary_column)
        
        # Initialize the appropriate trainer
        if self.use_lightning:
            if self.use_accelerate:
                if self.task_type in ["summarization", "translation"]:
                    self.trainer_class = AcceleratedNLPSeq2SeqTrainerWithLightning
                else:
                    self.trainer_class = AcceleratedNLPTrainerWithLightning
            else:
                if self.task_type in ["summarization", "translation"]:
                    self.trainer_class = NLPSeq2SeqTrainerWithLightning
                else:
                    self.trainer_class = NLPTrainerWithLightning
        else:
            if self.use_accelerate:
                if self.task_type in ["summarization", "translation"]:
                    self.trainer_class = AcceleratedNLPSeq2SeqTrainer
                else:
                    self.trainer_class = AcceleratedNLPTrainer
            else:
                if self.task_type in ["summarization", "translation"]:
                    self.trainer_class = NLPSeq2SeqTrainer
                else:
                    self.trainer_class = NLPTrainer
    
    def run(self):
        # Load model and tokenizer
        model = self.model_modules.load_model(self.task_type, self.quantization)
        tokenizer = self.tokenizer_modules.load_tokenizer()
        
        # Apply PEFT if specified
        if self.peft_method:
            model, tokenizer = self.model_modules.apply_peft(
                model=model,
                tokenizer=tokenizer,
                peft_method=self.peft_method,
                peft_config=self.peft_config,
                task_type=self.task_type
            )
        
        # Load dataset
        raw_dataset = self.dataset_modules.load_dataset()
        
        # Preprocess dataset
        if self.task_type == "masked_language_modeling":
            train_dataset, eval_dataset, data_collator = self.pretrain_modules.prepare_mlm(raw_dataset, tokenizer, self.specs)
        elif self.task_type == "token_classification":
            train_dataset, eval_dataset, data_collator = self.dataset_modules.prepare_token_classification(raw_dataset, tokenizer, self.specs)
        elif self.task_type == "translation":
            train_dataset, eval_dataset, data_collator = self.dataset_modules.prepare_translation(raw_dataset, tokenizer, self.specs)
        elif self.task_type == "summarization":
            train_dataset, eval_dataset, data_collator = self.dataset_modules.prepare_summarization(raw_dataset, tokenizer, self.specs)
        elif self.task_type == "text_generation":
            train_dataset, eval_dataset, data_collator = self.dataset_modules.prepare_text_generation(raw_dataset, tokenizer, self.specs)
        elif self.task_type == "question_answering":
            train_dataset, eval_dataset, data_collator = self.dataset_modules.prepare_question_answering(raw_dataset, tokenizer, self.specs)
        else:
            train_dataset, eval_dataset, data_collator = None, None, None
        
        # Define optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.specs["learning_rate"]))
        
        # Initialize the trainer
        trainer = self.trainer_class(
            args_dir=self.args_dir,
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            raw_dataset=raw_dataset,
            task_type=self.task_type,
            optimizer=optimizer,
            chosen_metric=self.chosen_metric
        )
        
        # Run the training process
        if self.use_lightning:
            trainer.fit()
        else:
            trainer.train()
        
        # Save model
        if self.peft_method:
            # For PEFT models, we need to save the adapter separately
            from .peft_modules import PEFTModules
            peft_modules = PEFTModules()
            peft_modules.save_peft_model(model, self.specs["output_dir"])
        else:
            # For regular models, we can use the trainer to save
            trainer.model.save_pretrained(self.specs["output_dir"])
            tokenizer.save_pretrained(self.specs["output_dir"])
        
        return model, tokenizer

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





    