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
from ..classes.checkpoint_manager import CheckpointManager
from ..classes.peft_callbacks import PeftAdapterMonitorCallback, PeftEarlyPruningCallback
from ..classes.quantization_manager import QuantizationManager
from ..classes.type_utils import TypeUtils

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
    def __init__(self, args_dir, task_type,
                chosen_metric=None, quantization=None,
                peft_method=None, peft_config=None, use_lightning=False, use_accelerate=False):
        """
        Initialize the fine-tuning pipeline.
        
        Args:
            args_dir: Path to the arguments file
            task_type: The type of task (e.g., 'masked_language_modeling', 'text_generation')
            chosen_metric: The evaluation metric to use
            quantization: Quantization settings
            peft_method: Parameter-efficient fine-tuning method to use (None, "lora", "qlora", etc.)
            peft_config: Configuration for the PEFT method
            use_lightning: Whether to use PyTorch Lightning
            use_accelerate: Whether to use Accelerate
        """
        # Initialize args, task_type, and metrics
        self.args_dir = args_dir
        self.task_type = task_type
        self.chosen_metric = chosen_metric
        self.quantization = quantization
        self.specs = {}
        
        # Initialize module helpers
        self.model_modules = ModelModules()
        self.dataset_modules = DatasetModules()
        self.tokenizer_modules = TokenizerModules()
        self.pretrain_modules = PretrainModules()
        
        # Initialize PEFT configuration
        self.peft_method = peft_method
        self.peft_config = peft_config or {}
        
        # Set framework flags
        self.use_lightning = use_lightning
        self.use_accelerate = use_accelerate
        
        # Initialize helper managers
        self.checkpoint_manager = CheckpointManager()
        self.quantization_manager = QuantizationManager()
        self.type_utils = TypeUtils()
        
        # Load specs
        self.load_specs()
        
        # Initialize the appropriate trainer class based on the task type and framework flags
        if self.task_type in ["summarization", "translation", "text_generation"]:
            if self.use_lightning:
                if self.use_accelerate:
                    from ..classes.accelerated_trainers_with_lightning import AcceleratedNLPSeq2SeqTrainer
                    self.trainer_class = AcceleratedNLPSeq2SeqTrainer
                else:
                    from ..classes.trainers_with_lightning import NLPSeq2SeqTrainer
                    self.trainer_class = NLPSeq2SeqTrainer
            else:
                if self.use_accelerate:
                    from ..classes.accelerated_trainers import AcceleratedNLPSeq2SeqTrainer
                    self.trainer_class = AcceleratedNLPSeq2SeqTrainer
                else:
                    from ..classes.trainers import NLPSeq2SeqTrainer
                    self.trainer_class = NLPSeq2SeqTrainer
        else:
            if self.use_lightning:
                if self.use_accelerate:
                    from ..classes.accelerated_trainers_with_lightning import AcceleratedNLPTrainer
                    self.trainer_class = AcceleratedNLPTrainer
                else:
                    from ..classes.trainers_with_lightning import NLPTrainer
                    self.trainer_class = NLPTrainer
            else:
                if self.use_accelerate:
                    from ..classes.accelerated_trainers import AcceleratedNLPTrainer
                    self.trainer_class = AcceleratedNLPTrainer
                else:
                    from ..classes.trainers import NLPTrainer
                    self.trainer_class = NLPTrainer
    
    def load_specs(self):
        """Load specifications from the arguments file."""
        import json
        from configs.default_config import DEFAULT_SPECS
        try:
            specs = json.load(open(self.args_dir, 'r'))
            self.specs = {**DEFAULT_SPECS, **specs}
        except Exception as e:
            print(f"Error loading specs: {e}")
            self.specs = DEFAULT_SPECS
    
    def run(self):
        """Run the fine-tuning pipeline."""
        # Load model and tokenizer with appropriate quantization
        model_kwargs = {}
        if self.quantization:
            model_kwargs = self.quantization_manager.prepare_model_for_quantization(
                model_name_or_path=self.specs.get("model_name_or_path", ""),
                quant_type=self.quantization,
                custom_quantization_config=self.specs.get("quantization_config", None)
            )
            
        model = self.model_modules.load_model(self.task_type, model_kwargs)
        tokenizer = self.tokenizer_modules.load_tokenizer()
        
        # Optimize memory layout
        model = self.type_utils.optimize_memory_layout(model, self.peft_method)
        
        # Apply PEFT if specified
        if self.peft_method:
            # Check compatibility with quantization
            if self.quantization:
                is_compatible, reason = self.quantization_manager.check_peft_compatible(
                    model, self.peft_method
                )
                if not is_compatible:
                    print(f"Warning: {reason}")
                
                # Optimize for PEFT with quantization
                model = self.quantization_manager.optimize_model_for_peft(
                    model, self.quantization, self.peft_method
                )
                
            model, tokenizer = self.model_modules.apply_peft(
                model=model,
                tokenizer=tokenizer,
                peft_method=self.peft_method,
                peft_config=self.peft_config,
                task_type=self.task_type
            )
            
            # Validate parameter types
            self.type_utils.check_parameter_types(model)
        
        # Load dataset
        raw_dataset = self.dataset_modules.load_dataset()
        
        # Preprocess dataset based on task type
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
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(self.specs["learning_rate"])
        )
        
        # Setup additional callbacks for Lightning
        callbacks = []
        if self.use_lightning and self.peft_method:
            # Add PEFT-specific callbacks
            peft_monitor = PeftAdapterMonitorCallback(
                monitor_gradients=True,
                monitor_weights=True,
                log_every_n_steps=100,
                save_path=os.path.join(self.specs["output_dir"], "peft_monitoring")
            )
            callbacks.append(peft_monitor)
            
            # Only add pruning if enabled in specs
            if self.specs.get("peft_pruning_enabled", False):
                peft_pruning = PeftEarlyPruningCallback(
                    start_pruning_epoch=1,
                    pruning_threshold=self.specs.get("peft_pruning_threshold", 0.01),
                    final_sparsity=self.specs.get("peft_pruning_sparsity", 0.3)
                )
                callbacks.append(peft_pruning)
        
        # Initialize the trainer
        trainer_kwargs = {
            "args_dir": self.args_dir,
            "model": model,
            "tokenizer": tokenizer,
            "data_collator": data_collator,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "raw_dataset": raw_dataset,
            "task_type": self.task_type,
            "optimizer": optimizer,
            "chosen_metric": self.chosen_metric
        }
        
        # Add callbacks for Lightning trainers
        if self.use_lightning and callbacks:
            trainer_kwargs["callbacks"] = callbacks
        
        trainer = self.trainer_class(**trainer_kwargs)
        
        # Run the training process
        if self.use_lightning:
            trainer.fit()
        else:
            trainer.train()
        
        # Save model and checkpoint
        self.save_model_and_checkpoint(model, tokenizer, trainer)
        
        return model, tokenizer
    
    def save_model_and_checkpoint(self, model, tokenizer, trainer):
        """Save the model, tokenizer, and create a checkpoint."""
        output_dir = self.specs["output_dir"]
        
        # For PEFT models, use different saving approach
        if self.peft_method:
            # Save using the checkpoint manager
            self.checkpoint_manager.save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                epoch=self.specs.get("num_train_epochs", 3),
                is_best=True
            )
        else:
            # For regular models, we can use the trainer to save
            if hasattr(trainer, "save_model"):
                trainer.save_model(output_dir)
            elif hasattr(model, "save_pretrained"):
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
            else:
                # Fallback for other model types
                torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                tokenizer.save_pretrained(output_dir)
    
    def load_checkpoint(self, checkpoint_dir, convert_precision=None):
        """
        Load a checkpoint using the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory containing the checkpoint
            convert_precision: Optional precision to convert to (fp16, bf16, fp32)
            
        Returns:
            Tuple of (loaded_model, loaded_tokenizer)
        """
        # Load specs for model initialization
        self.load_specs()
        
        # Load base model
        model = self.model_modules.load_model(self.task_type)
        tokenizer = self.tokenizer_modules.load_tokenizer()
        
        # Load checkpoint
        model, tokenizer, meta = self.checkpoint_manager.load_checkpoint(
            model=model,
            tokenizer=tokenizer,
            checkpoint_dir=checkpoint_dir,
            target_precision=convert_precision
        )
        
        return model, tokenizer, meta

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





    