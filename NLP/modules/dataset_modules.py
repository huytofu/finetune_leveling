from datasets import load_dataset
import polars as pl
from typing import Dict, List, Optional, Union, Any
import logging
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """Configuration for dataset processing.
    
    Attributes:
        max_length (int): Maximum sequence length
        chunk_size (int): Size of chunks for processing
        stride (int): Stride length for sliding window
        batch_size (int): Batch size for processing
        cache_dir (str): Directory for caching processed data
        use_streaming (bool): Whether to use streaming for large datasets
        memory_limit_gb (float): Memory limit in GB for Polars operations
    """
    max_length: int
    chunk_size: int
    stride: int
    batch_size: int = 1000
    cache_dir: str = ".cache"
    use_streaming: bool = False
    memory_limit_gb: float = 32.0

class DatasetModules:
    def __init__(self, dataset_dir: str, tokenizer: Any, specs: Dict[str, Any]):
        """Initialize the DatasetModules.
        
        Args:
            dataset_dir: Directory containing the dataset
            tokenizer: The tokenizer to use for data processing
            specs: Specifications for dataset processing
        
        Raises:
            ValueError: If required specifications are missing
            FileNotFoundError: If dataset directory doesn't exist
        """
        try:
            self.dataset_dir = dataset_dir
            self.tokenizer = tokenizer
            self.specs = specs
            self.config = DatasetConfig(
                max_length=specs.get('max_length', 512),
                chunk_size=specs.get('chunk_size', 128),
                stride=specs.get('stride', 64),
                batch_size=specs.get('batch_size', 1000),
                use_streaming=specs.get('use_streaming', False)
            )
            
            # Initialize Polars configuration for large datasets
            pl.Config.set_fmt_str_lengths(self.config.max_length)
            pl.Config.set_tbl_rows(self.config.batch_size)
            pl.Config.set_tbl_cols(100)
            
        except KeyError as e:
            raise ValueError(f"Missing required specification: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize DatasetModules: {e}")

    @lru_cache(maxsize=128)
    def align_labels_with_tokens(self, labels: tuple, word_ids: tuple) -> List[int]:
        """Align labels with tokenized word IDs with caching for efficiency.
        
        Args:
            labels: Tuple of original labels for the dataset
            word_ids: Tuple of word IDs from tokenization
        
        Returns:
            List of aligned labels
        
        Raises:
            ValueError: If labels and word_ids are incompatible
        """
        try:
            labels = list(labels)
            word_ids = list(word_ids)
            new_labels = []
            current_word = None
            
            for word_id in word_ids:
                if word_id != current_word:
                    current_word = word_id
                    label = -100 if word_id is None else labels[word_id]
                    new_labels.append(label)
                elif word_id is None:
                    new_labels.append(-100)
                else:
                    label = labels[word_id]
                    if label % 2 == 1:
                        label += 1
                    new_labels.append(label)
                    
            return new_labels
        except IndexError:
            raise ValueError("Labels and word_ids are incompatible")
        except Exception as e:
            logger.error(f"Error in align_labels_with_tokens: {e}")
            raise

    def _process_token_classification(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Process data for token classification task."""
        try:
            tokenized = self.tokenizer(
                examples['inputs'],
                is_split_into_words=True,
                truncation=True,
                return_offsets_mapping=True,
                padding=True,
                max_length=self.config.max_length
            )
            
            # Convert lists to tuples for caching
            new_labels = [
                self.align_labels_with_tokens(
                    tuple(labels),
                    tuple(tokenized.word_ids(i))
                )
                for i, labels in enumerate(examples['labels'])
            ]
            tokenized['labels'] = new_labels
            return tokenized
        except Exception as e:
            logger.error(f"Error in token classification processing: {e}")
            raise

    def _process_mlm(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Process data for masked language modeling task."""
        try:
            tokenized = self.tokenizer(
                examples['inputs'],
                is_split_into_words=True,
                truncation=True,
                return_offsets_mapping=True,
                padding=True,
                max_length=self.config.max_length
            )
            tokenized['word_ids'] = tokenized.word_ids()
            return tokenized
        except Exception as e:
            logger.error(f"Error in MLM processing: {e}")
            raise

    def _process_translation(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Process data for translation task."""
        try:
            return self.tokenizer(
                examples['inputs'],
                text_target=examples['targets'],
                max_length=self.config.max_length,
                truncation=True,
                padding=True
            )
        except Exception as e:
            logger.error(f"Error in translation processing: {e}")
            raise

    def _process_summarization(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Process data for summarization task."""
        try:
            tokenized = self.tokenizer(
                examples['inputs'],
                max_length=self.config.max_length,
                truncation=True,
                padding=True
            )
            
            labels = self.tokenizer(
                examples['targets'],
                max_length=self.config.max_length // 4,
                truncation=True,
                padding=True
            )
            
            tokenized['labels'] = labels['input_ids']
            return tokenized
        except Exception as e:
            logger.error(f"Error in summarization processing: {e}")
            raise

    def _process_qa_train(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Process data for question answering task - training mode."""
        try:
            questions = [q.strip() for q in examples['question']]
            inputs = self.tokenizer(
                questions,
                examples['context'],
                max_length=self.config.max_length,
                truncation="only_second",
                stride=self.config.stride,
                padding="max_length",
                return_overflowing_tokens=True,
                return_offsets_mapping=True
            )
            
            offset_mapping = inputs.pop("offset_mapping")
            sample_map = inputs.pop("overflow_to_sample_mapping")
            
            # Process answer positions for training
            start_positions = []
            end_positions = []
            
            for i, offset in enumerate(offset_mapping):
                sample_idx = sample_map[i]
                answer = examples["answers"][sample_idx]
                start_char = answer["answer_start"][0]
                end_char = start_char + len(answer["text"][0])
                
                sequence_ids = inputs.sequence_ids(i)
                context_start = next(idx for idx, val in enumerate(sequence_ids) if val == 1)
                context_end = next(idx for idx, val in reversed(list(enumerate(sequence_ids))) if val == 1)
                
                if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    idx_start = context_start
                    while idx_start <= context_end and offset[idx_start][0] <= start_char:
                        idx_start += 1
                    start_positions.append(idx_start - 1)
                    
                    idx_end = context_end
                    while idx_end >= context_start and offset[idx_end][1] >= end_char:
                        idx_end -= 1
                    end_positions.append(idx_end + 1)
            
            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions
            return inputs
        except Exception as e:
            logger.error(f"Error in QA training processing: {e}")
            raise

    def _process_qa_validation(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Process data for question answering task - validation mode."""
        try:
            questions = [q.strip() for q in examples['question']]
            inputs = self.tokenizer(
                questions,
                examples['context'],
                max_length=self.config.max_length,
                truncation="only_second",
                stride=self.config.stride,
                padding="max_length",
                return_overflowing_tokens=True,
                return_offsets_mapping=True
            )
            
            offset_mapping = inputs.pop("offset_mapping")
            sample_map = inputs.pop("overflow_to_sample_mapping")
            example_ids = []

            # Process validation specific data
            for i in range(len(inputs["input_ids"])):
                sample_idx = sample_map[i]
                example_ids.append(examples["id"][sample_idx])

                sequence_ids = inputs.sequence_ids(i)
                # Set to None for non-context tokens
                inputs["offset_mapping"][i] = [
                    o if sequence_ids[k] == 1 else None for k, o in enumerate(offset_mapping[i])
                ]

            inputs["example_id"] = example_ids
            return inputs
        except Exception as e:
            logger.error(f"Error in QA validation processing: {e}")
            raise

    def tokenize_dataset(self, examples: Dict[str, List], is_validation: bool = False) -> Dict[str, List]:
        """Tokenize dataset examples based on task type.
        
        Args:
            examples: Dataset examples to tokenize
            is_validation: Whether this is for validation dataset
        
        Returns:
            Tokenized dataset
            
        Raises:
            ValueError: If task_type is not supported
        """
        processors = {
            "token_classification": self._process_token_classification,
            "masked_language_modeling": self._process_mlm,
            "translation": self._process_translation,
            "summarization": self._process_summarization,
            "question_answering": self._process_qa_validation if is_validation else self._process_qa_train
        }
        
        try:
            processor = processors.get(self.task_type)
            if not processor:
                raise ValueError(f"Unsupported task type: {self.task_type}")
            
            return processor(examples)
        except Exception as e:
            logger.error(f"Error in tokenize_dataset: {e}")
            raise

    def group_texts(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Group texts for masked language modeling.
        
        Args:
            examples: Examples to group
            
        Returns:
            Grouped examples
        """
        try:
            # Use Polars for efficient concatenation
            df = pl.DataFrame(examples)
            
            # Concatenate all texts
            concatenated = {
                k: df[k].explode().to_list()
                for k in examples.keys()
            }
            
            total_length = len(concatenated[list(examples.keys())[0]])
            chunk_size = self.config.chunk_size
            total_length = (total_length // chunk_size) * chunk_size
            
            # Split by chunks efficiently
            results = {
                k: [
                    t[i:i + chunk_size]
                    for i in range(0, total_length, chunk_size)
                ]
                for k, t in concatenated.items()
            }
            
            results["labels"] = results["input_ids"].copy()
            return results
        except Exception as e:
            logger.error(f"Error in group_texts: {e}")
            raise

    def prepare_dataset(self, dataset: Any, task_type: str) -> Dict[str, Any]:
        """Prepare dataset for training and evaluation.
        
        Args:
            dataset: Raw dataset
            task_type: Type of NLP task
            
        Returns:
            Dictionary containing prepared train and eval datasets
        """
        try:
            self.task_type = task_type
            self.raw_dataset = dataset
            
            if 'eval' not in self.raw_dataset.keys():
                self.raw_dataset = self.raw_dataset["train"].train_test_split(
                    train_size=0.9,
                    seed=42
                )
            
            # Process datasets with progress bars
            logger.info("Processing training dataset...")
            train_dataset = self.raw_dataset['train'].map(
                lambda x: self.tokenize_dataset(x, is_validation=False),
                batched=True,
                batch_size=self.config.batch_size,
                remove_columns=self.raw_dataset['train'].column_names,
                desc="Processing training data"
            )
            
            logger.info("Processing evaluation dataset...")
            eval_dataset = self.raw_dataset['eval'].map(
                lambda x: self.tokenize_dataset(x, is_validation=True),
                batched=True,
                batch_size=self.config.batch_size,
                remove_columns=self.raw_dataset['eval'].column_names,
                desc="Processing evaluation data"
            )
            
            if self.task_type == "masked_language_modeling":
                logger.info("Grouping texts for MLM...")
                train_dataset = train_dataset.map(
                    self.group_texts,
                    batched=True,
                    batch_size=self.config.batch_size,
                    desc="Grouping training texts"
                )
                eval_dataset = eval_dataset.map(
                    self.group_texts,
                    batched=True,
                    batch_size=self.config.batch_size,
                    desc="Grouping evaluation texts"
                )
            
            return {
                "train": train_dataset,
                "eval": eval_dataset
            }
            
        except Exception as e:
            logger.error(f"Error in prepare_dataset: {e}")
            raise

    def prepare_dataset_from_dir(self, task_type: str) -> Dict[str, Any]:
        """Prepare dataset from directory.
        
        Args:
            task_type: Type of NLP task
            
        Returns:
            Dictionary containing prepared train and eval datasets
            
        Raises:
            FileNotFoundError: If dataset directory doesn't exist
        """
        try:
            logger.info(f"Loading dataset from {self.dataset_dir}...")
            if self.config.use_streaming:
                self.raw_dataset = load_dataset(
                    self.dataset_dir,
                    streaming=True
                )
            else:
                self.raw_dataset = load_dataset(self.dataset_dir)
            
            return self.prepare_dataset(self.raw_dataset, task_type)
            
        except Exception as e:
            logger.error(f"Error in prepare_dataset_from_dir: {e}")
            raise

    def load_dataset(self, name):
        self.raw_dataset = load_dataset(name)
        pass

    def save_dataset(self):
        pass