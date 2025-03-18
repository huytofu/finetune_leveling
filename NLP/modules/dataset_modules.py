from datasets import load_dataset

class DatasetModules():
    def __init__(self, dataset_dir, tokenizer, specs):
        """
        Initialize the DatasetModules.
        
        Args:
            dataset_dir: Directory containing the dataset.
            tokenizer: The tokenizer to use for data processing.
            specs: Specifications for dataset processing.
        """
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.specs = specs

    def align_labels_with_tokens(self, labels, word_ids):
        """
        Align labels with tokenized word IDs.
        
        Args:
            labels: Original labels for the dataset.
            word_ids: Word IDs from tokenization.
        
        Returns:
            List of aligned labels.
        """
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels
    
    def tokenize_dataset(self, examples):
        """
        Tokenize the dataset examples.
        
        Args:
            examples: Dataset examples to tokenize.
        
        Returns:
            Tokenized dataset.
        """
        if self.task_type == "token_classification":
            # Tokenize inputs for token classification
            self.tokenized_dataset = self.tokenizer(examples['inputs'],
                                                    is_split_into_words=True, truncation=True,
                                                    return_offsets_mapping=True)
            new_labels  = [self.align_labels_with_tokens(labels, self.tokenized_dataset.word_ids(i)) 
                           for i,labels in enumerate(examples['labels'])]
            self.tokenized_dataset['labels'] = new_labels
            return self.tokenized_dataset
        elif self.task_type == "masked_language_modeling":
            self.tokenized_dataset = self.tokenizer(examples['inputs'],
                                                    is_split_into_words=True, truncation=True,
                                                    return_offsets_mapping=True)
            self.tokenized_dataset['word_ids'] = self.tokenized_dataset.word_ids()
            return self.tokenized_dataset
        elif self.task_type == "translation":
            _from = examples['inputs']
            _to = examples['targets']
            self.tokenized_dataset = self.tokenizer(_from, text_target=_to, max_length=self.specs['max_length'],
                                               truncation=True)
            return self.tokenized_dataset
        elif self.task_type == "summarization":
            self.tokenized_dataset = self.tokenizer(examples['inputs'], max_length=self.specs['max_length'],
                                               truncation=True)
            labels = self.tokenizer(examples['targets'], max_length=self.specs['max_length']//4,
                                                truncation=True)
            self.tokenized_dataset['labels'] = labels['input_ids']
            return self.tokenized_dataset
        elif self.task_type == "question_answering":
            context = examples['context']
            question = [q.strip() for q in examples['question']]
            inputs = self.tokenizer(question, context, max_length=self.specs['max_length'],
                                    truncation="only_second", stride=self.specs['stride'], padding="max_length",
                                    return_overflowing_tokens=True, return_offsets_mapping=True)
            offset_mapping = inputs.pop("offset_mapping")
            sample_map = inputs.pop("overflow_to_sample_mapping")
            answers = examples["answers"]
            start_positions = []
            end_positions = []

            for i, offset in enumerate(offset_mapping):
                sample_idx = sample_map[i]
                answer = answers[sample_idx]
                start_char = answer["answer_start"][0]
                end_char = answer["answer_start"][0] + len(answer["text"][0])
                sequence_ids = inputs.sequence_ids(i)

                # Find the start and end of the context
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                # If the answer is not fully inside the context, label is (0, 0)
                if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # Otherwise it's the start and end token positions
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_positions.append(idx - 1)

                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)
            
            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions
            self.tokenized_dataset = inputs
            return self.tokenized_dataset
        else: pass

    def tokenize_validation_dataset(self, examples):
        # TO QC LATER
        if self.task_type == "token_classification":
            self.tokenized_dataset = self.tokenizer(examples['inputs'],
                                                    is_split_into_words=True, truncation=True,
                                                    return_offsets_mapping=True)
            new_labels  = [self.align_labels_with_tokens(labels, self.tokenized_dataset.word_ids(i)) 
                           for i,labels in enumerate(examples['labels'])]
            self.tokenized_dataset['labels'] = new_labels
            return self.tokenized_dataset
        elif self.task_type == "masked_language_modeling":
            self.tokenized_dataset = self.tokenizer(examples['inputs'],
                                                    is_split_into_words=True, truncation=True,
                                                    return_offsets_mapping=True)
            self.tokenized_dataset['word_ids'] = self.tokenized_dataset.word_ids()
            return self.tokenized_dataset
        elif self.task_type == "translation":
            _from = examples['inputs']
            _to = examples['targets']
            self.tokenized_dataset = self.tokenizer(_from, text_target=_to, max_length=self.specs['max_length'],
                                               truncation=True)
            return self.tokenized_dataset
        elif self.task_type == "summarization":
            self.tokenized_dataset = self.tokenizer(examples['inputs'], max_length=self.specs['max_length'],
                                               truncation=True)
            labels = self.tokenizer(examples['targets'], max_length=self.specs['max_length']//4,
                                                truncation=True)
            self.tokenized_dataset['labels'] = labels['input_ids']
            return self.tokenized_dataset
        elif self.task_type == "question_answering":
            context = examples['context']
            question = [q.strip() for q in examples['question']]
            inputs = self.tokenizer(question, context, max_length=self.specs['max_length'],
                                    truncation="only_second", stride=self.specs['stride'], padding="max_length",
                                    return_overflowing_tokens=True, return_offsets_mapping=True)
            offset_mapping = inputs.pop("offset_mapping")
            sample_map = inputs.pop("overflow_to_sample_mapping")
            example_ids = []

            for i in range(len(inputs["input_ids"])):
                sample_idx = sample_map[i]
                example_ids.append(examples["id"][sample_idx])

                sequence_ids = inputs.sequence_ids(i)
                inputs["offset_mapping"][i] = [
                    o if sequence_ids[k] == 1 else None for k, o in enumerate(offset_mapping)
                ]

            inputs["example_id"] = example_ids
            self.tokenized_dataset = inputs
            return self.tokenized_dataset
        else: pass
    
    def group_texts(self, examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        chunk_size = self.specs['chunk_size']
        total_length = (total_length // chunk_size) * chunk_size
        # Split by chunks of max_len
        results = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column
        results["labels"] = results["input_ids"].copy()
        return results

    def prepare_dataset(self, dataset, task_type):
        # TO ADD LATER
        self.task_type = task_type
        self.raw_dataset = dataset

        if 'eval' not in self.raw_dataset.keys():
            self.raw_datasets = self.raw_datasets["train"].train_test_split(train_size=0.9, seed=42)

        train_dataset = self.raw_dataset['train'].map(
            self.tokenize_dataset,
            batched=True,
            remove_columns=self.raw_dataset['train'].column_names
        )
        eval_dataset = self.raw_dataset['eval'].map(
            self.tokenize_dataset,
            batched=True,
            remove_columns=self.raw_dataset['eval'].column_names
        )
        if self.task_type == "masked_language_modeling":
            train_dataset = train_dataset.map(
                self.group_texts,
                batched=True,
            )
            eval_dataset = eval_dataset.map(
                self.group_texts,
                batched=True
            )
        return {
            "train": train_dataset,
            "eval": eval_dataset
        }

    def prepare_dataset_from_dir(self, task_type):
        # TO ADD LATER
        self.task_type = task_type
        self.raw_dataset = load_dataset(self.dataset_dir)

        if 'eval' not in self.raw_dataset.keys():
            self.raw_datasets = self.raw_datasets["train"].train_test_split(train_size=0.9, seed=42)

        train_dataset = self.raw_dataset['train'].map(
            self.tokenize_dataset,
            batched=True,
            remove_columns=self.raw_dataset['train'].column_names
        )
        eval_dataset = self.raw_dataset['eval'].map(
            self.tokenize_validation_dataset,
            batched=True,
            remove_columns=self.raw_dataset['eval'].column_names
        )
        return {
            "train": train_dataset,
            "eval": eval_dataset
        }

    def load_dataset(self, name):
        self.raw_dataset = load_dataset(name)
        pass

    def save_dataset(self):
        pass