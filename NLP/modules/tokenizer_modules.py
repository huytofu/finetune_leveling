from transformers import BertTokenizer, AutoTokenizer

class TokenizerModules():
    def __init__(self, checkpoint, use_bert=False):
        """
        Initialize the TokenizerModules.
        
        Args:
            checkpoint: Path to the model checkpoint.
            use_bert: Whether to use BERT-specific tokenizer.
        """
        self.checkpoint = checkpoint
        self.use_bert = use_bert
        
    def load_tokenizer(self):
        """
        Load the tokenizer based on the specified model checkpoint.
        
        Returns:
            The loaded tokenizer.
        """
        if self.use_bert:
            # Load BERT tokenizer
            tokenizer = BertTokenizer.from_pretrained(self.checkpoint)
        else:
            # Load tokenizer for other models
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        return tokenizer
