from transformers import BertTokenizer, AutoTokenizer

class TokenizerModules():
    def __init__(self, checkpoint, use_bert=False):
        self.checkpoint = checkpoint
        self.use_bert = use_bert
        
    def load_tokenizer(self):
        if self.use_bert:
            tokenizer = BertTokenizer.from_pretrained(self.checkpoint)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        return tokenizer
