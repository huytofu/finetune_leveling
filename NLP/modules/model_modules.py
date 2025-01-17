from transformers import BertForMaskedLM, BertForTokenClassification
from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering

class ModelModules():
    def __init__(self, checkpoint, use_bert=False):
        self.checkpoint = checkpoint
        self.use_bert = use_bert
        
    def load_model(self, task_type):
        if task_type == "masked_language_modeling":
            if self.use_bert:
                model = BertForMaskedLM.from_pretrained(self.checkpoint)
            else:
                model = AutoModelForMaskedLM.from_pretrained(self.checkpoint)
        elif task_type == "token_classification":
            if self.use_bert:
                model = BertForTokenClassification.from_pretrained(self.checkpoint)
            else:
                model = AutoModelForTokenClassification.from_pretrained(self.checkpoint)
        elif task_type == "translation":
            if self.use_bert:
                model = None
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint)
        elif task_type == "summarization":
            if self.use_bert:
                model = None
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint)
        elif task_type == "text_generation":
            if self.use_bert:
                model = None
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint)
        elif task_type == "question_answering":
            if self.use_bert:
                model = None
            else:
                model = AutoModelForQuestionAnswering.from_pretrained(self.checkpoint)
        else:
            model = None
        return model
