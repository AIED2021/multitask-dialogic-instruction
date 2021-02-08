from atc.models.hf_base import HFBase
from transformers import BertForSequenceClassification, BertModel, BertTokenizer,AutoTokenizer,AutoModelForSequenceClassification


class ALBERT(HFBase):
    def __init__(self,config):
        super().__init__(config)
        self.model_name = 'albert'

    def get_tokenizer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        except:
            tokenizer = BertTokenizer.from_pretrained(self.model_dir)
        return tokenizer
