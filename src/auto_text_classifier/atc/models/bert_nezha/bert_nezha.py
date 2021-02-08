from atc.models.bert4keras_base import Bert4KearsBase
import os

class NEZHA(Bert4KearsBase):
    def __init__(self,config):
        super().__init__(config)
        self.model_name = 'nezha'
        self.best_weights_path = os.path.join(self.save_dir,'best_model_{}.weights'.format(self.model_name))
        self.model_path = os.path.join(self.save_dir,'best_model_{}.model'.format(self.model_name))
        print(self.model_path)