from atc.models.base_model import BaseModel
from atc.utils.data_utils import init_dir
from atc.models.base_model import BaseModel
from atc.utils.metrics_utils import get_model_metrics
from atc.utils.data_utils import load_df
from atc.utils.emb_utils import init_emb_from_config
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from itertools import product
import os
from tqdm import tqdm

class GBDT(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        emb_config = config['emb_config']
        self.emb = init_emb_from_config(emb_config)
        self.model_name = "GBDT"
        self.model_path = os.path.join(self.save_dir, "gbdt.pkl")
        self.model_config = self.config.get("model_config",{})

    def process_one_data(self, path):
        df = load_df(path)
        x = self.emb.get_sentence_list_emb_mean(
            df['text'].tolist(), max_len=self.max_len)
        y = df['label']
        return x, y

    def process_data(self, train_path, dev_path, test_path):
        x_train, y_train = self.process_one_data(train_path)
        x_dev, y_dev = self.process_one_data(dev_path)
        x_test, y_test = self.process_one_data(test_path)
        return x_train, y_train, x_dev, y_dev, x_test, y_test

    def train(self, train_path, dev_path, test_path, model_config={},save_model=True):
        if len(model_config)==0:
            model_config = self.model_config
        x_train, y_train, x_dev, y_dev, x_test, y_test = self.process_data(
            train_path, dev_path, test_path)
        self.model = GradientBoostingClassifier(**model_config)
        self.model.fit(x_train, y_train)
        if save_model:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
        return self.evaluate(test_path)
    
    def search_best_params(self,train_path, dev_path, test_path,param_grid):
        param_grid = sorted(param_grid.items())
        param_name_list = [x[0] for x in param_grid]
        grid_list = [x[1] for x in param_grid]
        all_model_report = []
        for model_config in tqdm(product(*grid_list)):
            model_config = dict(zip(param_name_list,model_config))
            model_report = self.train(train_path, dev_path, dev_path,model_config=model_config,save_model=False)
            model_report.update(model_config)
            all_model_report.append(model_report)
        return all_model_report
      

    def demo(self, text):
        return self.demo_text_list([text])[0]

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def demo_text_list(self, text_list):
        x = self.emb.get_sentence_list_emb_mean(text_list,max_len=self.max_len)
        y_pred = self.model.predict_proba(x)[:, 1]
        return y_pred
