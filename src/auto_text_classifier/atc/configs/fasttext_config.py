
from atc.configs.log_config import *
from atc.configs.emb_config import *


fasttext_base_config = {
    "b_print": True,
    "run_logger": None,
    "stop_token": [""," "," ","\t","\r","\n"], 
    "max_len": 150,
    "embed_config": cn_wikipedia_emb_config,
    "hidden_size": 256,  # 隐藏层个数
    "num_labels": 2, # 类别个数
    "learn_rate": 0.001,    
    "dropout": 0.5, # 随即丢失 0.5
    "epochs": 100,
    "batch_size": 128,
    "patience": 5,
    "use_cuda": True,
    "update_embed": True,
    "save_dir": "model/fasttext"
}
