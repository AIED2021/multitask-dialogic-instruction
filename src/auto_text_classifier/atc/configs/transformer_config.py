import os
import numpy as np

from atc.configs.log_config import *
from atc.configs.emb_config import *
from atc.utils.emb_utils import *

S_BASE_PATH = os.path.dirname(os.path.realpath(__file__))

# 模型和日志默认保存地址
save_path = os.path.join(S_BASE_PATH, '../data/fasttext')
if not os.path.exists(save_path):
    os.makedirs(save_path)

transformer_base_config = {
    "b_print": True,
    "run_logger": None,
    # "run_logger": init_logger(os.path.join(S_FASTTEXT_SAV_DIR, "fasttext.log"), b_log_debug=True),
    "stop_token": [""," "," ","\t","\r","\n"], 

    "max_len": 150,

    "embed_config": cn_wikipedia_emb_config,
   
    "head_num": 5,      
    "dim_feedforward": 256,
    "encoder_layter_num": 2,
    
    "num_labels": 2, # 类别个数
    "learn_rate": 0.0001,    
    "dropout": 0.5, # 随即丢失 0.5

    "epochs": 100,
    "batch_size": 128,
    "patience": 15,
    "use_cuda": True,
    "update_embed": True,
    "save_dir": save_path
}