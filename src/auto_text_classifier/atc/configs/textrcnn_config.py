
from atc.configs.log_config import *
from atc.configs.emb_config import *

textrcnn_base_config = {
    "b_print": True,
    "run_logger": None,
    "stop_token": [""," "," ","\t","\r","\n"], 
    "max_len": 150,
    "embed_config": cn_wikipedia_emb_config,
    "hidden_size": 256,   # 隐藏层个数
    "layer_num": 1,      # LSTM层数
    "kernel_size": 100,  # pool池化的卷积核大小
    "num_labels": 2, # 类别个数
    "learn_rate": 0.001,    
    "dropout": 0.5, # 随即丢失 0.5
    "epochs": 100,
    "batch_size": 64,
    "patience": 12,
    "use_cuda": True,
    "update_embed": True,
    "save_dir": "model/textrcnn"
}
