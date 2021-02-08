from atc.configs.log_config import *
from atc.configs.emb_config import *

dpcnn_base_config = {
    "b_print": True,
    "run_logger": None,
    "stop_token": [""," "," ","\t","\r","\n"], 
    "max_len": 60,
    "embed_config": cn_wikipedia_emb_config,
    "filter_num": 16,  
    "pooling_stride": 2,  
    "kernel_size": 3,  
    "blocks": 2,    
    "num_labels": 2, # 类别个数
    "learn_rate": 0.0001,    
    "dropout": 0.2, # 随即丢失 0.5
    "epochs": 100,
    "batch_size": 32,
    "patience": 5,
    "use_cuda": True,
    "update_embed": True,  # embedding是否也一起跟着训练更新
    "save_dir": "model/dpcnn",
    "weight": [1.0, 5.072]
}