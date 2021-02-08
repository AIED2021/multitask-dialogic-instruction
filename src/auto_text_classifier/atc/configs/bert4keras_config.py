
import os
import copy
base_path = os.path.dirname(os.path.realpath(__file__))

# nezha_base
nezha_base_dir = os.path.join(base_path, '../data/nezha_base')
nezha_base_config = {"config_path": os.path.join(nezha_base_dir, 'bert_config.json'),
                     "checkpoint_path": os.path.join(nezha_base_dir, 'model.ckpt-900000'),
                     "dict_path": os.path.join(nezha_base_dir, 'vocab.txt'),
                     "save_dir": 'model/nezha_base/'}

# nezha_base_wwm
nezha_base_wwm_dir = os.path.join(base_path, '../data/nezha_base_wwm')
nezha_base_wwm_config = {"config_path": os.path.join(nezha_base_wwm_dir, 'bert_config.json'),
                     "checkpoint_path": os.path.join(nezha_base_wwm_dir, 'model.ckpt-691689'),
                     "dict_path": os.path.join(nezha_base_wwm_dir, 'vocab.txt'),
                     "save_dir": 'model/nezha_base_wwm/'}

# nezha_large
nezha_large_dir = os.path.join(base_path, '../data/nezha_large')
nezha_large_config = {"config_path": os.path.join(nezha_large_dir, 'bert_config.json'),
                     "checkpoint_path": os.path.join(nezha_large_dir, 'model.ckpt-325810'),
                     "dict_path": os.path.join(nezha_large_dir, 'vocab.txt'),
                     "save_dir": 'model/nezha_large/'}

# nezha_large_wwm
nezha_large_wwm_dir = os.path.join(base_path, '../data/nezha_large_wwm')
nezha_large_wwm_config = {"config_path": os.path.join(nezha_large_wwm_dir, 'bert_config.json'),
                     "checkpoint_path": os.path.join(nezha_large_wwm_dir, 'model.ckpt-346400'),
                     "dict_path": os.path.join(nezha_large_wwm_dir, 'vocab.txt'),
                     "save_dir": 'model/nezha_large_wwm/'}