import os
base_path = os.path.dirname(os.path.realpath(__file__))
from os.path import join
# 中文模型

# xlnet base

xlnet_base_dir = join(base_path, '../data/hfl_chinese_xlnet_base')
xlnet_base_config = {"model_dir": xlnet_base_dir,
                     "save_dir": 'model/xlnet_base'}
# xlnet mid
xlnet_mid_config = {"model_dir": join(base_path, '../data/hfl_chinese_xlnet_mid'),
                     "save_dir": 'model/xlnet_mid'}
# xlnet_large
xlnet_large_config = {"model_dir": join(base_path, '../data/clue_xlnet_chinese_large'),
                     "save_dir": 'model/xlnet_large'}

# bert base
bert_base_dir = join(base_path, '../data/bert_base_chinese')
bert_base_config = {"model_dir": bert_base_dir, "save_dir": 'model/bert_base'}

# bert base wwm_ext
hfl_chinese_bert_wwm_ext_dir = join(
    base_path, '../data/hfl_chinese_bert_wwm_ext')
bert_base_www_ext_config = {
    "model_dir": hfl_chinese_bert_wwm_ext_dir, "save_dir": 'model/bert_base_www_ext'}

# chinese-roberta-wwm-ext
chinese_roberta_wwm_ext_dir = join(
    base_path, '../data/chinese_roberta_wwm_ext')
chinese_roberta_wwm_ext_config = {
    "model_dir": chinese_roberta_wwm_ext_dir,
    # "save_dir": 'model/chinese_roberta_wwm_ext/',
    "save_dir": '/share/zjf/Models/roberta_neel_1845000/',
}


# hfl_chinese_roberta_wwm_ext_large
chinese_roberta_wwm_ext_large_dir = join(
    base_path, '../data/hfl_chinese_roberta_wwm_ext_large')
hfl_chinese_roberta_wwm_ext_large_config = {"model_dir": chinese_roberta_wwm_ext_large_dir,
                                     "save_dir": 'model/chinese_roberta_wwm_ext_large/',"use_bert_type":True}

# xlm_roberta_base
xlm_roberta_base_config = {"model_dir": join(base_path, '../data/xlm_roberta_base'),
                     "save_dir": 'model/xlm_roberta_base'}
# xlm_roberta_large
xlm_roberta_large_config = {"model_dir": join(base_path, '../data/xlm_roberta_large'),
                     "save_dir": 'model/xlm_roberta_large'}

# voidful albert base
voidful_albert_chinese_base_dir = join(
    base_path, '../data/voidful_albert_chinese_base')
voidful_albert_chinese_base_config = {"model_dir": voidful_albert_chinese_base_dir,
                                       "save_dir": 'model/voidful_albert_chinese_base/'}

# voidful albert small
voidful_albert_chinese_small_dir = join(
    base_path, '../data/voidful_albert_chinese_small')
voidful_albert_chinese_small_config = {"model_dir": voidful_albert_chinese_small_dir,
                                       "save_dir": 'model/voidful_albert_chinese_small/'}

# voidful albert tiny
voidful_albert_chinese_tiny_dir = join(
    base_path, '../data/voidful_albert_chinese_tiny')
voidful_albert_chinese_tiny_config = {"model_dir": voidful_albert_chinese_tiny_dir,
                                      "save_dir": 'model/voidful_albert_chinese_tiny/'}
# chinese_electra_large
hfl_chinese_electra_large_dir = join(
    base_path, '../data/hfl_chinese_electra_large_d')
hfl_chinese_electra_large_config = {"model_dir": hfl_chinese_electra_large_dir,
                                   "save_dir": 'model/electra_large/'}

# chinese_electra_base
hfl_chinese_electra_base_dir = join(
    base_path, '../data/hfl_chinese_electra_base_d')
hfl_chinese_electra_base_config = {"model_dir": hfl_chinese_electra_base_dir,
                                   "save_dir": 'model/electra_base/'}


# chinese_electra_small
hfl_chinese_electra_small_dir = join(
    base_path, '../data/hfl_chinese_electra_small_d')
hfl_chinese_electra_small_config = {"model_dir": hfl_chinese_electra_small_dir,
                                    "save_dir": 'model/electra_small'}


# 英文模型

## en_bert_base_cased
en_bert_base_cased_config = {"model_dir": join(base_path, '../data/en_bert_base_cased'),
                             "save_dir": 'model/en_bert_base_cased'}
# ## en_gpt2
# en_gpt2_config = {"model_dir": join(base_path, '../data/en_gpt2'),
#                              "save_dir": 'model/en_gpt2'}
# ## en_gpt2_large
# en_gpt2_large_config = {"model_dir": join(base_path, '../data/en_gpt2_large'),
#                              "save_dir": 'model/en_gpt2_large'}

## en_roberta_base
en_roberta_base_config = {"model_dir":  join(base_path, '../data/en_roberta_base'),
                             "save_dir": 'model/en_roberta_base'}

## en_roberta_large
en_roberta_large_config = {"model_dir": join(base_path, '../data/en_roberta_large'),
                             "save_dir": 'model/en_roberta_large'}

## en_xlnet_base_cased
en_xlnet_base_cased_config = {"model_dir": join(base_path, '../data/en_xlnet_base_cased'),
                             "save_dir": 'model/en_xlnet_base_cased'}
# en_xlnet_large_cased
en_xlnet_large_cased_config = {"model_dir": join(base_path, '../data/en_xlnet_large_cased'),
                             "save_dir": 'model/en_xlnet_large_cased'}

## en_albert_base_v2
en_albert_base_v2_config = {"model_dir": join(base_path, '../data/en_albert_base_v2'),
                             "save_dir": 'model/en_albert_base_v2'}

## en_albert_xxlarge_v2
# en_albert_xxlarge_v2_config = {"model_dir": join(base_path, '../data/en_albert_xxlarge_v2'),
#                              "save_dir": 'model/en_albert_xxlarge_v2'}

## google_electra_small
en_google_electra_small_config = {"model_dir": join(base_path, '../data/en_google_electra_small'),
                             "save_dir": 'model/en_google_electra_small'}
## google_electra_base
en_google_electra_base_config = {"model_dir": join(base_path, '../data/en_google_electra_base'),
                             "save_dir": 'model/en_google_electra_base'}
## google_electra_large
en_google_electra_large_config = {"model_dir": join(base_path, '../data/en_google_electra_large'),
                             "save_dir": 'model/en_google_electra_large'}
## en_bart_large
en_bart_large_config = {"model_dir": join(base_path, '../data/en_bart_large'),
                             "save_dir": 'model/en_bart_large',"token_type_ids_disable":True}