from atc.models import *
from atc.configs import *

model_dict = {
    "gbdt": {"model_class": GBDT, "config": gbdt_config},
    "albert_base": {"model_class": ALBERT,
                     "config": voidful_albert_chinese_base_config},
    "albert_small": {"model_class": ALBERT,
                     "config": voidful_albert_chinese_small_config},
    "albert_tiny": {"model_class": ALBERT,
                    "config": voidful_albert_chinese_tiny_config},
    "bert_base": {"model_class": BERT, "config": bert_base_config},
    "bert_base_wwm": {"model_class": BERT, "config": bert_base_www_ext_config},
    "nezha_base": {"model_class": NEZHA, "config": nezha_base_config},
    "nezha_base_wwm": {"model_class": NEZHA, "config": nezha_base_wwm_config},
    "nezha_large": {"model_class": NEZHA, "config": nezha_large_config},
    "nezha_large_wwm": {"model_class": NEZHA, "config": nezha_large_wwm_config},
    "roberta": {"model_class": ROBERTA, "config": chinese_roberta_wwm_ext_config},
    "roberta_wwm_ext_large": {"model_class": ROBERTA, "config": hfl_chinese_roberta_wwm_ext_large_config},
    # "clue_roberta_large":{"model_class": ROBERTA, "config": clue_roberta_large_config},
    "xlm_roberta_base":{"model_class": ROBERTA, "config": xlm_roberta_base_config},
    "xlm_roberta_large":{"model_class": ROBERTA, "config": xlm_roberta_large_config},
    "electra_large": {"model_class": ELECTRA, "config": hfl_chinese_electra_large_config},
    "electra_base": {"model_class": ELECTRA, "config": hfl_chinese_electra_base_config},
    "electra_small": {"model_class": ELECTRA, "config": hfl_chinese_electra_small_config},
    "xlnet_base": {"model_class": XLNet, "config": xlnet_base_config},
    "xlnet_mid": {"model_class": XLNet, "config": xlnet_mid_config},
    # "xlnet_large":{"model_class": XLNet, "config": xlnet_large_config},
    "bilstm": {"model_class": BiLSTM, "config": bilstm_base_config},
    "fasttext": {"model_class": FastText, "config": fasttext_base_config},
    "dpcnn": {"model_class": DPCNN, "config": dpcnn_base_config},
    "textcnn": {"model_class": TextCNN, "config": textcnn_base_config},
    "textrcnn": {"model_class": TextRCNN, "config": textrcnn_base_config}
}
en_model_dict = {"en_bert_base_cased": {"model_class": BERT, "config": en_bert_base_cased_config},
                 "en_roberta_base": {"model_class": ROBERTA, "config": en_roberta_base_config},
                 "en_roberta_large": {"model_class": ROBERTA, "config": en_roberta_large_config},
                 "en_xlnet_base_cased": {"model_class": XLNet, "config": en_xlnet_base_cased_config},
                 "en_xlnet_large_cased": {"model_class": XLNet, "config": en_xlnet_large_cased_config},
                 "en_albert_base_v2": {"model_class": ALBERT, "config": en_albert_base_v2_config},
                 "en_electra_small": {"model_class": ELECTRA, "config": en_google_electra_small_config},
                 "en_electra_base": {"model_class": ELECTRA, "config": en_google_electra_base_config},
                 "en_electra_large": {"model_class": ELECTRA, "config": en_google_electra_large_config},
                 "en_bart_large":{"model_class":BART,"config":en_bart_large_config}
                 }


model_dict.update(en_model_dict)

default_model_list = list(model_dict.keys())
