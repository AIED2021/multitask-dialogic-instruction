
from atc.utils.emb_utils import *
import os
import copy
base_path = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.join(base_path,"../data/word_vector/")

# 中文
# 腾讯中文词向量(缩小版)
cn_emb_tencent_small_path = os.path.join(base_dir, '中文腾讯_highfreq/normal_words_tencent.pkl')
cn_emb_tencent_small_config = {"emb_class": EmbDict,
                      "emb_path": cn_emb_tencent_small_path, "emb_dim": 200}

# 中文
# 腾讯中文词向量
cn_emb_tencent_path = os.path.join(base_dir, '中文腾讯')
cn_emb_tencent_config = {"emb_class": EmbTC,
                      "emb_path": cn_emb_tencent_path, "emb_dim": 200}

# 自研ASR
cn_emb_zy_asr_path = os.path.join(base_dir, '自研ASR/godeye_w2v_0426.bin')
cn_emb_zy_asr_config = {"emb_class":EmbGensimBin,"emb_path":cn_emb_zy_asr_path,"emb_dim":200}

# 搜狗输入法词向量
cn_emb_sgsrf_path = os.path.join(base_dir,"搜狗输入法词向量/shurufa_uft8_killSingleTerm.bin")
cn_emb_sgsrf_config = {"emb_class":EmbGensimBin,"emb_path":cn_emb_sgsrf_path,"emb_dim":400}

# 百度百科
cn_emb_bdbk_path = os.path.join(base_dir,"百度百科/baidubaike.pkl")
cn_emb_bdbk_config = {"emb_class":EmbDict,"emb_path":cn_emb_bdbk_path,"emb_dim":300}

# 维基百科
cn_emb_wibk_path = os.path.join(base_dir,"维基百科/wikibaike.pkl")
cn_emb_wibk_config = {"emb_class":EmbDict,"emb_path":cn_emb_wibk_path,"emb_dim":300}

# 人民日报
cn_emb_rmrb_path = os.path.join(base_dir,"人民日报/renmin.pkl")
cn_emb_rmrb_config = {"emb_class":EmbDict,"emb_path":cn_emb_rmrb_path,"emb_dim":300}

# 搜狗新闻字向量
cn_emb_sg_char_path = os.path.join(base_dir,"搜狗新闻字向量/sougounews_char.pkl")
cn_emb_sg_char_config = {"emb_class":EmbDict,"emb_path":cn_emb_sg_char_path,"emb_dim":300}

# 英文

# GoogleNews-EN300-Highfreq
en_emb_google_new_high_path = os.path.join(base_dir,"google_news/GoogleNews-EN300-Highfreq.pkl")
en_emb_google_new_high_config = {"emb_class": EmbDict,
                      "emb_path": en_emb_google_new_high_path, "emb_dim": 300}

# cn_wikipedia_emb_path = os.path.join(
#     base_path, '../data/word_vector/Word_Character_Ngram_sgns.wiki.bigram-char.pkl')
cn_wikipedia_emb_path = os.path.join(base_dir,"Word_Character_Ngram_sgns.wiki.bigram-char.pkl")

cn_wikipedia_emb_config = {"emb_class": EmbDict,
                      "emb_path": cn_wikipedia_emb_path, "emb_dim": 300}

# 英文的
#
# wikipedia
en_emb_wikipedia_path = os.path.join(base_dir,"wikipedia/wiki-news-300d-1M.pkl")
en_emb_wikipedia_config = {"emb_class": EmbDict,
                      "emb_path": en_emb_wikipedia_path, "emb_dim": 300}
