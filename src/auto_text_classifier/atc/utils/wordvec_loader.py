import os
import numpy as np

def load_cn_wiki_wordvec(s_word_vec_path):
    n_vocab_num = 0
    n_word_dim = 0
    m_word_vec ={}
    m_word_id = {}
    with open(s_word_vec_path, "r", encoding="utf-8") as fp:
        for i, s_line in enumerate(fp):
            if i == 0:
                ls_line = s_line.strip().split(" ")
                n_token_num = int(ls_line[0])
                n_word_dim = int(ls_line[1])
                print("token num ", n_vocab_num)
                print("word dim ", n_word_dim)
            else:
                ls_line = s_line.rstrip().split(" ")
                s_token = ls_line[0]
                lf_vec = [float(e) for e in ls_line[1:]] 
                np_vec = np.array(lf_vec)

                m_word_vec[s_token] = np_vec
                m_word_id[s_token] = i-1
                if i % 10000 == 0:
                    print("load %s %s" % (str(i), s_token))
                    # break

    o_out = {
        "word_vec":m_word_vec,
        "word_id": m_word_id,
        "word_dim": n_word_dim
    }                
    return o_out