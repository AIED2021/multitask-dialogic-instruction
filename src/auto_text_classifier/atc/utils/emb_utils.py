import os
import pickle
import gensim
import jieba
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer

class EmbCore():
    def __init__(self, emb_path, emb_dim):
        self.emb_dim = emb_dim
        self.default_emb = np.array(
            [0.] * emb_dim, dtype=np.float32)

    def get_vocabs(self):
        """get vocabs
        Returns
        -------
            vocabs:vocabs
        """
        raise NotImplementedError

    def is_in(self, word):
        """check word is in emb
        Parameters
        ----------
            word: word
        Returns
        -------
            state:True if word in emb else False.
        """
        raise NotImplementedError

    def get_word_emb(self, word):
        """get word embedding,if word in emb_dict else error
        Parameters
        ----------
            word: word
        Returns
        -------
            emb:word emb
        """
        raise NotImplementedError

    def get_tokenizer(self):
        vocabs = self.get_vocabs()
        tokenizer = Tokenizer()
        tokenizer.word_index = dict(zip(vocabs, range(1, len(vocabs)+1)))
        return tokenizer

    def get_emb_matrix(self):
        tokenizer = self.get_tokenizer()
        vocab = tokenizer.word_index
        emb_matrix = np.zeros((len(vocab) + 1, self.emb_dim))
        for word, i in vocab.items():
            emb_vector = self.get_w2v(word)
            emb_matrix[i, :] = emb_vector
        return emb_matrix

    def get_w2v(self, word):
        """get word embedding,if word in emb_dict return embedding else return default_embedding
        Parameters
        ----------
            word: word
        Returns
        -------
            emb:word emb or default_embedding
        """
        if self.is_in(word):
            embedding = self.get_word_emb(word)
        else:
            embedding = self.default_emb
        return embedding

    def get_sentence_emb_mean(self, sentence, max_len):
        words = jieba.lcut(sentence)[:max_len]
        emb_list = []
        for word in words:
            embedding = self.get_w2v(word)
            if len(embedding) != 0:
                emb_list.append(embedding)
        if len(emb_list) == 0:
            return self.default_emb
        else:
            return np.array(emb_list).mean(axis=0)

    def get_sentence_list_emb_mean(self, text_list, max_len):
        if len(text_list) == 0:
            raise Exception("text_list is empty")
        s2w = {}
        for sentence in set(text_list):
            s2w[sentence] = self.get_sentence_emb_mean(sentence, max_len)
        emb_list = []
        for sentence in text_list:
            emb_list.append(s2w[sentence])
        return np.array(emb_list).reshape(len(emb_list), self.emb_dim)


class EmbDict(EmbCore):
    def __init__(self, emb_path, emb_dim):
        super().__init__(emb_path, emb_dim)
        self.w2v = pickle.load(open(emb_path, 'rb'))

    def get_vocabs(self):
        return list(self.w2v.keys())

    def is_in(self, word):
        return word in self.w2v

    def get_word_emb(self,word):
        return self.w2v[word]


class EmbGensimBin(EmbCore):
    def __init__(self, emb_path, emb_dim):
        self.emb = gensim.models.KeyedVectors.load_word2vec_format(emb_path, binary=True)
        super().__init__(emb_path, emb_dim)
        
    def get_vocabs(self):
        vocabs = list(self.emb.vocab.keys())
        return vocabs

    def is_in(self, word):
        return word in self.emb.vocab
    
    def get_word_emb(self, word):
        return self.emb.get_vector(word)

class EmbTxtType_1(EmbCore):
    def __init__(self, emb_path, emb_dim):
        self.w2v = {}
        with open(emb_path, "r", encoding="utf-8") as fp:
            for i, s_line in enumerate(fp):
                if i == 0:
                    ls_line = s_line.strip().split(" ")
                    n_vocab_num = int(ls_line[0])
                    n_word_dim = int(ls_line[1])
                    print("token num ", n_vocab_num)
                    print("word dim ", n_word_dim)
                else:
                    ls_line = s_line.rstrip().split(" ")
                    s_token = ls_line[0]
                    lf_vec = [float(e) for e in ls_line[1:]] 
                    np_vec = np.array(lf_vec)

                    self.w2v[s_token] = np_vec
                    if i % 10000 == 0:
                        print("load %s %s" % (str(i), s_token))
                        # break

    def get_vocabs(self):
        return list(self.w2v.keys())

    def is_in(self, word):
        return word in self.w2v

class EmbTC(EmbCore):
    def __init__(self, emb_path, emb_dim):
        super().__init__(emb_path, emb_dim)
        emb_dir = emb_path
        names = pickle.load(open(os.path.join(emb_dir,'names.plk'),'rb'))
        self.words = dict(zip(names,range(len(names))))
        self.w2i = pickle.load(open(os.path.join(emb_dir,'w2i.plk'),'rb'))
        self.i2w = pickle.load(open(os.path.join(emb_dir,'i2w.plk'),'rb'))
        self.full_vector = np.load(os.path.join(emb_dir,'full.npy'))
                
    def get_vocabs(self):
        return self.words

    def is_in(self, word):
        return word in self.words
    
    def get_word_emb(self, word):
        return self.full_vector[self.w2i[word],:]


def shrink_emb(emb_config,text_list,save_path):
    """shrink emb use text_list.Use EmbDict to load shrink_emb.
    Parameters
    ----------
        emb_config: emb config
        text_list: text_list to get small embedding
        save_path: where to save
    Returns
    -------
        None
    """
    emb_model = emb_config['emb_class'](emb_config['emb_path'],emb_config['emb_dim'])
    words = []
    for text in text_list:
        words.extend(jieba.lcut(str(text)))
    words = list(set(words))
    small_emb = {}
    for word in words:
        if emb_model.is_in(word):
            small_emb[word] =  emb_model.get_word_emb(word)

    print("Total token in shrink_emb is {},ignore token num is {}.".format(
        len(small_emb),
        len(words)-len(small_emb)))

    print("Use the shrink embedding by `EmbDict(save_path,emb_dim={})`".format(emb_config['emb_dim']))

    with open(save_path,'wb') as f:
        pickle.dump(small_emb,f)

def init_emb_from_config(emb_config):
    emb = emb_config['emb_class'](emb_config['emb_path'],emb_config['emb_dim'])
    return emb
    
if __name__ == "__main__":
    embed_path = r"/share/small_project/auto_text_classifier/atc/data/word_vector/Word_Character_Ngram_sgns.wiki.bigram-char"
    emb = EmbTxtType_1(embed_path,300)

    # w2v = emb.w2v 

    import pickle

    # s_w2v_pkl = r"/share/small_project/auto_text_classifier/atc/data/word_vector/Word_Character_Ngram_sgns.wiki.bigram-char.pkl"
    # with open(s_w2v_pkl, "wb") as fp:
    #     pickle.dump(w2v, fp)

    # emb = EmbDict(s_w2v_pkl,300)
    # s_word = "词语"
    # np_vec = emb.get_word_emb(s_word)
    # print(np_vec)
    # print(np_vec.shape)