# -*- coding: utf-8 -*-

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from atc.models.torch_base import TorchBase

class BiLSTMNet(nn.Module):
    def __init__(self, O_CONFIG):
        super(BiLSTMNet, self).__init__()

        self.b_use_cuda = O_CONFIG["use_cuda"]
        self.n_embed_dim = O_CONFIG["embed_dim"]
        self.n_hidden_size = O_CONFIG["lstm_hidden_size"]
        self.n_layer_num = O_CONFIG["layer_num"]

        self.embedding = nn.Embedding.from_pretrained(O_CONFIG["embed_pretrained"], freeze=(not O_CONFIG["update_embed"]))

        # self.layer_norm = torch.nn.LayerNorm(self.n_embed_dim)

        # batch_first 第一个维度设为 batch, 即:(batch_size, seq_length, embedding_dim)
        self.lstm = nn.LSTM(self.n_embed_dim,  # 文本嵌入的维度
                            self.n_hidden_size,  # 隐藏层单元个数
                            self.n_layer_num,    # 隐藏层层数
                            bidirectional=True, dropout=O_CONFIG["dropout"], batch_first=True) # 
        
        # self.fc = nn.Sequential(
        #     nn.Linear(O_CONFIG["lstm_hidden_size"] * 2, O_CONFIG["fc_hidden_size"]),
        #     nn.BatchNorm1d(O_CONFIG["fc_hidden_size"]),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(O_CONFIG["fc_hidden_size"], O_CONFIG["num_labels"]),
        #     nn.Softmax(dim=1)
        # )

        # self.fc = nn.Sequential(
        #     nn.Linear(O_CONFIG["lstm_hidden_size"] * 2, O_CONFIG["num_labels"]),
        #     nn.Softmax(dim=1)
        # )
        self.dropout = nn.Dropout(O_CONFIG["dropout"])
        self.fc = nn.Linear(self.n_hidden_size * 2, O_CONFIG["num_labels"])
        self.softmax = nn.Softmax(dim=1)

    
    def forward(self, x):
        x = self.embedding(x)   # In: [batch, seq]            Out: [batch, seq, embed]

        # layer normalization 
        # batch, sentence_len, input_size = x.shape
        # x = x.view(-1, input_size)
        # x = self.layer_norm(x)
        # x = x.view(batch, sentence_len, input_size)

        h_0 = torch.zeros(2 * self.n_layer_num, x.size(0), self.n_hidden_size) # Initial hidden state of the LSTM
        c_0 = torch.zeros(2 * self.n_layer_num, x.size(0), self.n_hidden_size) # Initial cell state of the LSTM
        if self.b_use_cuda:
            h_0, c_0 = h_0.cuda(), c_0.cuda()

        x = self.dropout(x)

        # out, _ = self.lstm(x)   # In: [batch, seq, embed]            Out: [batch, seq, hidden]
        # h_n = out[:,-1,:]

        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))   # In: [batch, seq, embed]            Out: [batch, seq, hidden]
        h_n = torch.cat((h_n[0,:,:], h_n[1,:,:]), 1)

        h_n = self.dropout(h_n)  # 句子最后时刻的 hidden state
        out = self.fc(h_n)   # In: [batch, 1, hidden]            Out: [batch, class] 
        out = self.softmax(out)        # In: [batch, class]                  Out: [batch, class]  
        return out

class BiLSTM(TorchBase):
    def __init__(self, O_CONFIG):
        super().__init__(O_CONFIG)

        self._s_model_name = "bilstm"
        
        self._net = BiLSTMNet(O_CONFIG)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=O_CONFIG["learn_rate"])
        
        self._s_model_with_weight_path = os.path.join(self.save_dir, "%s_model_with_weight.pth" % (self._s_model_name))
        self._s_best_model_with_weight_path = os.path.join(self.save_dir, "%s_best_model_with_weight.pth" % (self._s_model_name))
        self._s_weight_file = os.path.join(self.save_dir, "%s_weight.pth" % (self._s_model_name))

        self.model_path = self._s_best_model_with_weight_path

    





