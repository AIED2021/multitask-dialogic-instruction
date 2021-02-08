# -*- coding: utf-8 -*-

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from atc.models.torch_base import TorchBase

# FastText网络
class TextCNNNet(nn.Module):
    def __init__(self, O_CONFIG):
        super(TextCNNNet, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(O_CONFIG["embed_pretrained"], freeze=(not O_CONFIG["update_embed"]))

        # nn.Conv2d(1, 16, (3, 100))  1个通道(文本是单通道), 16个卷积核， (3*100)的卷积核大小
        # nn.Conv2d(1, 16, (4, 100))  1个通道(文本是单通道), 16个卷积核， (4*100)的卷积核大小
        # nn.Conv2d(1, 16, (5, 100))  1个通道(文本是单通道), 16个卷积核， (5*100)的卷积核大小

        self.convs = nn.ModuleList([nn.Conv2d(1, O_CONFIG["filter_num"], (k, O_CONFIG["embed_dim"])) for k in O_CONFIG["filter_size"]])
        self.dropout = nn.Dropout(O_CONFIG["dropout"], inplace=True)
        self.fc = nn.Linear(O_CONFIG["filter_num"] * len(O_CONFIG["filter_size"]), O_CONFIG["num_labels"])
        self.softmax = nn.Softmax(dim=1)
        
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)              # In: [batch, 1, seq, embed]      [batch, kernel_num, seq - kernel_row, 1]    Out: [batch, kernel_num, seq - kernel_row]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)   # In: [batch, kernel_num, seq - kernel_row]     Out: [batch, kernel_num]
        # F.max_pool1d 对应维度数据组成数组取最大值，对应维度数据个数变为1，所以用squeeze压缩
        return x

    def forward(self, x):
        out = self.embedding(x) # In: [batch, seq]            Out: [batch, seq, embed]
        out = out.unsqueeze(1)  # In: [batch, seq, embed]     Out: [batch, 1, seq, embed]       
        # out = x.unsqueeze(1)  # In: [batch, seq, embed]     Out: [batch, 1, seq, embed]  
        out = self.dropout(out) # In: [batch, 1, seq, embed]  Out: [batch, 1, seq, embed]

        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1) # In: [batch, 1, seq, embed]  Out: [batch, kernel_num * filter_num]   
        out = self.fc(out)      # In: [batch, kernel_num * filter_num]   Out: [batch, class] 

        out = self.softmax(out) # In: [batch, class]                     Out: [batch, class] 
        return out

class TextCNN(TorchBase):
    def __init__(self, O_CONFIG):
        super().__init__(O_CONFIG)

        self._s_model_name = "textcnn"
        
        self._net = TextCNNNet(O_CONFIG)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=O_CONFIG["learn_rate"])
        
        self._s_model_with_weight_path = os.path.join(self.save_dir, "%s_model_with_weight.pth" % (self._s_model_name))
        self._s_best_model_with_weight_path = os.path.join(self.save_dir, "%s_best_model_with_weight.pth" % (self._s_model_name))
        self._s_weight_file = os.path.join(self.save_dir, "%s_weight.pth" % (self._s_model_name))
        self.model_path = self._s_best_model_with_weight_path

    

