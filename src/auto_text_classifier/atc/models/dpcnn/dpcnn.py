# -*- coding: utf-8 -*-

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from atc.models.torch_base import TorchBase

# DPCNN网络
# class DPCNNNet(nn.Module):
#     def __init__(self, O_CONFIG):
#         super(DPCNNNet, self).__init__()

#         self._n_filter_num = O_CONFIG["filter_num"]

#         self.embedding = nn.Embedding.from_pretrained(O_CONFIG["embed_pretrained"], freeze=(not O_CONFIG["update_embed"]))
#         self.conv_region = nn.Conv2d(1, self._n_filter_num, (3, O_CONFIG["embed_dim"]), stride=1)
#         self.conv = nn.Conv2d(self._n_filter_num, self._n_filter_num, (3, 1), stride=1)
#         self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
#         self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
#         self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
#         self.relu = nn.ReLU()
#         self.fc = nn.Linear(2 * self._n_filter_num, O_CONFIG["num_labels"])
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         batch_size = x.shape[0]
#         # x = x[0]

#         x = self.embedding(x)
#         x = x.unsqueeze(1)       # In: [batch_size, seq_len, embed_dim]      Out:[batch_size, 1, seq_len, embed_dim]
       
#         x = self.conv_region(x)  # In: [batch_size, 1, seq_len, embed_dim]   Out:[batch_size, filter_num, seq_len-3+1, 1]
#         x = self.padding1(x)  # In: [batch_size, filter_num, seq_len-3+1, 1]   Out:[batch_size, filter_num, seq_len, 1]
#         x = self.relu(x)   # In: [batch_size, filter_num, seq_len, 1]  Out: [batch_size, filter_num, seq_len, 1]
       
#         x = self.conv(x)  # In: [batch_size, filter_num, seq_len, 1]   Out:[batch_size, filter_num, seq_len-3+1, 1]
#         x = self.padding1(x)  # In: [batch_size, filter_num, seq_len-3+1, 1]   Out: [batch_size, filter_num, seq_len, 1]
#         x = self.relu(x) # In: [batch_size, filter_num, seq_len, 1]   Out: [batch_size, filter_num, seq_len, 1]
        
#         x = self.conv(x)  # In: [batch_size, filter_num, seq_len, 1]   Out:[batch_size, filter_num, seq_len-3+1, 1]
#         while x.size()[-2] > 2:
#             x = self._block(x)
#             # print(x.shape)
            
#         # x = x.squeeze()  # [batch_size, num_filters(250)]
#         # print(x.shape)
#         x = x.view(batch_size, 2*self._n_filter_num)
#         x = self.fc(x)


#         if len(x.shape) ==  1:
#             x = x.unsqueeze(0)
#         x = self.softmax(x)
#         return x

#     def _block(self, x):
#         x = self.padding2(x)
#         px = self.max_pool(x)

#         x = self.padding1(px)
#         x = F.relu(x)
#         x = self.conv(x)

#         x = self.padding1(x)
#         x = F.relu(x)
#         x = self.conv(x)

#         # Short Cut
#         x = x + px
#         return x


class DPCNNNet(nn.Module):
    def __init__(self, O_CONFIG):
        super(DPCNNNet, self).__init__()

        
        self.embedding = nn.Embedding.from_pretrained(O_CONFIG["embed_pretrained"], freeze=(not O_CONFIG["update_embed"]))
          
        self.num_kernels = O_CONFIG["filter_num"]
        self.pooling_stride = O_CONFIG["pooling_stride"]
        self.kernel_size = O_CONFIG["kernel_size"]
        self.radius = int(self.kernel_size / 2)
        self.input_size = O_CONFIG["embed_dim"]
        self.blocks = O_CONFIG["blocks"]
        self.dropout = O_CONFIG["dropout"]
        self.output_size = O_CONFIG["num_labels"]

        assert self.kernel_size % 2 == 1, "kernel should be odd!"
        self.convert_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                self.input_size, self.num_kernels,
                self.kernel_size, padding=self.radius)
        )

        self.convs = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                self.num_kernels, self.num_kernels,
                self.kernel_size, padding=self.radius),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                self.num_kernels, self.num_kernels,
                self.kernel_size, padding=self.radius)
        ) for _ in range(self.blocks + 1)])

        self.dropout = nn.Dropout(p=self.dropout, inplace=True)
        self.fc = torch.nn.Linear(self.num_kernels, self.output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedding = self.embedding(x)          # In: [batch_size, seq_len]      Out:[batch_size, seq_len, embed_dim] 
        embedding = embedding.permute(0, 2, 1) # In: [batch_size, seq_len, embed_dim]       Out:[batch_size, embed_dim, seq_len] 
        
        conv_embedding = self.convert_conv(embedding) # In: [batch_size, embed_dim, seq_len]       Out:[batch_size, kernel_num, seq_len]      
        conv_features = self.convs[0](conv_embedding) # In: [batch_size, kernel_num, seq_len]       Out:[batch_size, kernel_num, seq_len]   
        conv_features = conv_embedding + conv_features # In: [batch_size, kernel_num, seq_len]       Out:[batch_size, kernel_num, seq_len]
        
        for i in range(1, len(self.convs)):
            # In: [batch_size, kernel_num, seq_len]       Out:[batch_size, kernel_num, seq_len//2]
            block_features = F.max_pool1d(
                conv_features, self.kernel_size, self.pooling_stride) 
           
            conv_features = self.convs[i](block_features) # In: [batch_size, kernel_num, seq_len//2]       Out:[batch_size, kernel_num, seq_len//2]
            
            conv_features = conv_features + block_features # In: [batch_size, kernel_num, seq_len//2]       Out:[batch_size, kernel_num, seq_len//2]
            
        pool_out = F.max_pool1d(conv_features, conv_features.size(2)).squeeze() # In: [batch_size, kernel_num, seq_len//2]       Out:[batch_size, kernel_num]
        
        outputs = self.fc(self.dropout(pool_out))
        if len(outputs.shape) ==  1:
            outputs = outputs.unsqueeze(0)

        outputs = self.softmax(outputs)
        return outputs

  
class DPCNN(TorchBase):
    def __init__(self, O_CONFIG):
        super().__init__(O_CONFIG)

        self._s_model_name = "dpcnn"
        
        self._net = DPCNNNet(O_CONFIG)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=O_CONFIG["learn_rate"])
        
        self._s_model_with_weight_path = os.path.join(self.save_dir, "%s_model_with_weight.pth" % (self._s_model_name))
        self._s_best_model_with_weight_path = os.path.join(self.save_dir, "%s_best_model_with_weight.pth" % (self._s_model_name))
        self._s_weight_file = os.path.join(self.save_dir, "%s_weight.pth" % (self._s_model_name))

        self.model_path = self._s_best_model_with_weight_path

    

