# -*- coding: utf-8 -*-

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from atc.models.torch_base import TorchBase

# FastText网络
class FastTextNet(nn.Module):
    def __init__(self, O_CONFIG):
        super(FastTextNet, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(O_CONFIG["embed_pretrained"], freeze=(not O_CONFIG["update_embed"]))

        self.dropout = nn.Dropout(O_CONFIG["dropout"], inplace=True)
        self.fc = nn.Sequential(
            nn.Linear(O_CONFIG["embed_dim"], O_CONFIG["hidden_size"]),
            nn.BatchNorm1d(O_CONFIG["hidden_size"]),
            nn.ReLU(inplace=True),
            nn.Linear(O_CONFIG["hidden_size"], O_CONFIG["num_labels"]),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.embedding(x) # In: [bantch, seq]    Out: [bantch, seq, embed]
        out = out.mean(dim=1)   # In: [bantch, seq, embed]   Out: [bantch, embed]

        out = self.dropout(out) # In: [bantch, embed]        Out:  [bantch, embed]
        out = self.fc(out)      # In:  [bantch, embed]       Out:  [bantch, class]
        return out

class FastText(TorchBase):
    def __init__(self, O_CONFIG):
        super().__init__(O_CONFIG)
        self._s_model_name = "fasttext"
        
        self._net = FastTextNet(O_CONFIG)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=O_CONFIG["learn_rate"])
        
        self._s_model_with_weight_path = os.path.join(self.save_dir, "%s_model_with_weight.pth" % (self._s_model_name))
        self._s_best_model_with_weight_path = os.path.join(self.save_dir, "%s_best_model_with_weight.pth" % (self._s_model_name))
        self._s_weight_file = os.path.join(self.save_dir, "%s_weight.pth" % (self._s_model_name))
        
        self.model_path = self._s_best_model_with_weight_path
    

    

