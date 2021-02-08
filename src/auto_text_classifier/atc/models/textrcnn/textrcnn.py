# -*- coding: utf-8 -*-

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from atc.models.torch_base import TorchBase

# FastText网络
class TextRCNNNet(nn.Module):
    def __init__(self, O_CONFIG):
        super(TextRCNNNet, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(O_CONFIG["embed_pretrained"], freeze=(not O_CONFIG["update_embed"]))
        self.lstm = nn.LSTM(O_CONFIG["embed_dim"], O_CONFIG["hidden_size"], O_CONFIG["layer_num"],
                            bidirectional=True, batch_first=True, dropout=O_CONFIG["dropout"])
        self.maxpool = nn.MaxPool1d(O_CONFIG["kernel_size"], padding=0)
        self.fc = nn.Linear(O_CONFIG["hidden_size"] * 2 + O_CONFIG["embed_dim"], O_CONFIG["num_labels"])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embed = self.embedding(x)        # In: [batch, seq]                               Out: [batch, seq, embed]
        out, _ = self.lstm(embed)        # In: [batch, seq, embed]                        Out: [batch, seq, hidden]
        out = torch.cat((embed, out), 2) # In: [batch, seq, embed] [batch, seq, hidden]   Out: [batch, seq, embed + hidden]
        out = F.relu(out) # In: [batch, seq, embed + hidden]                              Out: [batch, seq, embed + hidden]
        out = out.permute(0, 2, 1)       # In: [batch, seq, embed + hidden]               Out: [batch, embed + hidden, seq]
        # print("out.permute", out.shape)

        out = self.maxpool(out).squeeze() # In: [batch, embed + hidden, seq]      Out: [batch, embed + hidden]
        out = self.fc(out)                # In: [batch, embed + hidden]       Out: [batch, class]

        if len(out.shape) ==  1:
            out = out.unsqueeze(0)
            
        out = self.softmax(out)           # In: [batch, class]                Out: [batch, class]
        return out

class TextRCNN(TorchBase):
    def __init__(self, O_CONFIG):
        super().__init__(O_CONFIG)

        self._s_model_name = "textrcnn"
        
        self._net = TextRCNNNet(O_CONFIG)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=O_CONFIG["learn_rate"])
        
        self._s_model_with_weight_path = os.path.join(self.save_dir, "%s_model_with_weight.pth" % (self._s_model_name))
        self._s_best_model_with_weight_path = os.path.join(self.save_dir, "%s_best_model_with_weight.pth" % (self._s_model_name))
        self._s_weight_file = os.path.join(self.save_dir, "%s_weight.pth" % (self._s_model_name))
        self.model_path = self._s_best_model_with_weight_path

    





