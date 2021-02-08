# -*- coding: utf-8 -*-

import os
import sys
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np

from atc.models.torch_base import TorchBase


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerNet(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""
    def __init__(self, O_CONFIG, use_cls_token=False, requires_grad=False):
        super(TransformerNet, self).__init__()
       
        self.embed_dim = O_CONFIG["embed_dim"]
        self.dropout = O_CONFIG["dropout"]
        self.max_len = O_CONFIG["max_len"]
        self.num_labels = O_CONFIG["num_labels"]

        self.head_num = O_CONFIG["head_num"]
        self.dim_feedforward = O_CONFIG["dim_feedforward"]
        self.encoder_layter_num = O_CONFIG["encoder_layter_num"]

        self.src_mask = None
        self.use_cls_token = use_cls_token
    
    
        self.embedding = nn.Embedding.from_pretrained(O_CONFIG["embed_pretrained"], freeze=(not O_CONFIG["update_embed"]))
        # self.embedding.weight = nn.Parameter(weights, requires_grad=requires_grad) # Assigning the look-up table to the pre-trained GloVe word embedding.

        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout)

        encoder_layers = TransformerEncoderLayer(self.embed_dim, self.head_num, self.dim_feedforward, self.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.encoder_layter_num)
        
        if(use_cls_token==True):
            print('Enable CLS token for classification... ')
            self.cls_token_vector = torch.empty(self.embed_dim).uniform_(-0.1, 0.1)
        else:
            print('Enable weighted sum hidden states for classification...')
            self.weighted_sum_layer = nn.Linear(self.max_len, 1, bias=False)

        self.linear = nn.Linear(self.embed_dim, self.num_labels)  
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_src_key_padding_mask(self, src):
        '''
        args:
        src shape: batch_size, max_time_step
        returns:
        boolean padding mask of shape: batch_size, max_time_step
        where True values are posititions taht should be masked with -inf
        and False values will be unchanged.
        '''
        return src==0

    def init_weights(self):
        initrange = 0.1
        if(not self.use_cls_token):
            self.weighted_sum_layer.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, batch_size=None, has_mask=True):
        '''
        Args:
        src: input sequence of shape: batch_size, max_time_step

        Returns:
        final_output of shape: batch_size, self.num_labels
        '''
        if(has_mask):
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self.generate_src_key_padding_mask(src)
                self.src_mask = mask
        else:
            self.src_mask = None

        if(self.use_cls_token==True):
            src = src[:, :-1] # batch_size, max_time_step-1
            cls_vector_repeat = self.cls_token_vector.repeat(src.shape[0], 1, 1)
            src = torch.cat((cls_vector_repeat, self.word_embeddings(src)), dim=1) # append cls token vector at the front
            src *= math.sqrt(self.embed_dim)
        else:
            src = self.embedding(src) * math.sqrt(self.embed_dim) #  batch_size, max_time_setp, embed_dim
            
        src = src.permute(1, 0, 2)  # max_time_setp, batch_size, embed_dim
        src = self.pos_encoder(src) # max_time_setp, batch_size, embed_dim

        output = self.transformer_encoder(src, src_key_padding_mask=self.src_mask) # max_time_step, batch_size, embed_dim

        if(self.use_cls_token): # use cls hidden state for classification
            final_hidden = output[0, :, :].view(-1, self.d_model) # batch_size, embed_dim
        else:
            # compute weighted sum of hidden states over each time_step
            output = torch.transpose(output, 0, 1) # batch_size, max_time_step, embed_dim
            output = torch.transpose(output, 1, 2) # batch_size, embed_dim, max_time_step
            final_hidden = self.weighted_sum_layer(output).view(-1, self.embed_dim) # batch_size, embed_dim
        
        final_output = self.linear(final_hidden) # batch_size, self.num_labels

        final_output = self.softmax(final_output)

        return final_output

class Transformer(TorchBase):
    def __init__(self, O_CONFIG):
        super().__init__(O_CONFIG)

        self._s_model_name = "transformer"
        
        self._net = TransformerNet(O_CONFIG)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=O_CONFIG["learn_rate"])
        
        self._s_model_with_weight_path = os.path.join(self.save_dir, "%s_model_with_weight.pth" % (self._s_model_name))
        self._s_best_model_with_weight_path = os.path.join(self.save_dir, "%s_best_model_with_weight.pth" % (self._s_model_name))
        self._s_weight_file = os.path.join(self.save_dir, "%s_weight.pth" % (self._s_model_name))

        self.model_path = self._s_best_model_with_weight_path
    

    

