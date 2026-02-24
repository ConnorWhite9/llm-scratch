# LayerNorm, FFN, embeddings

import torch.nn as nn
import torch 
import math
from math import sqrt
from .config import ModelConfig
#Embeddings
# pseudo-structure only, not full code

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        

    def forward(self, token_ids):  # [batch, seq_len] of ints
        # returns [batch, seq_len, d_model]
        return self.embedding(token_ids)

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(max_seq_len, d_model)
          # ---- Build the positional encoding matrix ----
        # We want a tensor of shape [max_seq_len, d_model]
        # where each row = position (0..max_seq_len-1)
        # and each column = one dimension of the embedding

        # position: tensor [0, 1, 2, ..., max_seq_len-1] with shape [max_seq_len]

        position = torch.arange(0, max_seq_len).unsqueeze(1)  # [max_seq_len, 1]

    
        # exp( -log(10000) * (2i / d_model) )

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # [d_model/2]

        # pe: positional encoding tensor we will fill:
        # start with zeros, shape [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)

        
        #For all even indexes use the sin position function. 
        pe[:, 0::2] = torch.sin(position * div_term)  # fill even dimensions with sin

        #For all odd indexes use the cos position function. 
        pe[:, 1::2] = torch.cos(position * div_term)  # fill odd dimensions with cos

        # register_buffer:
        # - Tells PyTorch this tensor is part of the module's state, but not a parameter.
        # - It will be moved to GPU/CPU with .to(device), saved in state_dict, etc.
        # - It will NOT be updated by the optimizer (no gradients).
        #
        self.register_buffer("pe", pe)  # shape: [max_seq_len, d_model]
     

    def forward(self, seq_len):  # or token_ids to infer seq_len
        # returns [1, seq_len, d_model]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} > max_seq_len {self.max_seq_len}"
            )

        # self.pe has shape [max_seq_len, d_model]
        # self.pe[:seq_len] -> [seq_len, d_model] (positions 0 .. seq_len-1)

        # .unsqueeze(0) adds a batch dimension at the front:
        # [seq_len, d_model] -> [1, seq_len, d_model]
        # so it matches token embeddings shape [batch, seq_len, d_model]
        return self.pe[:seq_len].unsqueeze(0)

class Embeddings(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_emb = TokenEmbedding(config.vocab_size, config.d_model)
        self.pos_emb = PositionalEmbedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.scale = sqrt(config.d_model)
    

    def forward(self, token_ids: torch.Tensor):
        # 1) lookup token_emb(token_ids)
        batch_size, seq_len = token_ids.size()
        device = token_ids.device
        tok = self.token_emb(token_ids)
        # 2) add pos_emb
        pos = self.pos_emb(seq_len).to(device)
        # 3) apply dropout
        # Add together token embedding and positional embedding to get final embedding to input directly into llm. 
        final_emb = tok * self.scale + pos
        return self.dropout(final_emb)