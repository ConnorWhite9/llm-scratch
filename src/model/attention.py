# Multi-head self-attention with causal mask

import torch.nn as nn
import torch
import math
from .config import ModelConfig
class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = self.d_model // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model)
        self.out = nn.Linear(self.d_model, self.d_model)
        self.register_buffer("mask", torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)))
    
    #Scaled dot product calculation
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        att_nlogits = att_nlogits /math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = torch.softmax(attn_logits, dim = 1)
        values = torch.matmul(attention, v)
        return values, attention 