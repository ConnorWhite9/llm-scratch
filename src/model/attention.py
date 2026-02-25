# Multi-head self-attention with causal mask

import math
import torch
import torch.nn as nn

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = self.d_model // self.n_heads

        # Precompute the 1/sqrt(d_k) scaling factor used in attention
        self.scale = self.head_dim ** -0.5

        # One projection layer that produces Q, K, V in a single matmul
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model)

        # Final projection after concatenating all heads
        self.out = nn.Linear(self.d_model, self.d_model)

        # Lower-triangular causal mask of shape [max_seq_len, max_seq_len]
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)),
            persistent=False,
        )

        # Optional dropout on attention weights and output
        self.attn_dropout = nn.Dropout(config.dropout)
        self.out_dropout = nn.Dropout(config.dropout)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        q, k, v: [batch, n_heads, seq_len, head_dim]
        mask: broadcastable to [batch, n_heads, seq_len, seq_len]
        """
        # Raw attention scores: [batch, n_heads, seq_len, seq_len]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))

        # Scale by 1/sqrt(d_k) to keep logits in a good range
        attn_logits = attn_logits * self.scale

        # Apply causal (and any additional) mask: very negative where mask == 0
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

        # Softmax over the key dimension (last dim)
        attention = torch.softmax(attn_logits, dim=-1)
        attention = self.attn_dropout(attention)

        # Weighted sum of values: [batch, n_heads, seq_len, head_dim]
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        returns: [batch, seq_len, d_model]
        """
        bsz, seq_len, _ = x.size()

        # Project once, then split into Q, K, V: [batch, seq_len, 3 * d_model]
        qkv = self.qkv(x)

        # Split last dimension into three: each is [batch, seq_len, d_model]
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to [batch, n_heads, seq_len, head_dim]
        def reshape_to_heads(t):
            return t.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = reshape_to_heads(q)
        k = reshape_to_heads(k)
        v = reshape_to_heads(v)

        # Build a causal mask for this sequence length and make it broadcastable
        # mask: [1, 1, seq_len, seq_len] -> broadcasts over batch and heads
        causal_mask = self.mask[:seq_len, :seq_len]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Compute attention per head
        attn_output, _ = self.scaled_dot_product_attention(q, k, v, mask=causal_mask)

        # Merge heads back: [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

        # Final linear + dropout
        out = self.out(attn_output)
        out = self.out_dropout(out)
        return out