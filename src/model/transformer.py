import torch
import torch.nn as nn

from .config import ModelConfig
from .layers import Embeddings
from .attention import CausalSelfAttention


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network used inside each Transformer block.
    Applies the same MLP to every token position independently.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # nn.Sequential lets us chain layers in the order they are called in forward()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),  # [B, T, d_model] -> [B, T, d_ff]
            nn.GELU(),  # non-linear activation (applied element-wise)
            nn.Linear(config.d_ff, config.d_model),  # back to [B, T, d_model]
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, d_model]
        return self.net(x)


class DecoderBlock(nn.Module):
    """
    One decoder block = LayerNorm + self-attention + residual,
    followed by LayerNorm + feed-forward + residual.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # LayerNorm normalizes features in the last dimension (d_model)
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]

        # "Pre-norm" pattern:
        # 1) normalize x
        # 2) apply attention
        # 3) add the result back to the original x (residual connection)
        x = x + self.attn(self.ln_1(x))

        # Same idea for the feed-forward network:
        # normalize -> MLP -> add residual
        x = x + self.ff(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """
    Decoder-only Transformer (GPT-style language model).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embeddings combines:
        # - token embeddings (what word/byte it is)
        # - positional encodings (where in the sequence it is)
        self.embed = Embeddings(config)

        # ModuleList is a trainable list of submodules.
        # Here we create n_layers copies of DecoderBlock and stack them.
        self.blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.n_layers)]
        )

        # Final LayerNorm applied after all blocks
        self.ln_f = nn.LayerNorm(config.d_model)

        # Language-modeling head:
        # takes hidden states [B, T, d_model] -> logits over vocabulary [B, T, vocab_size]
        # bias=False is common in GPT-style models.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: [batch, seq_len] integer token indices
        returns:
            logits: [batch, seq_len, vocab_size]

        This computes next-token logits for every position in the sequence.
        """
        # 1) Turn token ids into continuous embeddings (with positions added)
        x = self.embed(token_ids)  # [B, T, d_model]

        # 2) Pass through each Transformer layer in sequence.
        #    Each block does self-attention + feed-forward with residuals.
        for block in self.blocks:
            x = block(x)

        # 3) Final LayerNorm + projection to vocabulary size
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]
        return logits
