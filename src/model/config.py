from dataclasses import dataclass


@dataclass
class ModelConfig:
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int
    vocab_size: int
    max_seq_len: int
    dropout: float = 0.1
