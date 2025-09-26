from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 64
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = True
    ffn_hidden_size: int = 256
