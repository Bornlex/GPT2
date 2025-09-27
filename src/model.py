import numpy as np
import torch
from torch import nn

from src import utils
from src.attention import MultiHeadAttention, MultiQueryAttention
from src.ffn import FFN
from src.gpt_config import GPTConfig
from src.normalization import LayerNorm
from src.positional import PositionalEmbedding


class GPTBlock(nn.Module):
    def __init__(self, embedding_size: int, n_head: int, hidden_size: int, dropout: float):
        super().__init__()

        self.__attention = MultiQueryAttention(embedding_size, n_head)
        self.__norm1 = LayerNorm(embedding_size)
        self.__fc = FFN(embedding_size, hidden_size, embedding_size, dropout)
        self.__norm2 = LayerNorm(embedding_size)

    def forward(self, x, *args, **kwargs):
        dx1 = self.__attention(x)
        x = self.__norm1(x + dx1)
        dx2 = self.__fc(x)
        x = self.__norm2(x + dx2)

        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
        self.device = 'cpu'

        self.__positional_embedding = PositionalEmbedding(config.n_embd)
        self.__embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.__blocks = nn.ModuleList([
            GPTBlock(config.n_embd, config.n_head, config.ffn_hidden_size, config.dropout)
            for _ in range(config.n_layer)
        ])
        self.__fc = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x, *args, **kwargs):
        x = self.__embedding(x)
        positional_x = self.__positional_embedding(x)
        x = x + positional_x

        for block in self.__blocks:
            x = block(x)

        x = self.__fc(x)

        return x

    def generate(self, sentence: str, max_tokens: int = 256, temperature: float = 1.0) -> str:
        encoder, decoder = utils.get_encoder_decoder()

        encoded = encoder(sentence)
        encoded = np.array(encoded)
        indices = torch.from_numpy(encoded.astype(np.int64)).view(1, encoded.shape[0]).to(self.device)

        self.eval()
        with torch.no_grad():
            for _ in range(max_tokens):
                indices_cond = indices if indices.shape[1] <= self.config.block_size else indices[:, -self.config.block_size:]
                logits = self(indices_cond)
                logits = logits[:, -1, :] / temperature
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                next_index = torch.multinomial(probabilities, num_samples=1)
                indices = torch.cat((indices, next_index), dim=1)

        indices = indices.tolist()
        result = decoder(indices[0])

        return result

    def to(self, device, *args, **kwargs):
        self.device = device
        super().to(device, *args, **kwargs)


if __name__ == '__main__':
    model = GPT(GPTConfig())
    generation = model.generate('Shakespeare was born in ')
    print(generation)
