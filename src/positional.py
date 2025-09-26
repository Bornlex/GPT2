import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()

        self.__embedding_size = embedding_size

    def forward(self, x, *args, **kwargs):
        mask = torch.zeros_like(x)

        for pos in range(x.shape[1]):
            mask[:, pos, :] = pos

        for index in range(x.shape[-1]):
            mask[:, :, index] /= (10000 ** (2 * index / self.__embedding_size))

        mask[:, :, 0::2] = torch.sin(mask[:, :, 0::2])
        mask[:, :, 1::2] = torch.cos(mask[:, :, 1::2])

        return mask
