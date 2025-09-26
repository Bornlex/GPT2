import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, features: int):
        super().__init__()

        self.__epsilon = 1e-5
        self.__gamma = nn.Parameter(torch.ones(features))
        self.__beta = nn.Parameter(torch.zeros(features))

    def forward(self, x, *args, **kwargs):
        mean = torch.mean(x, -1).unsqueeze(-1)
        stdev = torch.std(x, -1).unsqueeze(-1)

        mean_tensor = torch.zeros_like(x)
        stdev_tensor = torch.zeros_like(x)

        mean_tensor[:, :, :] = mean
        stdev_tensor[:, :, :] = stdev

        normalized = (x - mean_tensor) / (stdev_tensor + self.__epsilon)

        return normalized * self.__gamma + self.__beta
