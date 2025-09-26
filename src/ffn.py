import torch
from torch import nn


class FFN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float):
        super().__init__()

        self.__fc1 = nn.Linear(input_size, hidden_size)
        self.__fc2 = nn.Linear(hidden_size, output_size)
        self.__dropout = nn.Dropout(dropout)

    def forward(self, x, *args, **kwargs):
        x = self.__fc1(x)
        x = torch.relu(x)
        x = self.__fc2(x)

        return x
