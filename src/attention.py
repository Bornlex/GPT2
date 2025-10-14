import torch
from torch import nn


class ScaledDotProduct(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer('mask', None, persistent=False)

    def create_causal_mask(self, inputs: torch.Tensor):
        mask = torch.ones_like(inputs, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)

        return mask

    def forward(self, keys, queries, values):
        x = torch.matmul(queries, keys.transpose(-1, -2)) / (keys.shape[-1] ** 0.5)

        if self.mask is None or self.mask.shape[-1] != x.shape[-1]:
            self.mask = self.create_causal_mask(x)

        x = torch.masked_fill(x, mask=self.mask, value=-torch.inf)

        x = torch.softmax(x, -1)
        x = torch.matmul(x, values)

        return x


class ScaledDotProductMHA(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer('mask', None, persistent=False)

    def create_causal_mask(self, inputs: torch.Tensor):
        mask = torch.ones_like(inputs, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)

        return mask

    def forward(self, keys, queries, values):
        b, n, h, d2 = keys.shape
        x = torch.matmul(
            queries.reshape((b, h, n, d2)),
            keys.reshape((b, h, n, d2)).transpose(-1, -2)
        ) / (d2 ** 0.5)

        if self.mask is None or self.mask.shape[-1] != x.shape[-1]:
            self.mask = self.create_causal_mask(x)

        x = torch.masked_fill(x, mask=self.mask, value=-torch.inf)

        x = torch.softmax(x, -1)
        x = torch.matmul(x, values.reshape((b, h, n, d2)))
        x = x.reshape((b, n, h, d2))

        return x


class Attention(nn.Module):
    """
    The Attention layer accepts a tensor of shape
    (b, n, d) as input, with :
    - b -> the batch size
    - n -> the sequence size
    - d -> the embedding size

    Then we define the layers dimensions :
    - dk
    - dq (= dk)
    - dv
    """
    def __init__(self, embedding_size: int, dk: int):
        super().__init__()

        self.__d = embedding_size
        self.__dk = dk

        self.__qkv = nn.Linear(self.__d, 2 * self.__dk + self.__d)
        self.__scaled_dot_product = ScaledDotProduct()

    def forward(self, x, kv_cache: dict | None = None, *args, **kwargs):
        """
        inputs (n, d)
        """
        qkv = self.__qkv(x)  # (n, 2 * dk + d)
        queries, keys, values = torch.split(qkv, [self.__dk, self.__dk, self.__d], -1)
        result = self.__scaled_dot_product(keys, queries, values)

        return result


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size: int, n_heads: int):
        assert embedding_size % n_heads == 0

        super().__init__()

        self.__embedding_size = embedding_size
        self.__number_heads = n_heads

        self.__qkv = nn.Linear(
            self.__embedding_size,
            self.__embedding_size * 3
        )
        self.__scaled_dot_product = ScaledDotProductMHA()
        self.__fc = nn.Linear(self.__embedding_size, self.__embedding_size)

    def forward(self, x, kv_cache: dict | None = None, *args, **kwargs):
        qkv = self.__qkv(x)
        queries, keys, values = torch.split(qkv, [self.__embedding_size] * 3, -1)

        head_size = self.__embedding_size // self.__number_heads
        queries = queries.view(x.shape[0], x.shape[1], self.__number_heads, head_size)
        keys = keys.view(x.shape[0], x.shape[1], self.__number_heads, head_size)
        values = values.view(x.shape[0], x.shape[1], self.__number_heads, head_size)

        scaled = self.__scaled_dot_product(keys, queries, values)
        scaled = scaled.view(x.shape[0], x.shape[1], self.__embedding_size)

        result = self.__fc(scaled)

        return result


class MultiQueryAttention(nn.Module):
    def __init__(self, embedding_size: int, n_heads: int, dropout: float = 0.2):
        assert embedding_size % n_heads == 0

        super().__init__()

        self.__embedding_size = embedding_size
        self.__number_heads = n_heads
        self.__head_dimension = self.__embedding_size // self.__number_heads

        self.__qkv = nn.Linear(
            self.__embedding_size,
            self.__embedding_size + 2 * self.__head_dimension
        )
        self.__fc = nn.Linear(self.__embedding_size, self.__embedding_size)
        self.__dropout = nn.Dropout(dropout)

        self.register_buffer('mask', None, persistent=False)

    def forward(self, x, kv_cache: dict | None = None, *args, **kwargs):
        qkv = self.__qkv(x)
        queries, keys, values = torch.split(qkv, [self.__embedding_size, self.__head_dimension, self.__head_dimension], -1)

        if kv_cache is not None:
            if kv_cache.get('k') is not None:
                keys = torch.cat((kv_cache['k'], keys), dim=1)
                values = torch.cat((kv_cache['v'], values), dim=1)

            kv_cache['k'] = keys
            kv_cache['v'] = values

        queries = queries.view(queries.shape[0], queries.shape[1], self.__number_heads, self.__head_dimension)
        keys = keys.unsqueeze(2)
        values = values.unsqueeze(2)

        use_causal_mask = x.shape[1] > 1

        scaled = self.scaled_dot_product(keys, queries, values, use_causal_mask=use_causal_mask)
        scaled = scaled.reshape(queries.shape[0], queries.shape[1], self.__embedding_size)

        result = self.__fc(scaled)
        result = self.__dropout(result)

        return result, kv_cache

    def scaled_dot_product(self, keys, queries, values, use_causal_mask: bool = True):
        b, n, h, d2 = queries.shape
        x = torch.matmul(
            queries.transpose(1, 2),
            keys.transpose(1, 2).transpose(-1, -2)
        ) / (d2 ** 0.5)

        if use_causal_mask:
            if self.mask is None or self.mask.shape[-1] != x.shape[-1]:
                self.mask = self.create_causal_mask(x)

            x = torch.masked_fill(x, mask=self.mask, value=-torch.inf)

        x = torch.softmax(x, -1)
        x = torch.matmul(x, values.transpose(1, 2))
        x = x.transpose(1, 2).contiguous()

        return x

    def create_causal_mask(self, x: torch.Tensor):
        mask = torch.ones_like(x, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)

        return mask
