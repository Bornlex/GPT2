import pytest
import torch

from src.attention import MultiHeadAttention
from src.gpt_config import GPTConfig
from src.normalization import LayerNorm
from src.model import GPT
from src.positional import PositionalEmbedding


torch.manual_seed(42)

GPT2_EMBEDDING_DIMENSION = 768


@pytest.fixture()
def random_tensor():
    """
    Returns a tensor of shape (1, 5, 3).
    With (b, n, d) being :
    - b : the batch size
    - n : the sequence length
    - d : the embedding dimension
    """
    return torch.rand((1, 5, 3)) * 2 - 1


def test_positional_embedding_shape(random_tensor):
    positional_embedding_layer = PositionalEmbedding(GPT2_EMBEDDING_DIMENSION)
    embedded = positional_embedding_layer(random_tensor)

    assert embedded.shape == random_tensor.shape


def test_positional_embedding_output(random_tensor):
    positional_embedding_layer = PositionalEmbedding(GPT2_EMBEDDING_DIMENSION)
    embedded = positional_embedding_layer(random_tensor)
    embedded = torch.round(embedded, decimals=3)

    expected_tensor = torch.tensor([[
        [0.000, 1.000, 0.000],
        [0.841, 0.560, 0.815],
        [0.909, -0.373, 0.944],
        [0.141, -0.977, 0.278],
        [-0.757, -0.722, -0.622],
    ]], dtype=embedded.dtype)

    assert torch.equal(embedded, expected_tensor)


def test_layer_normalization_shape(random_tensor: torch.Tensor):
    layer_norm = LayerNorm(random_tensor.shape[-1])
    layer_norm_output = layer_norm(random_tensor)

    assert layer_norm_output.shape == random_tensor.shape


def test_layer_normalization_output(random_tensor: torch.Tensor):
    layer_norm = LayerNorm(random_tensor.shape[-1])
    normalized = layer_norm(random_tensor)
    normalized = torch.round(normalized, decimals=3)

    expected_tensor = torch.tensor([[
        [0.522, 0.631, -1.153],
        [1.075, -0.903, -0.171],
        [-1.130, 0.361, 0.769],
        [-1.046, 0.947, 0.099],
        [0.947, -1.046, 0.099],
    ]], dtype=normalized.dtype)

    assert torch.equal(normalized, expected_tensor)


def test_multihead_self_attention_shape(random_tensor: torch.Tensor):
    mha = MultiHeadAttention(random_tensor.shape[-1], random_tensor.shape[-1])
    mha_output = mha(random_tensor)

    assert mha_output.shape == random_tensor.shape


def test_parameters_on_gpu():
    model_config = GPTConfig(
        block_size=10,
        n_layer=2,
        n_head=10,
        n_embd=10,
        dropout=0.2,
    )
    gpt_model = GPT(model_config)
    gpt_model.to('mps')
