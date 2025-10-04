import pytest

from src.utils import get_encoder_decoder


@pytest.fixture()
def prompt():
    return "KING ARTHUR:\n"


def test_encoding(prompt: str):
    encoder, _ = get_encoder_decoder()
    encoded = encoder(prompt)

    assert encoded
