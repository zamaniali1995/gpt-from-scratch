"""Tokenization helpers built on the GPT-2 BPE tokenizer (tiktoken).

The tokenizer is the bridge between raw text and the model: it converts
strings into integer token IDs before training/generation, and converts
token IDs back into readable text afterwards.
"""

import tiktoken
import torch

# Cache encoding objects so repeated calls don't re-load the BPE tables.
_tokenizer_cache: dict[str, tiktoken.Encoding] = {}


def load_tokenizer(encoding: str = "gpt2") -> tiktoken.Encoding:
    """Return the cached tiktoken encoding, loading it on first use.

    Args:
        encoding: the tiktoken encoding name (default "gpt2").

    Returns:
        The tiktoken Encoding object with a 50257-token vocabulary.
    """
    if encoding not in _tokenizer_cache:
        _tokenizer_cache[encoding] = tiktoken.get_encoding(encoding)
    return _tokenizer_cache[encoding]


def text_to_token_ids(text: str, encoding: str = "gpt2") -> torch.Tensor:
    """Encode a string into token IDs with a batch dimension.

    Args:
        text: the raw text to encode.
        encoding: the tiktoken encoding name.

    Returns:
        Integer tensor of shape (1, num_tokens).
    """
    tokenizer = load_tokenizer(encoding)
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded).unsqueeze(0)  # add batch dimension


def token_ids_to_text(token_ids: torch.Tensor, encoding: str = "gpt2") -> str:
    """Decode token IDs (with batch dimension) back into a string.

    Args:
        token_ids: integer tensor of shape (1, num_tokens).
        encoding: the tiktoken encoding name.

    Returns:
        The decoded text.
    """
    tokenizer = load_tokenizer(encoding)
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())
