"""Tests for gpt_from_scratch.model.GPTModel."""

import torch

from gpt_from_scratch.generate import generate_text, generate_text_simple
from gpt_from_scratch.model import GPTModel, count_parameters

GPT2_SMALL = dict(
    vocab_size=50257,
    emb_dim=768,
    context_length=1024,
    n_heads=12,
    n_layers=12,
    drop_rate=0.1,
    qkv_bias=False,
)

TINY = dict(
    vocab_size=512,
    emb_dim=64,
    context_length=32,
    n_heads=4,
    n_layers=2,
    drop_rate=0.0,
    qkv_bias=False,
)


def test_logits_shape_gpt2_small():
    """Full GPT-2 small config: (batch, seq) -> (batch, seq, vocab)."""
    torch.manual_seed(123)
    model = GPTModel(**GPT2_SMALL)
    model.eval()
    idx = torch.randint(0, 50257, (2, 16))
    with torch.no_grad():
        logits = model(idx)
    assert logits.shape == (2, 16, 50257)


def test_parameter_count_above_100m():
    """Sanity check: the GPT-2 small config must be a >100M-parameter model."""
    model = GPTModel(**GPT2_SMALL)
    n_params = count_parameters(model)
    print(f"\nGPT-2 small parameter count: {n_params:,}")
    assert n_params > 100_000_000


def test_generate_text_simple_output_length():
    torch.manual_seed(123)
    model = GPTModel(**TINY)
    idx = torch.randint(0, 512, (1, 5))
    out = generate_text_simple(model, idx=idx, max_new_tokens=7, context_size=32)
    assert out.shape == (1, 5 + 7)


def test_generate_text_respects_context_window():
    """Long inputs are cropped to context_size before the forward pass."""
    torch.manual_seed(0)
    model = GPTModel(**TINY)
    idx = torch.randint(0, 512, (1, 40))  # longer than context_length=32
    out = generate_text(model, idx=idx, max_new_tokens=3, context_size=32, temperature=0.0)
    assert out.shape == (1, 43)


def test_generate_text_sampling_stays_in_vocab():
    torch.manual_seed(0)
    model = GPTModel(**TINY)
    idx = torch.randint(0, 512, (1, 5))
    out = generate_text(
        model, idx=idx, max_new_tokens=10, context_size=32, temperature=1.0, top_k=10
    )
    assert out.shape == (1, 15)
    assert out.max() < 512 and out.min() >= 0
