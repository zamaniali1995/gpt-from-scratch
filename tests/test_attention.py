"""Tests for gpt_from_scratch.attention.MultiHeadAttention."""

import torch

from gpt_from_scratch.attention import MultiHeadAttention


def make_attn(**kwargs):
    defaults = dict(d_in=64, d_out=64, context_length=32, dropout=0.0, n_heads=4)
    defaults.update(kwargs)
    return MultiHeadAttention(**defaults)


def test_output_shape():
    attn = make_attn()
    attn.eval()
    x = torch.randn(2, 16, 64)
    out = attn(x)
    assert out.shape == (2, 16, 64)


def test_causal_masking():
    """Position 0 can only attend to token 0, so its output must be
    unchanged when later tokens are modified."""
    torch.manual_seed(0)
    attn = make_attn()
    attn.eval()

    x = torch.randn(1, 8, 64)
    x_modified = x.clone()
    x_modified[:, 1:, :] = torch.randn(1, 7, 64)  # scramble all later tokens

    out = attn(x)
    out_modified = attn(x_modified)

    # First position sees only token 0 -> outputs must match exactly
    assert torch.allclose(out[:, 0, :], out_modified[:, 0, :])

    # Later positions DO see the change -> outputs must differ
    assert not torch.allclose(out[:, -1, :], out_modified[:, -1, :])


def test_deterministic_with_dropout_zero():
    torch.manual_seed(123)
    attn = make_attn(dropout=0.0)
    attn.eval()
    x = torch.randn(2, 10, 64)
    assert torch.equal(attn(x), attn(x))


def test_dropout_active_in_train_mode():
    """With dropout on, two train-mode forward passes should differ."""
    torch.manual_seed(0)
    attn = make_attn(dropout=0.9)  # extreme dropout to make the test robust
    attn.train()
    x = torch.randn(2, 10, 64)
    assert not torch.equal(attn(x), attn(x))


def test_rejects_incompatible_head_count():
    try:
        make_attn(d_out=64, n_heads=5)
    except AssertionError:
        return
    raise AssertionError("expected AssertionError for d_out % n_heads != 0")
