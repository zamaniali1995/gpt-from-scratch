"""Tests for the transformer building blocks: LayerNorm, GELU, FeedForward, TransformerBlock."""

import torch

from gpt_from_scratch.transformer import GELU, FeedForward, LayerNorm, TransformerBlock


def test_layernorm_normalizes_to_zero_mean_unit_variance():
    norm = LayerNorm(emb_dim=16)
    x = torch.randn(4, 8, 16) * 10 + 5  # arbitrary scale/shift
    out = norm(x)

    mean = out.mean(dim=-1)
    var = out.var(dim=-1, unbiased=False)

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)


def test_layernorm_learnable_parameters():
    norm = LayerNorm(emb_dim=16)
    assert norm.scale.shape == (16,)
    assert norm.shift.shape == (16,)
    assert torch.all(norm.scale == 1.0)
    assert torch.all(norm.shift == 0.0)


def test_gelu_matches_pytorch_reference():
    gelu = GELU()
    ref = torch.nn.GELU(approximate="tanh")
    x = torch.linspace(-4, 4, 50)
    assert torch.allclose(gelu(x), ref(x), atol=1e-6)


def test_feedforward_expands_by_4x_internally():
    ff = FeedForward(emb_dim=32)
    assert ff.layers[0].out_features == 128  # 4x expansion
    assert ff.layers[2].out_features == 32  # contract back to emb_dim

    x = torch.randn(2, 10, 32)
    assert ff(x).shape == (2, 10, 32)


def test_transformer_block_preserves_shape():
    block = TransformerBlock(d_in=64, d_out=64, context_length=32, n_heads=4, drop_rate=0.0)
    block.eval()
    x = torch.randn(2, 16, 64)
    out = block(x)
    assert out.shape == x.shape


def test_transformer_block_residual_keeps_gradient_path():
    """The residual connection means the output still depends directly on the input."""
    block = TransformerBlock(d_in=64, d_out=64, context_length=32, n_heads=4, drop_rate=0.0)
    block.eval()
    x = torch.randn(1, 8, 64, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
