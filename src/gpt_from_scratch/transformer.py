import torch
import torch.nn as nn
from gpt_from_scratch.multi_head_attention import MultiHeadAttention

class LayerNorm(nn.Module):
    """
    Custom layer normalization, matching GPT-2's exact implementation
    (uses biased variance, i.e. divides by n, not n-1, for compatibility
    with OpenAI's pretrained weights).

    Normalizes each token's vector independently to mean 0, variance 1,
    then applies learnable scale and shift parameters.
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """GELU activation function (smooth alternative to ReLU, used throughout GPT-2)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (
            1.0 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi))
                * (x + 0.044715 * torch.pow(x, 3))
            )
        )


class FeedForward(nn.Module):
    """
    Position-wise feedforward network. Expands embedding dim by 4x,
    applies GELU, then contracts back down, per token independently.
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    One GPT transformer block: pre-layernorm multi-head attention,
    followed by pre-layernorm feedforward, each wrapped in a shortcut
    (residual) connection.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        n_heads: int,
        drop_rate: float,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.norm1 = LayerNorm(d_in)
        self.attn =MultiHeadAttention(
            d_in=d_in,
            d_out=d_out,
            context_length=context_length,
            dropout=drop_rate,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
        )
        self.drop_shortcut = nn.Dropout(drop_rate)

        self.norm2 = LayerNorm(d_out)
        self.ff = FeedForward(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Attention sub-block ---
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # residual connection

        # --- Feedforward sub-block ---
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # residual connection

        return x