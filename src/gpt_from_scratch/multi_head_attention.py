import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Splits the embedding dimension across multiple heads, computes scaled
    dot-product attention independently per head (with causal masking so
    each token only attends to itself and earlier tokens), then
    concatenates the head outputs and projects back to the embedding dim.

    Args:
        d_in: input embedding dimension
        d_out: output embedding dimension (must be divisible by n_heads)
        context_length: max sequence length, used to size the causal mask
        dropout: dropout probability applied to attention weights
        n_heads: number of attention heads
        qkv_bias: whether the Q/K/V linear projections use a bias term
    """
    def __init__(
            self, 
            d_in: int,
            d_out: int,
            context_length: int,
            dropout: float,
            n_heads: int,
            qkv_bias: bool,
    ):
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, num_tokens, d_in)

        Returns:
            Tensor of shape (batch_size, num_tokens, d_out)
        """
        batch_size, num_tokens, _ = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Split d_out into (n_heads, head_dim), then move heads before tokens
        queries = queries.view(batch_size, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)

        return self.out_proj(context_vec)