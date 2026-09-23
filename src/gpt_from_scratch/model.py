"""The GPT model: embeddings, a stack of transformer blocks, final norm, output head."""

import torch
from torch import nn

from gpt_from_scratch.transformer import LayerNorm, TransformerBlock


class GPTModel(nn.Module):
    """
    Full GPT model: token + positional embeddings, a stack of
    TransformerBlocks, a final LayerNorm, and a linear output head
    projecting back to vocabulary size (logits over next-token predictions).

    Args:
        vocab_size: number of tokens in the vocabulary
        emb_dim: embedding dimension
        context_length: max sequence length the model can handle
        n_heads: number of attention heads per block
        n_layers: number of transformer blocks
        drop_rate: dropout probability
        qkv_bias: whether the Q/K/V projections use bias terms
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_length: int,
        n_heads: int,
        n_layers: int,
        drop_rate: float,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)
        self.drop_emb = nn.Dropout(drop_rate)

        self.trf_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_in=emb_dim,
                    d_out=emb_dim,
                    context_length=context_length,
                    n_heads=n_heads,
                    drop_rate=drop_rate,
                    qkv_bias=qkv_bias,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            in_idx: token IDs of shape (batch_size, seq_len)

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)
        # Positional indices live on the same device as the input (CPU or GPU)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds

        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits


def count_parameters(model: nn.Module) -> int:
    """Return the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())
