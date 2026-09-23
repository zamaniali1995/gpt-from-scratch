import torch
from torch import nn
from gpt_from_scratch.transformer import LayerNorm, TransformerBlock

class GPTModel(nn.Module):
    """
    Full GPT model: token + positional embeddings, a stack of
    TransformerBlocks, a final LayerNorm, and a linear output head
    projecting back to vocabulary size (logits over next-token predictions).
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
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds

        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits


def generate_text_simple(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """
    Greedy autoregressive text generation: repeatedly predicts the next
    token (always picking the single most likely one) and appends it to
    the sequence, feeding the extended sequence back in each time.

    Args:
        model: the GPT model
        idx: starting token IDs, shape (batch_size, seq_len)
        max_new_tokens: how many new tokens to generate
        context_size: the model's max context length, used to truncate
                      idx if it grows longer than the model can handle

    Returns:
        Token IDs of shape (batch_size, seq_len + max_new_tokens)
    """
    model.eval()  # disable dropout during generation

    for _ in range(max_new_tokens):
        # Crop idx to the last `context_size` tokens if it's gotten too long
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # Only care about the prediction for the very last position
        logits = logits[:, -1, :]  # shape: (batch_size, vocab_size)

        probas = torch.softmax(logits, dim=-1)

        # Greedy: always pick the single highest-probability token
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # shape: (batch_size, 1)

        idx = torch.cat((idx, idx_next), dim=1)  # append to the running sequence

    return idx


import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

def text_to_token_ids(text: str) -> torch.Tensor:
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded).unsqueeze(0)  # add batch dimension

def token_ids_to_text(token_ids: torch.Tensor) -> str:
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)["model"]

torch.manual_seed(123)
model = GPTModel(
    vocab_size=config["vocab_size"],
    emb_dim=config["emb_dim"],
    context_length=config["context_length"],
    n_heads=config["n_heads"],
    n_layers=config["n_layers"],
    drop_rate=config["drop_rate"],
    qkv_bias=config["qkv_bias"],
)

# --- Try it ---
torch.manual_seed(123)

start_context = "what is your name?"
encoded = text_to_token_ids(start_context)
print("Encoded input:", encoded)

out = generate_text_simple(
    model=model,
    idx=encoded,
    max_new_tokens=10,
    context_size=1024,
)

print("Output token IDs:", out)
print("Decoded text:", token_ids_to_text(out))