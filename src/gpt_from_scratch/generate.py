"""Autoregressive text generation: greedy decoding and temperature/top-k sampling."""

import torch
from torch import nn


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


def generate_text(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """
    Autoregressive generation with temperature scaling and top-k sampling.

    Lower temperature makes the output more focused and deterministic
    (temperature=0 is greedy); higher temperature makes it more diverse.
    top_k restricts sampling to the k most likely tokens at each step,
    cutting off the long tail of improbable tokens.

    Args:
        model: the GPT model
        idx: starting token IDs, shape (batch_size, seq_len)
        max_new_tokens: how many new tokens to generate
        context_size: the model's max context length, used to truncate
                      idx if it grows longer than the model can handle
        temperature: >0 samples from scaled logits; 0 picks the argmax
        top_k: if set, only the k highest-scoring tokens are candidates
        eos_id: if set, stop generation early once every sequence in the
                batch has produced this token ID

    Returns:
        Token IDs of shape (batch_size, seq_len + <=max_new_tokens)
    """
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]  # (batch_size, vocab_size)

        if top_k is not None:
            # Zero out everything outside the top-k candidates
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf"), device=logits.device),
                logits,
            )

        if temperature > 0.0:
            logits = logits / temperature
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probas, num_samples=1)  # (batch_size, 1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)

        if eos_id is not None and bool((idx_next == eos_id).all()):
            break

    return idx
