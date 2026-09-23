"""Tests for the data pipeline: GPTDatasetV1 and create_dataloader."""

import torch

from gpt_from_scratch.data import GPTDatasetV1, create_dataloader
from gpt_from_scratch.tokenizer import load_tokenizer, text_to_token_ids, token_ids_to_text

SAMPLE_TEXT = (
    "Every effort moves you forward. Every setback teaches you something new. "
    "Keep going, keep learning, and the results will follow in time."
)


def test_tokenizer_roundtrip():
    ids = text_to_token_ids("Hello, world!")
    assert ids.shape[0] == 1  # batch dimension
    assert token_ids_to_text(ids) == "Hello, world!"


def test_tokenizer_vocab_size():
    assert load_tokenizer().n_vocab == 50257


def test_dataset_chunk_shapes():
    tokenizer = load_tokenizer()
    ds = GPTDatasetV1(SAMPLE_TEXT, tokenizer, max_length=8, stride=4)
    assert len(ds) > 0
    inp, tgt = ds[0]
    assert inp.shape == (8,)
    assert tgt.shape == (8,)
    assert inp.dtype == torch.int64


def test_target_is_input_shifted_by_one():
    """Next-token prediction: target[i] == input[i+1] for every chunk."""
    tokenizer = load_tokenizer()
    ds = GPTDatasetV1(SAMPLE_TEXT, tokenizer, max_length=8, stride=2)
    for i in range(len(ds)):
        inp, tgt = ds[i]
        assert torch.equal(tgt[:-1], inp[1:])


def test_dataloader_batch_shapes():
    loader = create_dataloader(
        SAMPLE_TEXT,
        batch_size=2,
        max_length=8,
        stride=4,
        shuffle=False,
        drop_last=True,
    )
    inp, tgt = next(iter(loader))
    assert inp.shape == (2, 8)
    assert tgt.shape == (2, 8)
