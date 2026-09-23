"""Data pipeline: loading the raw text, tokenizing it, and batching (input, target) pairs."""

import os
import urllib.request

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from gpt_from_scratch.tokenizer import load_tokenizer

# Edith Wharton's short story "The Verdict" — the same sample text used in
# Raschka's book. Small enough to train on a CPU in minutes.
DATA_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
    "ch02/01_main-chapter-code/the-verdict.txt"
)
DEFAULT_DATA_PATH = "data/the-verdict.txt"


def load_text(path: str = DEFAULT_DATA_PATH, url: str = DATA_URL) -> str:
    """Read the training text, downloading it first if the file is missing.

    Args:
        path: local path of the text file.
        url: download URL used only when `path` does not exist yet.

    Returns:
        The full raw text as a string.
    """
    if not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def train_val_split(text: str, train_ratio: float = 0.9) -> tuple[str, str]:
    """Split raw text into a training part and a validation part.

    Args:
        text: the full raw text.
        train_ratio: fraction of characters used for training.

    Returns:
        (train_text, val_text) tuple.
    """
    split_idx = int(train_ratio * len(text))
    return text[:split_idx], text[split_idx:]


class GPTDatasetV1(Dataset):
    """
    Slices tokenized text into fixed-length (input, target) chunks using
    a sliding window. Each target is the input shifted right by one
    token, i.e. next-token prediction.
    """

    def __init__(
        self,
        text: str,
        tokenizer: tiktoken.Encoding,
        max_length: int,
        stride: int,
    ):
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    text: str,
    batch_size: int = 2,
    max_length: int = 256,
    stride: int = 256,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Tokenize text and wrap the sliding-window dataset in a DataLoader.

    Args:
        text: raw training text.
        batch_size: number of (input, target) pairs per batch.
        max_length: sequence length of each chunk.
        stride: sliding-window step between consecutive chunks.
        shuffle: shuffle chunk order each epoch.
        drop_last: drop the final partial batch.
        num_workers: DataLoader worker processes.

    Returns:
        A DataLoader yielding (input_ids, target_ids) batches.
    """
    tokenizer = load_tokenizer()
    dataset = GPTDatasetV1(text, tokenizer, max_length=max_length, stride=stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
