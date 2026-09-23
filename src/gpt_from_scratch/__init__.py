"""gpt-from-scratch: a GPT-style language model built from first principles in PyTorch.

Every component — attention, normalization, the transformer block, the
training loop — is implemented by hand (no Hugging Face wrappers) while
working through Sebastian Raschka's "Build a Large Language Model
(From Scratch)".

Importing this package has no side effects: it never touches the
filesystem or network and never instantiates a model. All runnable
training/demo code lives in :mod:`gpt_from_scratch.train` and
``scripts/train.py``.
"""

from gpt_from_scratch.attention import MultiHeadAttention
from gpt_from_scratch.data import (
    GPTDatasetV1,
    create_dataloader,
    load_text,
    train_val_split,
)
from gpt_from_scratch.generate import generate_text, generate_text_simple
from gpt_from_scratch.model import GPTModel, count_parameters
from gpt_from_scratch.tokenizer import (
    load_tokenizer,
    text_to_token_ids,
    token_ids_to_text,
)
from gpt_from_scratch.train import (
    build_model_from_config,
    calc_loss_batch,
    calc_loss_loader,
    evaluate_model,
    generate_and_print_sample,
    train_model_simple,
)
from gpt_from_scratch.transformer import (
    GELU,
    FeedForward,
    LayerNorm,
    TransformerBlock,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # Model
    "GPTModel",
    "count_parameters",
    # Attention & transformer blocks
    "MultiHeadAttention",
    "TransformerBlock",
    "LayerNorm",
    "GELU",
    "FeedForward",
    # Tokenizer
    "load_tokenizer",
    "text_to_token_ids",
    "token_ids_to_text",
    # Data
    "load_text",
    "train_val_split",
    "GPTDatasetV1",
    "create_dataloader",
    # Generation
    "generate_text_simple",
    "generate_text",
    # Training
    "build_model_from_config",
    "calc_loss_batch",
    "calc_loss_loader",
    "evaluate_model",
    "generate_and_print_sample",
    "train_model_simple",
]
