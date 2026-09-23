# gpt-from-scratch

A from-scratch PyTorch implementation of a GPT-style language model, built line by line while working through Sebastian Raschka's *[Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)*.

This isn't a wrapper around Hugging Face — every component (attention, normalization, the transformer block, the training loop) is implemented directly to understand exactly what's happening at each step, from raw text to trained weights.

## What's implemented

**Architecture**
- `MultiHeadAttention` — causal multi-head self-attention, built from a single set of `W_query` / `W_key` / `W_value` projections split across heads, with a registered causal mask buffer and scaled dot-product attention
- `LayerNorm` — custom implementation using biased variance (`unbiased=False`) for compatibility with pretrained GPT-2 weights, with learnable `scale`/`shift` parameters
- `GELU` — the smooth activation function used in GPT-2's feed-forward layers
- `FeedForward` — the two-layer MLP expert inside each transformer block
- `TransformerBlock` — pre-LayerNorm design (Norm → sublayer → residual add), which keeps gradients flowing cleanly through a 12-layer stack, unlike the post-LN order in the original "Attention Is All You Need" encoder-decoder diagram
- `GPTModel` — token + positional embeddings, a stack of transformer blocks, final layer norm, and an untied output projection head

**Data & training**
- `GPTDatasetV1` — sliding-window tokenization over raw text, producing `(input, target)` pairs where the target is the input shifted by one token (next-token prediction, fully self-supervised — no labels needed)
- `create_dataloader` — wraps the dataset in a PyTorch `DataLoader` with configurable `batch_size`, `max_length`, and `stride`
- `calc_loss_batch` / `calc_loss_loader` — cross-entropy loss over flattened `(batch × seq_len)` predictions against flattened targets
- `train_model_simple` — the full training loop: zero grad → forward → loss → backward → optimizer step, with periodic train/val loss evaluation and qualitative text samples generated after each epoch

**Generation**
- `generate_text_simple` — greedy (argmax) autoregressive decoding, with `tiktoken`-based encode/decode helpers (`text_to_token_ids`, `token_ids_to_text`)

## Key concepts covered

Working through this repo meant actually implementing (not just reading about):

- Why attention scores are scaled by `sqrt(head_dim)` before softmax
- How multi-head attention splits a single set of learned projections across heads via `view` + `transpose`, then recombines them
- The difference between `nn.Embedding` (lookup table) and `nn.Linear` (matrix multiply) — and why they're mathematically equivalent to a one-hot vector times a weight matrix
- Why token and positional embeddings are both learned via backprop, not fixed
- Pre-LN vs. post-LN transformer design and why it matters for training stability in deep stacks
- Why `logits.flatten(0, 1)` and `target_batch.flatten()` are needed before `cross_entropy` — reshaping `(batch, seq_len, vocab_size)` into `(batch × seq_len, vocab_size)` so PyTorch sees one flat list of independent next-token predictions
- The effect of `stride` on sliding-window datasets — smaller stride means more overlap and more training examples from the same text, at the cost of redundancy
- Why validation loss rises while training loss keeps falling on a small dataset (overfitting on a single short story is expected, not a bug)

## Project structure

```
gpt-from-scratch/
├── config.yaml              # model hyperparameters
├── pyproject.toml           # dependencies, managed with uv
├── src/
│   └── gpt_from_scratch/
│       ├── gpt.py           # MultiHeadAttention, TransformerBlock, GPTModel, generation
│       └── train.py         # dataset, dataloader, loss functions, training loop
├── notebooks/                # prototyping / experiments
├── tests/
└── .gitignore
```

## Setup

Dependencies are managed with [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Usage

Train on the sample dataset (Edith Wharton's *The Verdict*, the same text used in the book):

```bash
uv run python src/gpt_from_scratch/train.py
```

This will tokenize the text, split it 90/10 into train/val sets, and train the model for the configured number of epochs, printing train/val loss and a generated sample after each epoch.

Model architecture is controlled entirely through `config.yaml`:

```yaml
model:
  vocab_size: 50257
  context_length: 1024
  emb_dim: 768
  n_heads: 12
  n_layers: 12
  drop_rate: 0.1
  qkv_bias: false
```

## Roadmap

- [ ] Load pretrained GPT-2 weights and compare against the from-scratch trained model
- [ ] Experiment with smaller `stride` values for more training examples from limited text
- [ ] Explore swapping the dense `FeedForward` layer for a Mixture-of-Experts (MoE) layer — router + multiple expert FFNs with top-k gating, similar to Mixtral

## Acknowledgments

Built while working through Sebastian Raschka's *Build a Large Language Model (From Scratch)*. Architecture and training approach closely follow the book; this repo exists as a hands-on implementation and understanding exercise, not an original contribution.

## License

MIT
