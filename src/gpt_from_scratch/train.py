"""Training loop: loss functions, evaluation, the optimizer loop, and the CLI.

Everything here is import-safe: defining functions and classes only.
Running training happens through ``main()`` (the ``train-gpt`` console
script) or by calling :func:`run_training` from another script.
"""

import argparse
import os

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from gpt_from_scratch.data import create_dataloader, load_text, train_val_split
from gpt_from_scratch.generate import generate_text_simple
from gpt_from_scratch.model import GPTModel
from gpt_from_scratch.tokenizer import text_to_token_ids, token_ids_to_text


def build_model_from_config(model_cfg: dict) -> GPTModel:
    """Instantiate a GPTModel from the ``model:`` section of a config dict.

    Args:
        model_cfg: mapping with keys vocab_size, emb_dim, context_length,
                   n_heads, n_layers, drop_rate, qkv_bias.

    Returns:
        An uninitialized GPTModel on CPU.
    """
    return GPTModel(
        vocab_size=model_cfg["vocab_size"],
        emb_dim=model_cfg["emb_dim"],
        context_length=model_cfg["context_length"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        drop_rate=model_cfg["drop_rate"],
        qkv_bias=model_cfg.get("qkv_bias", False),
    )


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Cross-entropy loss for a single batch of next-token predictions."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    # Flatten (batch, seq_len, vocab) -> (batch*seq_len, vocab) so every
    # position is treated as an independent next-token prediction.
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """Average loss over (up to) `num_batches` batches of a data loader."""
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")

    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches


def evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    eval_iter: int,
) -> tuple[float, float]:
    """Compute average train/val loss over a few batches, in eval mode."""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model: nn.Module, device: torch.device, start_context: str) -> None:
    """Generate a short sample so you can eyeball qualitative progress."""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
        )

    decoded_text = token_ids_to_text(token_ids)
    print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
) -> tuple[list[float], list[float], list[int]]:
    """
    Standard training loop: for each batch, zero gradients, forward pass,
    compute loss, backward pass, optimizer step. Periodically evaluates
    train/val loss and prints a text sample to track qualitative progress.

    Returns:
        (train_losses, val_losses, track_tokens_seen) recorded at each eval step.
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Epoch {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(model, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface shared by ``scripts/train.py`` and ``train-gpt``."""
    parser = argparse.ArgumentParser(description="Train a from-scratch GPT model on The Verdict.")
    parser.add_argument(
        "--config",
        default="configs/gpt2_small.yaml",
        help="Path to the YAML config file (default: configs/gpt2_small.yaml).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides the config value).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to save the trained weights (overrides the config value).",
    )
    return parser


def run_training(args: argparse.Namespace) -> dict:
    """Run the full training pipeline described by parsed CLI args.

    Args:
        args: namespace with config, epochs, out_dir (see build_parser).

    Returns:
        Dict with train_losses, val_losses, tokens_seen, checkpoint_path.
    """
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    tr_cfg = cfg.get("training", {})

    epochs = args.epochs if args.epochs is not None else tr_cfg.get("epochs", 1)
    out_dir = args.out_dir or tr_cfg.get("checkpoint_dir", "checkpoints")
    seed = tr_cfg.get("seed", 123)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model_from_config(model_cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    text = load_text(tr_cfg.get("data_path", "data/the-verdict.txt"))
    print(f"Total characters: {len(text)}")

    train_data, val_data = train_val_split(text, tr_cfg.get("train_ratio", 0.9))

    batch_size = tr_cfg.get("batch_size", 2)
    max_length = tr_cfg.get("max_length", 256)
    stride = tr_cfg.get("stride", 256)

    train_loader = create_dataloader(
        train_data,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader(
        val_data,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=False,
        drop_last=False,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tr_cfg.get("learning_rate", 4e-4),
        weight_decay=tr_cfg.get("weight_decay", 0.1),
    )

    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=epochs,
        eval_freq=tr_cfg.get("eval_freq", 50),
        eval_iter=tr_cfg.get("eval_iter", 1),
        start_context=tr_cfg.get("start_context", "Every effort moves you"),
    )

    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "gpt2_small.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "tokens_seen": tokens_seen,
        "checkpoint_path": checkpoint_path,
    }


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``train-gpt`` console script."""
    args = build_parser().parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
