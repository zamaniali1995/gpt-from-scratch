import yaml
import torch
import os
import urllib.request
from gpt_from_scratch.gpt import GPTModel, generate_text_simple, text_to_token_ids, token_ids_to_text


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

file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists(file_path):
    urllib.request.urlretrieve(url, file_path)

with open(file_path, "r", encoding="utf-8") as f:
    text_data = f.read()

print("Total characters:", len(text_data))
print(text_data[:200])

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GPTDatasetV1(Dataset):
    """
    Slices tokenized text into fixed-length (input, target) chunks using
    a sliding window. Each target is the input shifted right by one
    token, i.e. next-token prediction.
    """

    def __init__(self, text: str, tokenizer, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    text: str,
    batch_size: int,
    max_length: int,
    stride: int,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


tokenizer = tiktoken.get_encoding("gpt2")
dataset = GPTDatasetV1(text_data, tokenizer, max_length=4, stride=4)

print("Number of chunks:", len(dataset))

input_chunk, target_chunk = dataset[0]
print("Input: ", input_chunk)
print("Target:", target_chunk)

print("\nDecoded input: ", tokenizer.decode(input_chunk.tolist()))
print("Decoded target:", tokenizer.decode(target_chunk.tolist()))


train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))

train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = create_dataloader(
    train_data,
    batch_size=2,
    max_length=256,
    stride=256,
    shuffle=True,
    drop_last=True,
)

val_loader = create_dataloader(
    val_data,
    batch_size=2,
    max_length=256,
    stride=256,
    shuffle=False,
    drop_last=False,
)

print("Train batches:", len(train_loader))
print("Val batches:  ", len(val_loader))

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
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


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Compute average train/val loss over a few batches, in eval mode."""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """Generate a short sample so you can eyeball qualitative progress."""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size,
        )

    decoded_text = token_ids_to_text(token_ids)
    print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    """
    Standard training loop: for each batch, zero gradients, forward pass,
    compute loss, backward pass, optimizer step. Periodically evaluates
    train/val loss and prints a text sample to track qualitative progress.
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
                    f"Epoch {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)

torch.manual_seed(123)
train_losses, val_losses, tokens_seen = train_model_simple(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=1,
    eval_freq=5,
    eval_iter=1,
    start_context="Every effort moves you",
    tokenizer=tokenizer,
)