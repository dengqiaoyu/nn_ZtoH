from typing import Callable, Final
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# hyperparameters

# Debug parameters
# batch_size: Final[int] = 32
# block_size: Final[int] = 8
# n_embd: Final[int] = 32
# n_heads: Final[int] = 4
# n_blocks: Final[int] = 3
# max_steps: Final[int] = 5000
# eval_interval: Final[int] = 500
# eval_iters: Final[int] = 200
# dropout: Final[float] = 0.1
# learning_rate: Final[float] = 1e-3
# device: Final[str] = "cuda" if torch.cuda.is_available() else "cpu"

# The number of independent sequences to process in parallel
batch_size: Final[int] = 64
# The maximum context length for predictions
block_size: Final[int] = 256
n_embd: Final[int] = 384
n_heads: Final[int] = 6
n_blocks: Final[int] = 6
dropout: Final[float] = 0.2
max_steps: Final[int] = 5000
learning_rate: Final[float] = 3e-4

eval_interval: Final[int] = 500
eval_iters: Final[int] = 200
device: Final[str] = "cuda" if torch.cuda.is_available() else "cpu"

retrain_anyway: Final[bool] = False
checkpoint_path: Final[str] = "gpt_dev_joey/checkpoint.pt"

n_tokens_to_file: Final[int] = 5000
generation_file_path: Final[str] = "gpt_dev_joey/output.txt"

# -----------------------------------------------------------------------------
# Seed
torch.manual_seed(1337)

# -----------------------------------------------------------------------------
# Prepare the data.

with open("gpt_dev_joey/input.txt", "r", encoding="utf-8") as f:
    text: Final[str] = f.read()

chars: Final[list[str]] = sorted(list(set(text)))
vocab_size: Final[int] = len(chars)

# Create a mapping from characters to integers and vice versa
stoi: Final[dict[str, int]] = {ch: i for i, ch in enumerate(chars)}
itos: Final[dict[int, str]] = {i: ch for ch, i in stoi.items()}
encode: Final[Callable[[str], list[int]]] = lambda chars: [
    stoi[c] for c in chars
]
decode: Final[Callable[[list[int]], str]] = lambda tokens: "".join(
    [itos[token] for token in tokens]
)

# Train and test splits
data: Final[torch.Tensor] = torch.tensor(
    encode(text), dtype=torch.long, device=device
)
split_ratio: Final[float] = 0.9
assert 0 < split_ratio < 1, "Split ratio must be between 0 and 1"
data_size: Final[int] = len(data)
train_data_size: Final[int] = int(len(data) * split_ratio)
train_data: Final[torch.Tensor] = data[:train_data_size]
val_data: Final[torch.Tensor] = data[train_data_size:]


# Data load function
def get_batch(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    split: str,
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:

    data: Final[torch.Tensor] = (
        train_data if split == "train" else val_data
    ).to(device)
    data_size: Final[int] = len(data)
    ix: Final[torch.Tensor] = torch.randint(
        0, data_size - block_size, (batch_size,)
    )
    x: Final[torch.Tensor] = torch.stack(
        [data[i : i + block_size] for i in ix]
    )  # (B, T)
    y: Final[torch.Tensor] = torch.stack(
        [data[i + 1 : i + block_size + 1] for i in ix]
    )  # (B, T)

    return x, y


# -----------------------------------------------------------------------------
# Training helpers


# Calculate training loss and validation loss.
@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    eval_iters: int,
    block_size: int,
    batch_size: int,
    device: str,
) -> tuple[float, float]:
    """Estimate the loss on the training and validation data."""
    model.eval()
    out: Final[dict[str, float]] = {}
    for split in ["train", "val"]:
        losses: torch.Tensor = torch.zeros(eval_iters, device=device)
        for i in range(eval_iters):
            x, y = get_batch(
                train_data,
                val_data,
                split=split,
                batch_size=batch_size,
                block_size=block_size,
                device=device,
            )
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out["train"], out["val"]


# Define the head
class Head(nn.Module):
    """One head of self-attention"""

    tril: torch.Tensor

    def __init__(self, n_embd: int, head_size: int, dropout: float) -> None:
        super().__init__()
        self.t_to_q: nn.Linear = nn.Linear(n_embd, head_size, bias=False)
        self.t_to_k: nn.Linear = nn.Linear(n_embd, head_size, bias=False)
        self.t_to_v: nn.Linear = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        head_size: Final[int] = self.t_to_q.weight.shape[1]
        # C is the number of channels, which is the number of embeddings.
        # (B, T, C) @ (C, head_size) -> (B, T, head_size)
        k: torch.Tensor = self.t_to_k(x)  # (B, T, head_size)
        q: torch.Tensor = self.t_to_q(x)  # (B, T, head_size)
        v: torch.Tensor = self.t_to_v(x)  # (B, T, head_size)

        # Compute attention scores.
        # First we want to use q and k to compute the weights for the
        # attention scores.
        # q(B, T, head_size) @ k(B, T, head_size) is not calculable, so we
        # need to transpose k to (B, head_size, T) and then do a matrix
        # multiplication plus a scaling factor to avoid peaking.
        # Shape: (B, T, T)
        weight: torch.Tensor = q @ k.transpose(-2, -1) * (head_size**-0.5)

        # Mask out the future tokens
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # Apply softmax to get the attention weights
        weight = weight.softmax(dim=-1)  # calculate attention along the row.
        weight = self.dropout(weight)

        # Get the weighted attention score with Value matrix.
        # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        out: torch.Tensor = weight @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""

    def __init__(
        self, n_embd: int, head_size: int, n_head: int, dropout: float
    ) -> None:
        super().__init__()
        self.heads: nn.ModuleList = nn.ModuleList(
            [Head(n_embd, head_size, dropout) for _ in range(n_head)]
        )
        self.proj: nn.Linear = nn.Linear(n_embd, n_embd)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate multiple heads with their channel dimension.
        # (B, T, head_size) -> (B, T, n_head * head_size)
        sa_out: torch.Tensor = torch.cat([h(x) for h in self.heads], dim=-1)
        out: torch.Tensor = self.dropout(self.proj(sa_out))  # (B, T, n_embd)
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        # The 4 below makes the MLP 4 times wider than the input.
        # which help the model to involve more computation.
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # This is the projection layer
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, n_embd) -> (B, T, n_embd)
        return self.net(x)


class Block(nn.Module):
    """A transformer block: communication followed by computation."""

    def __init__(self, n_embd: int, n_heads: int) -> None:
        super().__init__()
        # Multiple heads of self-attention.
        # (n_embd // n_heads) is the channel size of each head.
        # n_heads is the number of heads.
        # After concatenation, the output will be
        # (B, T, n_embd // n_heads * n_heads) -> (B, T, n_embd), which is the
        # same as the input.
        self.sa: MultiHeadAttention = MultiHeadAttention(
            n_embd, n_embd // n_heads, n_heads, dropout
        )
        # Feed forward layer: MLP
        # (B, T, n_embd) -> (B, T, n_embd)
        self.ffwd: FeedForward = FeedForward(n_embd, dropout)

        # Layer normalization to normalize the input across the feature.
        self.ln1: nn.LayerNorm = nn.LayerNorm(n_embd)
        self.ln2: nn.LayerNorm = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, n_embd) -> (B, T, n_embd)
        # Plus the residual connection
        x = self.sa(self.ln1(x)) + x
        x = self.ffwd(self.ln2(x)) + x
        return x


# Define the model
class TransformerModel(nn.Module):
    """A transformer-based language model."""

    def __init__(self, vocab_size: int, n_embd: int, n_heads: int) -> None:
        super().__init__()
        # Each token directly reads off the logits for the next token from a
        # lookup table.
        self.token_embedding_table: nn.Embedding = nn.Embedding(
            vocab_size, n_embd
        )
        # Encode the position of the token in the sequence
        self.position_embedding_table: nn.Embedding = nn.Embedding(
            block_size, n_embd
        )

        self.blocks: nn.Sequential = nn.Sequential(
            *[Block(n_embd, n_heads) for _ in range(n_blocks)],
        )
        # Final layer normalization.
        self.ln_f: nn.LayerNorm = nn.LayerNorm(n_embd)

        # Convert the token embeddings to logits for the next token
        # Shape: (n_embd, vocab_size)
        self.lm_head: nn.Linear = nn.Linear(n_embd, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers.
        #: token_embd is (B, T, n_embd) tensor.
        token_embd: torch.Tensor = self.token_embedding_table(idx)

        # pos is (B, T) tensor of integers.
        # pos_embd is (B, T, n_embd) tensor.
        pos: torch.Tensor = torch.arange(T, device=token_embd.device)
        pos_embd: torch.Tensor = self.position_embedding_table(pos)

        x: torch.Tensor = token_embd + pos_embd  # (B, T, n_embd)
        # (B, T, n_embd) -> (B, T, head_size), where head_size = n_embd
        # -> (B, T, n_embd)
        # (B, T, n_embd) -> (B, T, n_embd) for multiple times.
        x = self.blocks(x)
        x = self.ln_f(x)

        # (B, T, n_embd) @ (n_embd, vocab_size) -> (B, T, vocab_size)
        logits: torch.Tensor = self.lm_head(x)  # (B, T, vocab_size)

        loss: torch.Tensor | None = None
        if targets is None:
            loss = None
        else:
            # reshape the logits to (B * T, C) tensor
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # calculate the loss
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate a sequence of tokens."""
        self.eval()
        # idx is (B, T) tensor of integers.
        for _ in range(max_new_tokens):
            # Crop idx to be within the block size: always choose the last
            # block_size tokens.
            idx_crop: torch.Tensor = idx[:, -block_size:]
            # get the logits for the next token
            logits, _ = self(idx_crop)  # (B, T, C)
            # focus on the last time step
            logits = logits[:, -1, :]  # (B, C)
            # apply softmax to get probabilities
            probs: torch.Tensor = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            # idx_next shape (B, 1)
            idx_next: torch.Tensor = torch.multinomial(probs, num_samples=1)
            # append the sampled token to the input
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        self.train()
        return idx


# -----------------------------------------------------------------------------
# Doing the training

model: TransformerModel = TransformerModel(vocab_size, n_embd, n_heads).to(
    device
)

# Check if there is any existing weights to load

last_step: int
optimizer: torch.optim.AdamW = torch.optim.AdamW(
    model.parameters(), lr=learning_rate
)

if not retrain_anyway and os.path.exists(checkpoint_path):
    checkpoint: Final[dict] = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    last_step = checkpoint["step"]
    train_loss = checkpoint["train_loss"]
    val_loss = checkpoint["val_loss"]
    print(
        f"Loaded checkpoint at step {last_step}, "
        f"train loss {train_loss:.4f}, val loss {val_loss:.4f}"
    )
else:
    last_step = 0

should_train: Final[bool] = last_step < (max_steps - 1)
if should_train:
    print(f"Start training from step {last_step} to {max_steps - 1}")

# Training loop
step: int = 0
train_loss: float = 0.0
val_loss: float = 0.0
for step in range(last_step + 1, max_steps):
    # Sample a batch of data
    x, y = get_batch(
        train_data,
        val_data,
        split="train",
        batch_size=batch_size,
        block_size=block_size,
        device=device,
    )

    # Evaluate the loss
    if step % eval_interval == 0 or step == max_steps - 1:
        train_loss, val_loss = estimate_loss(
            model,
            train_data,
            val_data,
            eval_iters,
            block_size,
            batch_size,
            device,
        )
        print(
            f"step {step}: "
            f"train loss {train_loss:.4f}, val loss {val_loss:.4f}"
        )

    # Forward pass
    _, loss = model(x, y)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the model
if should_train:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        checkpoint_path,
    )
    print(f"Model saved to {checkpoint_path}")

# -----------------------------------------------------------------------------
# Sample from the model

model.eval()


def generate_text(model: TransformerModel, max_new_tokens: int) -> str:
    context: Final[torch.Tensor] = torch.zeros(
        (1, 1), dtype=torch.long, device=device
    )
    out_idx: Final = model.generate(context, max_new_tokens=max_new_tokens)[
        0
    ].tolist()
    return decode(out_idx)


# starting from time step 0
out: Final[str] = generate_text(model, max_new_tokens=500)
print("Generated text:")
print("=" * 20)
print(out)
print("=" * 20)

if n_tokens_to_file > 0:
    long_out: Final[str] = generate_text(
        model, max_new_tokens=n_tokens_to_file
    )
    with open(generation_file_path, "w", encoding="utf-8") as f:
        f.write(long_out)
        print(f"Output saved to {generation_file_path}")
