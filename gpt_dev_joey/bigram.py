from typing import Callable, Final
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# hyperparameters

# The number of independent sequences to process in parallel
batch_size: Final[int] = 32
# The maximum context length for predictions
block_size: Final[int] = 8
max_steps: Final[int] = 3000
eval_interval: Final[int] = 100
eval_iters: Final[int] = 200
learning_rate: Final[float] = 1e-2
device: Final[str] = "cuda" if torch.cuda.is_available() else "cpu"


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


# Define the model
class BigramLanguageModel(nn.Module):
    """A bigram language model."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        # Each token directly reads off the logits for the next token from a
        # lookup table.
        self.token_embedding_table: nn.Embedding = nn.Embedding(
            vocab_size, vocab_size
        )

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        # idx and targets are both (B, T) tensor of integers.
        # logits: (B, T, C) tensor.
        logits: torch.Tensor = self.token_embedding_table(idx)
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
        # idx is (B, T) tensor of integers.
        for _ in range(max_new_tokens):
            # get the logits for the next token
            logits, _ = self(idx)  # (B, T, C)
            # focus on the last time step
            logits = logits[:, -1, :]  # (B, C)
            # apply softmax to get probabilities
            probs: torch.Tensor = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            # idx_next shape (B, 1)
            idx_next: torch.Tensor = torch.multinomial(probs, num_samples=1)
            # append the sampled token to the input
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx


# -----------------------------------------------------------------------------
# Doing the training

model: BigramLanguageModel = BigramLanguageModel(vocab_size).to(device)
optimizer: torch.optim.AdamW = torch.optim.AdamW(
    model.parameters(), lr=learning_rate
)

# Training loop
for step in range(max_steps):
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
    if step % eval_interval == 0:
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
    logits, loss = model(x, y)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# -----------------------------------------------------------------------------
# Sample from the model

context: Final[torch.Tensor] = torch.zeros(
    (1, 1), dtype=torch.long, device=device
)  # starting from time step 0
out_idx = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(out_idx))
