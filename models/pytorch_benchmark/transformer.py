from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import inspect
import os
from pathlib import Path
from collections.abc import Iterator
from typing import Tuple
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

vocab_size = int(os.getenv("PYTORCH_VOCAB_SIZE", "65"))
dim = int(os.getenv("PYTORCH_DIM", "32"))
lr = float(os.getenv("PYTORCH_LR", "0.0003"))
sigma = float(os.getenv("PYTORCH_SIGMA", "0.02"))

seq_len = int(os.getenv("PYTORCH_SEQ_LEN", "32"))
num_heads = int(os.getenv("PYTORCH_NUM_HEADS", "8"))
num_epochs = int(os.getenv("PYTORCH_EPOCHS", "3"))
batch_size = int(os.getenv("PYTORCH_BATCH_SIZE", "512"))
max_val_sequences = int(os.getenv("PYTORCH_MAX_VAL_SEQUENCES", "1024"))
compile_mode = os.getenv("PYTORCH_COMPILE_MODE", "reduce-overhead")
materialize_data_on_device = os.getenv("PYTORCH_DATA_ON_DEVICE", "1") == "1"

torch.set_float32_matmul_precision("highest")
if hasattr(torch.backends, "cuda"):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@dataclass
class TrainProfile:
    num_epochs: int
    train_losses: list[float] = field(default_factory=list)
    warmup_wall_seconds: float = 0.0
    total_wall_seconds: float = 0.0
    gpu_seconds: float | None = None


def download_tiny_shakespeare(data_root: Path = Path("~/data")) -> Path:
    data_dir = data_root / "tinyshakespeare"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "input.txt"
    if out_path.exists():
        return out_path
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, out_path)
    return out_path


def build_windows(
    tokens: np.ndarray, seq_len: int, max_sequences: int
) -> Tuple[np.ndarray, np.ndarray]:
    total = len(tokens) - seq_len
    if total <= 0:
        raise ValueError(f"Need more than {seq_len} tokens, got {len(tokens)}")
    count = min(total, max_sequences)
    xs = np.empty((count, seq_len), dtype=np.int64)
    ys = np.empty((count, seq_len), dtype=np.int64)
    for i in range(count):
        xs[i] = tokens[i : i + seq_len]
        ys[i] = tokens[i + 1 : i + seq_len + 1]
    return xs, ys


def sample_index(
    logits: np.ndarray,
    rng: np.random.Generator,
    temperature: float,
    top_k: int | None,
) -> int:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    values = np.asarray(logits, dtype=np.float32) / temperature
    if top_k is not None and 0 < top_k < values.shape[0]:
        top_idx = np.argpartition(values, -top_k)[-top_k:]
        top_vals = values[top_idx]
        top_vals = top_vals - top_vals.max()
        probs = np.exp(top_vals)
        probs = probs / probs.sum()
        return int(rng.choice(top_idx, p=probs))
    values = values - values.max()
    probs = np.exp(values)
    probs = probs / probs.sum()
    return int(rng.choice(values.shape[0], p=probs))


class TransformerBlock(nn.Module):
    def __init__(self, width: int, n_heads: int):
        super().__init__()
        if width % n_heads != 0:
            raise ValueError(f"width {width} must be divisible by n_heads {n_heads}")
        self.width = width
        self.n_heads = n_heads
        self.head_dim = width // n_heads
        self.layer_norm1 = nn.LayerNorm(width)
        self.layer_norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.ff = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.ReLU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        attn_in = self.layer_norm1(x)
        q, k, v = self.qkv(attn_in).chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.width)
        x = x + self.proj(attn_out)
        x = x + self.ff(self.layer_norm2(x))
        return x


class BigramTransformer(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, width: int, n_heads: int):
        super().__init__()
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, width)
        self.position_emb = nn.Embedding(seq_len, width)
        self.block1 = TransformerBlock(width, n_heads)
        self.block2 = TransformerBlock(width, n_heads)
        self.layer_norm = nn.LayerNorm(width)
        self.linear = nn.Linear(width, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=sigma)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.shape
        if t > self.seq_len:
            raise ValueError(f"sequence length {t} exceeds model limit {self.seq_len}")
        positions = torch.arange(t, device=idx.device)
        x = self.token_emb(idx) + self.position_emb(positions).unsqueeze(0)
        x = self.block1(x)
        x = self.block2(x)
        x = self.layer_norm(x)
        return self.linear(x)


def make_batches(
    xs: torch.Tensor,
    ys: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    num_rows = xs.size(0)
    limit = num_rows if not drop_last else num_rows - (num_rows % batch_size)
    if limit <= 0:
        raise ValueError(
            f"Need at least one full batch, got {num_rows} rows and batch_size {batch_size}"
        )
    indices = torch.randperm(num_rows, device=xs.device) if shuffle else None

    def iterator():
        for start in range(0, limit, batch_size):
            end = min(start + batch_size, limit)
            if indices is not None:
                batch_idx = indices[start:end]
                yield xs.index_select(0, batch_idx), ys.index_select(0, batch_idx)
            else:
                yield xs[start:end], ys[start:end]

    return iterator()


def train_epoch(
    model: nn.Module,
    xs: torch.Tensor,
    ys: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    model.train()
    total_loss = torch.zeros((), device=device)
    total = 0
    batches = make_batches(xs, ys, batch_size, shuffle=True, drop_last=True)
    while True:
        try:
            x, y = next(batches)
        except StopIteration:
            break
        optimizer.zero_grad(set_to_none=True)
        if x.device != device:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.detach() * x.size(0)
        total += x.size(0)
    return total_loss / total


def synchronize_device(device: torch.device) -> None:
    if device.type == "cpu":
        return
    if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "synchronize"):
        torch.accelerator.synchronize()
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def warmup_train_step(
    model: nn.Module,
    xs: torch.Tensor,
    ys: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> None:
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    model_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    optimizer_state = deepcopy(optimizer.state_dict())
    x = xs[:batch_size]
    y = ys[:batch_size]
    if x.device != device:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    synchronize_device(device)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    torch.set_rng_state(cpu_rng_state)
    if cuda_rng_states is not None:
        torch.cuda.set_rng_state_all(cuda_rng_states)


def benchmark_train(
    model: nn.Module,
    xs: torch.Tensor,
    ys: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    num_epochs: int,
) -> TrainProfile:
    profile = TrainProfile(num_epochs=num_epochs)
    warmup_start = time.perf_counter()
    warmup_train_step(model, xs, ys, optimizer, loss_fn, device)
    profile.warmup_wall_seconds = time.perf_counter() - warmup_start
    synchronize_device(device)
    start_event = end_event = None
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    start = time.perf_counter()
    if start_event is not None:
        start_event.record()
    epoch_losses = []
    for _ in range(num_epochs):
        train_loss = train_epoch(
            model,
            xs,
            ys,
            optimizer,
            loss_fn,
            device,
        )
        epoch_losses.append(train_loss.detach())
    if end_event is not None:
        end_event.record()
    synchronize_device(device)
    profile.total_wall_seconds = time.perf_counter() - start
    profile.train_losses = [float(loss.item()) for loss in epoch_losses]
    if start_event is not None and end_event is not None:
        profile.gpu_seconds = start_event.elapsed_time(end_event) / 1000.0
    return profile


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    xs: torch.Tensor,
    ys: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = torch.zeros((), device=device)
    total = 0
    batches = make_batches(xs, ys, batch_size, shuffle=False, drop_last=False)
    while True:
        try:
            x, y = next(batches)
        except StopIteration:
            break
        if x.device != device:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.detach() * x.size(0)
        total += x.size(0)
    synchronize_device(device)
    return float((total_loss / total).item())


def maybe_compile_model(model: nn.Module, device: torch.device) -> nn.Module:
    if device.type != "cuda" or not hasattr(torch, "compile"):
        return model
    return torch.compile(model, mode=compile_mode)


def build_optimizer(model: nn.Module, device: torch.device) -> torch.optim.Optimizer:
    kwargs = {"lr": lr}
    adamw_params = inspect.signature(torch.optim.AdamW).parameters
    if device.type == "cuda" and "fused" in adamw_params:
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


@torch.inference_mode()
def generate_text(
    prefix: str,
    num_new_tokens: int,
    model: BigramTransformer,
    stoi: dict[str, int],
    itos: list[str],
    seq_len: int,
    device: torch.device,
    temperature: float = 0.9,
    top_k: int | None = 16,
    seed: int = 0,
) -> str:
    rng = np.random.default_rng(seed)
    if not prefix:
        prefix = " "
    fallback_token = stoi.get(" ", 0)
    token_ids = [stoi.get(ch, fallback_token) for ch in prefix]

    model.eval()
    for _ in range(num_new_tokens):
        ctx = token_ids[-seq_len:]
        if len(ctx) < seq_len:
            ctx = [fallback_token] * (seq_len - len(ctx)) + ctx
        x = torch.tensor([ctx], device=device, dtype=torch.long)
        logits = model(x)[0, seq_len - 1].float().cpu().numpy()
        next_id = sample_index(logits, rng, temperature, top_k)
        token_ids.append(next_id)
    return "".join(itos[i] for i in token_ids)


def print_timing_report(
    stage_timings: list[tuple[str, float]],
    train_profile: TrainProfile,
    generation_timings: list[tuple[str, float]],
) -> None:
    e2e_total = sum(seconds for _, seconds in stage_timings)
    print("\n[e2e profiler]")
    for label, seconds in stage_timings:
        share = 100.0 * seconds / e2e_total if e2e_total else 0.0
        print(f"{label:>20}: {seconds:.6f} s ({share:5.1f}%)")

    print("\n[train warmup]")
    print(f"{'warmup wall':>20}: {train_profile.warmup_wall_seconds:.6f} s")

    print("\n[train benchmark]")
    print(f"{'train wall total':>20}: {train_profile.total_wall_seconds:.6f} s")
    if train_profile.gpu_seconds is not None:
        share = (
            100.0 * train_profile.gpu_seconds / train_profile.total_wall_seconds
            if train_profile.total_wall_seconds
            else 0.0
        )
        print(f"{'train gpu region':>20}: {train_profile.gpu_seconds:.6f} s ({share:5.1f}%)")
    avg_epoch_wall = (
        train_profile.total_wall_seconds / train_profile.num_epochs
        if train_profile.num_epochs
        else 0.0
    )
    print(f"{'avg epoch wall':>20}: {avg_epoch_wall:.6f} s")

    if generation_timings:
        print("\n[generation profiler]")
        total_generation = sum(seconds for _, seconds in generation_timings)
        for label, seconds in generation_timings:
            share = 100.0 * seconds / total_generation if total_generation else 0.0
            print(f"{label:>20}: {seconds:.6f} s ({share:5.1f}%)")


def main() -> None:
    if dim % num_heads != 0:
        raise ValueError(
            f"dim must be divisible by num_heads, got dim={dim}, num_heads={num_heads}"
        )
    head_size = dim // num_heads

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device.type}")
    print(
        "config: "
        f"dim={dim}, seq_len={seq_len}, heads={num_heads}, head_size={head_size}, "
        f"batch_size={batch_size}, epochs={num_epochs}, "
        f"compile={device.type == 'cuda' and hasattr(torch, 'compile')}, "
        f"compile_mode={compile_mode}, precision=float32, "
        f"data_on_device={device.type == 'cuda' and materialize_data_on_device}"
    )

    stage_timings = []

    data_start = time.perf_counter()
    path = download_tiny_shakespeare()
    text = path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    if len(chars) != vocab_size:
        raise ValueError(
            f"TinyShakespeare vocab is {len(chars)}, but model expects {vocab_size}. "
            "Adjust `vocab_size` and related parameter shapes."
        )

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = chars
    encoded = np.array([stoi[ch] for ch in text], dtype=np.int64)
    split = int(len(encoded) * 0.9)
    train_tokens = encoded[:split]
    val_tokens = encoded[split:]
    max_train_sequences = len(train_tokens) - seq_len

    train_xs, train_ys = build_windows(train_tokens, seq_len, max_train_sequences)
    val_xs, val_ys = build_windows(val_tokens, seq_len, max_val_sequences)
    stage_timings.append(("data prep", time.perf_counter() - data_start))

    print(f"dataset path: {path}")
    print(
        f"train windows: {train_xs.shape[0]}, val windows: {val_xs.shape[0]}, "
        f"seq_len: {seq_len}, vocab: {vocab_size}"
    )

    loader_start = time.perf_counter()
    data_device = device if device.type == "cuda" and materialize_data_on_device else torch.device("cpu")
    train_xs_tensor = torch.from_numpy(train_xs).to(device=data_device)
    train_ys_tensor = torch.from_numpy(train_ys).to(device=data_device)
    val_xs_tensor = torch.from_numpy(val_xs).to(device=data_device)
    val_ys_tensor = torch.from_numpy(val_ys).to(device=data_device)
    stage_timings.append(("tensor setup", time.perf_counter() - loader_start))

    synchronize_device(device)
    model_start = time.perf_counter()
    model = BigramTransformer(vocab_size, seq_len, dim, num_heads).to(device)
    model = maybe_compile_model(model, device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, device)
    synchronize_device(device)
    stage_timings.append(("model setup", time.perf_counter() - model_start))

    train_profile = benchmark_train(
        model,
        train_xs_tensor,
        train_ys_tensor,
        optimizer,
        loss_fn,
        device,
        num_epochs,
    )
    stage_timings.append(("train warmup", train_profile.warmup_wall_seconds))
    stage_timings.append(("training", train_profile.total_wall_seconds))
    for epoch, train_loss in enumerate(train_profile.train_losses, start=1):
        print(f"epoch {epoch}/{num_epochs} train loss: {train_loss:.6f}")
    print(f"training wall {train_profile.total_wall_seconds:.6f} seconds")
    if train_profile.gpu_seconds is not None:
        print(f"training gpu {train_profile.gpu_seconds:.6f} seconds")

    synchronize_device(device)
    validation_start = time.perf_counter()
    val_loss = evaluate(model, val_xs_tensor, val_ys_tensor, loss_fn, device)
    synchronize_device(device)
    validation_seconds = time.perf_counter() - validation_start
    stage_timings.append(("validation", validation_seconds))
    print(f"validation loss: {val_loss:.6f}")

    generation_timings = []
    synchronize_device(device)
    generation_start = time.perf_counter()
    for idx, prefix in enumerate(["To be, or not to be", "ROMEO:", "JULIET:"]):
        synchronize_device(device)
        sample_start = time.perf_counter()
        sample = generate_text(
            prefix,
            120,
            model,
            stoi,
            itos,
            seq_len,
            device,
            temperature=0.9,
            top_k=16,
            seed=idx + 1,
        )
        synchronize_device(device)
        generation_timings.append((f"sample {idx + 1}", time.perf_counter() - sample_start))
        print(f"\n[prefix] {prefix}\n{sample}\n")
    synchronize_device(device)
    stage_timings.append(("generation", time.perf_counter() - generation_start))
    print_timing_report(stage_timings, train_profile, generation_timings)


if __name__ == "__main__":
    main()
