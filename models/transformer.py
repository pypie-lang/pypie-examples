from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple
import time
import urllib.request

import numpy as np

from pypie import (
    Tensor,
    Model,
    op,
    iota,
    larger,
    smaller,
    randn,
    rep,
    sqrt,
    f16,
    cast,
)

from libs.cross_entropy import cross_entropy
from libs.emb import emb
from libs.corr2d import dot
from libs.linear import linear, LinearParams
from libs.softmax import softmax

lr = float(os.getenv("PYPIE_LR", "0.003"))
sigma = float(os.getenv("PYPIE_SIGMA", "0.02"))


class LayerNormParams[T, d]:
    weight: Tensor[T][[d]]
    bias: Tensor[T][[d]]


@op
def layer_norm[T, l: int, d: int](
    x: Tensor[T][[l, d]],
    p: LayerNormParams[T, d],
) -> Tensor[T][[l, d]]:
    eps = 1e-5
    means = x.avg(1).reshape([l, 1])
    variances = ((x - means) ** 2).avg(1).reshape([l, 1])
    return ((x - means) / (variances + eps).sqrt()) * p.weight + p.bias


class SelfAttentionParams[T, n, d]:
    q: LinearParams[T, n, d]
    k: LinearParams[T, n, d]
    v: LinearParams[T, n, d]


@op
def causal_mask[T, l: int](scores: Tensor[T][[l, l]]) -> Tensor[T][[l, l]]:
    blocked = -1e9
    return [
        [scores[row][col] if col <= row else blocked for col in iota(l)]
        for row in iota(l)
    ]


@op
def self_attention[T, l: int, n: int, d: int](
    x: Tensor[T][[l, d]],
    p: SelfAttentionParams[T, n, d],
) -> Tensor[T][[l, n]]:
    q = linear(x, p.q)
    k = linear(x, p.k)
    v = linear(x, p.v)
    weights = (q @ k.permute(1, 0)) / (cast(n, T).sqrt())
    weights = causal_mask(weights)
    weights = softmax(weights)
    return weights @ v


class MultiHeadsAttentionParams[T, h, d, n]:
    qw: Tensor[T][[h, d, n]]
    kw: Tensor[T][[h, d, n]]
    vw: Tensor[T][[h, d, n]]
    proj: Tensor[T][[h * n, d]]


@op
def multi_heads_attention[T, l: int, h: int, d: int, n: int](
    x: Tensor[T][[l, d]],
    p: MultiHeadsAttentionParams[T, h, d, n],
) -> Tensor[T][[l, d]]:
    q = x @ p.qw
    k = x @ p.kw
    v = x @ p.vw
    weights = (q @ k.permute(0, 2, 1)) / (cast(n, T).sqrt()) 
    weights = softmax(causal_mask(weights))
    weights = weights @ v
    merged = weights.permute(1, 0, 2).reshape([l, h * n])
    return merged @ p.proj


class FeedForwardParams[T, d]:
    linear1: LinearParams[T, 4 * d, d]
    linear2: LinearParams[T, d, 4 * d]


@op
def feed_forward[T, l: int, d: int](
    x: Tensor[T][[l, d]],
    p: FeedForwardParams[T, d],
) -> Tensor[T][[l, d]]:
    x = larger(linear(x, p.linear1), 0.0)
    return linear(x, p.linear2)


class BlockParams[T, n, d, h]:
    multi_heads: MultiHeadsAttentionParams[T, n, d, h]
    feed_forward: FeedForwardParams[T, d]
    layer_norm1: LayerNormParams[T, d]
    layer_norm2: LayerNormParams[T, d]


@op
def block[T, l: int, d: int, h: int, n: int](
    x: Tensor[T][[l, d]],
    p: BlockParams[T, n, d, h],
) -> Tensor[T][[l, d]]:
    x = layer_norm(x, p.layer_norm1)
    x = x + multi_heads_attention(
        x,
        p.multi_heads,
    )
    x = layer_norm(x, p.layer_norm2)
    x = x + feed_forward(
        x,
        p.feed_forward,
    )
    return x


beta = 0.9
epsilon = 1e-08
mu = 0.85


@op
def smooth[T](decay: T, avg: T, g: T) -> T:
    return (decay * avg) + ((1 - decay) * g)


class ModelParams[l, n, h, v, d]:
    token_emb: Tensor[float][[v, d]]
    position_emb: Tensor[float][[l, d]]
    block1: BlockParams[float, n, d, h]
    block2: BlockParams[float, n, d, h]
    layer_norm: LayerNormParams[float, d]
    linear: LinearParams[float, v, d]


class LittleTransformer(Model):
    def predict[l: int, n: int, h: int, v: int, d: int](
        indices: Tensor[int][[l]],
        p: ModelParams[l, n, h, v, d],
    ) -> Tensor[float][[l, v]]:
        token_emb = emb(indices, p.token_emb)
        position_emb = emb(iota(l), p.position_emb)
        x = token_emb + position_emb
        x = block(x, p.block1)
        x = block(x, p.block2)
        x = layer_norm(x, p.layer_norm)
        logits = linear(x, p.linear)
        return logits

    def loss[b: int, l: int, v: int](
        ys_pred: Tensor[float][[b, l, v]],
        ys: Tensor[int][[b, l]],
    ) -> float:
        ys_pred = ys_pred.reshape([b * l, v])
        ys = ys.reshape([b * l])
        return cross_entropy(ys_pred, ys)

    def update(
        self, P: Tuple[float, float, float], g: float
    ) -> Tuple[float, float, float]:
        r = smooth(beta, P[2], g**2.0)
        alpha1 = 0.01 / (sqrt(r) + epsilon)
        v = smooth(mu, P[1], g)
        return (P[0] - (alpha1 * v), v, r)

    def inflate(self, p: float) -> Tuple[float, float, float]:
        return (p, 0.0, 0.0)

    def deflate(self, P: Tuple[float, float, float]) -> float:
        return P[0]


@op
def gen_params(l: int, n: int, h: int, v: int, d: int) -> ModelParams[l, n, h, v, d]:
    zero = 0.0
    one = 1.0
    return ModelParams(
        randn([v, d], zero, sigma),
        randn([l, d], zero, sigma),
        BlockParams(
            MultiHeadsAttentionParams(
                randn([n, d, h], zero, sigma),
                randn([n, d, h], zero, sigma),
                randn([n, d, h], zero, sigma),
                randn([n * h, d], zero, sigma),
            ),
            FeedForwardParams(
                LinearParams(
                    randn([4 * d, d], zero, sigma),
                    rep([4 * d], zero),
                ),
                LinearParams(
                    randn([d, 4 * d], zero, sigma),
                    rep([d], zero),
                ),
            ),
            LayerNormParams(
                rep([d], one),
                rep([d], zero),
            ),
            LayerNormParams(
                rep([d], one),
                rep([d], zero),
            ),
        ),
        BlockParams(
            MultiHeadsAttentionParams(
                randn([n, d, h], zero, sigma),
                randn([n, d, h], zero, sigma),
                randn([n, d, h], zero, sigma),
                randn([n * h, d], zero, sigma),
            ),
            FeedForwardParams(
                LinearParams(
                    randn([4 * d, d], zero, sigma),
                    rep([4 * d], zero),
                ),
                LinearParams(
                    randn([d, 4 * d], zero, sigma),
                    rep([d], zero),
                ),
            ),
            LayerNormParams(
                rep([d], one),
                rep([d], zero),
            ),
            LayerNormParams(
                rep([d], one),
                rep([d], zero),
            ),
        ),
        LayerNormParams(
            rep([d], one),
            rep([d], zero),
        ),
        LinearParams(
            randn([v, d], zero, sigma),
            rep([v], zero),
        ),
    )


def download_tiny_shakespeare(data_root: Path = Path("~/data")) -> Path:
    data_dir = data_root / "tinyshakespeare"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "input.txt"
    if out_path.exists():
        return out_path
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, out_path)
    return out_path


def sample_index(
    logits,
    rng: np.random.Generator,
    temperature: float,
    top_k: int | None,
) -> int:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    values = np.asarray(logits, dtype=np.float64) / temperature
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


def generate_text(
    prefix: str,
    num_new_tokens: int,
    params,
    stoi: dict[str, int],
    itos: list[str],
    seq_len: int,
    temperature: float = 0.9,
    top_k: int | None = 16,
    seed: int = 0,
) -> str:
    rng = np.random.default_rng(seed)
    if not prefix:
        prefix = " "
    fallback_token = stoi.get(" ", 0)
    token_ids = [stoi.get(ch, fallback_token) for ch in prefix]
    for _ in range(num_new_tokens):
        ctx = token_ids[-seq_len:]
        if len(ctx) < seq_len:
            ctx = [fallback_token] * (seq_len - len(ctx)) + ctx
        logits = LittleTransformer.predict(Tensor(ctx), params)
        next_id = sample_index(logits[seq_len - 1], rng, temperature, top_k)
        token_ids.append(next_id)
    return "".join(itos[i] for i in token_ids)


def main() -> None:
    path = download_tiny_shakespeare()
    text = path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    vocab_size = int(os.getenv("PYPIE_VOCAB_SIZE", "65"))
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

    seq_len = int(os.getenv("PYPIE_SEQ_LEN", "32"))
    dim = int(os.getenv("PYPIE_DIM", "32"))
    num_heads = int(os.getenv("PYPIE_NUM_HEADS", "8"))
    default_head_size = dim // num_heads if dim % num_heads == 0 else 8
    head_size = int(os.getenv("PYPIE_HEAD_SIZE", str(default_head_size)))
    if num_heads * head_size != dim:
        raise ValueError(
            f"num_heads * head_size must equal dim ({dim}), got {num_heads * head_size}"
        )
    num_epochs = int(os.getenv("PYPIE_EPOCHS", "3"))
    batch_size = int(os.getenv("PYPIE_BATCH_SIZE", "512"))
    max_train_sequences = len(train_tokens) - seq_len
    max_val_sequences = int(os.getenv("PYPIE_MAX_VAL_SEQUENCES", "1024"))

    train_xs, train_ys = build_windows(train_tokens, seq_len, max_train_sequences)
    val_xs, val_ys = build_windows(val_tokens, seq_len, max_val_sequences)

    print(f"dataset path: {path}")
    print(
        f"train windows: {train_xs.shape[0]}, val windows: {val_xs.shape[0]}, "
        f"seq_len: {seq_len}, dim: {dim}, vocab: {vocab_size}, "
        f"num_heads: {num_heads}, head_size: {head_size}"
    )

    params = gen_params(seq_len, num_heads, head_size, vocab_size, dim)
    start = time.perf_counter()
    params = LittleTransformer.learn(
        Tensor(train_xs),
        Tensor(train_ys),
        params,
        num_epochs,
        batch_size,
        True,
    )
    print(f"training {time.perf_counter() - start:.6f} seconds")

    val_logits = LittleTransformer.predict(Tensor(val_xs), params)
    val_loss = LittleTransformer.loss(val_logits, Tensor(val_ys))
    print(f"validation loss: {val_loss}")

    for idx, prefix in enumerate(["To be, or not to be", "ROMEO:", "JULIET:"]):
        sample = generate_text(
            prefix,
            120,
            params,
            stoi,
            itos,
            seq_len,
            temperature=0.9,
            top_k=16,
            seed=idx + 1,
        )
        print(f"\n[prefix] {prefix}\n{sample}\n")


if __name__ == "__main__":
    main()
