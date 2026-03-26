from collections.abc import Iterator
from copy import deepcopy
import inspect
import os
import time

import torch
from torch import nn
from torchvision import datasets

lr = 0.1
batch_size = 1024
revs = 10
compile_mode = os.getenv("PYTORCH_COMPILE_MODE", "reduce-overhead")
data_on_device = os.getenv("PYTORCH_DATA_ON_DEVICE", "1") == "1"


def resolve_device() -> torch.device:
    try:
        if torch.cuda.is_available():
            # Force a minimal CUDA allocation so broken runtimes fail here and
            # the benchmark can fall back to CPU instead of crashing later.
            torch.empty(1, device="cuda")
            return torch.device("cuda")
    except RuntimeError as exc:
        reason = str(exc).splitlines()[0]
        print(f"CUDA unavailable ({reason}); falling back to CPU")
    return torch.device("cpu")


device = resolve_device()
print(f"Using {device.type}")

training_data = datasets.FashionMNIST(
    root="~/data",
    train=True,
    download=True,
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="~/data",
    train=False,
    download=True,
)


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def make_dataset_tensors(
    dataset: datasets.FashionMNIST,
    data_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs = dataset.data.unsqueeze(1).to(dtype=torch.float32).div(255.0)
    ys = dataset.targets.to(dtype=torch.long)
    if data_device.type != "cpu":
        xs = xs.to(device=data_device, non_blocking=True)
        ys = ys.to(device=data_device, non_blocking=True)
    return xs, ys


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
            f"Need at least one batch, got {num_rows} rows and batch_size {batch_size}"
        )
    indices = torch.randperm(num_rows, device=xs.device) if shuffle else None

    def iterator() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for start in range(0, limit, batch_size):
            end = min(start + batch_size, limit)
            if indices is not None:
                batch_idx = indices[start:end]
                yield xs.index_select(0, batch_idx), ys.index_select(0, batch_idx)
            else:
                yield xs[start:end], ys[start:end]

    return iterator()


def maybe_compile_train_step(train_step, device: torch.device):
    if device.type != "cuda" or not hasattr(torch, "compile"):
        return train_step
    return torch.compile(train_step, mode=compile_mode)


def build_optimizer(model: nn.Module, device: torch.device) -> torch.optim.Optimizer:
    kwargs = {"lr": lr}
    sgd_params = inspect.signature(torch.optim.SGD).parameters
    if device.type == "cuda" and "fused" in sgd_params:
        kwargs["fused"] = True
    return torch.optim.SGD(model.parameters(), **kwargs)


@torch.inference_mode()
def evaluate(model, xs, ys, loss_fn):
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device, dtype=torch.int64)
    total = 0
    for x, y in make_batches(xs, ys, batch_size, shuffle=False, drop_last=False):
        if x.device != device:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.detach() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum()
        total += x.size(0)
    synchronize_device()
    return float((total_loss / total).item()), float((total_correct.float() / total).item())


def train(model, xs, ys, train_step, revs):
    model.train()
    for _ in range(revs):
        batches = make_batches(xs, ys, batch_size, shuffle=True, drop_last=False)
        for x, y in batches:
            if x.device != device:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
            train_step(x, y)


def synchronize_device():
    if device.type == "cpu":
        return
    if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "synchronize"):
        torch.accelerator.synchronize()
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def warmup_train_step(model, xs, ys, optimizer, train_step):
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    model_state = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    optimizer_state = deepcopy(optimizer.state_dict())
    x = xs[:batch_size]
    y = ys[:batch_size]
    if x.device != device:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
    train_step(x, y)
    optimizer.zero_grad(set_to_none=True)
    synchronize_device()
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    torch.set_rng_state(cpu_rng_state)
    if cuda_rng_states is not None:
        torch.cuda.set_rng_state_all(cuda_rng_states)


def benchmark_train(model, xs, ys, optimizer, train_step, revs):
    warmup_train_step(model, xs, ys, optimizer, train_step)
    synchronize_device()
    start_event = end_event = None
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    if start_event is not None:
        start_event.record()
    train(model, xs, ys, train_step, revs)
    if end_event is not None:
        end_event.record()
    synchronize_device()
    wall_seconds = time.perf_counter() - t0
    gpu_seconds = None
    if start_event is not None and end_event is not None:
        gpu_seconds = start_event.elapsed_time(end_event) / 1000.0
    return wall_seconds, gpu_seconds


data_device = device if device.type == "cuda" and data_on_device else torch.device("cpu")
train_xs, train_ys = make_dataset_tensors(training_data, data_device)
test_xs, test_ys = make_dataset_tensors(test_data, data_device)


model = LeNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = build_optimizer(model, device)


def train_step(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()
    return loss.detach()


compiled_train_step = maybe_compile_train_step(train_step, device)
using_compiled_train_step = compiled_train_step is not train_step
wall_seconds, cuda_seconds = benchmark_train(
    model,
    train_xs,
    train_ys,
    optimizer,
    compiled_train_step,
    revs,
)
print(
    "config: "
    f"batch_size={batch_size}, revs={revs}, lr={lr}, "
    f"compile_step={using_compiled_train_step}, "
    f"compile_mode={compile_mode}, data_on_device={device.type == 'cuda' and data_on_device}"
)
print(f"training wall {wall_seconds:.6f} seconds")
if cuda_seconds is not None:
    print(f"training gpu {cuda_seconds:.6f} seconds")
test_loss, test_acc = evaluate(model, test_xs, test_ys, loss_fn)
print(f"loss {test_loss}, correct: {test_acc}")
