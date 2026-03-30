from collections.abc import Iterator
from copy import deepcopy
import inspect
import math
import operator
import os
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.fx import Graph, GraphModule
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


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def init_linear_params(
    in_features: int,
    out_features: int,
    bias: bool = True,
) -> tuple[nn.Parameter, nn.Parameter | None]:
    A = nn.Parameter(torch.empty(out_features, in_features))
    bound = 1 / math.sqrt(in_features)
    nn.init.uniform_(A, -bound, bound)

    b = None
    if bias:
        b = nn.Parameter(torch.empty(out_features))
        nn.init.uniform_(b, -bound, bound)
    return A, b


def init_conv2d_params(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    bias: bool = True,
) -> tuple[nn.Parameter, nn.Parameter | None]:
    kernel_size = _pair(kernel_size)
    fan_in = in_channels * kernel_size[0] * kernel_size[1]
    bound = 1 / math.sqrt(fan_in)

    w = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
    nn.init.uniform_(w, -bound, bound)

    b = None
    if bias:
        b = nn.Parameter(torch.empty(out_channels))
        nn.init.uniform_(b, -bound, bound)
    return w, b


class LeNetGraphParams:
    conv1_w: nn.Parameter
    conv1_b: nn.Parameter | None
    conv2_w: nn.Parameter
    conv2_b: nn.Parameter | None
    linear1_A: nn.Parameter
    linear1_b: nn.Parameter | None
    linear2_A: nn.Parameter
    linear2_b: nn.Parameter | None
    linear3_A: nn.Parameter
    linear3_b: nn.Parameter | None

    def __init__(
        self,
        conv1_w: nn.Parameter,
        conv1_b: nn.Parameter | None,
        conv2_w: nn.Parameter,
        conv2_b: nn.Parameter | None,
        linear1_A: nn.Parameter,
        linear1_b: nn.Parameter | None,
        linear2_A: nn.Parameter,
        linear2_b: nn.Parameter | None,
        linear3_A: nn.Parameter,
        linear3_b: nn.Parameter | None,
    ) -> None:
        self.conv1_w = conv1_w
        self.conv1_b = conv1_b
        self.conv2_w = conv2_w
        self.conv2_b = conv2_b
        self.linear1_A = linear1_A
        self.linear1_b = linear1_b
        self.linear2_A = linear2_A
        self.linear2_b = linear2_b
        self.linear3_A = linear3_A
        self.linear3_b = linear3_b


class VmapModule(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.vmap(self.module)(x)


def build_linear_graph_module(
    A: nn.Parameter,
    b: nn.Parameter | None = None,
) -> GraphModule:
    root = nn.Module()
    root.A = A
    if b is not None:
        root.b = b
    else:
        root.register_parameter("b", None)

    graph = Graph()
    x = graph.placeholder("x")
    y = graph.call_method("unsqueeze", args=(x, -2))
    A = graph.get_attr("A")
    y = graph.call_function(operator.mul, args=(y, A))
    y = graph.call_method("sum", args=(y, -1))
    if root.b is not None:
        b = graph.get_attr("b")
        y = graph.call_function(operator.add, args=(y, b))
    graph.output(y)
    graph.lint()
    return GraphModule(root, graph, "MyLinear")

def batched_view(g, x, to_shape):
    from_shape = g.call_function(getattr, args=(x, "shape"))
    to_shape = tuple(to_shape)
    batch_shape = g.call_function(
        operator.getitem,
        args=(from_shape, slice(None, None if not to_shape else -len(to_shape))),
    )
    shape = g.call_function(operator.add, args=(batch_shape, to_shape))
    return g.call_method("view", args=(x, shape))

def build_conv2d_graph_module(
    w: nn.Parameter,
    b: nn.Parameter | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> GraphModule:
    out_channels, _, kernel_h, kernel_w = w.shape
    kernel_size = (kernel_h, kernel_w)
    stride = _pair(stride)
    padding = _pair(padding)

    root = nn.Module()
    root.w = w
    if b is not None:
        root.b = b
    else:
        root.register_parameter("b", None)

    graph = Graph()
    x = graph.placeholder("x")
    x_shape = graph.call_function(getattr, args=(x, "shape"))
    height = graph.call_function(operator.getitem, args=(x_shape, -2))
    width = graph.call_function(operator.getitem, args=(x_shape, -1))
    x = graph.call_method("unsqueeze", args=(x, -4))

    patches = graph.call_function(
        F.unfold,
        args=(x, kernel_size),
        kwargs={"dilation": 1, "padding": padding, "stride": stride},
    )
    patches = graph.call_method("squeeze", args=(patches, 0))
    w = graph.get_attr("w")
    weight = graph.call_method("view", args=(w, out_channels, -1))
    y = graph.call_method("matmul", args=(weight, patches))

    pad_h, pad_w = padding
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    out_h = graph.call_function(operator.add, args=(height, 2 * pad_h))
    out_h = graph.call_function(operator.sub, args=(out_h, kernel_h))
    out_h = graph.call_function(operator.floordiv, args=(out_h, stride_h))
    out_h = graph.call_function(operator.add, args=(out_h, 1))
    out_w = graph.call_function(operator.add, args=(width, 2 * pad_w))
    out_w = graph.call_function(operator.sub, args=(out_w, kernel_w))
    out_w = graph.call_function(operator.floordiv, args=(out_w, stride_w))
    out_w = graph.call_function(operator.add, args=(out_w, 1))

    y = graph.call_method(
        "view",
        args=(y, out_channels, out_h, out_w),
    )
    if root.b is not None:
        b = graph.get_attr("b")
        bias_view = graph.call_method("view", args=(b, -1, 1, 1))
        y = graph.call_function(operator.add, args=(y, bias_view))
    graph.output(y)
    graph.lint()
    return GraphModule(root, graph, "MyConv2d")


def build_relu_graph_module() -> GraphModule:
    root = nn.Module()
    graph = Graph()
    x = graph.placeholder("x")
    graph.output(graph.call_method("clamp", args=(x,), kwargs={"min": 0.0}))
    graph.lint()
    return GraphModule(root, graph, "MyRelu")


def build_avg_pool2d_graph_module(
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
) -> GraphModule:
    kernel_size = _pair(kernel_size)
    stride = kernel_size if stride is None else _pair(stride)
    padding = _pair(padding)

    graph = Graph()
    x = graph.placeholder("x")
    pad_h, pad_w = padding
    if pad_h != 0 or pad_w != 0:
        x = graph.call_function(F.pad, args=(x, (pad_w, pad_w, pad_h, pad_h)))

    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    y = graph.call_method("unfold", args=(x, -2, kernel_h, stride_h))
    y = graph.call_method("unfold", args=(y, -2, kernel_w, stride_w))
    graph.output(graph.call_method("mean", args=(y, (-2, -1))))
    graph.lint()
    return GraphModule(nn.Module(), graph, "MyAvgPool2d")


def build_flatten_graph_module(
    start_dim: int = 0,
    end_dim: int = -1,
) -> GraphModule:
    graph = Graph()
    x = graph.placeholder("x")
    graph.output(graph.call_method("flatten", args=(x, start_dim, end_dim)))
    graph.lint()
    return GraphModule(nn.Module(), graph, "MyFlatten")


def build_lenet_graph_module(params: LeNetGraphParams) -> GraphModule:
    root = nn.Module()
    root.conv1 = VmapModule(build_conv2d_graph_module(params.conv1_w, params.conv1_b, padding=2))
    root.relu1 = build_relu_graph_module()
    root.pool1 = build_avg_pool2d_graph_module(kernel_size=2, stride=2)
    root.conv2 = VmapModule(build_conv2d_graph_module(params.conv2_w, params.conv2_b))
    root.relu2 = build_relu_graph_module()
    root.pool2 = build_avg_pool2d_graph_module(kernel_size=2, stride=2)
    root.flatten = build_flatten_graph_module(1, -1)
    root.linear1 = build_linear_graph_module(params.linear1_A, params.linear1_b)
    root.relu3 = build_relu_graph_module()
    root.linear2 = build_linear_graph_module(params.linear2_A, params.linear2_b)
    root.relu4 = build_relu_graph_module()
    root.linear3 = build_linear_graph_module(params.linear3_A, params.linear3_b)

    graph = Graph()
    x = graph.placeholder("x")
    x = graph.call_module("conv1", args=(x,))
    x = graph.call_module("relu1", args=(x,))
    x = graph.call_module("pool1", args=(x,))
    x = graph.call_module("conv2", args=(x,))
    x = graph.call_module("relu2", args=(x,))
    x = graph.call_module("pool2", args=(x,))
    x = graph.call_module("flatten", args=(x,))
    x = graph.call_module("linear1", args=(x,))
    x = graph.call_module("relu3", args=(x,))
    x = graph.call_module("linear2", args=(x,))
    x = graph.call_module("relu4", args=(x,))
    x = graph.call_module("linear3", args=(x,))
    graph.output(x)
    graph.lint()
    return GraphModule(root, graph, "LeNet")


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


conv1_w, conv1_b = init_conv2d_params(1, 6, kernel_size=5)
conv2_w, conv2_b = init_conv2d_params(6, 16, kernel_size=5)
linear1_A, linear1_b = init_linear_params(400, 120)
linear2_A, linear2_b = init_linear_params(120, 84)
linear3_A, linear3_b = init_linear_params(84, 10)
lenet_params = LeNetGraphParams(
    conv1_w,
    conv1_b,
    conv2_w,
    conv2_b,
    linear1_A,
    linear1_b,
    linear2_A,
    linear2_b,
    linear3_A,
    linear3_b,
)

model = build_lenet_graph_module(lenet_params).to(device)
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
