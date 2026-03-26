from __future__ import annotations
from math import sqrt
from torchvision import datasets
from torchvision.transforms import ToTensor

from pypie import (
    Model,
    op,
    rand,
    Tensor,
    larger,
)
from libs.linear import linear, LinearParams
from libs.cross_entropy import cross_entropy
from libs.corr2d import corr2d_multi_in_out, pool2d, Corr2dParams

lr = 0.1

t = float


class LeNetParams:
    corr1: Corr2dParams[t, 6, 1, 5, 5]
    corr2: Corr2dParams[t, 16, 6, 5, 5]
    linear1: LinearParams[t, 120, 400]
    linear2: LinearParams[t, 84, 120]
    linear3: LinearParams[t, 10, 84]


class LeNet(Model):
    def predict(
        x: Tensor[t][[1, 28, 28]],
        p: LeNetParams,
    ) -> Tensor[t][[10]]:
        layer1 = larger(corr2d_multi_in_out(x, p.corr1, 1, 1, 2, 2), 0.0)
        layer2 = pool2d(layer1, 2, 2, 2, 2, 0, 0)
        layer3 = larger(corr2d_multi_in_out(layer2, p.corr2, 1, 1, 0, 0), 0.0)
        layer4 = pool2d(layer3, 2, 2, 2, 2, 0, 0)
        flat = layer4.reshape([16 * 5 * 5])
        layer_5 = larger(linear(flat, p.linear1), 0.0)
        layer_6 = larger(linear(layer_5, p.linear2), 0.0)
        layer_7 = linear(layer_6, p.linear3)
        return layer_7

    def loss[n: int](ys_pred: Tensor[t][[n, 10]], ys: Tensor[int][[n]]) -> t:
        return cross_entropy(ys_pred, ys)

    def update(p: t, g: t) -> t:
        return p - lr * g


range1 = sqrt(6.0 / (1 * 5 * 5 + 6 * 5 * 5))
range2 = sqrt(6.0 / (6 * 5 * 5 + 16 * 5 * 5))
range3 = sqrt(6.0 / (400 + 120))
range4 = sqrt(6.0 / (120 + 84))
range5 = sqrt(6.0 / (84 + 10))


@op
def gen_params() -> LeNetParams:
    corr1 = Corr2dParams(
        rand([6, 1, 5, 5], -range1, range1),
        rand([6], 0.0, 0.0),
    )
    corr2 = Corr2dParams(
        rand([16, 6, 5, 5], -range2, range2),
        rand([16], 0.0, 0.0),
    )
    linear1 = LinearParams(
        rand([120, 400], -range3, range3),
        rand([120], 0.0, 0.0),
    )
    linear2 = LinearParams(
        rand([84, 120], -range4, range4),
        rand([84], 0.0, 0.0),
    )
    linear3 = LinearParams(
        rand([10, 84], -range5, range5),
        rand([10], 0.0, 0.0),
    )
    return LeNetParams(
        corr1,
        corr2,
        linear1,
        linear2,
        linear3,
    )


training_data = datasets.FashionMNIST(
    root="~/data",
    train=True,
    download=True,
    transform=ToTensor(),
)
xs = Tensor([pair[0] for pair in training_data])
ys = Tensor([pair[1] for pair in training_data])

batch_size = 1024

params = gen_params()
params = LeNet.learn(xs, ys, params, 10, batch_size, True)

test_data = datasets.FashionMNIST(
    root="~/data",
    train=False,
    download=True,
    transform=ToTensor(),
)
test_xs = [pair[0] for pair in test_data]
test_ys = [pair[1] for pair in test_data]

result = LeNet.predict(Tensor(test_xs), params)
total_loss = LeNet.loss(result, Tensor(test_ys))
print(f"loss: {total_loss}")


def max_index(xs) -> int:
    index = 0
    max_value = xs[0]
    for i, x in enumerate(xs):
        if x > max_value:
            max_value = x
            index = i
    return index


ys_pred = Tensor([max_index(ls) for ls in result])

correct = 0
for y_pred, y in zip(ys_pred, test_ys):
    if y_pred == y:
        correct += 1

print(f"correctness: {correct}/{len(test_data)}")
