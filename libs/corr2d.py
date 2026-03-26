from __future__ import annotations

from pypie import op, Tensor, iota
from libs.seq import seq


@op
def dot[n: int](s: Tensor[float][[n]], p: Tensor[float][[n]]) -> float:
    return (s * p).sum(0)


@op
def corr1d[w: int, n: int](
    s: Tensor[float][[w]], p: Tensor[float][[n]]
) -> Tensor[float][[w - n + 1]]:
    return [dot(p, s[idx : idx + n]) for idx in iota(w - n + 1)]


@op
def pad1d[n: int](
    xs: Tensor[float][[n]], padding: int
) -> Tensor[float][[n + 2 * padding]]:
    return [
        xs[i - padding] if (padding <= i and i < n + padding) else 0.0
        for i in iota(n + 2 * padding)
    ]


# signal = Tensor([0.0, 1.0, 0.0, 0.0, 0.0])
# pattern1 = Tensor([0.0, 1.0, 0.0])
# img = Tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
# kernel = Tensor([[1.0, 0.0], [0.0, -1.0]])

# print(pad1d(signal, 2))


@op
def corr1d_padded[w: int, n: int](
    s: Tensor[float][[w]], p: Tensor[float][[n]], padding: int
) -> Tensor[float][[w + 2 * padding - n + 1]]:
    return corr1d(pad1d(s, padding), p)


signal = Tensor([0.0, 1.0, 0.0, 0.0, 0.0])
pattern1 = Tensor([0.0, 1.0, 0.0])
# print(pad1d(signal, 2))
# print(corr1d(pattern1, signal))
# print(corr1d_padded(signal, pattern1, 1))
# print(corr1d_padded(signal, pattern1, 2))
# print(dot(pattern, Tensor([0.0, 1.0, 0.0])))


@op
def corr1d_stride[w: int, n: int](
    s: Tensor[float][[w]], p: Tensor[float][[n]], stride: int
) -> Tensor[float][[(w - n + stride) / stride]]:
    return [dot(p, s[idx : idx + n]) for idx in seq(0, w - n + stride, stride)]


@op
def corr1d_padded_stride[w: int, n: int](
    s: Tensor[float][[w]], p: Tensor[float][[n]], padding: int, stride: int
) -> Tensor[float][[(w + 2 * padding - n + stride) / stride]]:
    return corr1d_stride(pad1d(s, padding), p, stride)


# print(corr1d_padded_stride(pattern1, signal, 1, 1))
# print(corr1d_padded_stride(pattern1, signal, 1, 2))


@op
def pad2d[T, h: int, w: int](
    xs: Tensor[T][[h, w]], padding0: int, padding1: int
) -> Tensor[T][[h + 2 * padding0, w + 2 * padding1]]:
    return [
        [
            (
                xs[j - padding0][i - padding1]
                if (
                    padding1 <= i
                    and i < w + padding1
                    and padding0 <= j
                    and j < h + padding0
                )
                else 0.0
            )
            for i in iota(w + 2 * padding1)
        ]
        for j in iota(h + 2 * padding0)
    ]


@op
def dot2d[T, h: int, w: int, m: int, n: int](
    s: Tensor[T][[h, w]], p: Tensor[T][[m, n]], s_j: int, s_i: int
) -> T:
    return [[s[s_j + j][s_i + i] * p[j][i] for i in iota(n)] for j in iota(m)].sum(0, 1)


@op
def corr2d[T, h: int, w: int, m: int, n: int](
    s: Tensor[T][[h, w]],
    p: Tensor[T][[m, n]],
    bias: T,
    stride0: int,
    stride1: int,
    padding0: int,
    padding1: int,
) -> Tensor[T][
    [
        (h + 2 * padding0 - m + stride0) / stride0,
        ((w + 2 * padding1 - n + stride1) / stride1),
    ]
]:
    padded = pad2d(s, padding0, padding1)
    return [
        [
            dot2d(padded, p, j, i) + bias
            for i in seq(0, w + 2 * padding1 - n + stride1, stride1)
        ]
        for j in seq(0, h + 2 * padding0 - m + stride0, stride0)
    ]

@op
def corr2d_multi_in[T, i: int, h: int, w: int, m: int, n: int](
    s: Tensor[T][[i, h, w]],
    p: Tensor[T][[i, m, n]],
    bias: T,
    stride0: int,
    stride1: int,
    padding0: int,
    padding1: int,
) -> Tensor[T][
    [
        (h + 2 * padding0 - m + stride0) / stride0,
        ((w + 2 * padding1 - n + stride1) / stride1),
    ]
]:
    return corr2d(s, p, bias, stride0, stride1, padding0, padding1).sum(0)

class Corr2dParams[T, o, i, m, n]:
    w: Tensor[T][[o, i, m, n]]
    b: Tensor[T][[o]]

@op
def corr2d_multi_in_out[T, o: int, i: int, h: int, w: int, m: int, n: int](
    s: Tensor[T][[i, h, w]],
    p: Corr2dParams[T, o, i, m, n],
    stride0: int,
    stride1: int,
    padding0: int,
    padding1: int,
) -> Tensor[T][
    [
        o,
        (h + 2 * padding0 - m + stride0) / stride0,
        ((w + 2 * padding1 - n + stride1) / stride1),
    ]
]:
    return corr2d_multi_in(s, p.w, p.b, stride0, stride1, padding0, padding1)


@op
def avg2d[T, m: int, n: int](
    x: Tensor[T][[m, n]],
    h: int,
    w: int,
    x_j: int,
    x_i: int,
) -> T:
    return [[x[j + x_j, i + x_i] for i in iota(w)] for j in iota(h)].avg(0, 1)


@op
def pool2d[T, m: int, n: int](
    x: Tensor[T][[m, n]],
    h: int,
    w: int,
    stride0: int,
    stride1: int,
    padding0: int,
    padding1: int,
) -> Tensor[T][
    [
        (m + 2 * padding0 - h + stride0) / stride0,
        (n + 2 * padding1 - w + stride1) / stride1,
    ]
]:
    padded = pad2d(x, padding0, padding1)
    return [
        [
            avg2d(padded, h, w, j, i)
            for i in seq(0, n + 2 * padding1 - w + stride1, stride1)
        ]
        for j in seq(0, m + 2 * padding0 - h + stride0, stride0)
    ]
