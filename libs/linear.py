from __future__ import annotations

from pypie import Tensor, op


class LinearParams[T, o: int, i: int]:
    A: Tensor[T][[o, i]]
    b: Tensor[T][[o]]


@op
def linear[i: int, o: int, T](
    x: Tensor[T][[i]], p: LinearParams[T, o, i]
) -> Tensor[T][[o]]:
    return (x * p.A).sum(1) + p.b


@op
def linear2d[m: int, i: int, o: int, T](
    x: Tensor[T][[m, i]], p: LinearParams[T, o, i]
) -> Tensor[T][[m, o]]:
    return (x @ p.A.transpose()) + p.b


@op
def linear3d[b: int, m: int, i: int, o: int, T](
    x: Tensor[T][[b, m, i]], p: LinearParams[T, o, i]
) -> Tensor[T][[b, m, o]]:
    flat = x.reshape([b * m, i])
    return linear2d(flat, p).reshape([b, m, o])
