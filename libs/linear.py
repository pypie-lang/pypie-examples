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