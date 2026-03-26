from __future__ import annotations

from pypie import Tensor, op, iota

@op
def seq(start: int, end: int, stride: int) -> Tensor[int][[(end - start) / stride]]:
    size = (end - start) / stride
    return iota(size) * stride + start