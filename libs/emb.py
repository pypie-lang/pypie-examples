from pypie import Tensor, op

@op
def emb[T, n: int, v: int, d: int](
    indices: Tensor[int][[n]], embedding: Tensor[T][[v, d]]
) -> Tensor[T][[n, d]]:
    return [embedding[idx] for idx in indices]
