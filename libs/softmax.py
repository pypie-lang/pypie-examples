from pypie import Tensor, op, Var

@op
def softmax[T, n: int](x: Tensor[T][[n]]) -> Tensor[T][[n]]:
    x_exp = (x - x.max(0)).exp()
    return x_exp / x_exp.sum(0)
