from pypie import Tensor, op
from libs.softmax import softmax


@op
def cross_entropy[T, m: int, n: int](
    ys_pred: Tensor[T][[m, n]], ys: Tensor[int][[m]]
) -> T:
    target_probs = [probs[idx] for (probs, idx) in zip(softmax(ys_pred), ys)]
    return -target_probs.log().avg(0)
