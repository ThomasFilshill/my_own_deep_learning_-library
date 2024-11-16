from typing import Iterator, NamedTuple
import numpy as np
from neuronix.tensor import Tensor

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])

class DataIterator:
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        raise NotImplementedError

class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        batch_starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(batch_starts)

        for batch_start in batch_starts:
            batch_end = batch_start + self.batch_size
            batch_inputs = inputs[batch_start:batch_end]
            batch_targets = targets[batch_start:batch_end]
            yield Batch(batch_inputs, batch_targets)
