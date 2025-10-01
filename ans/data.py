import math
from typing import Iterator

import torch


class BatchLoader:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int | None = None,
        shuffle: bool = False,
        device: str = 'cpu',
    ) -> None:
        """
        Args:
            dataset: indexable torch Dataset returning either a tuple (input,) or (input, target)
            batch_size: How many samples in a batch
            shuffle: If True, then the data should be randomly reordered on each __iter__ (before each epoch)
            device: torch device name, e.g. 'cpu' or 'cuda:0', 'cuda:1', etc.
        """
        self.dataset = dataset
        self.batch_size = batch_size or len(dataset)
        self.shuffle = shuffle
        self.device = device

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        """
        A single epoch over the dataset

        Returns:
            batch: If unsupervised (i.e. self.y is None), return single element tuple (x[batch_ids],). If supervised
                   (i.e. self.y is torch.Tensor), return the pair (x[batch_ids], y[batch_ids])
        """

        ########################################
        # TODO: implement

        # Recommended approach:
        # 1. Create tensor of indices into self.dataset. If self.shuffle, then the indices should be randomly reodered.
        # 2. Loop over the indices in groups (i.e. batches) of size self.batch_size
        #    2.1 pack items of self.dataset as indexed by the current batch of indices into two torch.Tensors x and y
        #    2.2 `yield` tuple (x, y)
        #    2.2 stop if there are no more batches

        raise NotImplementedError

        # ENDTODO
        ########################################

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def __repr__(self) -> str:
        return f'{self.__class__.__qualname__}:\n    num_batches: {len(self)}\n    batch_size: {self.batch_size}\n'
