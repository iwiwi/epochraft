from __future__ import annotations

from typing import Optional

from ...base import CheckpointableDataset, CheckpointableIterator, CollateFn, Sample, StateDict


class BatchIterator(CheckpointableIterator):
    def __init__(self, dataset: BatchDataset, source: CheckpointableIterator) -> None:
        self.dataset = dataset
        self.source = source

    def __next__(self) -> Sample:
        batch = []
        try:
            for _ in range(self.dataset.batch_size):
                batch.append(next(self.source))
        except StopIteration:
            if self.dataset.drop_last or not batch:
                raise

        return self.dataset.collate_fn(batch)

    def state_dict(self) -> StateDict:
        return self.source.state_dict()


class BatchDataset(CheckpointableDataset):
    def __init__(
        self,
        source: CheckpointableDataset,
        batch_size: int,
        collate_fn: CollateFn,
        drop_last: bool,
    ) -> None:
        self.source = source
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def iter(self, state_dict: Optional[StateDict] = None) -> CheckpointableIterator:
        return BatchIterator(self, self.source.iter(state_dict=state_dict))
