from __future__ import annotations

from typing import Optional

from ..base import CheckpointableDataset, CheckpointableIterator, Sample, StateDict


class CountIterator(CheckpointableIterator):
    def __init__(self, dataset: Count, source: CheckpointableIterator, start_count: int) -> None:
        self.count = start_count
        self.dataset = dataset
        self.source = source

    def __next__(self) -> Sample:
        count = self.count
        self.count += 1
        if self.dataset.max_count is not None and self.count > self.dataset.max_count:
            raise StopIteration

        sample = next(self.source)

        if self.dataset.count_column:
            sample[self.dataset.count_column] = count

        return sample

    def state_dict(self) -> StateDict:
        return {
            "source": self.source.state_dict(),
            "count": self.count,
        }


class Count(CheckpointableDataset):
    def __init__(
        self,
        source: CheckpointableDataset,
        count_column: Optional[str] = None,
        max_count: Optional[int] = None,
    ) -> None:
        self.source = source
        self.count_column = count_column
        self.max_count = max_count

    def iter(self, state_dict: Optional[StateDict] = None) -> CheckpointableIterator:
        if state_dict:
            source_state_dict = state_dict.pop("source")
            count = state_dict.pop("count")
            if state_dict:
                raise ValueError(f"Unexpected keys in state_dict: {state_dict.keys()}")
        else:
            source_state_dict = None
            count = 0

        return CountIterator(self, self.source.iter(state_dict=source_state_dict), count)
