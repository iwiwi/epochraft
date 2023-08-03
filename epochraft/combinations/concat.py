from __future__ import annotations

from typing import Optional, Sequence

from ..base import CheckpointableDataset, CheckpointableIterator, Sample, StateDict


class ConcatIterator(CheckpointableIterator):
    def __init__(
        self, dataset: ConcatDataset, source_iter: CheckpointableIterator, source_index: int
    ) -> None:
        self.dataset = dataset
        self.source_iter = source_iter
        self.source_index = source_index

    def __next__(self) -> Sample:
        while self.source_index < len(self.dataset.sources):
            try:
                return next(self.source_iter)
            except StopIteration:
                self.source_index += 1
                if self.source_index >= len(self.dataset.sources):
                    break
                self.source_iter = self.dataset.sources[self.source_index].iter()
        raise StopIteration()

    def state_dict(self) -> StateDict:
        return {
            "source_iter": self.source_iter.state_dict(),
            "source_index": self.source_index,
        }


class ConcatDataset(CheckpointableDataset):
    def __init__(self, sources: Sequence[CheckpointableDataset]) -> None:
        if len(sources) == 0:
            raise ValueError("ConcatDataset must have at least one source")
        self.sources = sources

    def iter(self, state_dict: Optional[StateDict] = None) -> CheckpointableIterator:
        if state_dict:
            source_index = state_dict.pop("source_index")
            if source_index >= len(self.sources):
                raise ValueError(
                    f"source_index out of range: {source_index} >= {len(self.sources)}"
                )
            source_iter_state = state_dict.pop("source_iter")
            if state_dict:
                raise ValueError(f"Unexpected keys in state_dict: {state_dict.keys()}")
        else:
            source_index = 0
            source_iter_state = None

        source = self.sources[source_index]
        source_iter = source.iter(source_iter_state)
        return ConcatIterator(self, source_iter, source_index)


def concat_datasets(sources: Sequence[CheckpointableDataset]) -> ConcatDataset:
    return ConcatDataset(sources)
