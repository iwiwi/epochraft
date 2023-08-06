from __future__ import annotations

from typing import Any, Optional

from ...base import CheckpointableDataset, CheckpointableIterator, FilterMapFn, Sample, StateDict


class FilterMapIterator(CheckpointableIterator):
    def __init__(self, source: CheckpointableIterator, fn: FilterMapFn):
        self.source = source
        self.fn = fn

    def __next__(self) -> Sample:
        while True:
            sample = self.fn(next(self.source))
            if sample is not None:
                return sample

    def state_dict(self) -> StateDict:
        return self.source.state_dict()


class FilterMapDataset(CheckpointableDataset):
    def __init__(self, source: CheckpointableDataset, fn: FilterMapFn):
        self.source = source
        self.fn = fn

    def iter(self, state_dict: Optional[dict[str, Any]] = None) -> CheckpointableIterator:
        iter = self.source.iter(state_dict=state_dict)
        return FilterMapIterator(iter, self.fn)
