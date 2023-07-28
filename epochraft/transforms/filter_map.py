from typing import Any, Callable, Optional

from ..base import (CheckpointableDataset, CheckpointableIterator, Sample,
                    StateDict)

FilterMapFn = Callable[[Sample], Optional[Sample]]


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


class FilterMap(CheckpointableDataset):
    def __init__(self, source: CheckpointableDataset, fn: FilterMapFn):
        self.source = source
        self.fn = fn

    def iter(self, state_dict: Optional[dict[str, Any]] = None) -> CheckpointableIterator:
        iter = self.source.iter(state_dict=state_dict)
        return FilterMapIterator(iter, self.fn)
