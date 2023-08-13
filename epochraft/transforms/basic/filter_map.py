from __future__ import annotations

import functools
from typing import Any, Optional

from ...base import (
    CheckpointableDataset,
    CheckpointableIterator,
    FilterFn,
    FilterMapFn,
    MapFn,
    Sample,
    StateDict,
)


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


def _map_fn_adapter(sample: Sample, map_fn: MapFn) -> Optional[Sample]:
    sample = map_fn(sample)
    if sample is None:
        raise ValueError("map_fn must not return None")
    return sample


def _filter_fn_adapter(sample: Sample, filter_fn: FilterFn) -> Optional[Sample]:
    return sample if filter_fn(sample) else None


def adapt_map_fn(map_fn: MapFn) -> FilterMapFn:
    return functools.partial(_map_fn_adapter, map_fn=map_fn)


def adapt_filter_fn(filter_fn: FilterFn) -> FilterMapFn:
    return functools.partial(_filter_fn_adapter, filter_fn=filter_fn)
