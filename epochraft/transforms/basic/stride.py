from __future__ import annotations

from typing import Optional

from ...base import CheckpointableDataset, CheckpointableIterator, Sample, StateDict


class StrideIterator(CheckpointableIterator):
    def __init__(
        self, source: CheckpointableIterator, interval: int, offset: int, start_count: int
    ) -> None:
        self.source = source
        self.interval = interval
        self.offset = offset
        self.count = start_count

    def __next__(self) -> Sample:
        while self.count % self.interval != self.offset:
            self.count += 1
            next(self.source)

        self.count += 1
        return next(self.source)

    def state_dict(self) -> StateDict:
        return {
            "source": self.source.state_dict(),
            "count": self.count,
        }

    def close(self) -> None:
        self.source.close()


class StrideDataset(CheckpointableDataset):
    def __init__(self, source: CheckpointableDataset, interval: int, offset: int) -> None:
        if interval <= 0:
            raise ValueError(f"interval must be positive: {interval}")
        if offset < 0 or offset >= interval:
            raise ValueError(f"offset must be in [0, interval): {offset}")

        self.source = source
        self.interval = interval
        self.offset = offset

    def iter(self, state_dict: Optional[StateDict] = None) -> CheckpointableIterator:
        if state_dict is not None:
            source = state_dict.pop("source")
            count = state_dict.pop("count")
        else:
            source = None
            count = 0

        return StrideIterator(
            source=self.source.iter(source),
            interval=self.interval,
            offset=self.offset,
            start_count=count,
        )
