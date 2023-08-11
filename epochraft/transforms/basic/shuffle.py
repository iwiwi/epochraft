from __future__ import annotations

import random
from typing import Optional

from ...base import CheckpointableDataset, CheckpointableIterator, Sample, StateDict


class ShuffleIterator(CheckpointableIterator):
    def __init__(
        self,
        source: CheckpointableIterator,
        buffer_size: int,
        rng: random.Random,
        buffer: list[Sample],
    ) -> None:
        self.source = source
        self.buffer_size = buffer_size
        self.rng = rng
        self.buffer = buffer

    def _refill_buffer(self) -> None:
        while len(self.buffer) < self.buffer_size:
            try:
                self.buffer.append(next(self.source))
            except StopIteration:
                break

    def __next__(self) -> Sample:
        self._refill_buffer()
        if len(self.buffer) == 0:
            raise StopIteration()

        index = self.rng.randrange(len(self.buffer))
        self.buffer[index], self.buffer[-1] = self.buffer[-1], self.buffer[index]
        return self.buffer.pop()

    def state_dict(self) -> StateDict:
        return {
            "source": self.source.state_dict(),
            "buffer": self.buffer.copy(),
            "rng": self.rng.getstate(),
        }


class ShuffleDataset(CheckpointableDataset):
    def __init__(self, source: CheckpointableDataset, buffer_size: int, seed: int) -> None:
        if buffer_size < 1:
            raise ValueError(f"Invalid buffer_size (got {buffer_size}, expected >= 1)")

        self.source = source
        self.buffer_size = buffer_size
        self.seed = seed

    def iter(self, state_dict: Optional[StateDict] = None) -> CheckpointableIterator:
        rng = random.Random(self.seed)
        if state_dict is not None:
            source = state_dict.pop("source")
            buffer = state_dict.pop("buffer")
            rng.setstate(state_dict.pop("rng"))
            if state_dict:
                raise ValueError("Unexpected state_dict keys: {}".format(state_dict.keys()))
        else:
            source = None
            buffer = []

        return ShuffleIterator(
            source=self.source.iter(source),
            buffer_size=self.buffer_size,
            rng=rng,
            buffer=buffer,
        )
