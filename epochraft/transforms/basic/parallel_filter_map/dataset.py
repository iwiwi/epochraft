from __future__ import annotations

import itertools
import os
from logging import getLogger
from typing import Any, Generator, Optional, Union

from ....base import (
    CheckpointableDataset,
    CheckpointableIterator,
    FilterMapFn,
    ParallelExecutorType,
    Sample,
    StateDict,
)
from .imap import IMapOrdered, IMapUnordered


logger = getLogger(__name__)


class ParallelFilterMapIterator(CheckpointableIterator):
    def __init__(
        self,
        source: CheckpointableIterator,
        dataset: ParallelFilterMapDataset,
        unconsumed_outputs: list[Sample],
    ) -> None:
        self.source = source
        self.dataset = dataset
        self._closing = False
        self._iter = self._start(unconsumed_outputs)

        self.pool: Union[IMapOrdered, IMapUnordered]
        if self.dataset.ordered:
            self.pool = IMapOrdered(
                self.dataset.fn,
                self.dataset.max_workers,
                self.dataset.queue_len,
                self.dataset.executor_type,
            )
        else:
            self.pool = IMapUnordered(
                self.dataset.fn,
                self.dataset.max_workers,
                self.dataset.queue_len,
                self.dataset.executor_type,
            )

    def _source_iter(self) -> Generator[Sample, None, None]:
        while not self._closing:
            try:
                yield next(self.source)
            except StopIteration:
                break

    def _output_iter(self, unconsumed_outputs: list[Sample]) -> Generator[Sample, None, None]:
        # `self.closing = False` should not be in this method; it should be done before calling
        # this method. This is because the execution is delayed until the first `next()` call.
        # In the edge case where `state_dict` (and thus `_close`) is called before `next`,
        # `_source_iter` cannot be closed properly.

        it = self.pool(self._source_iter())
        yield from itertools.chain(unconsumed_outputs, it)

    def __next__(self) -> Sample:
        while True:
            sample = next(self._iter)
            if sample is not None:
                return sample

    def _start(self, unconsumed_outputs: list[Sample]) -> Generator[Sample, None, None]:
        self._closing = False
        return self._output_iter(unconsumed_outputs)

    def _close(self) -> list[Sample]:
        self._closing = True
        return list(self._iter)

    def state_dict(self) -> StateDict:
        unconsumed_outputs = self._close()
        state_dict = {
            "source": self.source.state_dict(),
            "unconsumed_outputs": unconsumed_outputs,
        }
        self._closing = False
        self._iter = self._output_iter(unconsumed_outputs)
        return state_dict

    def close(self) -> None:
        self.pool.close()


def _get_default_max_workers() -> int:
    n_cpus = os.cpu_count() or 1
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    return max(n_cpus // local_world_size, 1)


class ParallelFilterMapDataset(CheckpointableDataset):
    def __init__(
        self,
        source: CheckpointableDataset,
        fn: FilterMapFn,
        max_workers: Optional[int],
        prefetch_factor: int,
        ordered: bool,
        executor_type: ParallelExecutorType,
    ):
        self.source = source
        self.fn = fn

        if max_workers is None:
            self.max_workers = _get_default_max_workers()
        else:
            self.max_workers = max_workers
        if self.max_workers < 1:
            raise ValueError("max_workers must be greater than 0: {self.max_workers}")

        if prefetch_factor < 1:
            raise ValueError("prefetch_factor must be greater than 0: {prefetch_factor}")
        self.queue_len = self.max_workers * prefetch_factor

        self.ordered = ordered
        self.executor_type = executor_type

    def iter(self, state_dict: Optional[dict[str, Any]] = None) -> CheckpointableIterator:
        if state_dict is not None:
            unconsumed_outputs = state_dict.pop("unconsumed_outputs")
            source_state_dict = state_dict.pop("source")
            if state_dict:
                raise ValueError(f"Unexpected keys in state_dict: {state_dict.keys()}")
        else:
            unconsumed_outputs = []
            source_state_dict = None
        iter = self.source.iter(state_dict=source_state_dict)
        return ParallelFilterMapIterator(iter, self, unconsumed_outputs)
