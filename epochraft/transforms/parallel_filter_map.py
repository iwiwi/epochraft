from __future__ import annotations

import functools
import itertools
import os
import uuid
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Generator, Iterator, Literal, Optional

from ..base import CheckpointableDataset, CheckpointableIterator, Sample, StateDict


FilterMapFn = Callable[[Sample], Optional[Sample]]
ExecutorType = Literal["process", "thread"]


_registered_fns: dict[str, FilterMapFn] = {}


def _apply_registered_fn(sample: Sample, fn_name: str) -> Optional[Sample]:
    callable = _registered_fns[fn_name]
    return callable(sample)


@contextmanager
def _register_fn(fn: FilterMapFn) -> Generator[Callable[[Sample], Optional[Sample]], None, None]:
    name = str(uuid.uuid4())[:8]
    _registered_fns[name] = fn
    try:
        yield functools.partial(_apply_registered_fn, fn_name=name)
    finally:
        del _registered_fns[name]


def _imap(
    fn: FilterMapFn,
    source: Iterator[Sample],
    max_workers: Optional[int],
    queue_len: int,
    executor_type: ExecutorType,
) -> Generator[Sample, None, None]:
    executor_class: Any
    if executor_type == "process":
        executor_class = ProcessPoolExecutor
    elif executor_type == "thread":
        executor_class = ThreadPoolExecutor
    else:
        raise ValueError('Invalid executor_type. Choose either "process" or "thread".')

    with executor_class(max_workers=max_workers) as executor:
        # Submit initial tasks
        futures = deque(executor.submit(fn, x) for x in itertools.islice(source, queue_len))

        # Loop over tasks as they complete
        while futures:
            done = futures.popleft()
            yield done.result()
            try:
                futures.append(executor.submit(fn, next(source)))
            except StopIteration:
                # All tasks submitted, just wait for them to complete
                pass


class ParallelFilterMapIterator(CheckpointableIterator):
    def __init__(
        self,
        source: CheckpointableIterator,
        dataset: ParallelFilterMap,
        unconsumed_outputs: list[Sample],
    ) -> None:
        self.source = source
        self.dataset = dataset
        self.closing = False
        self.it = self._start(unconsumed_outputs)

    def _source_iter(self) -> Generator[Sample, None, None]:
        while not self.closing:
            try:
                yield next(self.source)
            except StopIteration:
                break

    def _start(self, unconsumed_outputs: list[Sample]) -> Generator[Sample, None, None]:
        self.closing = False
        with _register_fn(self.dataset.fn) as fn:
            yield from itertools.chain(
                unconsumed_outputs,
                _imap(
                    fn,
                    self._source_iter(),
                    self.dataset.max_workers,
                    self.dataset.queue_len,
                    self.dataset.executor_type,
                ),
            )

    def __next__(self) -> Sample:
        while True:
            sample = next(self.it)
            if sample is not None:
                return sample

    def _close(self) -> list[Sample]:
        self.closing = True
        return list(self.it)

    def state_dict(self) -> StateDict:
        unconsumed_outputs = self._close()
        state_dict = {
            "source": self.source.state_dict(),
            "unconsumed_outputs": unconsumed_outputs,
        }
        self.it = self._start(unconsumed_outputs)
        return state_dict


class ParallelFilterMap(CheckpointableDataset):
    def __init__(
        self,
        source: CheckpointableDataset,
        fn: FilterMapFn,
        max_workers: Optional[int],
        prefetch_factor: int,
        executor_type: ExecutorType,
    ):
        self.source = source
        self.fn = fn
        self.max_workers = max_workers or os.cpu_count() or 1
        self.queue_len = self.max_workers * prefetch_factor
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
