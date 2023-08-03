from __future__ import annotations

import concurrent.futures
import functools
import itertools
import os
import uuid
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Generator, Iterator, Optional, Type, Union

from ..base import (
    CheckpointableDataset,
    CheckpointableIterator,
    FilterMapFn,
    ParallelExecutorType,
    Sample,
    StateDict,
)


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


def _get_executor_class(
    executor_type: ParallelExecutorType,
) -> Union[Type[ProcessPoolExecutor], Type[ThreadPoolExecutor]]:
    if executor_type == "process":
        return ProcessPoolExecutor
    elif executor_type == "thread":
        return ThreadPoolExecutor
    else:
        raise ValueError('Invalid executor_type. Choose either "process" or "thread".')


def _imap(
    fn: FilterMapFn,
    source: Iterator[Sample],
    max_workers: Optional[int],
    queue_len: int,
    executor_type: ParallelExecutorType,
) -> Generator[Optional[Sample], None, None]:
    executor_class = _get_executor_class(executor_type)

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


def _imap_unordered(
    fn: FilterMapFn,
    source: Iterator[Sample],
    max_workers: Optional[int],
    queue_len: int,
    executor_type: ParallelExecutorType,
) -> Generator[Optional[Sample], None, None]:
    executor_class = _get_executor_class(executor_type)

    with executor_class(max_workers=max_workers) as executor:
        # Submit initial tasks
        futures = {executor.submit(fn, x) for x in itertools.islice(source, queue_len)}

        while futures:
            # Wait for the first future to complete
            done, futures = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                yield future.result()

                # Try to add new tasks
                try:
                    futures.add(executor.submit(fn, next(source)))
                except StopIteration:
                    # All tasks submitted, just wait for them to complete
                    pass


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

        with _register_fn(self.dataset.fn) as fn:
            if self.dataset.ordered:
                it = _imap(
                    fn,
                    self._source_iter(),
                    self.dataset.max_workers,
                    self.dataset.queue_len,
                    self.dataset.executor_type,
                )
            else:
                it = _imap_unordered(
                    fn,
                    self._source_iter(),
                    self.dataset.max_workers,
                    self.dataset.queue_len,
                    self.dataset.executor_type,
                )
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
