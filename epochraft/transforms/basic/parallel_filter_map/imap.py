from __future__ import annotations

import itertools
import multiprocessing
import os
import threading
import traceback
from logging import getLogger
from multiprocessing import Process, Queue
from multiprocessing.context import ForkServerProcess, SpawnProcess
from threading import Thread
from types import TracebackType
from typing import Generator, Iterator, Optional, Type, Union

from ....base import FilterMapFn, ParallelExecutorType, Sample


logger = getLogger(__name__)
mp_ctx = multiprocessing.get_context("forkserver")  # TODO: customization?


class StopToken:
    pass


WorkerClass = Union[Type[Process], Type[Thread], Type[ForkServerProcess], Type[SpawnProcess]]
WorkerInput = Union[Sample, StopToken]


class WorkerResult:
    # (Exception, traceback string) --- The actual traceback is not picklable
    error: Optional[tuple[Exception, str]]
    result: Optional[Sample]

    def __init__(self, error: Optional[tuple[Exception, str]], result: Optional[Sample]) -> None:
        self.error = error
        self.result = result

    def log_error(self) -> None:
        assert self.error
        e, exc_traceback = self.error
        logger.error(
            f"Exception in worker: {repr(e)}. The actual traceback is as follows:\n"
            f"{'=' * 80}\n"
            f"{exc_traceback}"
            f"(The exception is going to be raised again, but its traceback can be misleading.)\n"
            f"{'=' * 80}"
        )


def _get_worker_class(executor_type: ParallelExecutorType) -> WorkerClass:
    if executor_type == "process":
        return mp_ctx.Process
    elif executor_type == "thread":
        return Thread
    else:
        raise ValueError('Invalid executor_type. Choose either "process" or "thread".')


def _worker(fn: FilterMapFn, rx: Queue[WorkerInput], tx: Queue[WorkerResult]) -> None:
    process_id = os.getpid()
    thread_id = threading.get_ident()
    logger.debug(f"Worker starting. Process ID: {process_id}, Thread ID: {thread_id}.")

    while True:
        try:
            item = rx.get()
            if isinstance(item, StopToken):
                break

            result = fn(item)
            tx.put(WorkerResult(error=None, result=result))
        except Exception as e:
            logger.exception(
                f"Exception in worker. Process ID: {process_id}, Thread ID: {thread_id}."
            )
            # Traceback objects are not picklable, so we need to convert them to strings
            exc_traceback = traceback.format_exc()
            tx.put(WorkerResult(error=(e, exc_traceback), result=None))

    logger.debug(f"Worker ending. Process ID: {process_id}, Thread ID: {thread_id}.")


class IMapOrdered:
    def __init__(
        self,
        fn: FilterMapFn,
        max_workers: int,
        queue_len: int,
        executor_type: ParallelExecutorType,
    ):
        self.executor_type = executor_type
        self.max_workers = max_workers
        self.queue_len = queue_len

        logger.debug(f"Starting {max_workers} workers. This may take some time.")
        worker_class = _get_worker_class(executor_type)
        self.workers = []
        for _ in range(max_workers):
            worker_queue_len = queue_len // max_workers + 1
            rx: Queue[WorkerInput] = mp_ctx.Queue(worker_queue_len)
            tx: Queue[WorkerResult] = mp_ctx.Queue(worker_queue_len)
            worker = worker_class(target=_worker, args=(fn, rx, tx), daemon=True)
            worker.start()
            self.workers.append((worker, rx, tx))
        logger.debug(f"Started {max_workers} workers.")

        # Sample index to read and put
        self.get_index = 0
        self.put_index = 0

    def __call__(self, source: Iterator[Sample]) -> Generator[Optional[Sample], None, None]:
        self.flush()

        # Fill the queues
        for x in itertools.islice(source, self.queue_len):
            _, rx, _ = self.workers[self.put_index % self.max_workers]
            rx.put(x)
            self.put_index += 1

        # Read from the queues
        while self.get_index < self.put_index:
            _, _, tx = self.workers[self.get_index % self.max_workers]
            self.get_index += 1
            result = tx.get()
            if result.error:
                result.log_error()
                raise result.error[0]
            yield result.result

            _, rx, _ = self.workers[self.put_index % self.max_workers]
            try:
                rx.put(next(source))
                self.put_index += 1
            except StopIteration:
                # All tasks submitted, just wait for them to complete
                continue

    def flush(self) -> list[Optional[Sample]]:
        # Read from the queues
        results = []
        while self.get_index < self.put_index:
            _, _, tx = self.workers[self.get_index % self.max_workers]
            result = tx.get()
            if result.error:
                result.log_error()
                raise result.error[0]
            results.append(result.result)
            self.get_index += 1

        self.get_index = 0
        self.put_index = 0

        return results

    def close(self) -> None:
        if not self.workers:
            return

        logger.debug(f"Terminating {len(self.workers)} workers.")
        if self.executor_type == "process":
            for worker, _, _ in self.workers:
                assert not isinstance(worker, Thread)
                worker.terminate()
        else:
            # Thread does not have terminate method
            logger.debug("Flushing the queue. This may take some time.")
            self.flush()
            logger.debug("Terminating the threads.")
            for _, rx, _ in self.workers:
                rx.put(StopToken())
            for worker, _, _ in self.workers:
                worker.join()

        logger.debug(f"Terminated {len(self.workers)} workers.")
        self.workers = []

    def __enter__(self) -> IMapOrdered:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class IMapUnordered:
    def __init__(
        self,
        fn: FilterMapFn,
        max_workers: int,
        queue_len: int,
        executor_type: ParallelExecutorType,
    ):
        self.queue_len = queue_len
        self.max_workers = max_workers
        self.executor_type = executor_type

        worker_class = _get_worker_class(executor_type)
        self.rx: Queue[WorkerInput] = mp_ctx.Queue(queue_len + 1)
        self.tx: Queue[WorkerResult] = mp_ctx.Queue(queue_len + 1)

        self.workers = []
        logger.debug(f"Starting {max_workers} workers. This may take some time.")
        for _ in range(max_workers):
            worker = worker_class(target=_worker, args=(fn, self.rx, self.tx), daemon=True)
            worker.start()
            self.workers.append((worker, self.rx, self.tx))

        self.get_index = 0
        self.put_index = 0

    def __call__(self, source: Iterator[Sample]) -> Generator[Optional[Sample], None, None]:
        # Fill the queues
        for x in itertools.islice(source, self.queue_len):
            self.rx.put(x)
            self.put_index += 1

        # Read from the queues
        while self.get_index < self.put_index:
            result = self.tx.get()
            self.get_index += 1
            if result.error:
                result.log_error()
                raise result.error[0]
            yield result.result

            try:
                self.rx.put(next(source))
                self.put_index += 1
            except StopIteration:
                # All tasks submitted, just wait for them to complete
                continue

    def flush(self) -> list[Optional[Sample]]:
        # Read from the queues
        results = []
        while self.get_index < self.put_index:
            result = self.tx.get()
            if result.error:
                result.log_error()
                raise result.error[0]
            results.append(result.result)
            self.get_index += 1

        self.get_index = 0
        self.put_index = 0

        return results

    def close(self) -> None:
        if not self.workers:
            return

        logger.debug(f"Terminating {len(self.workers)} workers.")
        if self.executor_type == "process":
            for worker, _, _ in self.workers:
                assert not isinstance(worker, Thread)
                worker.terminate()
        else:
            # Thread does not have terminate method
            logger.debug("Flushing the queue. This may take some time.")
            self.flush()
            logger.debug("Terminating the threads.")
            for _, rx, _ in self.workers:
                rx.put(StopToken())
            for worker, _, _ in self.workers:
                worker.join()

        logger.debug(f"Terminated {len(self.workers)} workers.")
        self.workers = []

    def __enter__(self) -> IMapUnordered:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
