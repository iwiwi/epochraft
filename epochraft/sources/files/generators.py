from __future__ import annotations

import json
import threading
from logging import getLogger
from queue import Empty, Queue
from typing import IO, Any, Generator, Optional

import smart_open

from ...base import FileFormat, Sample
from .delay_handler import ProtocolDelayHandler


logger = getLogger(__name__)
delay_handler = ProtocolDelayHandler()


class ReaderThreadResult:
    error: Optional[Exception]
    result: Optional[Sample]
    end: bool

    def __init__(
        self, error: Optional[Exception] = None, result: Optional[Sample] = None, end: bool = False
    ) -> None:
        self.error = error
        self.result = result
        self.end = end


def _deduce_format(url: str) -> FileFormat:
    url = url.lower()
    if url.endswith(".cbor"):
        return "cbor"
    elif url.endswith(".jsonl"):
        return "jsonl"
    else:
        raise ValueError(f"Could not deduce format from url: {url}")


def _is_text(format: FileFormat) -> bool:
    return format in ["jsonl"]


def _generate_from_stream_jsonl(
    url: str,
    stream: IO[str],
    n_samples_to_skip: int,
    max_consecutive_errors: int = 10,
) -> Generator[Sample, None, None]:
    line_count = 0
    n_consecutive_errors = 0
    while True:
        line = stream.readline()

        # EOF
        if not line:
            break
        line_count += 1

        # Empty line
        line = line.strip()
        if not line:
            continue

        # Decode and yield
        try:
            sample = json.loads(line)
            n_consecutive_errors = 0

            if n_samples_to_skip > 0:
                n_samples_to_skip -= 1
            else:
                yield sample
        except json.decoder.JSONDecodeError:
            n_consecutive_errors += 1
            logger.exception(
                f"Decoding error encountered on line {line_count} of URL {url}. "
                f"Skipping this line and moving on to the next. "
                f"This is error {n_consecutive_errors}/{max_consecutive_errors} in a row. "
                f"If {max_consecutive_errors} consecutive errors occur, it will be aborted."
            )
            logger.debug(
                f"Content of the problematic line (line={line_count}, url={url}):\n"
                f"{'-' * 80}\n"
                f"{line}\n"
                f"{'-' * 80}\n"
            )
            if n_consecutive_errors >= max_consecutive_errors:
                # TODO: we probably don't need to retry for this error
                raise ValueError(
                    f"Encountered {max_consecutive_errors} consecutive decoding errors "
                    f"for URL {url}. Terminating the generator."
                )

    logger.debug(f"JSONL EOF: line={line_count}, url={url}")


def _generate_form_stream_cbor(
    url: str,
    stream: IO[bytes],
    n_samples_to_skip: int,
) -> Generator[Sample, None, None]:
    import cbor2

    try:
        while True:
            sample = cbor2.load(stream)  # type: ignore

            if n_samples_to_skip > 0:
                n_samples_to_skip -= 1
            else:
                yield sample  # type: ignore
    except EOFError:
        pass


def _generate_from_stream(
    url: str,
    format: FileFormat,
    stream: IO[Any],
    n_samples_to_skip: int,
) -> Generator[Sample, None, None]:
    if format == "jsonl":
        yield from _generate_from_stream_jsonl(url, stream, n_samples_to_skip)
    elif format == "cbor":
        yield from _generate_form_stream_cbor(url, stream, n_samples_to_skip)
    else:
        raise ValueError(f"Unknwon format: {format}")


def _reader_thread(
    url: str,
    format: FileFormat,
    n_samples_to_skip: int,
    queue: Queue[ReaderThreadResult],
    finish_event: threading.Event,
) -> None:
    mode = "r" if _is_text(format) else "rb"

    try:
        logger.debug(f"Read thread starting: url={url}")
        delay_handler(url)
        with smart_open.open(url, mode) as f:
            gen = _generate_from_stream(
                url=url,
                format=format,
                stream=f,
                n_samples_to_skip=n_samples_to_skip,
            )
            for sample in gen:
                if finish_event.is_set():
                    logger.debug(f"Read thread interrupted: url={url}")
                    return
                queue.put(ReaderThreadResult(result=sample))

        logger.debug(f"Read thread completed: url={url}")
        queue.put(ReaderThreadResult(end=True))
    except Exception as e:
        logger.exception(f"Exception in read thread: url={url}")
        queue.put(ReaderThreadResult(error=e))


def _generator(
    thread: threading.Thread,
    queue: Queue[ReaderThreadResult],
    finish_event: threading.Event,
    timeout: float,
) -> Generator[Sample, None, None]:
    try:
        while True:
            result = queue.get(timeout=timeout)
            if result.end:
                break
            elif result.error:
                raise result.error
            else:
                assert result.result is not None
                yield result.result
    finally:
        finish_event.set()
        try:
            while True:
                queue.get_nowait()
        except Empty:
            pass
        thread.join()


def yield_samples(
    url: str,
    format: FileFormat,
    n_samples_to_skip: int,
    n_prefetch_samples: int,
    timeout: float,
) -> Generator[Sample, None, None]:
    if format == "auto":
        format = _deduce_format(url)
    format = format.lower()  # type: ignore

    queue: Queue[ReaderThreadResult] = Queue(n_prefetch_samples)
    finish_event = threading.Event()

    thread = threading.Thread(
        target=_reader_thread,
        daemon=True,
        kwargs={
            "url": url,
            "format": format,
            "n_samples_to_skip": n_samples_to_skip,
            "queue": queue,
            "finish_event": finish_event,
        },
    )
    thread.start()

    # `_generator` should be a different method because we want to launch the thread
    # before `next` is called on the generator for the first time.
    return _generator(thread, queue, finish_event, timeout)
