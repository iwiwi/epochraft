from __future__ import annotations

import functools
import io
import json
import locale
import subprocess
import urllib.parse
from logging import getLogger
from typing import IO, Any, BinaryIO, Callable, Generator, Union

from ...base import FileFormat, Sample


logger = getLogger(__name__)


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


def generate_from_jsonl_stream(
    url: str,
    stream: IO[str],
    stream_check_fn: Callable[[], None],
    n_samples_to_skip: int,
    max_consecutive_errors: int = 10,
) -> Generator[Sample, None, None]:
    line_count = 0
    n_consecutive_errors = 0
    while True:
        stream_check_fn()
        line = stream.readline()
        stream_check_fn()

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
                f"If {max_consecutive_errors} consecutive errors occur, this generator will be aborted."
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
                    f"Encountered {max_consecutive_errors} consecutive decoding errors for URL {url}. "
                    f"Terminating the generator."
                )

    logger.debug(f"JSONL EOF: line={line_count}, url={url}")


def generate_from_stream(
    url: str,
    format: FileFormat,
    stream: IO[Any],
    stream_check_fn: Callable[[], None],
    n_samples_to_skip: int,
) -> Generator[Sample, None, None]:
    if format == "jsonl":
        yield from generate_from_jsonl_stream(url, stream, stream_check_fn, n_samples_to_skip)
    elif format == "cbor":
        # TODO!!
        raise NotImplemented()
    else:
        raise ValueError(f"Unknwon format: {format}")


def generate_by_open(
    path: str,
    format: FileFormat,
    n_samples_to_skip: int,
) -> Generator[Sample, None, None]:
    mode = "r" if _is_text(format) else "rb"

    with open(path, mode) as fp:
        yield from generate_from_stream(
            url=path,
            format=format,
            stream=fp,
            stream_check_fn=lambda: None,
            n_samples_to_skip=n_samples_to_skip,
        )


def generate_by_popen(
    url: str,
    command: list[str],
    format: FileFormat,
    n_samples_to_skip: int,
) -> Generator[Sample, None, None]:
    # We would like to eargerly evaluate Popen
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, text=_is_text(format))

    def _check_status() -> None:
        if proc.poll() is not None:
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, proc.args)

    # TODO: potential leak when `_generator()` is called but no element is consumed?
    def _generator() -> Generator[Sample, None, None]:
        with proc:
            assert proc.stdout is not None
            check_fn = _check_status

            # yield from generate_from_stream(proc.stdout, format, n_samples_to_skip)
            yield from generate_from_stream(
                url=url,
                format=format,
                stream=proc.stdout,
                stream_check_fn=check_fn,
                n_samples_to_skip=n_samples_to_skip,
            )

            retcode = proc.wait()
            if retcode != 0:
                raise subprocess.CalledProcessError(retcode, command)

    return _generator()


def yield_samples(
    url: str,
    format: FileFormat = "auto",
    n_samples_to_skip: int = 0,
) -> Generator[Sample, None, None]:
    if format == "auto":
        format = _deduce_format(url)
    format = format.lower()  # type: ignore

    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme == "s3":
        # https://github.com/webdataset/webdataset/issues/21
        cmd = ["aws", "s3", "cp", url, "-"]
        yield from generate_by_popen(url, cmd, format, n_samples_to_skip)
    elif parsed_url.scheme == "":
        yield from generate_by_open(url, format, n_samples_to_skip)
    else:
        raise ValueError(f"Unknown scheme: {parsed_url.scheme} (url: {url})")
