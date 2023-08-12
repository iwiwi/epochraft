from __future__ import annotations

import io
import json
from typing import IO, BinaryIO, Generator, Union

import webdataset
from webdataset.gopen import Pipe

from ...base import FileFormat, Sample


# Return type of `webdataset.gopen`
GopenStream = Union[Pipe, IO[bytes], BinaryIO]


def _yield_samples_cbor(
    stream: GopenStream,
    n_samples_to_skip: int = 0,
) -> Generator[Sample, None, None]:
    import cbor2

    try:
        while True:
            sample = cbor2.load(stream)
            if n_samples_to_skip > 0:
                n_samples_to_skip -= 1
            else:
                yield sample
    except EOFError:
        return


def _yield_samples_jsonl(
    stream: GopenStream, n_samples_to_skip: int = 0
) -> Generator[Sample, None, None]:
    # `webdataset.gopen` opens anything in the binary mode.
    text_stream = io.TextIOWrapper(stream)

    for line in text_stream:
        line = line.strip()
        if line:
            if n_samples_to_skip > 0:
                n_samples_to_skip -= 1
            else:
                yield json.loads(line)


def _deduce_format(url: str) -> FileFormat:
    url = url.lower()
    if url.endswith(".cbor"):
        return "cbor"
    elif url.endswith(".jsonl"):
        return "jsonl"
    else:
        raise ValueError(f"Could not deduce format from url: {url}")


def yield_samples(
    url: str,
    format: FileFormat = "auto",
    n_samples_to_skip: int = 0,
) -> Generator[Sample, None, None]:
    if format == "auto":
        format = _deduce_format(url)
    format = format.lower()  # type: ignore

    # We need to call `gopen` here, not in the actual generators. This is because we want to
    # immediately start the read process for prefetching.
    stream: GopenStream = webdataset.gopen(url)

    if format == "jsonl":
        return _yield_samples_jsonl(stream, n_samples_to_skip)
    elif format == "cbor":
        return _yield_samples_cbor(stream, n_samples_to_skip)
    else:
        raise ValueError(f"Unknown format: {format}")
