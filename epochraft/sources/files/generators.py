from __future__ import annotations

import json
from logging import getLogger
from typing import Generator

import webdataset

from ...base import FileFormat, Sample


logger = getLogger()

ENCODING = "utf-8"


def _yield_samples_cbor(
    url: str,
    n_samples_to_skip: int = 0,
) -> Generator[Sample, None, None]:
    import cbor2

    stream = webdataset.gopen(url)

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
    url: str, n_samlpes_to_skip: int = 0, buffer_size: int = 1048576
) -> Generator[Sample, None, None]:
    pipe = webdataset.gopen(url)
    buffer = b""
    eol = b"\n"

    while True:
        while eol not in buffer:
            chunk = pipe.read(buffer_size)

            # EOF
            if not chunk:
                if buffer and n_samlpes_to_skip == 0:
                    yield json.loads(buffer.decode(ENCODING))
                return

            buffer += chunk

        line, buffer = buffer.split(eol, 1)

        if n_samlpes_to_skip > 0:
            n_samlpes_to_skip -= 1
        else:
            yield json.loads(line.decode(ENCODING))


def _deduce_format(url: str) -> FileFormat:
    url = url.lower()
    if url.endswith(".cbor"):
        return "cbor"
    elif url.endswith(".jsonl"):
        return "jsonl"
    else:
        raise ValueError(f"Unknown format for {url}")


def yield_samples(
    url: str,
    format: FileFormat = "auto",
    n_samples_to_skip: int = 0,
) -> Generator[Sample, None, None]:
    if format == "auto":
        format = _deduce_format(url)
    format = format.lower()  # type: ignore

    if format == "jsonl":
        return _yield_samples_jsonl(url, n_samples_to_skip)
    elif format == "cbor":
        return _yield_samples_cbor(url, n_samples_to_skip)
    else:
        raise ValueError(f"Unknown format: {format}")
