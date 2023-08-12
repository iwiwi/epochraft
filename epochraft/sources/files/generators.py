from __future__ import annotations

import json
import locale
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
    stream: GopenStream, n_samlpes_to_skip: int = 0, buffer_size: int = 1048576
) -> Generator[Sample, None, None]:
    encoding = locale.getpreferredencoding(False)

    # We cannot use `io.TextIOWrapper`, because `webdataset.gopen.Pipe` does not fully implement
    # the IO protocol.
    buffer = b""
    eol = b"\n"

    while True:
        while eol not in buffer:
            chunk = stream.read(buffer_size)

            # EOF
            if not chunk:
                if buffer and n_samlpes_to_skip == 0:
                    yield json.loads(buffer.decode(encoding))
                return

            buffer += chunk

        line, buffer = buffer.split(eol, 1)

        if n_samlpes_to_skip > 0:
            n_samlpes_to_skip -= 1
        else:
            yield json.loads(line.decode(encoding))


def _deduce_format(url: str) -> FileFormat:
    url = url.lower()
    if url.endswith(".cbor"):
        return "cbor"
    elif url.endswith(".jsonl"):
        return "jsonl"
    else:
        raise ValueError(f"Could not deduce format from url: {url}")


def _convert_url(url: str) -> str:
    # https://github.com/webdataset/webdataset/issues/21
    if url.startswith("s3://"):
        return f"pipe:aws s3 cp {url} -"
    else:
        return url


def yield_samples(
    url: str,
    format: FileFormat = "auto",
    n_samples_to_skip: int = 0,
) -> Generator[Sample, None, None]:
    if format == "auto":
        format = _deduce_format(url)
    format = format.lower()  # type: ignore

    url = _convert_url(url)

    # We need to call `gopen` here, not in the actual generators. This is because we want to
    # immediately start the read process for prefetching.
    stream: GopenStream = webdataset.gopen(url)

    if format == "jsonl":
        return _yield_samples_jsonl(stream, n_samples_to_skip)
    elif format == "cbor":
        return _yield_samples_cbor(stream, n_samples_to_skip)
    else:
        raise ValueError(f"Unknown format: {format}")
