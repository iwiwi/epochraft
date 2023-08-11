from __future__ import annotations

import json
import os
import tempfile
from typing import Generator, Iterable

import cbor2
import pytest

from epochraft import Sample


LOREM = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "Vivamus lacinia odio vitae vestibulum.",
    "Donec in efficitur leo.",
    "Morbi pharetra lacus nec arcu rutrum, in imperdiet libero tempus.",
    "Sed vel ipsum est.",
]

SAMPLES_PER_FILE = [5, 3, 20, 4, 0, 5]  # Intentionally skewed


def texts_to_samples(texts: Iterable[str], file_index: int) -> list[Sample]:
    return [
        {
            "text": text,
            "file_index": file_index,
        }
        for text in texts
    ]


@pytest.fixture(scope="session")
def jsonl_paths() -> Generator[list[str], None, None]:
    paths = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for file_idx, count in enumerate(SAMPLES_PER_FILE):
            path = os.path.join(tmpdir, f"file_{file_idx}.jsonl")
            with open(path, "w") as file:
                samples = texts_to_samples((LOREM[i % len(LOREM)] for i in range(count)), file_idx)
                for sample in samples:
                    file.write(json.dumps(sample) + "\n")
            paths.append(path)

        yield paths


@pytest.fixture(scope="session")
def cbor_paths() -> Generator[list[str], None, None]:
    paths = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for file_idx, count in enumerate(SAMPLES_PER_FILE):
            path = os.path.join(tmpdir, f"file_{file_idx}.cbor")
            with open(path, "wb") as file:
                samples = texts_to_samples((LOREM[i % len(LOREM)] for i in range(count)), file_idx)
                for sample in samples:
                    file.write(cbor2.dumps(sample))
            paths.append(path)

        yield paths
