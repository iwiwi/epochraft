from __future__ import annotations

from typing import Generator
from unittest.mock import Mock, call, patch

import pytest

from epochraft import FileFormat, Sample
from epochraft.sources.files.shard_reader import ShardReader

from .conftest import LOREM, texts_to_samples


@pytest.fixture
def mock_yield_samples() -> Generator[Mock, None, None]:
    """Mock yield_samples to raise exceptions"""

    samples = texts_to_samples(LOREM, 0)
    error_counts = {2: 2, 3: 1}  # Number of errors for the 2nd and 3rd iterations

    def _yield_samples(
        url: str, format: FileFormat, n_samples_to_skip: int
    ) -> Generator[Sample, None, None]:
        it = iter(list(enumerate(samples))[n_samples_to_skip:])

        for index, sample in it:
            if index in error_counts and error_counts[index] > 0:
                error_counts[index] -= 1
                raise Exception(
                    f"Mocked exception from yield_samples during the {index}-th iteration"
                )
            else:
                yield sample

    with patch(
        "epochraft.sources.files.shard_reader.yield_samples",
        side_effect=_yield_samples,
    ) as mock:
        yield mock


@pytest.fixture
def mock_sleep() -> Generator[Mock, None, None]:
    """Mock time.sleep to avoid waiting during tests"""

    with patch("time.sleep", return_value=None) as mock:
        yield mock


def test_retry_logic_during_iteration(
    jsonl_paths: list[str], mock_yield_samples: Mock, mock_sleep: Mock
) -> None:
    url = jsonl_paths[0]
    reader = ShardReader(url, format="auto", n_samples_yielded=0, epoch=0, index_in_epoch=0)

    texts = [sample["text"] for sample in list(reader)]

    assert texts == LOREM

    # Check that the retry logic was triggered correctly
    assert mock_sleep.call_count == 3
    assert mock_yield_samples.call_count == 4
    assert mock_sleep.call_args_list == [
        call(1.0),
        call(2.0),
        call(1.0),
    ]
