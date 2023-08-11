from __future__ import annotations

import pytest

from epochraft import CheckpointableDataset, testing


SAMPLES = [{"id": i} for i in range(30)]


@pytest.mark.parametrize("buffer_size", [2, 10, 100])
def test_shuffle(buffer_size: int) -> None:
    samples = list(CheckpointableDataset.from_sequence(SAMPLES).shuffle(buffer_size))
    assert samples != SAMPLES


def test_shuffle_buffer_size_one() -> None:
    samples = list(CheckpointableDataset.from_sequence(SAMPLES).shuffle(1))
    assert samples == SAMPLES


@pytest.mark.parametrize("buffer_size", [1, 2, 10, 100])
def test_shuffle_resumption(buffer_size: int) -> None:
    dataset = CheckpointableDataset.from_sequence(SAMPLES, repeat=True).shuffle(buffer_size)

    testing.check_resumption(dataset, dataset, 0)
    testing.check_resumption(dataset, dataset, 1)
    testing.check_resumption(dataset, dataset, 2)
    testing.check_resumption(dataset, dataset, 3)
    testing.check_resumption(dataset, dataset, 10)
    testing.check_resumption(dataset, dataset, 13)
    testing.check_resumption(dataset, dataset, 100)
