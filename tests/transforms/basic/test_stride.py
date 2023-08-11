from __future__ import annotations

import pytest

from epochraft import CheckpointableDataset


SAMPLES = [{"id": i} for i in range(30)]


@pytest.mark.parametrize(
    ("interval", "offset"),
    [
        (1, 0),
        (2, 0),
        (2, 1),
        (5, 0),
        (5, 2),
        (5, 4),
        (100, 0),
        (100, 4),
        (100, 99),
    ],
)
def test_stride(interval: int, offset: int) -> None:
    samples = list(CheckpointableDataset.from_sequence(SAMPLES).stride(interval, offset))
    assert samples == SAMPLES[offset::interval]
