from __future__ import annotations

import itertools
from typing import Optional

import pytest

from epochraft import CheckpointableDataset, Sample, testing

from .conftest import SAMPLES_PER_FILE


def test_from_files(jsonl_paths: list[str]) -> None:
    samples = list(
        CheckpointableDataset.from_files(
            jsonl_paths,
            repeat=False,
            shuffle_shards=False,
            n_active_shards=2,
            n_standby_shards=1,
        )
    )

    file_indices = [sample["file_index"] for sample in samples]

    # Should produce all samples in all shards
    for file_index, expected_count in enumerate(SAMPLES_PER_FILE):
        assert file_indices.count(file_index) == expected_count

    # Two active shards should be read alternately
    assert file_indices[:6] == [0, 1, 0, 1, 0, 1]


def test_from_files_repeat(cbor_paths: list[str]) -> None:
    n_epochs = 5

    it = CheckpointableDataset.from_files(
        cbor_paths,
        repeat=True,
        shuffle_shards=False,
        n_active_shards=2,
        n_standby_shards=1,
    )
    n_samples_per_epoch = sum(SAMPLES_PER_FILE)

    for _ in range(n_epochs):
        samples = list(itertools.islice(it, n_samples_per_epoch))

        file_indices = [sample["file_index"] for sample in samples]

        # Should produce all samples in all shards
        for file_index, expected_count in enumerate(SAMPLES_PER_FILE):
            assert file_indices.count(file_index) == expected_count

        # Two active shards should be read alternately
        assert file_indices[:6] == [0, 1, 0, 1, 0, 1]


@pytest.mark.parametrize(
    ("n_active_shards", "n_standby_shards"),
    [
        (1, 1),
        (1, 2),
        (1, 10),
        (3, 1),
        (3, 3),
        (3, 10),
        (10, 10),
        (1, 0),
        (3, 0),
        (10, 0),
    ],
)
def test_from_files_repeat_shuffle(
    jsonl_paths: list[str],
    n_active_shards: int,
    n_standby_shards: int,
) -> None:
    n_epochs = 5

    it = iter(
        CheckpointableDataset.from_files(
            jsonl_paths,
            repeat=True,
            shuffle_shards=True,
            n_active_shards=n_active_shards,
            n_standby_shards=n_standby_shards,
            seed=42,
        )
    )
    n_samples_per_epoch = sum(SAMPLES_PER_FILE)
    prv_samples: Optional[list[Sample]] = None

    for _ in range(n_epochs):
        samples = list(itertools.islice(it, n_samples_per_epoch))

        file_indices = [sample["file_index"] for sample in samples]

        # Should produce all samples in all shards
        for file_index, expected_count in enumerate(SAMPLES_PER_FILE):
            assert file_indices.count(file_index) == expected_count

        # Should produce different order of samples
        if prv_samples is not None:
            assert prv_samples != samples
        prv_samples = samples


@pytest.mark.parametrize(
    ("n_active_shards", "n_standby_shards"),
    [
        (1, 1),
        (1, 2),
        (1, 10),
        (3, 1),
        (3, 3),
        (3, 10),
        (10, 10),
        (1, 0),
        (3, 0),
        (10, 0),
    ],
)
def test_from_files_resumption(
    cbor_paths: list[str],
    n_active_shards: int,
    n_standby_shards: int,
) -> None:
    dataset = CheckpointableDataset.from_files(
        cbor_paths,
        repeat=True,
        shuffle_shards=True,
        n_active_shards=n_active_shards,
        n_standby_shards=n_standby_shards,
    )

    testing.check_resumption(dataset, dataset, 0)
    testing.check_resumption(dataset, dataset, 1)
    testing.check_resumption(dataset, dataset, 2)
    testing.check_resumption(dataset, dataset, 3)
    testing.check_resumption(dataset, dataset, 10)
    testing.check_resumption(dataset, dataset, 13)
    testing.check_resumption(dataset, dataset, 100)
