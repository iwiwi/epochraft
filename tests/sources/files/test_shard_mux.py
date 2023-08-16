from __future__ import annotations

import itertools
from typing import Optional

from epochraft import Sample
from epochraft.sources.files.shards_mux import ShardsMux

from .conftest import SAMPLES_PER_FILE


def test_shards_mux(jsonl_paths: list[str]) -> None:
    it = ShardsMux(
        jsonl_paths,
        format="auto",
        repeat=False,
        shuffle=False,
        n_active_shards=2,
        n_standby_shards=1,
        timeout=10.0,
        n_prefetch_samples=4,
        seed=42,
    )
    samples = list(it)

    file_indices = [sample["file_index"] for sample in samples]

    # Should produce all samples in all shards
    for file_index, expected_count in enumerate(SAMPLES_PER_FILE):
        assert file_indices.count(file_index) == expected_count

    # Two active shards should be read alternately
    assert file_indices[:6] == [0, 1, 0, 1, 0, 1]


def test_shards_mux_repeat(jsonl_paths: list[str]) -> None:
    n_epochs = 5

    it = ShardsMux(
        jsonl_paths,
        format="auto",
        repeat=True,
        shuffle=False,
        n_active_shards=2,
        n_standby_shards=1,
        timeout=10.0,
        n_prefetch_samples=4,
        seed=42,
    )
    n_samples_per_epoch = sum(SAMPLES_PER_FILE)

    for _ in range(n_epochs):
        samples = itertools.islice(it, n_samples_per_epoch)

        file_indices = [sample["file_index"] for sample in samples]

        # Should produce all samples in all shards
        for file_index, expected_count in enumerate(SAMPLES_PER_FILE):
            assert file_indices.count(file_index) == expected_count

        # Two active shards should be read alternately
        assert file_indices[:6] == [0, 1, 0, 1, 0, 1]


def test_shards_mux_shuffle(jsonl_paths: list[str]) -> None:
    n_epochs = 5

    it = ShardsMux(
        jsonl_paths,
        format="auto",
        repeat=True,
        shuffle=True,
        n_active_shards=2,
        n_standby_shards=1,
        timeout=10.0,
        n_prefetch_samples=4,
        seed=42,
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
