from __future__ import annotations

import itertools
import tempfile
from typing import Generator

import pytest
import streaming

from epochraft import CheckpointableDataset, Sample, testing


@pytest.fixture
def samples_and_path() -> Generator[tuple[list[Sample], str], None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        samples = testing.generate_example_sequence()
        columns = {"id": "int"}
        with streaming.MDSWriter(out=tmp_dir, columns=columns) as mds:
            for sample in samples:
                mds.write(sample)

        yield samples, tmp_dir


def test_mosaicml_dataset_construction(samples_and_path: tuple[list[Sample], str]) -> None:
    samples, mds_path = samples_and_path

    mosaicml_dataset = streaming.StreamingDataset(local=mds_path, num_canonical_nodes=1)
    dataset = CheckpointableDataset.from_mosaicml(mosaicml_dataset)
    samples_generated = list(dataset)

    assert samples_generated == samples


def test_mosaicml_dataset_repeat(samples_and_path: tuple[list[Sample], str]) -> None:
    n_epochs = 3
    samples, mds_path = samples_and_path

    mosaicml_dataset = streaming.StreamingDataset(local=mds_path, num_canonical_nodes=1)
    dataset = CheckpointableDataset.from_mosaicml(mosaicml_dataset, repeat=True)
    samples_generated = list(itertools.islice(dataset, len(samples) * n_epochs))

    assert samples_generated == samples * n_epochs


def test_mosaicml_dataset_repeat_shuffle(samples_and_path: tuple[list[Sample], str]) -> None:
    n_epochs = 3
    samples, mds_path = samples_and_path

    mosaicml_dataset = streaming.StreamingDataset(
        local=mds_path,
        num_canonical_nodes=1,
        shuffle=True,
    )
    dataset = CheckpointableDataset.from_mosaicml(mosaicml_dataset, repeat=True)

    for _ in range(n_epochs):
        samples_generated = list(itertools.islice(dataset, len(samples)))
        assert samples != samples_generated
        assert samples == sorted(samples_generated, key=testing.sort_key_fn)


@pytest.mark.parametrize(
    "ckpt_index",
    [0, 1, 19, 20, 21, 119, 120, 121, 125],
)
def test_mosaicml_dataset_resumption(
    samples_and_path: tuple[list[Sample], str], ckpt_index: int
) -> None:
    _, mds_path = samples_and_path

    def dataset_factory_fn() -> CheckpointableDataset:
        return CheckpointableDataset.from_mosaicml(
            streaming.StreamingDataset(local=mds_path, num_canonical_nodes=1, shuffle=False),
            repeat=True,
        )

    testing.check_resumption_with_instantiation(dataset_factory_fn, ckpt_index)
