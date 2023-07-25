from __future__ import annotations

import itertools

import pytest

from cpds import CheckpointableDataset, Sample
from cpds.testing import generate_example_sequence


def generate_example_sequence(n_samples: int = 20) -> list[Sample]:
    return [{"id": i} for i in range(n_samples)]


def key_fn(sample: Sample) -> int:
    """Key function for sorting samples by their id"""
    return sample["id"]  # type: ignore


def test_sequence_dataset_construction() -> None:
    samples_original = generate_example_sequence()
    dataset = CheckpointableDataset.from_sequence(samples_original)
    samples_generated = list(dataset)

    # Should generate the same samples
    assert samples_generated == samples_original


def test_sequence_dataset_shuffle() -> None:
    samples_original = generate_example_sequence()
    dataset = CheckpointableDataset.from_sequence(samples_original, shuffle=True)
    samples_generated = list(dataset)

    # Shuffled, so the order should be different
    assert samples_generated != samples_original

    # However, the set of the samples should be the same
    assert sorted(samples_generated, key=key_fn) == sorted(samples_original, key=key_fn)


def test_sequence_dataset_repeat() -> None:
    n_samples = 5
    n_epochs = 4
    samples_original = generate_example_sequence(n_samples=n_samples)
    dataset = CheckpointableDataset.from_sequence(samples_original, repeat=True)
    samples_generated = list(itertools.islice(dataset, n_samples * n_epochs))

    # Should generate the same samples
    assert samples_generated == samples_original * n_epochs


def test_sequence_dataset_shuffle_repeat() -> None:
    n_samples = 5
    n_epochs = 4
    samples_original = generate_example_sequence(n_samples=n_samples)
    dataset = CheckpointableDataset.from_sequence(samples_original, repeat=True, shuffle=True)
    samples_generated = list(itertools.islice(dataset, n_samples * n_epochs))

    # Shuffled, so the order should be different
    assert samples_generated != samples_original * n_epochs

    # However, the set of the samples should be the same
    assert sorted(samples_generated, key=key_fn) == sorted(samples_original * n_epochs, key=key_fn)


@pytest.mark.parametrize(
    "n_samples, ckpt_index, shuffle, repeat",
    [
        (20, 5, False, False),
        (20, 5, True, False),
        (20, 19, True, False),
        (5, 2, False, True),
        (5, 10, False, True),
        (5, 24, False, True),
        (5, 2, True, True),
        (5, 10, True, True),
        (5, 24, True, True),
    ],
)
def test_sequence_dataset_resumption(
    n_samples: int, ckpt_index: int, shuffle: bool, repeat: bool
) -> None:
    n_subsequent_samples = 3  # How many samples to check after resumption
    samples = generate_example_sequence(n_samples=n_samples)
    dataset = CheckpointableDataset.from_sequence(samples, repeat=repeat, shuffle=shuffle)

    # Consuming the first `ckpt_index` samples and get the state dict
    it = iter(dataset)
    for _ in range(ckpt_index):
        next(it)
    state_dict = it.state_dict()
    subsequent_samples_original = list(itertools.islice(it, n_subsequent_samples))

    # Recreate the dataset and resume from the state dict
    dataset = CheckpointableDataset.from_sequence(samples, repeat=repeat, shuffle=shuffle)
    it = dataset.iter(state_dict=state_dict)
    subsequent_samples_resumed = list(itertools.islice(it, n_subsequent_samples))

    assert subsequent_samples_resumed == subsequent_samples_original
