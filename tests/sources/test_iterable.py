from __future__ import annotations

from typing import Generator

import pytest

from epochraft import CheckpointableDataset, Sample, testing


def test_iterable_dataset() -> None:
    iterable = [{"data": "dataset_1", "value": i} for i in range(3)]

    dataset = CheckpointableDataset.from_iterable(iterable, repeat=True)
    dataset_iter = dataset.iter()

    # Extract the first 5 samples and check if they are as expected.
    # If the `repeat` option is working,
    # the samples should cycle back to the start after the end of the iterable.
    for i in range(5):
        sample = next(dataset_iter)
        expected_sample = {"data": "dataset_1", "value": i % len(iterable)}
        assert sample == expected_sample


def test_iterable_dataset_no_repeat() -> None:
    iterable = [{"data": "dataset_1", "value": i} for i in range(3)]

    dataset = CheckpointableDataset.from_iterable(iterable, repeat=False)
    dataset_iter = dataset.iter()

    # Extract the first 4 samples.
    # Since `repeat` is False, a StopIteration exception should be
    # raised after the end of the iterable.
    samples = []
    with pytest.raises(StopIteration):
        for _ in range(4):
            samples.append(next(dataset_iter))

    # Check if the extracted samples are as expected.
    expected_samples = [{"data": "dataset_1", "value": i} for i in range(3)]
    assert samples == expected_samples


def test_iterable_dataset_infinite() -> None:
    # Define an infinite generator.
    def infinite_generator() -> Generator[Sample, None, None]:
        i = 0
        while True:
            yield {"data": "dataset_1", "value": i}
            i += 1

    dataset = CheckpointableDataset.from_iterable(infinite_generator())
    dataset_iter = dataset.iter()

    # Extract the first 10 samples.
    # Since the generator is infinite, the iterable should not run out of samples.
    samples = [next(dataset_iter) for _ in range(10)]

    # Check if the extracted samples are as expected.
    expected_samples = [{"data": "dataset_1", "value": i} for i in range(10)]
    assert samples == expected_samples


def test_iterable_dataset_resumption_no_repeat() -> None:
    iterable = [{"data": "dataset_1", "value": i} for i in range(3)]

    dataset = CheckpointableDataset.from_iterable(iterable, repeat=False)

    testing.check_resumption(dataset, dataset, 0)
    testing.check_resumption(dataset, dataset, 1)
    testing.check_resumption(dataset, dataset, 2)
    testing.check_resumption(dataset, dataset, 3)


def test_iterable_dataset_resumption_repeat() -> None:
    iterable = [{"data": "dataset_1", "value": i} for i in range(3)]

    dataset = CheckpointableDataset.from_iterable(iterable, repeat=True)

    testing.check_resumption(dataset, dataset, 0)
    testing.check_resumption(dataset, dataset, 1)
    testing.check_resumption(dataset, dataset, 2)
    testing.check_resumption(dataset, dataset, 3)
    testing.check_resumption(dataset, dataset, 4)
    testing.check_resumption(dataset, dataset, 6)
    testing.check_resumption(dataset, dataset, 100)


def test_iterable_dataset_resumption_repeat_single() -> None:
    iterable = [{"data": "dataset_1", "value": i} for i in range(1)]

    dataset = CheckpointableDataset.from_iterable(iterable, repeat=True)

    testing.check_resumption(dataset, dataset, 0)
    testing.check_resumption(dataset, dataset, 1)
    testing.check_resumption(dataset, dataset, 2)
    testing.check_resumption(dataset, dataset, 3)
    testing.check_resumption(dataset, dataset, 4)
    testing.check_resumption(dataset, dataset, 6)
    testing.check_resumption(dataset, dataset, 100)
