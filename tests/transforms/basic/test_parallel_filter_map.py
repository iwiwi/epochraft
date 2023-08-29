from typing import Optional

import pytest

from epochraft import CheckpointableDataset, ParallelExecutorType, Sample, testing


parameterize_parallel_config = pytest.mark.parametrize(
    "max_workers, prefetch_factor, executor_type",
    [
        (None, 2, "process"),
        (2, 1, "process"),
        (4, 5, "process"),
        (None, 2, "thread"),
        (2, 1, "thread"),
        (4, 5, "thread"),
    ],
)


def _map_fn(sample: Sample) -> Sample:
    return {"id": sample["id"] * 2}


@parameterize_parallel_config
def test_parallel_map(
    max_workers: int, prefetch_factor: int, executor_type: ParallelExecutorType
) -> None:
    samples = testing.generate_example_sequence()
    dataset = CheckpointableDataset.from_sequence(samples).parallel_map(
        _map_fn,
        max_workers=max_workers,
        prefetch_factor=prefetch_factor,
        executor_type=executor_type,
    )
    samples_generated = list(dataset)

    # Should generate the same samples
    assert samples_generated == list(map(_map_fn, samples))


def _filter_fn(sample: Sample) -> bool:
    return sample["id"] % 2 == 0  # type: ignore


@parameterize_parallel_config
def test_parallel_filter(
    max_workers: int, prefetch_factor: int, executor_type: ParallelExecutorType
) -> None:
    samples = testing.generate_example_sequence()
    dataset = CheckpointableDataset.from_sequence(samples).parallel_filter(
        _filter_fn,
        max_workers=max_workers,
        prefetch_factor=prefetch_factor,
        executor_type=executor_type,
    )
    samples_generated = list(dataset)

    # Should generate the same samples
    assert samples_generated == list(filter(_filter_fn, samples))


def _filter_map_fn(sample: Sample) -> Optional[Sample]:
    if sample["id"] % 2 == 0:
        return {"id": sample["id"] * 3}
    else:
        return None


@parameterize_parallel_config
def test_parallel_filter_map(
    max_workers: int, prefetch_factor: int, executor_type: ParallelExecutorType
) -> None:
    samples = testing.generate_example_sequence()
    dataset = CheckpointableDataset.from_sequence(samples).parallel_filter_map(
        _filter_map_fn,
        max_workers=max_workers,
        prefetch_factor=prefetch_factor,
        executor_type=executor_type,
    )
    samples_generated = list(dataset)

    # Should generate the same samples
    samples_expected = list(filter(None, map(_filter_map_fn, samples)))
    assert samples_generated == samples_expected


@parameterize_parallel_config
def test_unorderd_parallel_filter_map(
    max_workers: int, prefetch_factor: int, executor_type: ParallelExecutorType
) -> None:
    samples = testing.generate_example_sequence()
    dataset = CheckpointableDataset.from_sequence(samples).parallel_filter_map(
        _filter_map_fn,
        max_workers=max_workers,
        prefetch_factor=prefetch_factor,
        executor_type=executor_type,
        ordered=False,
    )

    # Sorting is the necessary because the order of the samples is not guaranteed
    samples_generated = sorted(dataset, key=lambda x: x["id"])

    # Should generate the same samples
    samples_expected = list(filter(None, map(_filter_map_fn, samples)))
    assert samples_generated == samples_expected


@parameterize_parallel_config
def test_parallel_filter_map_checkpointing(
    max_workers: int, prefetch_factor: int, executor_type: ParallelExecutorType
) -> None:
    samples = testing.generate_example_sequence()
    dataset = CheckpointableDataset.from_sequence(
        samples, repeat=True, shuffle=True
    ).parallel_filter_map(
        _filter_map_fn,
        max_workers=max_workers,
        prefetch_factor=prefetch_factor,
        executor_type=executor_type,
    )

    testing.check_resumption(dataset, dataset, 0)
    testing.check_resumption(dataset, dataset, 1)
    testing.check_resumption(dataset, dataset, 123)


class ExampleException(Exception):
    pass


def _map_fn_with_exception(sample: Sample) -> Sample:
    if sample["id"] >= 10:
        raise ExampleException()
    else:
        return sample


@pytest.mark.parametrize(
    "executor_type, ordered",
    [
        ("process", True),
        ("thread", True),
        ("process", False),
        ("thread", False),
    ],
)
def test_parallel_filter_map_exception(executor_type: ParallelExecutorType, ordered: bool) -> None:
    samples = testing.generate_example_sequence()
    dataset = CheckpointableDataset.from_sequence(samples).parallel_filter_map(
        _map_fn_with_exception,
        max_workers=2,
        prefetch_factor=1,
        executor_type=executor_type,
    )
    with pytest.raises(ExampleException):
        list(dataset)
