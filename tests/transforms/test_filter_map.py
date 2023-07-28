from typing import Optional

from epochraft import CheckpointableDataset, Sample, testing


def test_map() -> None:
    def fn(sample: Sample) -> Sample:
        return {"id": sample["id"] * 2}

    samples_original = testing.generate_example_sequence()
    dataset = CheckpointableDataset.from_sequence(samples_original)
    dataset = dataset.map(fn)
    samples_generated = list(dataset)

    # Should generate the same samples
    assert samples_generated == list(map(fn, samples_original))


def test_filter() -> None:
    def fn(sample: Sample) -> bool:
        return sample["id"] % 2 == 0  # type: ignore

    samples_original = testing.generate_example_sequence()
    dataset = CheckpointableDataset.from_sequence(samples_original)
    dataset = dataset.filter(fn)
    samples_generated = list(dataset)

    # Should generate the same samples
    assert samples_generated == list(filter(fn, samples_original))


def test_filter_map() -> None:
    def fn(sample: Sample) -> Optional[Sample]:
        if sample["id"] % 2 == 0:
            return {"id": sample["id"] * 3}
        else:
            return None

    samples_original = testing.generate_example_sequence()
    dataset = CheckpointableDataset.from_sequence(samples_original)
    dataset = dataset.filter_map(fn)
    samples_generated = list(dataset)

    # Should generate the same samples
    samples_expected = list(filter(None, map(fn, samples_original)))
    assert samples_generated == samples_expected


def test_filter_map_checkpointing() -> None:
    def fn(sample: Sample) -> Optional[Sample]:
        if sample["id"] % 2 == 0:
            return {"id": sample["id"] * 3}
        else:
            return None

    samples = testing.generate_example_sequence()
    dataset = CheckpointableDataset.from_sequence(samples, repeat=True, shuffle=True).filter_map(fn)

    testing.check_resumption(dataset, dataset, 0)
    testing.check_resumption(dataset, dataset, 1)
    testing.check_resumption(dataset, dataset, 123)
