from epochraft import CheckpointableDataset, testing


def test_enumerate() -> None:
    sequence = [{"data": i} for i in range(5)]
    dataset = CheckpointableDataset.from_sequence(sequence)

    enumerated_dataset = dataset.enumerate(count_column="index")

    for i, sample in enumerate(enumerated_dataset):
        assert sample["index"] == i
        assert sample["data"] == i


def test_take() -> None:
    sequence = [{"data": i} for i in range(10)]
    dataset = CheckpointableDataset.from_sequence(sequence)

    limited_dataset = dataset.take(max_count=5)

    limited_data = list(limited_dataset)
    assert len(limited_data) == 5


def test_take_enumerate_checkpointing() -> None:
    sequence = [{"data": i} for i in range(5)]
    dataset = (
        CheckpointableDataset.from_sequence(sequence, repeat=True, shuffle=True)
        .enumerate()
        .take(35)
    )

    testing.check_resumption(dataset, dataset, 0)
    testing.check_resumption(dataset, dataset, 1)
    testing.check_resumption(dataset, dataset, 10)
    testing.check_resumption(dataset, dataset, 34)
    testing.check_resumption(dataset, dataset, 35)
