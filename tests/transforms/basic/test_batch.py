import torch

from epochraft import CheckpointableDataset, testing


def test_batch() -> None:
    sequence = [{"data": torch.tensor(i)} for i in range(10)]
    dataset = CheckpointableDataset.from_sequence(sequence)
    batched_dataset = dataset.batch(batch_size=2)

    for i, sample in enumerate(batched_dataset):
        assert len(sample["data"]) == 2
        assert torch.all(sample["data"] == torch.tensor([2 * i, 2 * i + 1]))


def test_batch_drop_last() -> None:
    # Total samples is divisible by batch_size
    sequence = [{"data": torch.tensor(i)} for i in range(10)]
    dataset = CheckpointableDataset.from_sequence(sequence)

    # Test without drop_last
    batched_dataset = dataset.batch(batch_size=3, drop_last=False)
    batched_data = list(batched_dataset)
    assert len(batched_data) == 4
    assert len(batched_data[-1]["data"]) == 1

    # Test with drop_last
    batched_dataset = dataset.batch(batch_size=3, drop_last=True)
    batched_data = list(batched_dataset)
    assert len(batched_data) == 3
    assert len(batched_data[-1]["data"]) == 3

    # Total samples is not divisible by batch_size
    sequence = [{"data": torch.tensor(i)} for i in range(9)]
    dataset = CheckpointableDataset.from_sequence(sequence)

    # Test without drop_last
    batched_dataset = dataset.batch(batch_size=3, drop_last=False)
    batched_data = list(batched_dataset)
    assert len(batched_data) == 3
    assert len(batched_data[-1]["data"]) == 3

    # Test with drop_last
    batched_dataset = dataset.batch(batch_size=3, drop_last=True)
    batched_data = list(batched_dataset)
    assert len(batched_data) == 3
    assert len(batched_data[-1]["data"]) == 3


def test_batch_checkpointing() -> None:
    sequence = [{"data": torch.tensor(i)} for i in range(10)]
    dataset = CheckpointableDataset.from_sequence(sequence, repeat=True, shuffle=True).batch(
        batch_size=3
    )

    testing.check_resumption(dataset, dataset, 0)
    testing.check_resumption(dataset, dataset, 1)
    testing.check_resumption(dataset, dataset, 2)
    testing.check_resumption(dataset, dataset, 10)
