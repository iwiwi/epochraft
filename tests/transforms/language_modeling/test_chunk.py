from __future__ import annotations

from epochraft import CheckpointableDataset, testing


SAMPLES = [
    {"input_ids": [0, 1, 2, 3, 4, 5]},
    {"input_ids": [14, 15, 16, 17, 18, 19, 20, 21]},
    {"input_ids": [0, 1, 2]},
    {"input_ids": [4, 5, 6, 7, 8, 9, 10, 11, 12]},
]


def test_chunk() -> None:
    dataset = (
        CheckpointableDataset.from_sequence(SAMPLES)
        .chunk(chunk_length=4)
        .pad(chunk_length=4, pad_values={"input_ids": -1})
    )
    chunks = list(dataset)

    assert len(chunks) == 5
    assert chunks[0]["input_ids"].tolist() == [0, 1, 2, 3]
    assert chunks[1]["input_ids"].tolist() == [14, 15, 16, 17]
    assert chunks[2]["input_ids"].tolist() == [18, 19, 20, 21]
    assert chunks[3]["input_ids"].tolist() == [4, 5, 6, 7]
    assert chunks[4]["input_ids"].tolist() == [8, 9, 10, 11]


def test_chunk_drop_remainder_false() -> None:
    dataset = CheckpointableDataset.from_sequence(SAMPLES).chunk(
        chunk_length=4,
        target_columns=["input_ids"],
        drop_remainder=False,
    )
    chunks = list(dataset)

    assert len(chunks) == 8
    assert chunks[0]["input_ids"].tolist() == [0, 1, 2, 3]
    assert chunks[1]["input_ids"].tolist() == [4, 5]
    assert chunks[2]["input_ids"].tolist() == [14, 15, 16, 17]
    assert chunks[3]["input_ids"].tolist() == [18, 19, 20, 21]
    assert chunks[4]["input_ids"].tolist() == [0, 1, 2]
    assert chunks[5]["input_ids"].tolist() == [4, 5, 6, 7]
    assert chunks[6]["input_ids"].tolist() == [8, 9, 10, 11]
    assert chunks[7]["input_ids"].tolist() == [12]


def test_chunk_resumption() -> None:
    dataset = CheckpointableDataset.from_sequence(SAMPLES, repeat=True, shuffle=True).chunk(5)

    testing.check_resumption(dataset, dataset, 0)
    testing.check_resumption(dataset, dataset, 1)
    testing.check_resumption(dataset, dataset, 10)
    testing.check_resumption(dataset, dataset, 20)
    testing.check_resumption(dataset, dataset, 123)


def test_chunk_resumption_drop_remainder_false() -> None:
    dataset = CheckpointableDataset.from_sequence(SAMPLES, repeat=True, shuffle=True).chunk(
        5, drop_remainder=False
    )

    testing.check_resumption(dataset, dataset, 0)
    testing.check_resumption(dataset, dataset, 1)
    testing.check_resumption(dataset, dataset, 10)
    testing.check_resumption(dataset, dataset, 20)
    testing.check_resumption(dataset, dataset, 123)
