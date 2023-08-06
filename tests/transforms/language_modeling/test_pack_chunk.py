from __future__ import annotations

from epochraft import CheckpointableDataset


def test_pack_chunk() -> None:
    samples = [
        {"input_ids": [0, 1]},
        {"input_ids": [3]},
        {"input_ids": [4, 5, 6, 7, 8, 9, 10, 11]},
        {"input_ids": [0, 1, 2]},
        {"input_ids": [0, 1, 3]},
        {"input_ids": [5]},
        {"input_ids": [1]},
        {"input_ids": [2]},
        {"input_ids": [3]},
        {"input_ids": [4]},
    ]
    dataset = CheckpointableDataset.from_sequence(samples).pack_chunk(
        chunk_length=4, target_columns=["input_ids"], pad_values={"input_ids": -1}
    )
    chunks = list(dataset)

    assert len(chunks) == 5
    assert chunks[0]["input_ids"].tolist() == [0, 1, 3, -1]
    assert chunks[1]["input_ids"].tolist() == [4, 5, 6, 7]
    assert chunks[2]["input_ids"].tolist() == [0, 1, 2, -1]
    assert chunks[3]["input_ids"].tolist() == [0, 1, 3, 5]
    assert chunks[4]["input_ids"].tolist() == [1, 2, 3, 4]


def test_pack_chunk_discard_long_samples() -> None:
    samples = [
        {"input_ids": [0, 1]},
        {"input_ids": [3]},
        {"input_ids": [4, 5, 6, 7, 8, 9, 10, 11]},
        {"input_ids": [0, 1, 2]},
        {"input_ids": [0, 1, 3]},
        {"input_ids": [5]},
        {"input_ids": [1]},
        {"input_ids": [2]},
        {"input_ids": [3]},
        {"input_ids": [4]},
    ]
    dataset = CheckpointableDataset.from_sequence(samples).pack_chunk(
        chunk_length=4,
        target_columns=["input_ids"],
        pad_values={"input_ids": -1},
        discard_long_samples=True,
    )
    chunks = list(dataset)

    assert len(chunks) == 4
    assert chunks[0]["input_ids"].tolist() == [0, 1, 3, -1]
    assert chunks[1]["input_ids"].tolist() == [0, 1, 2, -1]
    assert chunks[2]["input_ids"].tolist() == [0, 1, 3, 5]
    assert chunks[3]["input_ids"].tolist() == [1, 2, 3, 4]


def test_pack_chunk_multiple_columns() -> None:
    samples = [
        {"input_ids": [0, 1, 2], "labels": [3, 4, 5]},
        {"input_ids": [6], "labels": [7]},
        {"input_ids": [8, 9], "labels": [10, 11]},
    ]
    dataset = CheckpointableDataset.from_sequence(samples).pack_chunk(
        chunk_length=4,
        target_columns=["input_ids", "labels"],
        pad_values={"input_ids": -1, "labels": -2},
    )
    chunks = list(dataset)

    assert len(chunks) == 2

    assert chunks[0]["input_ids"].tolist() == [0, 1, 2, 6]
    assert chunks[0]["labels"].tolist() == [3, 4, 5, 7]
    assert chunks[1]["input_ids"].tolist() == [8, 9, -1, -1]
    assert chunks[1]["labels"].tolist() == [10, 11, -2, -2]
