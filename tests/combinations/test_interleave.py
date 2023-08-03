from epochraft import CheckpointableDataset, interleave_datasets, testing


def test_interleave_simple() -> None:
    sequences = [
        [
            {"data": "dataset_1", "value": 0},
            {"data": "dataset_1", "value": 1},
            {"data": "dataset_1", "value": 2},
        ],
        [
            {"data": "dataset_2", "value": 3},
            {"data": "dataset_2", "value": 4},
            {"data": "dataset_2", "value": 5},
        ],
    ]
    datasets = [
        CheckpointableDataset.from_sequence(sequence, repeat=True) for sequence in sequences
    ]

    interleaved_dataset = interleave_datasets(datasets).take(10)
    interleaved_data = list(interleaved_dataset)

    # Expected samples
    expected_samples = [
        {"data": "dataset_1", "value": 0},
        {"data": "dataset_2", "value": 3},
        {"data": "dataset_1", "value": 1},
        {"data": "dataset_2", "value": 4},
        {"data": "dataset_1", "value": 2},
        {"data": "dataset_2", "value": 5},
        {"data": "dataset_1", "value": 0},
        {"data": "dataset_2", "value": 3},
        {"data": "dataset_1", "value": 1},
        {"data": "dataset_2", "value": 4},
    ]

    for sample, expected_sample in zip(interleaved_data, expected_samples):
        assert sample == expected_sample


def test_interleave_weighted() -> None:
    sequences = [
        [
            {"data": "dataset_1", "value": 0},
            {"data": "dataset_1", "value": 1},
        ],
        [
            {"data": "dataset_2", "value": 3},
            {"data": "dataset_2", "value": 4},
        ],
    ]
    datasets = [
        CheckpointableDataset.from_sequence(sequence, repeat=True) for sequence in sequences
    ]

    interleaved_dataset = interleave_datasets(datasets, weights=[1, 2]).take(9)
    interleaved_data = list(interleaved_dataset)

    # Count occurrences of each dataset
    dataset_counts = {"dataset_1": 0, "dataset_2": 0}
    for sample in interleaved_data:
        dataset_counts[sample["data"]] += 1

    # Verify counts
    assert dataset_counts["dataset_1"] == 3
    assert dataset_counts["dataset_2"] == 6


def test_interleave_checkpointing() -> None:
    sequences = [
        [
            {"data": "dataset_1", "value": 0},
            {"data": "dataset_1", "value": 1},
        ],
        [
            {"data": "dataset_2", "value": 3},
            {"data": "dataset_2", "value": 4},
        ],
    ]
    datasets = [
        CheckpointableDataset.from_sequence(sequence, repeat=True, shuffle=True)
        for sequence in sequences
    ]
    dataset = interleave_datasets(datasets, weights=[1, 2])

    testing.check_resumption(dataset, dataset, 0)
    testing.check_resumption(dataset, dataset, 1)
    testing.check_resumption(dataset, dataset, 10)
    testing.check_resumption(dataset, dataset, 34)
    testing.check_resumption(dataset, dataset, 35)
