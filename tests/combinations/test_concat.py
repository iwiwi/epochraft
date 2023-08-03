from epochraft import CheckpointableDataset, concat_datasets, testing


def test_concat_simple() -> None:
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
        CheckpointableDataset.from_sequence(sequence, repeat=False) for sequence in sequences
    ]

    concatenated_dataset = concat_datasets(datasets).take(6)
    concatenated_data = list(concatenated_dataset)

    print(concatenated_data)

    # Expected samples
    expected_samples = [
        {"data": "dataset_1", "value": 0},
        {"data": "dataset_1", "value": 1},
        {"data": "dataset_1", "value": 2},
        {"data": "dataset_2", "value": 3},
        {"data": "dataset_2", "value": 4},
        {"data": "dataset_2", "value": 5},
    ]

    for sample, expected_sample in zip(concatenated_data, expected_samples):
        assert sample == expected_sample


def test_concat_checkpointing() -> None:
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
        CheckpointableDataset.from_sequence(sequence, repeat=False, shuffle=True)
        for sequence in sequences
    ]
    dataset = concat_datasets(datasets)

    testing.check_resumption(dataset, dataset, 0)
    testing.check_resumption(dataset, dataset, 1)
    testing.check_resumption(dataset, dataset, 2)
    testing.check_resumption(dataset, dataset, 3)
    testing.check_resumption(dataset, dataset, 4)
