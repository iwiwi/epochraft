import itertools

from .base import CheckpointableDataset, Sample


def generate_example_sequence(n_samples: int = 20) -> list[Sample]:
    return [{"id": i} for i in range(n_samples)]


def check_resumption(
    original_dataset: CheckpointableDataset,
    resuming_dataset: CheckpointableDataset,
    ckpt_index: int,
    n_subsequent_samples: int = 10,
) -> None:
    # Consuming the first `ckpt_index` samples and get the state dict
    it = iter(original_dataset)
    for _ in range(ckpt_index):
        next(it)
    state_dict = it.state_dict()
    subsequent_samples_original = list(itertools.islice(it, n_subsequent_samples))

    # Resume from the state dict
    it = resuming_dataset.iter(state_dict=state_dict)
    subsequent_samples_resumed = list(itertools.islice(it, n_subsequent_samples))

    assert subsequent_samples_original == subsequent_samples_resumed
