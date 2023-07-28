import itertools
import random
from typing import Callable

import numpy as np
import torch

from .base import CheckpointableDataset, Sample, TokenArray


def generate_example_sequence(n_samples: int = 20) -> list[Sample]:
    return [{"id": i} for i in range(n_samples)]


def sort_key_fn(sample: Sample) -> int:
    """Key function for sorting samples by their id"""
    return sample["id"]  # type: ignore


def generate_tokenized_samples(
    min_length: int = 0,
    max_length: int = 10,
    n_examples: int = 20,
    vocab_size: int = 100,
    seed: int = 42,
) -> list[dict[str, TokenArray]]:
    rng = random.Random(seed)
    i = 0
    samples = []
    for _ in range(n_examples):
        seq_len = rng.randint(min_length, max_length)
        seq = []
        for _ in range(seq_len):
            seq.append(i % vocab_size)
            i += 1
        sample: dict[str, TokenArray] = {"input_ids": seq}
        samples.append(sample)
    return samples


def assert_samples_equal(sample1: Sample, sample2: Sample) -> None:
    assert sample1.keys() == sample2.keys(), "Samples' keys are different"
    for key in sample1.keys():
        item1 = sample1[key]
        item2 = sample2[key]
        if isinstance(item1, torch.Tensor) and isinstance(item2, torch.Tensor):
            assert torch.equal(item1, item2), f"Tensors do not match for key {key}"
        elif isinstance(item1, np.ndarray) and isinstance(item2, np.ndarray):
            assert np.array_equal(item1, item2), f"Arrays do not match for key {key}"
        else:
            assert item1 == item2, f"Items do not match for key {key}"


def assert_sample_lists_equal(samples1: list[Sample], samples2: list[Sample]) -> None:
    assert len(samples1) == len(samples2), "Samples have different lengths"
    for sample1, sample2 in zip(samples1, samples2):
        assert_samples_equal(sample1, sample2)


def check_resumption_with_instantiation(
    dataset_factory: Callable[[], CheckpointableDataset],
    ckpt_index: int,
    n_subsequent_samples: int = 10,
) -> None:
    # Consuming the first `ckpt_index` samples and get the state dict
    dataset = dataset_factory()
    it = iter(dataset)
    for _ in range(ckpt_index):
        next(it)
    state_dict = it.state_dict()
    subsequent_samples_original = list(itertools.islice(it, n_subsequent_samples))

    # Clean up the dataset
    del dataset, it

    # Resume from the state dict
    dataset = dataset_factory()
    it = dataset.iter(state_dict=state_dict)
    subsequent_samples_resumed = list(itertools.islice(it, n_subsequent_samples))

    assert_sample_lists_equal(subsequent_samples_original, subsequent_samples_resumed)


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

    assert_sample_lists_equal(subsequent_samples_original, subsequent_samples_resumed)
