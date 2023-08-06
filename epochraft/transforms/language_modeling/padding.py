from __future__ import annotations

import functools
from typing import Mapping

import torch

from ...base import CheckpointableDataset, Sample
from .tokenizer_utils import tensor_from_token_array


def _pad_fn(
    sample: Sample,
    pad_values: Mapping[str, int],
    chunk_length: int,
) -> Sample:
    sample = sample.copy()
    for column, pad_value in pad_values.items():
        if column not in sample:
            raise ValueError(f"Column {column} not found in sample (columns: {sample.keys()})")

        input_tensor = tensor_from_token_array(sample[column])
        input_length = len(input_tensor)
        if input_length > chunk_length:
            raise ValueError(f"Input sequence too long ({input_length} > {chunk_length})")

        pad_length = chunk_length - input_length
        pad_tensor = torch.full((pad_length,), pad_value, dtype=torch.long)
        sample[column] = torch.cat((input_tensor, pad_tensor))

    return sample


def pad(
    source: CheckpointableDataset,
    pad_values: Mapping[str, int],
    chunk_length: int,
) -> CheckpointableDataset:
    return source.map(
        functools.partial(
            _pad_fn,
            pad_values=pad_values,
            chunk_length=chunk_length,
        )
    )
