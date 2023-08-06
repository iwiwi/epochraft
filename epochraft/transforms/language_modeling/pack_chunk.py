from __future__ import annotations

import copy
from typing import Any, Optional, Sequence

import torch

from ...base import CheckpointableDataset, CheckpointableIterator, Sample, StateDict
from .tokenizer_utils import TokensQueue


class PackChunkIterator(CheckpointableIterator):
    def __init__(
        self,
        dataset: PackChunkDataset,
        source: CheckpointableIterator,
        buffers: Optional[dict[str, torch.Tensor]],
    ) -> None:
        self.dataset = dataset
        self.queue = TokensQueue(columns=self.dataset.target_columns, buffers=buffers)
        self.source = source

    def __next__(self) -> Sample:
        # Add samples until the buffer exceeds the chunk length
        while True:
            try:
                input_sample = next(self.source)
            except StopIteration:
                input_tensor_dict = None
                break
            input_tensor_dict = self.queue.tensor_dict_from_sample(input_sample)
            input_length = len(input_tensor_dict[self.dataset.target_columns[0]])

            # Discard too long samples
            if self.dataset.discard_long_samples and input_length > self.dataset.chunk_length:
                continue

            buffer_length = self.queue.length()

            if input_length + buffer_length > self.dataset.chunk_length:
                break
            else:
                self.queue.push_from_tensor_dict(input_tensor_dict)

        # Take the sample from the buffer, then add the remaining sample
        output_sample = self.queue.pop_all()
        if input_tensor_dict is None:
            output_length = len(output_sample[self.dataset.target_columns[0]])
            # If the source is exhausted and the buffer is empty, then we are done
            if output_length == 0:
                raise StopIteration()
        else:
            self.queue.push_from_tensor_dict(input_tensor_dict)

        # Truncate the output sample
        for column in self.dataset.target_columns:
            if len(output_sample[column]) > self.dataset.chunk_length:
                # Truncate
                assert not self.dataset.discard_long_samples  # This should have been discarded
                output_sample[column] = output_sample[column][: self.dataset.chunk_length]

        return output_sample

    def state_dict(self) -> StateDict:
        return {
            "source": self.source.state_dict(),
            "buffers": self.queue.buffers.copy(),
        }


class PackChunkDataset(CheckpointableDataset):
    def __init__(
        self,
        source: CheckpointableDataset,
        chunk_length: int,
        target_columns: Sequence[str],
        discard_long_samples: bool = False,
    ) -> None:
        self.source = source
        self.target_columns = target_columns
        self.chunk_length = chunk_length
        self.discard_long_samples = discard_long_samples

    def iter(self, state_dict: Optional[dict[str, Any]] = None) -> CheckpointableIterator:
        if state_dict:
            source_state_dict = state_dict.pop("source")
            buffers = copy.copy(state_dict.pop("buffers"))
            if state_dict:
                raise ValueError(f"Unexpected keys in state_dict: {state_dict.keys()}")
        else:
            source_state_dict = None
            buffers = None

        source = self.source.iter(state_dict=source_state_dict)
        return PackChunkIterator(self, source, buffers)
