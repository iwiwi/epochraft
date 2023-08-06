from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch

from ...base import CheckpointableDataset, CheckpointableIterator, Sample, StateDict
from .tokenizer_utils import BufferDict


class PackChunkIterator(CheckpointableIterator):
    def __init__(
        self,
        dataset: PackChunkDataset,
        source: CheckpointableIterator,
        buffers: dict[str, torch.Tensor],
    ) -> None:
        self.dataset = dataset
        self.buffers = BufferDict(columns=self.dataset.target_columns, buffers=buffers)
        self.source = source

    def __next__(self) -> Sample:
        # Add samples until the buffer exceeds the chunk length
        while True:
            try:
                input_sample = next(self.source)
            except StopIteration:
                input_tensor_dict = None
                break
            input_tensor_dict = self.buffers.tensor_dict_from_sample(input_sample)
            input_length = len(input_tensor_dict[self.dataset.target_columns[0]])

            # Discard too long samples
            if self.dataset.discard_long_samples and input_length > self.dataset.chunk_length:
                continue

            buffer_length = self.buffers.buffer_length()

            if input_length + buffer_length > self.dataset.chunk_length:
                break
            else:
                self.buffers.append_from_tensor_dict(input_tensor_dict)

        # Take the sample from the buffer, then add the remaining sample
        output_sample = self.buffers.take_all()
        if input_tensor_dict is None:
            output_length = len(output_sample[self.dataset.target_columns[0]])
            # If the source is exhausted and the buffer is empty, then we are done
            if output_length == 0:
                raise StopIteration()
        else:
            self.buffers.append_from_tensor_dict(input_tensor_dict)

        # Pad or truncate the output sample
        for column in self.dataset.target_columns:
            pad_value = self.dataset.pad_values[column]

            if len(output_sample[column]) > self.dataset.chunk_length:
                # Truncate
                assert not self.dataset.discard_long_samples  # This should have been discarded
                output_sample[column] = output_sample[column][: self.dataset.chunk_length]
            else:
                # Pad
                pad_length = self.dataset.chunk_length - len(output_sample[column])
                assert pad_length >= 0
                pad_tensor = torch.full((pad_length,), pad_value, dtype=torch.long)
                output_sample[column] = torch.cat((output_sample[column], pad_tensor))

        return output_sample

    def state_dict(self) -> StateDict:
        raise NotImplementedError()


class PackChunkDataset(CheckpointableDataset):
    def __init__(
        self,
        source: CheckpointableDataset,
        chunk_length: int,
        target_columns: Sequence[str],
        pad_values: Mapping[str, int],
        discard_long_samples: bool = False,
    ) -> None:
        self.source = source
        self.target_columns = target_columns
        self.chunk_length = chunk_length
        self.pad_values = pad_values
        self.discard_long_samples = discard_long_samples

        for column in self.target_columns:
            if column not in self.pad_values:
                raise ValueError(f"Missing pad_value for column: {column}")

    def iter(self, state_dict: Optional[dict[str, Any]] = None) -> CheckpointableIterator:
        if state_dict:
            source_state_dict = state_dict.pop("source")
            buffers = state_dict.pop("buffers")
            if state_dict:
                raise ValueError(f"Unexpected keys in state_dict: {state_dict.keys()}")
        else:
            source_state_dict = None
            buffers = None

        source = self.source.iter(state_dict=source_state_dict)
        return PackChunkIterator(self, source, buffers)
