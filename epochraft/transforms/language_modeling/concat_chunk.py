from __future__ import annotations

from typing import Any, Optional, Union

import torch

from ...base import CheckpointableDataset, CheckpointableIterator, Sample, StateDict, TokenArray
from .tokenizer_utils import tensor_from_token_array


class ConcatChunkIterator(CheckpointableIterator):
    def __init__(
        self,
        dataset: ConcatChunkDataset,
        source: CheckpointableIterator,
        buffer: Optional[torch.LongTensor],
    ) -> None:
        self.dataset = dataset
        self.source = source
        self.buffer = torch.empty((0,), dtype=torch.long) if buffer is None else buffer

    def __next__(self) -> Sample:
        while len(self.buffer) < self.dataset.chunk_length:
            source_sample = next(self.source)
            tokens = tensor_from_token_array(source_sample[self.dataset.column])
            self.buffer = torch.cat((self.buffer, tokens))
            print(tokens.shape, self.buffer.shape)

        y = self.buffer[: self.dataset.chunk_length]
        self.buffer = self.buffer[self.dataset.chunk_length :]
        return {self.dataset.column: y}

    def state_dict(self) -> StateDict:
        return {
            "source": self.source.state_dict(),
            "buffer": self.buffer,
        }


class ConcatChunkDataset(CheckpointableDataset):
    def __init__(
        self,
        source: CheckpointableDataset,
        chunk_length: int,
        target_column: str,
    ) -> None:
        self.source = source
        self.column = target_column
        self.chunk_length = chunk_length

    def iter(self, state_dict: Optional[dict[str, Any]] = None) -> CheckpointableIterator:
        if state_dict:
            source_state_dict = state_dict.pop("source")
            buffer = state_dict.pop("buffer")
            if state_dict:
                raise ValueError(f"Unexpected keys in state_dict: {state_dict.keys()}")
        else:
            source_state_dict = None
            buffer = None

        source = self.source.iter(state_dict=source_state_dict)
        return ConcatChunkIterator(self, source, buffer)
