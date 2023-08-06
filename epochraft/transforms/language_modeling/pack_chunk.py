from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch

from ...base import CheckpointableDataset, CheckpointableIterator, Sample, StateDict, TokenArray
from .tokenizer_utils import tensor_from_token_array


class _Buffer:
    def __init__(self, buffer: torch.LongTensor) -> None:
        self.buffer = buffer

    def add(self, tokens: torch.LongTensor) -> None:
        self.buffer = torch.cat([self.buffer, tokens])

    def len(self) -> int:
        return len(self.buffer)


class PackChunkIterator(CheckpointableIterator):
    def __init__(self, dataset: PackChunkDataset) -> None:
        self.dataset = dataset
        self._buffers = {
            column: torch.empty(0, dtype=torch.long) for column in self.dataset.target_columns
        }

    def _buffer_len(self) -> int:
        return len(next(self._buffers.keys()))

    def __next__(self) -> Sample:
        while len(self.buffer) < self.dataset.chunk_length:
            source_sample = next(self.source)

            tokens = tensor_from_token_array(source_sample[self.dataset.column])
            self.buffer.add(tokens)


class PackChunkDataset(CheckpointableDataset):
    def __init__(
        self,
        source: CheckpointableDataset,
        chunk_length: int,
        target_columns: str,
        bos_tokens: Optional[Union[int, TokenArray]],
        eos_tokens: Optional[Union[int, TokenArray]],
    ) -> None:
        self.source = source
        self.target_columns = target_columns
        self.chunk_length = chunk_length
        self.bos_tokens = tensor_from_token_array(bos_tokens)
        self.eos_tokens = tensor_from_token_array(eos_tokens)

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
