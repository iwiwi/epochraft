from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch

from ..base import (CheckpointableDataset, CheckpointableIterator, Sample,
                    StateDict, TokenArray)


def tensor_from_token_array(data: Optional[Union[int, TokenArray]]) -> torch.Tensor:
    if data is None:
        data = []
    if isinstance(data, int):
        data = [data]

    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, np.ndarray):
        if np.issubdtype(data.dtype, np.integer):
            tensor = torch.from_numpy(data).long()
        else:
            raise ValueError(f"Expected integer ndarray, got ndarray of {data.dtype}")
    else:
        tensor = torch.tensor(data, dtype=torch.long)

    if tensor.dtype != torch.long:
        raise ValueError(f"Expected long tensor, got {tensor.dtype}")
    if tensor.dim() != 1:
        raise ValueError("input_ids must be 1-dimensional tensor")
    return tensor


class ConcatChunkIterator(CheckpointableIterator):
    def __init__(
        self,
        dataset: ConcatChunk,
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
            self.buffer = torch.cat(
                [
                    self.buffer,
                    self.dataset.bos_tokens,
                    tokens,
                    self.dataset.eos_tokens,
                ]
            )

        y = self.buffer[: self.dataset.chunk_length]
        self.buffer = self.buffer[self.dataset.chunk_length :]
        return {self.dataset.column: y}

    def state_dict(self) -> StateDict:
        return {
            "source": self.source.state_dict(),
            "buffer": self.buffer,
        }


class ConcatChunk(CheckpointableDataset):
    def __init__(
        self,
        source: CheckpointableDataset,
        chunk_length: int,
        column: str,
        bos_tokens: Optional[Union[int, TokenArray]],
        eos_tokens: Optional[Union[int, TokenArray]],
    ) -> None:
        self.source = source
        self.column = column
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
