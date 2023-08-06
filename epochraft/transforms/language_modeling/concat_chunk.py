from __future__ import annotations

import copy
from typing import Any, Optional, Sequence

import torch

from ...base import CheckpointableDataset, CheckpointableIterator, Sample, StateDict
from .tokenizer_utils import TokensQueue


class ConcatChunkIterator(CheckpointableIterator):
    def __init__(
        self,
        dataset: ConcatChunkDataset,
        source: CheckpointableIterator,
        buffers: Optional[dict[str, torch.Tensor]],
    ) -> None:
        self.dataset = dataset
        self.source = source
        self.queue = TokensQueue(columns=self.dataset.target_columns, buffers=buffers)

    def __next__(self) -> Sample:
        while self.queue.length() < self.dataset.chunk_length:
            in_sample = next(self.source)
            self.queue.push_from_sample(in_sample)

        return self.queue.pop_by_length(self.dataset.chunk_length)

    def state_dict(self) -> StateDict:
        return {
            "source": self.source.state_dict(),
            "buffers": self.queue.buffers.copy(),
        }


class ConcatChunkDataset(CheckpointableDataset):
    def __init__(
        self,
        source: CheckpointableDataset,
        chunk_length: int,
        target_columns: Sequence[str],
    ) -> None:
        self.source = source
        self.target_columns = target_columns
        self.chunk_length = chunk_length

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
        return ConcatChunkIterator(self, source, buffers)
