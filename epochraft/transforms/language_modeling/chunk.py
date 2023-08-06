from __future__ import annotations

from typing import Optional, Sequence

import torch

from ...base import CheckpointableDataset, CheckpointableIterator, Sample, StateDict
from .tokenizer_utils import BufferDict


class ChunkIterator(CheckpointableIterator):
    def __init__(
        self,
        dataset: ChunkDataset,
        source: CheckpointableIterator,
        buffers: Optional[dict[str, torch.Tensor]],
    ) -> None:
        self.dataset = dataset
        self.source = source
        self.buffers = BufferDict(columns=self.dataset.target_columns, buffers=buffers)

    def __next__(self) -> Sample:
        # This is while-loop as we may receive empty samples from the source
        while self.buffers.buffer_length() == 0:
            in_sample = next(self.source)
            self.buffers.append_from_sample(in_sample)

        if self.buffers.buffer_length() < self.dataset.chunk_length:
            out_sample = self.buffers.take_all()
            if self.dataset.drop_remainder:
                return next(self)
            else:
                return out_sample
        else:
            return self.buffers.take(self.dataset.chunk_length)

    def state_dict(self) -> StateDict:
        return {
            "source": self.source.state_dict(),
            "buffers": self.buffers.buffers.copy(),
        }


class ChunkDataset(CheckpointableDataset):
    def __init__(
        self,
        source: CheckpointableDataset,
        chunk_length: int,
        target_columns: Sequence[str],
        drop_remainder: bool,
    ) -> None:
        self.source = source
        self.chunk_length = chunk_length
        self.target_columns = target_columns
        self.drop_remainder = drop_remainder

    def iter(self, state_dict: Optional[StateDict] = None) -> CheckpointableIterator:
        if state_dict is not None:
            source_state_dict = state_dict.pop("source")
            buffers = state_dict.pop("buffers")
            if state_dict:
                raise ValueError(f"Unexpected state_dict keys: {state_dict.keys()}")
        else:
            source_state_dict = None
            buffers = None

        source = self.source.iter(state_dict=source_state_dict)
        return ChunkIterator(self, source, buffers)
