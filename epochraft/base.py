from __future__ import annotations

import abc
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional,
                    Sequence, Union)

import numpy as np
import torch

if TYPE_CHECKING:
    import streaming
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

    Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


Sample = Dict[str, Any]
StateDict = Dict[str, Any]
TokenArray = Union[List[int], np.ndarray, torch.Tensor]
CollateFnType = Callable[[List[Sample]], Sample]


class CheckpointableIterator(abc.ABC):
    def __iter__(self) -> CheckpointableIterator:
        return self

    @abc.abstractmethod
    def __next__(self) -> Sample:
        raise NotImplementedError()

    @abc.abstractmethod
    def state_dict(self) -> StateDict:
        raise NotImplementedError


class CheckpointableDataset(abc.ABC):
    def __iter__(self) -> CheckpointableIterator:
        return self.iter(state_dict=None)

    @abc.abstractmethod
    def iter(self, state_dict: Optional[StateDict] = None) -> CheckpointableIterator:
        raise NotImplementedError()

    @staticmethod
    def from_sequence(
        sequence: Sequence[Sample],
        repeat: bool = False,
        shuffle: bool = False,
        shuffle_seed: int = 42,
    ) -> CheckpointableDataset:
        from .sources import SequenceDataset

        return SequenceDataset(
            sequence=sequence, repeat=repeat, shuffle=shuffle, shuffle_seed=shuffle_seed
        )

    @staticmethod
    def from_mosaicml(
        mosaicml_dataset: streaming.StreamingDataset,
        repeat: bool = False,
    ) -> CheckpointableDataset:
        from .sources import MosaicmlDataset

        return MosaicmlDataset(mosaicml_dataset, repeat=repeat)

    def filter_map(self, fn: Callable[[Sample], Optional[Sample]]) -> CheckpointableDataset:
        from .transforms import FilterMap

        return FilterMap(self, fn)

    def map(self, fn: Callable[[Sample], Sample]) -> CheckpointableDataset:
        from .transforms import FilterMap

        return FilterMap(self, fn)

    def filter(self, fn: Callable[[Sample], bool]) -> CheckpointableDataset:
        from .transforms import FilterMap

        def _fn(sample: Sample) -> Optional[Sample]:
            return sample if fn(sample) else None

        return FilterMap(self, _fn)

    def enumerate(self, count_column: str = "step") -> CheckpointableDataset:
        from .transforms import Count

        return Count(self, count_column=count_column)

    def take(
        self,
        max_count: int,
    ) -> CheckpointableDataset:
        from .transforms import Count

        return Count(self, max_count=max_count)

    def batch(
        self,
        batch_size: int,
        collate_fn: CollateFnType = torch.utils.data.default_collate,
        drop_last: bool = False,
    ) -> CheckpointableDataset:
        from .transforms import Batch

        return Batch(self, batch_size=batch_size, collate_fn=collate_fn, drop_last=drop_last)

    def concat_chunk(
        self,
        chunk_length: int,
        column: str = "input_ids",
        bos_tokens: Optional[TokenArray] = None,
        eos_tokens: Optional[TokenArray] = None,
    ) -> CheckpointableDataset:
        from .transforms import ConcatChunk

        return ConcatChunk(
            self,
            chunk_length=chunk_length,
            column=column,
            bos_tokens=bos_tokens,
            eos_tokens=eos_tokens,
        )
