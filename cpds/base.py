from __future__ import annotations

import abc
from typing import Any, Optional, Sequence

Sample = dict[str, Any]
StateDict = dict[str, Any]


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
