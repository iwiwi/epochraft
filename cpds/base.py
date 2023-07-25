from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

if TYPE_CHECKING:
    import streaming


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
