from __future__ import annotations

import abc
from typing import Any, Optional, Union

from .base import CheckpointableDataset, CheckpointableIterator, Sample, StateDict


class CacheStorage(abc.ABC):
    @abc.abstractmethod
    def load_sample(self, index: int) -> Union[None, Sample, StopIteration]:
        raise NotImplementedError()

    @abc.abstractmethod
    def save_sample(self, index: int, sample: Union[Sample, StopIteration]) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def load_state_dict(self) -> Optional[tuple[int, StateDict]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def save_state_dict(self, sample_index: int, state_dict: StateDict) -> None:
        raise NotImplementedError()


class InMemoryCacheStorage(CacheStorage):
    def __init__(self) -> None:
        self.samples: dict[int, Union[Sample, StopIteration]] = {}
        self.state_dict: Optional[tuple[int, StateDict]] = None

    def load_sample(self, index: int) -> Union[None, Sample, StopIteration]:
        return self.samples.get(index, None)

    def save_sample(self, index: int, sample: Union[Sample, StopIteration]) -> None:
        self.samples[index] = sample

    def load_state_dict(self) -> Optional[tuple[int, StateDict]]:
        return self.state_dict

    def save_state_dict(self, sample_index: int, state_dict: StateDict) -> None:
        self.state_dict = (sample_index, state_dict)


def _initialize_iterator(
    source: CheckpointableDataset, storage: CacheStorage, sample_index: int
) -> CheckpointableIterator:
    cached_sample_index, cached_state_dict = storage.load_state_dict() or (0, None)
    if cached_sample_index > sample_index:
        cached_sample_index = 0
        cached_state_dict = None

    it = source.iter(state_dict=cached_state_dict)

    # Fast-forward iterator to the correct position
    while cached_sample_index < sample_index:
        next(it)
        cached_sample_index += 1

    return it


class CacheIterator(CheckpointableIterator):
    def __init__(
        self, dataset: CacheDataset, index: int, source: Optional[CheckpointableIterator]
    ):
        self.dataset = dataset
        self.index = index
        self.source = source

    def __next__(self) -> Sample:
        # TODO: store state_dict in cache for specified frequency
        storage = self.dataset.storage

        # Read from cache
        if self.source is None:
            sample = storage.load_sample(self.index)
            if isinstance(sample, StopIteration):
                raise StopIteration()
            elif sample is not None:
                self.index += 1
                return sample
            else:
                # Cache exhausted, initialize an iterator and switch to generate from it
                self.source = _initialize_iterator(
                    self.dataset.source, self.dataset.storage, self.index
                )

        # Read from source
        assert self.source
        try:
            sample = next(self.source)
            storage.save_sample(self.index, sample)
            self.index += 1
            return sample
        except StopIteration:
            storage.save_sample(self.index, StopIteration())
            raise

    def state_dict(self) -> StateDict:
        state_dict = {
            "index": self.index,
            "source": self.source.state_dict() if self.source else None,
        }
        return state_dict

    def close(self) -> None:
        if self.source is not None:
            self.source.close()


class CacheDataset(CheckpointableDataset):
    def __init__(self, source: CheckpointableDataset):
        self.source = source
        self.storage = InMemoryCacheStorage()  # TODO: customizable

    def iter(self, state_dict: Optional[dict[str, Any]] = None) -> CheckpointableIterator:
        if state_dict is not None:
            index = state_dict.pop("index")
            source_state_dict = state_dict.pop("source")
            if state_dict:
                raise ValueError(f"Unexpected keys in state_dict: {state_dict.keys()}")
            if source_state_dict is None:
                source = None
            else:
                source = self.source.iter(state_dict=source_state_dict)
        else:
            index = 0
            source = None

        return CacheIterator(self, index, source)
