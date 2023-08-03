from typing import Iterable, Iterator, Optional

from ..base import CheckpointableDataset, CheckpointableIterator, Sample, StateDict


class IterableIterator(CheckpointableIterator):
    def __init__(self, dataset: "IterableDataset", start_index: int) -> None:
        self.dataset = dataset
        self.index = start_index
        self.source_iter = self._initialize_source_iter(start_index)

    def _initialize_source_iter(self, start_index: int) -> Iterator[Sample]:
        it = iter(self.dataset.iterable)

        for _ in range(start_index):
            try:
                next(it)
            except StopIteration:
                raise ValueError(f"start_index larger than iterable length: {start_index}")

        return it

    def __next__(self) -> Sample:
        try:
            sample = next(self.source_iter)
            self.index += 1
            return sample
        except StopIteration:
            if not self.dataset.repeat:
                raise StopIteration()

            self.source_iter = iter(self.dataset.iterable)
            sample = next(self.source_iter)
            self.index = 1
            return sample

    def state_dict(self) -> StateDict:
        return {
            "index": self.index,
        }


class IterableDataset(CheckpointableDataset):
    def __init__(
        self,
        iterable: Iterable[Sample],
        repeat: bool,
    ) -> None:
        self.iterable = iterable
        self.repeat = repeat

    def iter(self, state_dict: Optional[StateDict] = None) -> CheckpointableIterator:
        if state_dict:
            if set(state_dict.keys()) != {"index"}:
                raise ValueError(
                    f"Invalid state_dict (expect keys: 'index', keys: {state_dict.keys()})"
                )
            index = state_dict["index"]
        else:
            index = 0

        return IterableIterator(self, start_index=index)
