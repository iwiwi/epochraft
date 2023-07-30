import random
from typing import Optional, Sequence

from ..base import CheckpointableDataset, CheckpointableIterator, Sample, StateDict


class SequenceIterator(CheckpointableIterator):
    def __init__(self, dataset: "SequenceDataset", start_index: int) -> None:
        self.dataset = dataset
        self.index = start_index
        self.epoch = 0
        self.rng = random.Random(dataset.shuffle_seed)
        self.order = list(range(len(self.dataset.sequence)))
        self._reshuffle()

    def _reshuffle(self) -> None:
        if self.dataset.shuffle:
            self.rng.shuffle(self.order)

    def __next__(self) -> Sample:
        while True:
            index_in_epoch = self.index - self.epoch * len(self.dataset.sequence)
            if index_in_epoch < len(self.dataset.sequence):
                break

            if not self.dataset.repeat:
                raise StopIteration()

            self.epoch += 1
            self._reshuffle()

        sample = self.dataset.sequence[self.order[index_in_epoch]]
        self.index += 1
        return sample

    def state_dict(self) -> StateDict:
        return {
            "index": self.index,
        }


class SequenceDataset(CheckpointableDataset):
    def __init__(
        self,
        sequence: Sequence[Sample],
        repeat: bool,
        shuffle: bool,
        shuffle_seed: int,
    ) -> None:
        self.sequence = sequence
        self.repeat = repeat
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed

    def iter(self, state_dict: Optional[StateDict] = None) -> CheckpointableIterator:
        if state_dict:
            if set(state_dict.keys()) != {"index"}:
                raise ValueError(
                    f"Invalid state_dict (expect keys: 'index', keys: {state_dict.keys()})"
                )
            index = state_dict["index"]
        else:
            index = 0

        return SequenceIterator(self, start_index=index)
