from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..base import (CheckpointableDataset, CheckpointableIterator, Sample,
                    StateDict)

if TYPE_CHECKING:
    import streaming


class MosaicmlIterator(CheckpointableIterator):
    def __init__(self, dataset: MosaicmlDataset, start_index: int) -> None:
        self.dataset = dataset
        self.iter = iter(dataset.source)
        self.index = start_index

    def __next__(self) -> Sample:
        self.index += 1

        try:
            sample = next(self.iter)
        except StopIteration:
            if not self.dataset.repeat:
                raise

            # Starting the next epoch
            self.iter = iter(self.dataset.source)
            sample = next(self.iter)

        return sample

    def state_dict(self) -> StateDict:
        # The StreamingDataset in MosaicML is semi-stateful. It remembers the epoch, but it doesn't
        # remember the position within an epoch. We need to provide the position within the epoch.
        index_in_epoch = self.index % len(self.dataset.source)
        return {
            "index": self.index,
            "dataset": self.dataset.source.state_dict(index_in_epoch, from_beginning=True),
        }


class MosaicmlDataset(CheckpointableDataset):
    def __init__(self, source: streaming.StreamingDataset, repeat: bool) -> None:
        self.source = source
        self.repeat = repeat
        self.iter_called = False

    def iter(self, state_dict: Optional[dict[str, Any]] = None) -> CheckpointableIterator:
        if state_dict:
            if self.iter_called:
                msg = (
                    "At least in the current version (0.5.1), in order to resume a MosaicML's "
                    "StreamingDataset with a state_dict, it looks that it must not have accessed "
                    "the data before. The state_dict is ignored for the subsequent calls."
                )
                raise ValueError(msg)

            self.source.load_state_dict(state_dict["dataset"])
            index = state_dict["index"]
        else:
            index = 0

        self.iter_called = True
        return MosaicmlIterator(self, start_index=index)
