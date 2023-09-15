from __future__ import annotations

import random
from collections import deque
from logging import getLogger
from typing import Generic, Optional, Sequence, TypeVar

from ...base import CheckpointableIterator, FileFormat, Sample, StateDict
from .shard_reader import ShardReader


logger = getLogger()


T = TypeVar("T")


class EpochShuffleList(Generic[T]):
    def __init__(self, items: Sequence[T], shuffle: bool, seed: int) -> None:
        self.items = list(items)
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.epoch = 0

        if shuffle:
            self.rng.shuffle(self.items)

    def __getitem__(self, index: int) -> T:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    def advance_epoch(self, epoch: int) -> None:
        if self.epoch > epoch:
            raise ValueError(f"Cannot decrease epoch from {self.epoch} to {epoch}")

        while self.epoch < epoch:
            if self.shuffle:
                self.rng.shuffle(self.items)
            self.epoch += 1


class ShardsMux(CheckpointableIterator):
    def __init__(
        self,
        urls: Sequence[str],
        format: FileFormat,
        repeat: bool,
        shuffle: bool,
        n_active_shards: int,
        n_standby_shards: int,
        timeout: float,
        n_prefetch_samples: int,
        seed: int,
        next_active_shard: Optional[tuple[int, int]] = None,  # (epoch, index in epoch)
        active_shard_states: Optional[Sequence[StateDict]] = None,
    ) -> None:
        self.urls = EpochShuffleList(urls, shuffle=shuffle, seed=seed)
        self.format = format
        self.repeat = repeat
        self.shuffle = shuffle
        self.timeout = timeout
        self.n_prefetch_samples = n_prefetch_samples
        self.n_active_shards = min(n_active_shards, len(self.urls))

        # When we handle the boundary of epochs, we have smaller number of active shards.
        # So, instead of limiting the number of stand-by shards,
        # we limit the total number of open shards.
        self.n_open_shards = self.n_active_shards + n_standby_shards

        # We have `next_active_shard` as an argument, because it is more clear for checkpointing.
        # However, internally, it is more useful to manage `next_standby_shard`.
        # As the stand-by queue is initially empty, `next_standby_shard` is `next_active_shard`,
        # because the stand-by queue and we will refill the active shards from the stand-by queue.
        self.next_standby_shard = next_active_shard or (0, 0)

        self.active_shards = deque(
            ShardReader.from_state_dict(
                state_dict=state_dict, timeout=timeout, n_prefetch_samples=n_prefetch_samples
            )
            for state_dict in (active_shard_states or [])
        )
        self.standby_shards: deque[ShardReader] = deque()

        # Start the prefetching
        self._refill_active_shards()

    def _refill_standby_shards(self) -> None:
        while len(self.active_shards) + len(self.standby_shards) < self.n_open_shards:
            epoch, index = self.next_standby_shard
            if index >= len(self.urls):
                self.next_standby_shard = (epoch + 1, 0)
                continue
            if epoch > 0 and not self.repeat:
                break

            self.urls.advance_epoch(epoch)
            assert self.urls.epoch == epoch
            url = self.urls[index]

            logger.debug(f"New stand-by shard: {url}")
            shard_reader = ShardReader(
                url,
                self.format,
                timeout=self.timeout,
                n_prefetch_samples=self.n_prefetch_samples,
                n_samples_yielded=0,
                epoch=epoch,
                index_in_epoch=index,
            )
            self.standby_shards.append((shard_reader))
            self.next_standby_shard = (epoch, index + 1)

    def _refill_active_shards(self) -> None:
        self._refill_standby_shards()
        while len(self.active_shards) < self.n_active_shards:
            if len(self.standby_shards) == 0:
                break
            standby_shard = self.standby_shards[0]

            # We don't want to mix shards from different epochs
            if len(self.active_shards) == 0:
                epoch = None
            else:
                epoch = self.active_shards[0].epoch
            if epoch is not None and epoch != standby_shard.epoch:
                break

            # Move the shard from the stand-by queue to the active shards
            logger.debug(f"New active shard: {standby_shard.url}")
            self.standby_shards.popleft()
            self.active_shards.append(standby_shard)

            self._refill_standby_shards()

    def __next__(self) -> Sample:
        while True:
            self._refill_active_shards()

            # Done with all shards?
            if len(self.active_shards) == 0:
                raise StopIteration

            shard = self.active_shards.popleft()
            try:
                sample = next(shard)
                self.active_shards.append(shard)
                return sample
            except StopIteration:
                logger.debug(f"Complete shard: {shard.url}")
                continue

    @property
    def _next_active_shard(self) -> tuple[int, int]:
        if self.standby_shards:
            return (self.standby_shards[0].epoch, self.standby_shards[0].index_in_epoch)
        else:
            return self.next_standby_shard

    def state_dict(self) -> StateDict:
        return {
            "next_active_shard": self._next_active_shard,
            "active_shards": [active_shard.state_dict() for active_shard in self.active_shards],
        }

    def close(self) -> None:
        for shard in self.active_shards:
            shard.close()
        for shard in self.standby_shards:
            shard.close()
