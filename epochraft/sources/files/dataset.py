from __future__ import annotations

from typing import Optional, Sequence, Union

import braceexpand

from ...base import CheckpointableDataset, CheckpointableIterator, FileFormat, StateDict
from .shards_mux import ShardsMux


class FilesDataset(CheckpointableDataset):
    def __init__(
        self,
        urls: Union[str, Sequence[str]],
        format: FileFormat,
        repeat: bool,
        shuffle_shards: bool,
        n_active_shards: int,
        n_standby_shards: int,
        timeout: float,
        n_prefetch_samples: int,
        seed: int,
    ) -> None:
        if isinstance(urls, str):
            urls = [urls]
        self.urls: list[str] = sum((list(braceexpand.braceexpand(url)) for url in urls), [])

        self.format = format
        self.n_acive_shards = n_active_shards
        self.n_standby_shards = n_standby_shards
        self.timeout = timeout
        self.n_prefetch_samples = n_prefetch_samples
        self.repeat = repeat
        self.shuffle_shards = shuffle_shards
        self.seed = seed

    def iter(self, state_dict: Optional[StateDict] = None) -> CheckpointableIterator:
        if state_dict is not None:
            next_active_shard = state_dict.pop("next_active_shard")
            active_shard_states = state_dict.pop("active_shards")
            if state_dict:
                raise ValueError(f"Unexpected keys in state_dict: {state_dict.keys()}")
        else:
            next_active_shard = (0, 0)
            active_shard_states = None

        return ShardsMux(
            urls=self.urls,
            format=self.format,
            repeat=self.repeat,
            shuffle=self.shuffle_shards,
            n_active_shards=self.n_acive_shards,
            n_standby_shards=self.n_standby_shards,
            timeout=self.timeout,
            n_prefetch_samples=self.n_prefetch_samples,
            seed=self.seed,
            next_active_shard=next_active_shard,
            active_shard_states=active_shard_states,
        )
