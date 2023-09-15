from __future__ import annotations

import time
from logging import getLogger

from ...base import CheckpointableIterator, FileFormat, Sample, StateDict
from .generators import yield_samples


logger = getLogger()


class ShardReader(CheckpointableIterator):
    def __init__(
        self,
        url: str,
        format: FileFormat,
        timeout: float,
        n_prefetch_samples: int,
        n_samples_yielded: int,
        epoch: int,
        index_in_epoch: int,
        max_retries: int = 10,
        wait_time_base: float = 1.0,
        wait_time_max: float = 600.0,
        wait_time_multiplier: float = 2.0,
    ):
        self.url = url
        self.format = format
        self.timeout = timeout
        self.n_prefetch_samples = n_prefetch_samples
        self.epoch = epoch
        self.index_in_epoch = index_in_epoch
        self.max_retries = max_retries
        self.wait_time_base = wait_time_base
        self.wait_time_max = wait_time_max
        self.wait_time_multiplier = wait_time_multiplier

        self.n_samples_yielded = n_samples_yielded
        self.iter = yield_samples(
            self.url,
            self.format,
            self.n_samples_yielded,
            self.n_prefetch_samples,
            self.timeout,
        )

    def __next__(self) -> Sample:
        # For exponential backoff
        wait_time = self.wait_time_base

        n_retries = 0
        while n_retries < self.max_retries:
            try:
                sample = next(self.iter)
                self.n_samples_yielded += 1
                wait_time = self.wait_time_base  # Reset exponential backoff
                return sample
            except StopIteration:
                raise
            except Exception:
                n_retries += 1
                logger.exception(
                    f"Error while reading from {self.url}. "
                    f"Retry {n_retries}/{self.max_retries} in {wait_time} seconds."
                )
                time.sleep(wait_time)
                wait_time = min(self.wait_time_max, wait_time * self.wait_time_multiplier)

                self.iter = yield_samples(
                    self.url,
                    self.format,
                    self.n_samples_yielded,
                    self.n_prefetch_samples,
                    self.timeout,
                )

        raise Exception(f"Failed to read from {self.url} after {self.max_retries} attempts.")

    def state_dict(self) -> StateDict:
        return {
            "url": self.url,
            "format": self.format,
            "n_samples_yielded": self.n_samples_yielded,
            "epoch": self.epoch,
            "index_in_epoch": self.index_in_epoch,
        }

    @staticmethod
    def from_state_dict(
        state_dict: StateDict,
        timeout: float,
        n_prefetch_samples: int,
    ) -> ShardReader:
        url = state_dict.pop("url")
        format = state_dict.pop("format")
        n_samples_yielded = state_dict.pop("n_samples_yielded")
        epoch = state_dict.pop("epoch")
        index_in_epoch = state_dict.pop("index_in_epoch")
        if state_dict:
            raise ValueError(f"Unused keys in state_dict: {state_dict.keys()}")

        return ShardReader(
            url=url,
            format=format,
            n_samples_yielded=n_samples_yielded,
            timeout=timeout,
            n_prefetch_samples=n_prefetch_samples,
            epoch=epoch,
            index_in_epoch=index_in_epoch,
        )

    def close(self) -> None:
        # TODO: Close the underlying file handle
        pass
