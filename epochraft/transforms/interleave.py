from __future__ import annotations

from typing import Optional

from ..base import (CheckpointableDataset, CheckpointableIterator, Sample,
                    StateDict)


def generate_stratified_sampling_order(weights: list[float], length: int) -> list[int]:
    total_weights = sum(weights)
    weights = [weight / total_weights for weight in weights]

    order = []
    score = [0.0] * len(weights)
    for _ in range(length):
        for i, w in enumerate(weights):
            score[i] += w

        source_id = max(range(len(weights)), key=lambda i: score[i])
        score[source_id] -= 1.0
        order.append(source_id)

    return order


class InterleaveIterator(CheckpointableIterator):
    def __init__(
        self, dataset: Interleave, sources: list[CheckpointableIterator], start_index: int
    ) -> None:
        self.dataset = dataset
        self.sources = sources
        self.index = start_index
        self.source_ids = generate_stratified_sampling_order(
            self.dataset.weights, self.dataset.chunk_size
        )

    def __next__(self) -> Sample:
        source_id = self.source_ids[self.index % self.dataset.chunk_size]
        self.index += 1

        # TODO: we may want to have option to continue generation until all sources are exhausted
        # (currently we stop when the first source is exhausted)
        return next(self.sources[source_id])

    def state_dict(self) -> StateDict:
        state_dict: StateDict = {
            f"source{source_id}": source.state_dict()
            for source_id, source in enumerate(self.sources)
        }
        state_dict["index"] = self.index
        return state_dict


class Interleave(CheckpointableDataset):
    def __init__(
        self,
        sources: list[CheckpointableDataset],
        weights: Optional[list[float]],
        chunk_size: int = 1024,
    ):
        if weights is not None:
            if len(weights) != len(sources):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match number"
                    f" of sources ({len(sources)})"
                )
            if sum(weights) == 0:
                raise ValueError("Sum of weights must be greater than 0")

        self.sources = sources
        self.weights = weights or ([1] * len(sources))
        self.chunk_size = chunk_size

    def iter(self, state_dict: Optional[StateDict] = None) -> InterleaveIterator:
        if state_dict:
            source_state_dicts = [
                state_dict.pop(f"source{source_id}") for source_id in range(len(self.sources))
            ]
            index = state_dict.pop("index")
            if state_dict:
                raise ValueError(f"Unexpected keys in state_dict: {state_dict.keys()}")
        else:
            source_state_dicts = [None] * len(self.sources)
            index = 0

        iters = [
            source.iter(state_dict=state_dict)
            for source, state_dict in zip(self.sources, source_state_dicts)
        ]
        return InterleaveIterator(self, iters, start_index=index)


def interleave(
    sources: list[CheckpointableDataset],
    weights: Optional[list[float]] = None,
    chunk_size: int = 1024,
) -> CheckpointableDataset:
    return Interleave(sources, weights, chunk_size=chunk_size)
