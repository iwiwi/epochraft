from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Sequence, Union, overload

from epochraft import CheckpointableDataset, Sample, testing


class ExampleSequence(Sequence[Sample]):
    def __init__(self, n: int) -> None:
        self.n = n
        self.called: defaultdict[int, int] = defaultdict(int)

    @overload
    def __getitem__(self, index: int) -> Sample:
        ...

    @overload
    def __getitem__(self, s: slice) -> Sequence[Sample]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Sample, Sequence[Sample]]:
        if isinstance(index, slice):
            raise NotImplementedError()

        self.called[index] += 1

        if index >= self.n:
            raise IndexError()
        return {"id": index}

    def __len__(self) -> int:
        return self.n


def test_cache_trivial() -> None:
    seq = ExampleSequence(10)
    dataset = CheckpointableDataset.from_sequence(seq).cache()
    assert all(seq.called[i] == 0 for i in range(10))

    # 1st sweep
    out = list(dataset)
    assert out == [{"id": i} for i in range(10)]
    assert all(seq.called[i] == 1 for i in range(10))

    # 2nd sweep
    out = list(dataset)
    assert out == [{"id": i} for i in range(10)]
    assert all(seq.called[i] == 1 for i in range(10))


def test_cache_midway() -> None:
    seq = ExampleSequence(10)
    dataset = CheckpointableDataset.from_sequence(seq).cache()
    assert all(seq.called[i] == 0 for i in range(10))

    # 1st sweep --- stopping at 5th element
    out = list(itertools.islice(dataset, 5))
    assert out == [{"id": i} for i in range(5)]
    assert all(seq.called[i] == 1 for i in range(5))
    assert all(seq.called[i] == 0 for i in range(5, 10))

    # 2nd sweep
    out = list(dataset)
    assert out == [{"id": i} for i in range(10)]
    assert all(seq.called[i] == 2 for i in range(5))
    assert all(seq.called[i] == 1 for i in range(5, 10))


def test_cache_resumption() -> None:
    seq = ExampleSequence(10)
    dataset = CheckpointableDataset.from_sequence(seq, repeat=True).cache()

    testing.check_resumption(dataset, dataset, 0)
    testing.check_resumption(dataset, dataset, 1)
    testing.check_resumption(dataset, dataset, 2)
    testing.check_resumption(dataset, dataset, 3)
    testing.check_resumption(dataset, dataset, 10)
    testing.check_resumption(dataset, dataset, 100)
