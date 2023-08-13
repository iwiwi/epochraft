from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from ...base import CheckpointableDataset, ParallelExecutorType, Sample


if TYPE_CHECKING:
    from ...base import Tokenizer


def _map_fn(
    sample: Sample, tokenizer: Tokenizer, target_column: str, tokenizer_kwargs: dict[str, Any]
) -> Sample:
    sample = sample.copy()
    sample.update(tokenizer(sample[target_column], **tokenizer_kwargs))
    return sample


def tokenize(
    source: CheckpointableDataset,
    tokenizer: Tokenizer,
    tokenizer_kwargs: Optional[Dict[str, Any]],
    target_column: str,
    parallel: bool,
    max_workers: Optional[int],
    prefetch_factor: int,
    ordered: bool,
    executor_type: ParallelExecutorType,
) -> CheckpointableDataset:
    fn = functools.partial(
        _map_fn,
        tokenizer=tokenizer,
        target_column=target_column,
        tokenizer_kwargs=tokenizer_kwargs or {},
    )

    if parallel:
        # TODO: show some warning on this
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        return source.parallel_map(
            fn,
            max_workers=max_workers,
            prefetch_factor=prefetch_factor,
            ordered=ordered,
            executor_type=executor_type,
        )
    else:
        return source.map(fn)
