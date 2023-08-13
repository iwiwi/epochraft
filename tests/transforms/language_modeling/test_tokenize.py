from __future__ import annotations

from typing import Optional

import pytest
import torch
from transformers import AutoTokenizer

from epochraft import CheckpointableDataset


SAMPLES = [
    {"text": "hello world"},
    {"text": "this is a long sentence that will be tokenized"},
    {"text": "this is another long sentence that will be tokenized"},
]


def test_tokenize() -> None:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    samples_tokenized = list(
        CheckpointableDataset.from_sequence(SAMPLES).tokenize(tokenizer, max_workers=2)
    )

    assert len(samples_tokenized) == 3
    for sample in samples_tokenized:
        assert sample.keys() == {"text", "input_ids", "attention_mask"}
        assert isinstance(sample["input_ids"], list)
        assert len(sample["input_ids"]) > 0
        assert isinstance(sample["attention_mask"], list)
        assert len(sample["input_ids"]) == len(sample["attention_mask"])


def test_tokenize_return_pt() -> None:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    samples_tokenized = list(
        CheckpointableDataset.from_sequence(SAMPLES).tokenize(
            tokenizer, tokenizer_kwargs={"return_tensors": "pt"}, max_workers=2
        )
    )

    assert len(samples_tokenized) == 3
    for sample in samples_tokenized:
        assert sample.keys() == {"text", "input_ids", "attention_mask"}
        assert isinstance(sample["input_ids"], torch.Tensor)
        assert sample["input_ids"].dim() == 2
        assert sample["input_ids"].shape[0] == 1
        assert sample["input_ids"].shape[1] > 0
        assert isinstance(sample["attention_mask"], torch.Tensor)
        assert sample["input_ids"].shape == sample["attention_mask"].shape


@pytest.mark.parametrize(
    "tokenizer_kwargs",
    [
        None,
        {"return_tensors": "pt"},
        {"return_tensors": "np"},
    ],
)
def test_tokenize_concat_chunk(tokenizer_kwargs: Optional[dict]) -> None:
    chunk_length = 5
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    out_samples = list(
        CheckpointableDataset.from_sequence(SAMPLES)
        .tokenize(tokenizer, tokenizer_kwargs=tokenizer_kwargs, max_workers=2)
        .concat_chunk(chunk_length=chunk_length)
    )
    assert len(out_samples) > 0
    for sample in out_samples:
        assert sample.keys() == {"input_ids"}
        assert isinstance(sample["input_ids"], torch.Tensor)
        assert sample["input_ids"].dim() == 1
        assert sample["input_ids"].shape[0] == chunk_length
