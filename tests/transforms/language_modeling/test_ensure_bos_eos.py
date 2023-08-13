from __future__ import annotations

import torch
from transformers import (
    AutoTokenizer,
    GPT2Tokenizer,
    GPT2TokenizerFast,
    LlamaTokenizer,
    T5Tokenizer,
)

from epochraft import CheckpointableDataset
from epochraft.transforms.language_modeling.bos_eos import TokenizerBehavior


def test_tokenizer_behavior_gpt2() -> None:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    behavior = TokenizerBehavior.from_tokenizer(tokenizer)
    assert behavior.bos_eos_equal
    assert not behavior.bos_token_added
    assert not behavior.eos_token_added


def test_tokenizer_behavior_gpt2_fast() -> None:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    behavior = TokenizerBehavior.from_tokenizer(tokenizer)
    assert behavior.bos_eos_equal
    assert not behavior.bos_token_added
    assert not behavior.eos_token_added


def test_tokenizer_behavior_neox() -> None:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    behavior = TokenizerBehavior.from_tokenizer(tokenizer)
    assert behavior.bos_eos_equal
    assert not behavior.bos_token_added
    assert not behavior.eos_token_added


def test_tokenizer_behavior_novelai() -> None:
    tokenizer = LlamaTokenizer.from_pretrained("NovelAI/nerdstash-tokenizer-v1")
    behavior = TokenizerBehavior.from_tokenizer(tokenizer)
    assert not behavior.bos_eos_equal
    assert behavior.bos_token_added
    assert not behavior.eos_token_added


def test_tokenizer_behavior_t5() -> None:
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    behavior = TokenizerBehavior.from_tokenizer(tokenizer)
    assert not behavior.bos_eos_equal
    assert not behavior.bos_token_added
    assert behavior.eos_token_added


SAMPLES = [
    {"text": "hello world"},
    {"text": "this is a long sentence that will be tokenized"},
    {"text": "this is another long sentence that will be tokenized"},
]


def test_ensure_bos_eos_neox() -> None:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    outputs = list(
        CheckpointableDataset.from_sequence(SAMPLES)
        .tokenize(tokenizer, max_workers=2)
        .ensure_bos_eos(tokenizer)
    )

    assert len(outputs) == 3
    for sample in outputs:
        assert "input_ids" in sample.keys()
        assert isinstance(sample["input_ids"], torch.Tensor)
        assert sample["input_ids"].dim() == 1
        assert sample["input_ids"].shape[0] > 0

        # EOS token should exist
        assert sample["input_ids"][-1] == tokenizer.eos_token_id

        # BOS token shouldn't exist, because BOS == EOS for this tokenizer
        assert sample["input_ids"][0] != tokenizer.bos_token_id


def test_ensure_bos_eos_novelai() -> None:
    tokenizer = LlamaTokenizer.from_pretrained("NovelAI/nerdstash-tokenizer-v1")
    outputs = list(
        CheckpointableDataset.from_sequence(SAMPLES)
        .tokenize(tokenizer, max_workers=2)
        .ensure_bos_eos(tokenizer)
    )

    assert len(outputs) == 3
    for sample in outputs:
        assert "input_ids" in sample.keys()
        assert isinstance(sample["input_ids"], torch.Tensor)
        assert sample["input_ids"].dim() == 1
        assert sample["input_ids"].shape[0] > 0

        # BOS token should exist
        assert sample["input_ids"][0] == tokenizer.bos_token_id

        # EOS token should exist
        assert sample["input_ids"][-1] == tokenizer.eos_token_id
