from __future__ import annotations

from epochraft.sources.files.generators import yield_samples

from .conftest import LOREM


def test_yield_samples_jsonl(jsonl_paths: list[str]) -> None:
    samples = list(yield_samples(jsonl_paths[0]))
    assert [sample["text"] for sample in samples] == LOREM


def test_yield_samples_jsonl_with_skip(jsonl_paths: list[str]) -> None:
    samples = list(yield_samples(jsonl_paths[0], n_samples_to_skip=2))
    assert [sample["text"] for sample in samples] == LOREM[2:]

    samples = list(yield_samples(jsonl_paths[0], n_samples_to_skip=5))
    assert [sample["text"] for sample in samples] == []


def test_yield_samples_cbor(cbor_paths: list[str]) -> None:
    samples = list(yield_samples(cbor_paths[0]))
    assert [sample["text"] for sample in samples] == LOREM


def test_yield_samples_cbor_with_skip(cbor_paths: list[str]) -> None:
    samples = list(yield_samples(cbor_paths[0], n_samples_to_skip=2))
    assert [sample["text"] for sample in samples] == LOREM[2:]

    samples = list(yield_samples(cbor_paths[0], n_samples_to_skip=5))
    assert [sample["text"] for sample in samples] == []
