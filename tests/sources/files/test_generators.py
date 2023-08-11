from __future__ import annotations

from epochraft.sources.files.generators import (
    _yield_samples_cbor,
    _yield_samples_jsonl,
    yield_samples,
)

from .conftest import LOREM


def test_yield_samples_jsonl(jsonl_paths: list[str]) -> None:
    samples = list(_yield_samples_jsonl(jsonl_paths[0]))
    assert [sample["text"] for sample in samples] == LOREM


def test_yield_samples_jsonl_small_buffer(jsonl_paths: list[str]) -> None:
    samples = list(_yield_samples_jsonl(jsonl_paths[0], buffer_size=5))
    assert [sample["text"] for sample in samples] == LOREM


def test_yield_samples_jsonl_with_skip(jsonl_paths: list[str]) -> None:
    samples = list(_yield_samples_jsonl(jsonl_paths[0], n_samlpes_to_skip=2))
    assert [sample["text"] for sample in samples] == LOREM[2:]

    samples = list(_yield_samples_jsonl(jsonl_paths[0], n_samlpes_to_skip=5))
    assert [sample["text"] for sample in samples] == []


def test_yield_samples_cbor(cbor_paths: list[str]) -> None:
    samples = list(_yield_samples_cbor(cbor_paths[0]))
    assert [sample["text"] for sample in samples] == LOREM


def test_yield_samples_cbor_with_skip(cbor_paths: list[str]) -> None:
    samples = list(_yield_samples_cbor(cbor_paths[0], n_samples_to_skip=2))
    assert [sample["text"] for sample in samples] == LOREM[2:]

    samples = list(_yield_samples_cbor(cbor_paths[0], n_samples_to_skip=5))
    assert [sample["text"] for sample in samples] == []


def test_yield_samples(jsonl_paths: list[str], cbor_paths: list[str]) -> None:
    samples = list(yield_samples(jsonl_paths[0]))
    assert [sample["text"] for sample in samples] == LOREM

    samples = list(yield_samples(cbor_paths[0], n_samples_to_skip=2))
    assert [sample["text"] for sample in samples] == LOREM[2:]
