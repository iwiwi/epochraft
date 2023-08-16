from __future__ import annotations

from epochraft.sources.files.generators import yield_samples

from .conftest import LOREM


def test_yield_samples_jsonl(jsonl_paths: list[str]) -> None:
    samples = list(
        yield_samples(
            jsonl_paths[0],
            format="auto",
            n_samples_to_skip=0,
            timeout=10.0,
            n_prefetch_samples=1,
        )
    )
    assert [sample["text"] for sample in samples] == LOREM


def test_yield_samples_jsonl_with_skip(jsonl_paths: list[str]) -> None:
    samples = list(
        yield_samples(
            jsonl_paths[0],
            format="auto",
            n_samples_to_skip=2,
            timeout=10.0,
            n_prefetch_samples=1,
        )
    )
    assert [sample["text"] for sample in samples] == LOREM[2:]

    samples = list(
        yield_samples(
            jsonl_paths[0],
            format="auto",
            n_samples_to_skip=5,
            timeout=10.0,
            n_prefetch_samples=1,
        )
    )
    assert [sample["text"] for sample in samples] == []


def test_yield_samples_cbor(cbor_paths: list[str]) -> None:
    samples = list(
        yield_samples(
            cbor_paths[0],
            format="auto",
            n_samples_to_skip=0,
            timeout=10.0,
            n_prefetch_samples=1,
        )
    )
    assert [sample["text"] for sample in samples] == LOREM


def test_yield_samples_cbor_with_skip(cbor_paths: list[str]) -> None:
    samples = list(
        yield_samples(
            cbor_paths[0],
            format="auto",
            n_samples_to_skip=2,
            timeout=10.0,
            n_prefetch_samples=1,
        )
    )
    assert [sample["text"] for sample in samples] == LOREM[2:]

    samples = list(
        yield_samples(
            cbor_paths[0],
            format="auto",
            n_samples_to_skip=5,
            timeout=10.0,
            n_prefetch_samples=1,
        )
    )
    assert [sample["text"] for sample in samples] == []
