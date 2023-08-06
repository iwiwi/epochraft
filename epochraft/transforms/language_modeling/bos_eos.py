from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

import torch

from ...base import CheckpointableDataset, Sample
from .tokenizer_utils import tensor_from_token_array


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


TOKENIZATION_EXAMPLE_STR = "Hello world!"


@dataclasses.dataclass
class TokenizerBehavior:
    bos_token: Optional[str]
    eos_token: Optional[str]
    bos_token_id: Optional[int]
    eos_token_id: Optional[int]

    # Some tokenizers (like "gpt-neox-20b" and "gpt2") have the same token for bos and eos.
    bos_eos_equal: bool

    # Tokenization result for `TOKENIZATION_EXAMPLE_STR`
    ids_example: list[int]

    # Does the tokenizer add the BOS and EOS tokens automatically?
    bos_token_added: bool
    eos_token_added: bool

    def from_tokenizer(tokenizer: PreTrainedTokenizerBase) -> TokenizerBehavior:
        ids_example = tokenizer.encode(TOKENIZATION_EXAMPLE_STR)
        if not isinstance(ids_example, list):
            raise ValueError(f"Tokenizer returned {type(ids_example)}, expected list")
        if len(ids_example) == 0:
            raise ValueError(
                f"Tokenizer returned empty list for input string: {TOKENIZATION_EXAMPLE_STR}"
            )

        if tokenizer(TOKENIZATION_EXAMPLE_STR)["input_ids"] != ids_example:
            raise ValueError("Tokenizer returned different results for `encode` and `__call__`")

        if len(ids_example) > 0 and ids_example[0] == tokenizer.bos_token_id:
            bos_token_added = True
        else:
            bos_token_added = False

        if len(ids_example) > 0 and ids_example[-1] == tokenizer.eos_token_id:
            eos_token_added = True
        else:
            eos_token_added = False

        return TokenizerBehavior(
            bos_token=tokenizer.bos_token,
            eos_token=tokenizer.eos_token,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_eos_equal=(tokenizer.bos_token_id == tokenizer.eos_token_id),
            ids_example=ids_example,
            bos_token_added=bos_token_added,
            eos_token_added=eos_token_added,
        )


def add_bos_eos(
    source: CheckpointableDataset,
    bos_token_id: Optional[int],
    eos_token_id: Optional[int],
    target_column: str,
) -> CheckpointableDataset:
    if bos_token_id is not None:
        bos_tensor = tensor_from_token_array([bos_token_id])
    else:
        bos_tensor = tensor_from_token_array([])

    if eos_token_id is not None:
        eos_tensor = tensor_from_token_array([eos_token_id])
    else:
        eos_tensor = tensor_from_token_array([])

    def _fn(sample: Sample) -> Sample:
        sample = sample.copy()
        tokens = tensor_from_token_array(sample[target_column])
        sample[target_column] = torch.cat((bos_tensor, tokens, eos_tensor))
        return sample

    return source.map(_fn)


def ensure_bos_eos(
    source: CheckpointableDataset,
    tokenizer: PreTrainedTokenizerBase,
    target_column: str,
) -> CheckpointableDataset:
    behavior = TokenizerBehavior.from_tokenizer(tokenizer)

    # Some tokenizers (like "gpt2" and "gpt-neox-20b") have the same token for bos and eos.
    # In this case, we don't have to add both.
    if behavior.bos_eos_equal:
        if behavior.bos_token_added or behavior.eos_token_added:
            # Either of BOS or EOS is already added, so we don't need to add anything.
            return source
        else:
            # Either of BOS or EOS should be added, but not both.
            # Here, we choose to add EOS.
            return add_bos_eos(source, None, tokenizer.eos_token_id, target_column)
    else:
        if not behavior.bos_token_added and tokenizer.bos_token:
            bos_token_to_add = tokenizer.bos_token_id
        else:
            bos_token_to_add = None

        if not behavior.eos_token_added and tokenizer.eos_token:
            eos_token_to_add = tokenizer.eos_token_id
        else:
            bos_token_to_add = None

        return add_bos_eos(source, bos_token_to_add, eos_token_to_add, target_column)
