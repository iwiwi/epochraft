from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

from ...base import TokenArray


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


TOKENIZATION_EXAMPLE_STR = "Hello world!"


@dataclasses.dataclass
class TokenizerBehavior:
    bos_token: str
    eos_token: str
    bos_token_id: int
    eos_token_id: int

    # Some tokenizers (like `EleutherAI/gpt-neox-20b`) have the same token for bos and eos.
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


def tensor_from_token_array(data: Optional[Union[int, TokenArray]]) -> torch.Tensor:
    if data is None:
        data = []
    if isinstance(data, int):
        data = [data]

    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, np.ndarray):
        if np.issubdtype(data.dtype, np.integer):
            tensor = torch.from_numpy(data).long()
        else:
            raise ValueError(f"Expected integer ndarray, got ndarray of {data.dtype}")
    else:
        tensor = torch.tensor(data, dtype=torch.long)

    if tensor.dtype != torch.long:
        raise ValueError(f"Expected long tensor, got {tensor.dtype}")

    if tensor.dim() == 2:
        if tensor.shape[0] != 1:
            raise ValueError(
                "input_ids must be 1-dimensional tensor or 2-dimensional tensor with batch size 1"
            )
        tensor = tensor[0]
    elif tensor.dim() != 1:
        raise ValueError("input_ids must be 1-dimensional tensor")
    return tensor
