from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

from ...base import TokenArray


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
