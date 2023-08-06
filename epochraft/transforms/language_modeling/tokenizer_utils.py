from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch

from ...base import Sample, TokenArray


def tensor_from_token_array(data: Optional[Union[int, TokenArray]]) -> torch.Tensor:
    """
    Converts token arrays, typically the output from tokenizers, into a 1D PyTorch tensor.

    This function handles multiple input types including integers, torch tensors, and numpy arrays.
    The goal is to produce a consistent 1D tensor output for token data.

    Args:
        data (Optional[Union[int, TokenArray]]): Token data to be converted. It can be `None`,
        an integer, a torch tensor, or a numpy array.

    Returns:
        torch.Tensor: A 1-dimensional tensor representation of the input token data.

    Raises:
        ValueError: If the provided numpy array is not of an integer type.
        ValueError: If the resulting tensor is not of dtype `torch.long`.
        ValueError: If the tensor dimensions are not compatible with the expected shape.
    """
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


class BufferDict:
    """
    A buffer dictionary class to manage, concatenate, pack, and chunk token arrays on the fly.
    """

    def __init__(self, columns: Sequence[str], buffers: Optional[dict[str, torch.Tensor]]) -> None:
        if len(columns) == 0:
            raise ValueError("columns must be non-empty")

        self.columns = columns

        if buffers is not None:
            if set(columns) != set(buffers.keys()):
                raise ValueError(
                    f"columns and buffers keys must match: {columns} != {buffers.keys()}"
                )
            self.buffers = buffers
        else:
            self.buffers = {column: torch.empty(0, dtype=torch.long) for column in columns}

    def buffer_length(self) -> int:
        return len(next(iter(self.buffers.values())))

    def tensor_dict_from_sample(self, sample: Sample) -> dict[str, torch.Tensor]:
        return {column: tensor_from_token_array(sample[column]) for column in self.columns}

    def append_from_tensor_dict(self, tensor_dict: dict[str, torch.Tensor]) -> None:
        input_length: Optional[int] = None
        for column in self.columns:
            tokens = tensor_dict[column]

            if input_length is None:
                input_length = len(tokens)
            else:
                if input_length != len(tokens):
                    raise ValueError("All columns must have the same length")

            self.buffers[column] = torch.cat([self.buffers[column], tokens])

    def append_from_sample(self, sample: Sample) -> None:
        self.append_from_tensor_dict(self.tensor_dict_from_sample(sample))

    def take(self, length: int) -> Sample:
        output = {column: self.buffers[column][:length] for column in self.columns}
        self.buffers = {column: self.buffers[column][length:] for column in self.columns}
        return output

    def take_all(self) -> Sample:
        output = {column: self.buffers[column] for column in self.columns}
        self.buffers = {column: torch.empty(0, dtype=torch.long) for column in self.columns}
        return output
