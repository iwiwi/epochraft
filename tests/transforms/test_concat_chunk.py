import numpy as np
import pytest
import torch

from cpds import CheckpointableDataset, TokenArray, testing
from cpds.transforms.concat_chunk import tensor_from_token_array


@pytest.mark.parametrize(
    "token_array, expected_output",
    [
        # Testing list of integers
        ([1, 2, 3], torch.tensor([1, 2, 3], dtype=torch.long)),
        # Testing numpy array of integers
        (np.array([1, 2, 3]), torch.tensor([1, 2, 3], dtype=torch.long)),
        # Testing numpy array of int8
        (np.array([1, 2, 3], dtype=np.int8), torch.tensor([1, 2, 3], dtype=torch.long)),
        # Testing numpy array of int16
        (np.array([1, 2, 3], dtype=np.int16), torch.tensor([1, 2, 3], dtype=torch.long)),
        # Testing numpy array of int32
        (np.array([1, 2, 3], dtype=np.int32), torch.tensor([1, 2, 3], dtype=torch.long)),
        # Testing numpy array of int64
        (np.array([1, 2, 3], dtype=np.int64), torch.tensor([1, 2, 3], dtype=torch.long)),
        # Testing tensor of integers
        (torch.tensor([1, 2, 3], dtype=torch.long), torch.tensor([1, 2, 3], dtype=torch.long)),
        # Testing empty list
        ([], torch.tensor([], dtype=torch.long)),
        # Testing empty numpy array
        (np.array([], dtype=np.int64), torch.tensor([], dtype=torch.long)),
        # Testing empty tensor
        (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)),
    ],
)
def test_token_from_tensor_array_valid(
    token_array: TokenArray, expected_output: torch.LongTensor
) -> None:
    assert torch.all(tensor_from_token_array(token_array) == expected_output)


@pytest.mark.parametrize(
    "token_array",
    [
        # Testing numpy array of float
        np.array([1.0, 2.0, 3.0]),
        # Testing tensor of float
        torch.tensor([1.0, 2.0, 3.0]),
        # Testing multi-dimensional numpy array
        np.array([[1, 2, 3], [4, 5, 6]]),
        # Testing multi-dimensional tensor
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
        # Testing zero-dimensional numpy array
        np.array(1),
        # Testing zero-dimensional tensor
        torch.tensor(1, dtype=torch.long),
        # Testing zero-length numpy array
        np.array([], dtype=np.float64),
        # Testing zero-length tensor
        torch.tensor([], dtype=torch.float64),
        # Testing 2D zero-length numpy array
        np.array([[], []], dtype=np.int64),
        # Testing 2D zero-length tensor
        torch.tensor([[], []], dtype=torch.long),
    ],
)
def test_token_from_tensor_array_invalid(token_array: TokenArray) -> None:
    with pytest.raises(ValueError):
        tensor_from_token_array(token_array)


def test_concat_chunk_simple() -> None:
    samples = [
        {"input_ids": [0, 1, 2]},
        {"input_ids": [3]},
        {"input_ids": [4, 5, 6, 7, 8, 9, 10, 11]},
    ]
    dataset = CheckpointableDataset.from_sequence(samples).concat_chunk(
        chunk_length=4,
        bos_tokens=[-1],
        eos_tokens=[-2],
    )
    chunks = list(dataset)

    # Check that the number of chunks is correct
    assert len(chunks) == 4

    # Check that the chunks are dictionaries with the correct keys and dtypes
    for chunk in chunks:
        assert isinstance(chunk, dict)
        assert set(chunk.keys()) == {"input_ids"}
        assert chunk["input_ids"].dtype == torch.long
        assert chunk["input_ids"].dim() == 1

    # Check that the chunks are correct
    assert chunks[0]["input_ids"].tolist() == [-1, 0, 1, 2]
    assert chunks[1]["input_ids"].tolist() == [-2, -1, 3, -2]
    assert chunks[2]["input_ids"].tolist() == [-1, 4, 5, 6]
    assert chunks[3]["input_ids"].tolist() == [7, 8, 9, 10]


def test_concat_chunk_resumption() -> None:
    samples_original = testing.generate_tokenized_samples()
    dataset = CheckpointableDataset.from_sequence(
        samples_original, repeat=True, shuffle=True
    ).concat_chunk(
        chunk_length=5,
        eos_tokens=[-1],
        bos_tokens=[-2],
    )

    testing.check_resumption(dataset, dataset, 0)
    testing.check_resumption(dataset, dataset, 1)
    testing.check_resumption(dataset, dataset, 10)
    testing.check_resumption(dataset, dataset, 20)
    testing.check_resumption(dataset, dataset, 123)
