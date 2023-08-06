# Epochraft

[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/optuna/optuna)
[![Checks status](https://github.com/iwiwi/epochraft/actions/workflows/checks.yml/badge.svg?branch=main)](https://github.com/iwiwi/epochraft/actions)
[![Tests status](https://github.com/iwiwi/epochraft/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/iwiwi/epochraft/actions)
[![pypi](https://img.shields.io/pypi/v/epochraft.svg)](https://pypi.python.org/pypi/epochraft)


Supercharge your LLM training with checkpointable data loading.

## Key Features

* **Checkpointing** - Epochraft operates completely deterministically, and allows for a full restoration of state through checkpointing.
* **Simple** - It's a minimally readable implementation that makes it easy for users to add sources and transforms.
* **LLM-Ready** - It is equipped out of the box with features necessary for pre-training and SFT of LLMs.


## Quick Start

### Installation

```
pip install epochraft
```

### Example

This is an example of building a typical pretraining dataset. We will soon add other examples such as SFT.

```python
from datasets import load_dataset
from epochraft import CheckpointableDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

# Various data sources are ssupported. Refer to the explanation below for more details.
source = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)

train_dataset = (
    CheckpointableDataset
    .from_iterable(source, repeat=True)  # Create a CheckpointableDataset from the source
    .tokenize(tokenizer)                 # Tokenize the texts
    .ensure_bos_eos(tokenizer)           # Add BOS and EOS tokens where necessary
    .concat_chunk(1024)                  # Concatenate and chunk the tokens into a fixed length of 1024 tokens
    .batch(8)                            # Group the data into mini-batches with a batch size of 8
    .take(10_000)                        # Limit the dataset to the first 10,000 batches
    .enumerate()                         # Add a "step" field to keep track of the training step
)

for batch in train_dataset:
    step = batch["step"]            # Current number of iteration (int)
    input_ids = batch["input_ids"]  # Input data for this iteration (torch.Tensor)

    # Implement the `step`-th training iteration using `input_ids` here
    ...
```

### Checkpointing

Normally, you would obtain and save the `state_dict` of the model and optimizer. In addition to that, please also obtain and save the `state_dict` of the iterator

```python
train_iter = train_dataset.iter()  # `iter(train_dataset)` would also work

for batch in train_iter:
    step = batch["step"]
    ...

    if step % ckpt_freq == 0:
        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": train_iter.state_dict(),
        }
        torch.save(state_dict, ckpt_path)
```

You can restore the state of the iterator by passing the `state_dict` to the iter method of the `CheckpointableDataset` instance.


```python
state_dict = torch.load(ckpt_path)
train_iter = train_dataset.iter(state_dict=state_dict["iter"])
```

## Development


```
pip install -e .[development]
mypy .; black .; flake8 .; isort .
pytest tests
```
