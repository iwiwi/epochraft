# Epochraft

[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/optuna/optuna)
[![Checks status](https://github.com/iwiwi/epochraft/actions/workflows/checks.yml/badge.svg?branch=main)](https://github.com/iwiwi/epochraft/actions)
[![Tests status](https://github.com/iwiwi/epochraft/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/iwiwi/epochraft/actions)
[![pypi](https://img.shields.io/pypi/v/epochraft.svg)](https://pypi.python.org/pypi/epochraft)



## Introduction

*Epochraft* is a data loader library designed with a focus on **on-the-fly tokenization** and **checkpointing**, specifically for the streamlined training of LLMs.


### Why On-the-Fly Tokenization?

Previous frmaeworks like [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) requires pre-tokenization. That is, we need to tokenize the training data and store it before pretraining. However, this method is cumbersome and requires additional steps. Training can't begin until this is completed. If you change the dataset or the tokenizer, you'll have to recreate again. And, we need to manage the tokenized data.

You may ask "But, isn't on-the-fly tokenization too slow?" The answer is a definitive no.

For instance, the training of Llama2-7B is conducted at the speed of about 3K tokens/sec per GPU (see [Table 2](https://arxiv.org/abs/2307.09288)). The tokenizer of Llama2 can tokenize at a rate of about 1M tokens/sec with a single CPU process. Even if tokenizing in real-time in the background, you can easily saturate the GPUs. Larger models are even easier. With 13B, you can saturate each GPU by providing 1.5K tokens/sec, and with 70B, by just 300 tokens/sec.



### Why Data Loader Checkpointing?

The standard practice of checkpointing in PyTorch involves saving the `state_dict` of the model and optimizer. However, as we are training LLMs, we should also want to save the `state_dict` of the data loader.

In the era of training ResNets for 90 epochs, there was no such need. Simply checkpointing at the end of each epoch was enough. But now, in the age of LLMs, we often train around 1 epoch.

In training for 1 epoch, it's necessary to ensure that the data loader can continue from the middle of an epoch as well. After resuming the training, we want to correctly use only the data that has not been used up to that point. Moreover, since the data is quite large, an efficient resumption is needed, not an inefficient method that reads and discards all the data up to that point.




### Epochraft: On-the-Fly Tokenization + Checkpointing

Epochraft is designed with the aim of achieving both on-the-fly tokenization and checkpointing. Neither on-the-fly tokenization nor checkpointing are exceptionally difficult features in themselves. However, when attempting to realize both simultaneously, significant constraints arise at the core of the design. That's why no existing libraries are compatible with both features.

In Epochraft, a variety of existing datasets can be used as sources, so it supports a wide range of data formats. Particularly, when using [MosaicML Streaming](https://github.com/mosaicml/streaming) as a source, you can train directly by streaming data from S3,
and resumption is efficient.

As Epochraft is a library focused on the training of LLMs, it is equipped with features that are necessary for pretraining and SFT of LLMs. Operations like tokenization and chunking are available out of the box. Additionally, tokenization is performed efficiently using multi-processes.








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
