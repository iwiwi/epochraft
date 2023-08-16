# Epochraft

[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/optuna/optuna)
[![Checks status](https://github.com/iwiwi/epochraft/actions/workflows/checks.yml/badge.svg?branch=main)](https://github.com/iwiwi/epochraft/actions)
[![Tests status](https://github.com/iwiwi/epochraft/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/iwiwi/epochraft/actions)
[![pypi](https://img.shields.io/pypi/v/epochraft.svg)](https://pypi.python.org/pypi/epochraft)



## Introduction

*Epochraft* is a data loader library optimized for the streamlined training of LLMs, featuring **streaming from cloud storage**, **on-the-fly tokenization**, and **iterator checkpointing**. The name comes from a fusion of "epoch" and "craft".


### Streaming from Cloud Stoarge

Storing the vast datasets required for pretraining LLMs on local disks can be daunting. Even when it is feasible, transferring the data prior to training can be cumbersome and time consuming.

Epochraft offers a wide array of storage solutions, including S3, GCS, Azure Blob Storage, HDFS, WebHDFS, HTTP, HTTPS, SFTP, and the local filesystem (facilitated by [smart-open](https://github.com/RaRe-Technologies/smart_open/)). One of its salient features is the ability to train while concurrently downloading data. Due to its streaming-based architecture, a complete shuffle of data isn't possible. However, Epochraft achieves a level of shuffling by simultaneously accessing multiple data shards, intermixing the incoming data, and then performing an additional shuffle within a predetermined buffer size.

Additionally, it also supports Python's sequential or iterable interfaces. For instance, it can utilize the [Hugging Face Datasets](https://github.com/huggingface/datasets). While it might seem there's little benefit to using Epochraft with such small datasets, this enables the use of the same codebase for both SFT and pretraining.




### On-the-Fly Tokenization

Some of previous frameworks require pre-tokenization. This means that one has to tokenize the training data and then store it before pretraining. This is cumbersome. Training cannot begin until this step is completed. Moreover, if there are changes to the dataset or the tokenizer, you have to repeat this step again. Furthermore, there's added responsibility of managing tokenized data.

Now, you might wonder, "Isn't on-the-fly tokenization too slow?" The answer is a resounding no.

For instance, the training of Llama2-7B is conducted at the speed of approximately 3K tokens/sec per GPU (as seen in [Table 2](https://arxiv.org/abs/2307.09288)). The tokenizer of Llama2 can process at a rate of neraly 1M tokens/sec with a single CPU process. This means that even when tokenizing in real-time, the GPUs can be fully utilized without a bottleneck. And for larger models, the situation becomes even more favorable. For a 13B model, a rate of 1.5K tokens/sec is sufficient to saturate each GPU, while for a 70B model, only 300 tokens/sec is necessary.



### Data Loader Checkpointing

Beyond the state_dicts of models and optimizers, shouldn't we consider saving the state_dict of the data loader as well?

During the times when training ResNets for 90 epochs was the norm, this wasn’t a concern. A checkpoint at the end of each epoch sufficed. However, in the current era of LLMs, training often revolves around a single epoch.

When training for just 1 epoch, it becomes crucial to ensure that the data loader can pick up from where it left off in the middle of an epoch. Upon resuming training, it's vital to process only the data that hasn't been utilized up to that interruption point. Given the vastness of the data, an efficient resumption mechanism is essential.





## Quick Start

### Installation

```
pip install epochraft
```

### Example

This is an example of building a typical pretraining dataset. We will soon add other examples such as SFT.

```python
from epochraft import CheckpointableDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

# `{00..99}` will be expanded (see `braceexpand`)
url = "s3://.../cc-100/cc-100_{00..99}.jsonl"

train_dataset = (
    CheckpointableDataset
    .from_files(url, repeat=True, shuffle_shards=True)
    .tokenize(tokenizer)        # Tokenize the texts
    .ensure_bos_eos(tokenizer)  # Add BOS and EOS tokens where necessary
    .concat_chunk(1024)         # Concatenate and chunk the tokens into a fixed length of 1024 tokens
    .shuffle(1000)              # Shuffle the sequences using a buffer of size 1000
    .batch(8)                   # Group the data into mini-batches with a batch size of 8
)

for batch in train_dataset:
    input_ids = batch["input_ids"]  # Input data for this iteration (torch.Tensor)

    # Implement the training iteration using `input_ids` here
    ...

```

### Checkpointing

Normally, you would obtain and save the `state_dict` of the model and optimizer. In addition to that, please also obtain and save the `state_dict` of the iterator

```python
train_iter = train_dataset.iter()  # Same meaning as `iter(train_dataset)`

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

### Resumption

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
