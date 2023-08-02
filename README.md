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



## Development


```
pip install -e .[development]
mypy .; black .; flake8 .; isort .
pytest tests
```
