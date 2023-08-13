import gc
from typing import Generator

import pytest


# We need to run GC every time. This is because the shutdown of child processes in
# ParallelFilterMap does not occur unless GC is executed if its source is infinite.
@pytest.fixture(autouse=True)
def run_gc() -> Generator[None, None, None]:
    gc.collect()
    yield
