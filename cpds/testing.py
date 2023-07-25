from .base import Sample


def generate_example_sequence(n_samples: int = 20) -> list[Sample]:
    return [{"id": i} for i in range(n_samples)]
