from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import numpy as np

from .sources import Source

ResultType = TypeVar("ResultType")


def propagate(
    rng: np.random.Generator,
    func: Callable[..., ResultType],
    inputs: Sequence[Source[Any]],
    n_samples: int = 1_000,
) -> list[ResultType]:
    """Propagate uncertainties through a function using Monte Carlo sampling.

    Args:
        rng: A numpy random number generator.
        func: A callable function that takes as many arguments as there are inputs.
        inputs: A sequence of Source objects representing the uncertain inputs.
        n_samples: The number of samples to draw for the propagation.

    Returns:
        A list of results obtained by evaluating the function with sampled inputs.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")

    results: list[ResultType] = []
    for _ in range(n_samples):
        sampled_inputs = [source.sample(rng) for source in inputs]
        result = func(*sampled_inputs)
        results.append(result)
    return results


def propagate_pool(
    seed: int,
    func: Callable[..., ResultType],
    inputs: Sequence[Source[Any]],
    workers: int | None = None,
    n_samples: int = 1_000,
) -> list[ResultType]: ...
