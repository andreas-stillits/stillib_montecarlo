from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .sources import Source


def propagate(
    rng: np.random.Generator,
    func: Callable[..., Any],
    inputs: Sequence[Source],
    n_samples: int = 1_000,
) -> list[Any]:
    """Propagate uncertainties through a function using Monte Carlo sampling.

    Args:
        rng: A numpy random number generator.
        func: A callable function that takes as many arguments as there are inputs.
        inputs: A sequence of Source objects representing the uncertain inputs.
        n_samples: The number of samples to draw for the propagation.

    Returns:
        A list of results obtained by evaluating the function with sampled inputs.
    """
    results = []
    for _ in range(n_samples):
        sampled_inputs = [source.sample_once(rng) for source in inputs]
        result = func(*sampled_inputs)
        results.append(result)
    return results
