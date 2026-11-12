from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from .sources import NumpySource, Source

ResultType = TypeVar("ResultType")
ResultNDType = TypeVar("ResultNDType", bound=np.generic)


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
        sampled_inputs = [source.sample_once(rng) for source in inputs]
        result = func(*sampled_inputs)
        results.append(result)
    return results


def propagate_numpy(
    rng: np.random.Generator,
    func: Callable[..., NDArray[ResultNDType]],
    inputs: Sequence[NumpySource],
    n_samples: int = 1_000,
) -> NDArray[ResultNDType]:
    """Propagate uncertainties through a function using Monte Carlo sampling with NumPy arrays.

    Args:
        rng: A numpy random number generator.
        func: A callable function that takes as many arguments as there are inputs and returns a NumPy array.
        inputs: A sequence of NumpySource objects representing the uncertain inputs.
        n_samples: The number of samples to draw for the propagation.

    Returns:
        A NumPy array of results obtained by evaluating the function with sampled inputs.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")

    sampled_inputs = tuple(source.sample_numpy(rng, n_samples) for source in inputs)

    for position, samples in enumerate(sampled_inputs):
        if samples.ndim == 0 or samples.shape[0] != n_samples:
            raise ValueError(
                f"Input source {position} returned shape {samples.shape}; "
                f"expected first dimension to be {n_samples}."
            )

    results = func(*sampled_inputs)

    if results.ndim == 0 or results.shape[0] != n_samples:
        raise ValueError(
            "func must return an array whose first dimension is "
            f"n_samples={n_samples}; got shape {results.shape}."
        )

    return results
