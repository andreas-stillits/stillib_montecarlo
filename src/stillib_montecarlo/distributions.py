from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .sources import NumpySource, Source


@dataclass(frozen=True, slots=True)
class Normal(Source[float], NumpySource):
    """A source that returns values sampled from a specified Normal distribution."""

    mean: float
    sigma: float

    def __post_init__(self) -> None:
        if not self.sigma >= 0:
            raise ValueError("Standard deviation sigma must be non-negative.")

    def sample_once(self, rng: np.random.Generator) -> float:
        return rng.normal(loc=self.mean, scale=self.sigma)

    def sample_numpy(self, rng: np.random.Generator, n_samples: int) -> NDArray[Any]:
        return rng.normal(loc=self.mean, scale=self.sigma, size=n_samples)


@dataclass(frozen=True, slots=True)
class Uniform(Source[float], NumpySource):
    """A source that returns values sampled from a specified Uniform distribution."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.low <= self.high:
            raise ValueError("Lower bound must be less than upper bound.")

    def sample_once(self, rng: np.random.Generator) -> float:
        return rng.uniform(low=self.low, high=self.high)

    def sample_numpy(self, rng: np.random.Generator, n_samples: int) -> NDArray[Any]:
        return rng.uniform(low=self.low, high=self.high, size=n_samples)


@dataclass(frozen=True, slots=True)
class Poisson(Source[int], NumpySource):
    """A source that returns values sampled from a specified Poisson distribution."""

    mean: float

    def __post_init__(self) -> None:
        if not self.mean >= 0:
            raise ValueError("Mean must be non-negative.")

    def sample_once(self, rng: np.random.Generator) -> int:
        return rng.poisson(lam=self.mean)

    def sample_numpy(self, rng: np.random.Generator, n_samples: int) -> NDArray[Any]:
        return rng.poisson(lam=self.mean, size=n_samples)


@dataclass(frozen=True, slots=True)
class Binomial(Source[int], NumpySource):
    """A source that returns values sampled from a specified Binomial distribution."""

    n: int
    p: float

    def __post_init__(self) -> None:
        if not (0 <= self.p <= 1):
            raise ValueError("Probability p must be in the range [0, 1].")
        if not self.n >= 0:
            raise ValueError("Number of trials n must be non-negative.")

    def sample_once(self, rng: np.random.Generator) -> int:
        return rng.binomial(n=self.n, p=self.p)

    def sample_numpy(self, rng: np.random.Generator, n_samples: int) -> NDArray[Any]:
        return rng.binomial(n=self.n, p=self.p, size=n_samples)
