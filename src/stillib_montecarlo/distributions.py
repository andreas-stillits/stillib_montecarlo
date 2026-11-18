from dataclasses import dataclass

import numpy as np

from .sources import Source


@dataclass(frozen=True, slots=True)
class Normal(Source[float]):
    """A source that returns values sampled from a specified Normal distribution."""

    mean: float
    sigma: float

    def __post_init__(self) -> None:
        if not self.sigma >= 0:
            raise ValueError("Standard deviation sigma must be non-negative.")

    def sample(self, rng: np.random.Generator) -> float:
        return rng.normal(loc=self.mean, scale=self.sigma)


@dataclass(frozen=True, slots=True)
class Uniform(Source[float]):
    """A source that returns values sampled from a specified Uniform distribution."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.low <= self.high:
            raise ValueError("Lower bound must be less than upper bound.")

    def sample(self, rng: np.random.Generator) -> float:
        return rng.uniform(low=self.low, high=self.high)


@dataclass(frozen=True, slots=True)
class Poisson(Source[int]):
    """A source that returns values sampled from a specified Poisson distribution."""

    mean: float

    def __post_init__(self) -> None:
        if not self.mean >= 0:
            raise ValueError("Mean must be non-negative.")

    def sample(self, rng: np.random.Generator) -> int:
        return rng.poisson(lam=self.mean)


@dataclass(frozen=True, slots=True)
class Binomial(Source[int]):
    """A source that returns values sampled from a specified Binomial distribution."""

    n: int
    p: float

    def __post_init__(self) -> None:
        if not (0 <= self.p <= 1):
            raise ValueError("Probability p must be in the range [0, 1].")
        if not self.n >= 0:
            raise ValueError("Number of trials n must be non-negative.")

    def sample(self, rng: np.random.Generator) -> int:
        return rng.binomial(n=self.n, p=self.p)
