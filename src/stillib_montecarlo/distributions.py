from dataclasses import dataclass

import numpy as np

from .sources import Source


@dataclass(frozen=True, slots=True)
class Normal(Source):
    """A source that returns values sampled from a specified Normal distribution."""

    mean: float
    sigma: float

    def sample_once(self, rng: np.random.Generator) -> float:
        return rng.normal(loc=self.mean, scale=self.sigma)


@dataclass(frozen=True, slots=True)
class Uniform(Source):
    """A source that returns values sampled from a specified Uniform distribution."""

    low: float
    high: float

    def sample_once(self, rng: np.random.Generator) -> float:
        return rng.uniform(low=self.low, high=self.high)


@dataclass(frozen=True, slots=True)
class Poisson(Source):
    """A source that returns values sampled from a specified Poisson distribution."""

    mean: float

    def sample_once(self, rng: np.random.Generator) -> int:
        return rng.poisson(lam=self.mean)


@dataclass(frozen=True, slots=True)
class Binomial(Source):
    """A source that returns values sampled from a specified Binomial distribution."""

    n: int
    p: float

    def sample_once(self, rng: np.random.Generator) -> int:
        return rng.binomial(n=self.n, p=self.p)
