from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

SampleType = TypeVar("SampleType")


class Source(ABC, Generic[SampleType]):
    @abstractmethod
    def sample(self, rng: np.random.Generator) -> SampleType: ...


@dataclass(frozen=True, slots=True)
class Constant(Source[SampleType], Generic[SampleType]):
    """A source that always returns the same value."""

    _value: SampleType

    def sample(self, rng: np.random.Generator) -> SampleType:
        return self._value


@dataclass(frozen=True, slots=True, init=False)
class Empirical(Source[SampleType], Generic[SampleType]):
    """A source that returns values sampled with replacement from an empirical distribution."""

    _values: tuple[SampleType, ...]

    def __init__(self, values: Iterable[SampleType]) -> None:
        snapshot = tuple(values)

        if not snapshot:
            raise ValueError(
                "Empirical source must be initialized with a non-empty iterable of values."
            )

        object.__setattr__(self, "_values", snapshot)

    def sample(self, rng: np.random.Generator) -> SampleType:
        index = int(
            rng.integers(len(self._values))
        )  # prefer sampling by index over np.random.choice to avoid casting issues
        return self._values[index]


@dataclass(frozen=True, slots=True)
class Model(Source[SampleType], Generic[SampleType]):
    """A source created from a scalar sampling model, i.e. any callable taking an RNG as its argument and returning a single sample."""

    model: Callable[[np.random.Generator], SampleType]

    def sample(self, rng: np.random.Generator) -> SampleType:
        return self.model(rng)
