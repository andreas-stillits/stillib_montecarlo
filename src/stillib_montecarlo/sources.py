from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

SampleType = TypeVar("SampleType")


class Source(ABC, Generic[SampleType]):
    @abstractmethod
    def sample_once(self, rng: np.random.Generator) -> SampleType: ...


class NumpySource(ABC):
    # inherits sample_once from Source as well
    @abstractmethod
    def sample_numpy(
        self, rng: np.random.Generator, n_samples: int
    ) -> NDArray[Any]: ...


@dataclass(frozen=True, slots=True)
class Constant(Source[SampleType], NumpySource, Generic[SampleType]):
    """A source that always returns the same value."""

    _value: SampleType

    def sample_once(self, rng: np.random.Generator) -> SampleType:
        return self._value

    def sample_numpy(self, rng: np.random.Generator, n_samples: int) -> NDArray[Any]:

        value = np.asarray(self._value)
        shape = (n_samples, *value.shape)
        return np.broadcast_to(value, shape).copy()


@dataclass(frozen=True, slots=True, init=False)
class Empirical(Source[SampleType], NumpySource, Generic[SampleType]):
    """A source that returns values sampled with replacement from an empirical distribution."""

    _values: tuple[SampleType, ...]

    def __init__(self, values: Iterable[SampleType]) -> None:
        snapshot = tuple(values)

        if not snapshot:
            raise ValueError(
                "Empirical source must be initialized with a non-empty iterable of values."
            )

        object.__setattr__(self, "_values", snapshot)

    def sample_once(self, rng: np.random.Generator) -> SampleType:
        index = int(
            rng.integers(len(self._values))
        )  # prefer sampling by index over np.random.choice to avoid casting issues
        return self._values[index]

    def sample_numpy(self, rng: np.random.Generator, n_samples: int) -> NDArray[Any]:

        population = np.asarray(self._values)
        indices = rng.integers(len(population), size=n_samples)
        return population[indices]


@dataclass(frozen=True, slots=True)
class Model(Source[SampleType], Generic[SampleType]):
    """A source created from a scalar sampling model, i.e. any callable taking an RNG as its argument and returning a single sample."""

    model: Callable[[np.random.Generator], SampleType]

    def sample_once(self, rng: np.random.Generator) -> SampleType:
        return self.model(rng)


@dataclass(frozen=True, slots=True)
class NumpyModel(NumpySource):
    """A source created from a NumPy batch-sampling model, i.e. any callable taking an RNG and n_samples as its arguments and returning a numpy array."""

    model: Callable[[np.random.Generator, int], NDArray[Any]]

    def sample_numpy(self, rng: np.random.Generator, n_samples: int) -> NDArray[Any]:

        samples = np.asarray(self.model(rng, n_samples))

        if samples.ndim == 0:
            raise ValueError(
                "The model must return a numpy array with at least one dimension."
            )

        if samples.shape[0] != n_samples:
            raise ValueError(
                f"The model must return a numpy array with the first dimension equal to n_samples ({n_samples}), but got shape {samples.shape}."
            )

        return samples
