"""Abstract filter of sampled signals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from numpy.typing import ArrayLike

from ._samples import as_samples
from .array_types import FloatArray


@dataclass(frozen=True, kw_only=True)
class SignalFilter(ABC):
    """
    Filter that maps a sampled signal to a filtered signal of the same shape.

    Samples run along the first axis; any further axes are independent channels
    (``(n_samples,)``, ``(n_samples, n_channels)``, or more). Filters are
    immutable and keep no state between calls, so one instance can filter any
    number of signals.
    """

    @property
    @abstractmethod
    def minimum_length(self) -> int:
        """
        Get the fewest samples a signal needs to be filtered.

        Returns
        -------
        int
            The number of samples along the first axis.
        """

    def apply(self, signal: ArrayLike) -> FloatArray:
        """
        Filter a signal.

        Parameters
        ----------
        signal : ArrayLike
            Real samples with shape ``(n_samples, *channel_shape)``.

        Returns
        -------
        FloatArray
            The filtered float64 signal with the same shape.

        Raises
        ------
        TypeError
            If the signal is not real-valued numbers.
        ValueError
            If the signal is 0-D, shorter than ``minimum_length``, or not finite.
        """
        samples = as_samples(signal, self.minimum_length)
        return self._filter(samples)

    @abstractmethod
    def _filter(self, samples: FloatArray) -> FloatArray:
        """
        Filter validated samples.

        Parameters
        ----------
        samples : FloatArray
            Finite float64 samples with shape ``(n_samples, *channel_shape)``,
            at least ``minimum_length`` long, owned by the filter.

        Returns
        -------
        FloatArray
            The filtered samples with the same shape.
        """
