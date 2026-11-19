"""Abstract filter that summarizes a window centered on each sample."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import override

import numpy as np

from ._parameters import require_integer
from .array_types import FloatArray
from .boundary import Boundary
from .signal_filter import SignalFilter


@dataclass(frozen=True, kw_only=True)
class SlidingWindowFilter(SignalFilter, ABC):
    """
    Filter whose output at each sample depends on ``window_length`` samples around it.

    The window is centered, so features are not delayed. Near the ends it
    reaches beyond the signal, where values are assumed as ``boundary`` says.

    Attributes
    ----------
    window_length : int
        The number of samples in each window, odd.
    boundary : Boundary
        The values assumed beyond the ends of the signal.

    Raises
    ------
    TypeError
        If ``window_length`` is not an integer.
    ValueError
        If ``window_length`` is not a positive odd number.
    """

    window_length: int
    boundary: Boundary = Boundary.REFLECT

    def __post_init__(self) -> None:
        require_integer(self.window_length, "window_length")
        if self.window_length < 1 or self.window_length % 2 == 0:
            raise ValueError(f"window_length must be positive and odd, got {self.window_length}")

    @property
    @override
    def minimum_length(self) -> int:
        """
        Get the fewest samples a signal needs to be filtered.

        Returns
        -------
        int
            One; windows longer than the signal are filled as ``boundary`` says.
        """
        return 1

    def _pad(self, samples: FloatArray) -> FloatArray:
        """
        Extend the signal by half a window at each end, as ``boundary`` says.

        Every window centered on an original sample then lies inside the padded
        signal, so filtering it in any mode and keeping :meth:`_crop` of the result
        follows ``boundary`` exactly, on every SciPy version and for windows longer
        than the signal.

        Returns
        -------
        FloatArray
            The padded samples with ``window_length - 1`` more along the first axis.
        """
        half_width = self.window_length // 2
        padding = [(half_width, half_width)] + [(0, 0)] * (samples.ndim - 1)
        match self.boundary:
            case Boundary.REFLECT:
                return np.pad(samples, padding, mode="reflect")
            case Boundary.NEAREST:
                return np.pad(samples, padding, mode="edge")
            case Boundary.WRAP:
                return np.pad(samples, padding, mode="wrap")

    def _crop(self, padded: FloatArray) -> FloatArray:
        """
        Keep the samples of a padded signal that correspond to the original ones.

        Returns
        -------
        FloatArray
            The padded signal without half a window at each end.
        """
        half_width = self.window_length // 2
        return padded[half_width : padded.shape[0] - half_width]

    def _windows(self, samples: FloatArray) -> FloatArray:
        """
        View the window centered on every sample.

        Returns
        -------
        FloatArray
            A read-only view with shape ``(n_samples, *channel_shape, window_length)``.
        """
        return np.lib.stride_tricks.sliding_window_view(
            self._pad(samples), self.window_length, axis=0
        )
