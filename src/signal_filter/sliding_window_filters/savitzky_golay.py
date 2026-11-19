from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import signal

from .._parameters import require_integer
from ..array_types import FloatArray
from ..sliding_window_filter import SlidingWindowFilter


@dataclass(frozen=True, kw_only=True)
class SavitzkyGolayFilter(SlidingWindowFilter):
    """
    Least-squares polynomial fitted to the window centered on each sample.

    Keeps the height and width of peaks better than a moving average of the same
    length, and reproduces polynomials up to ``polynomial_order`` exactly away
    from the ends. With ``derivative`` above 0 it returns a smoothed derivative
    instead.

    Attributes
    ----------
    window_length : int
        The number of samples each polynomial is fitted to, odd and greater than
        ``polynomial_order``.
    boundary : Boundary
        The values assumed beyond the ends of the signal.
    polynomial_order : int
        The degree of the fitted polynomial; 0 and 1 equal a moving average.
    derivative : int
        The derivative of the fit to return, at most ``polynomial_order``.
    sample_spacing : float
        The time between samples, which scales derivatives into units per
        second (or whatever unit it is given in).

    Raises
    ------
    TypeError
        If ``window_length``, ``polynomial_order``, or ``derivative`` is not an
        integer.
    ValueError
        If ``window_length`` is not a positive odd number greater than
        ``polynomial_order``, ``polynomial_order`` or ``derivative`` is negative,
        ``derivative`` exceeds ``polynomial_order``, or ``sample_spacing`` is not
        positive and finite.
    """

    polynomial_order: int = 2
    derivative: int = 0
    sample_spacing: float = 1.0

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        require_integer(self.polynomial_order, "polynomial_order")
        require_integer(self.derivative, "derivative")
        if not 0 <= self.polynomial_order < self.window_length:
            raise ValueError(
                "polynomial_order must be non-negative and smaller than window_length "
                + f"{self.window_length}, got {self.polynomial_order}"
            )
        if not 0 <= self.derivative <= self.polynomial_order:
            raise ValueError(
                "derivative must be non-negative and at most polynomial_order "
                + f"{self.polynomial_order}, got {self.derivative}"
            )
        if not (np.isfinite(self.sample_spacing) and self.sample_spacing > 0.0):
            raise ValueError(
                f"sample_spacing must be positive and finite, got {self.sample_spacing}"
            )

    @override
    def _filter(self, samples: FloatArray) -> FloatArray:
        # Any mode but "interp" filters the interior alike; "nearest" also accepts short signals.
        filtered = signal.savgol_filter(
            self._pad(samples),
            self.window_length,
            self.polynomial_order,
            deriv=self.derivative,
            delta=self.sample_spacing,
            axis=0,
            mode="nearest",
        )
        return self._crop(np.asarray(filtered, dtype=np.float64))
