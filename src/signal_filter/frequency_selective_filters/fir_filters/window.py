from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, override

import numpy as np
from scipy import signal

from ..._parameters import require_integer
from ...array_types import FloatArray
from ...band import Band
from ..fir_filter import FIRFilter
from ..window_function import WindowFunction


@dataclass(frozen=True, kw_only=True)
class WindowFIRFilter(FIRFilter):
    """
    FIR filter designed by tapering the ideal response with a window function.

    The gain is -6 dB at each band edge (-12 dB with ``Phase.ZERO``). The
    transition narrows as ``taps`` grows; the stopband attenuation is set by
    the window function.

    Attributes
    ----------
    band : Band
        The band to pass or reject.
    taps : int
        The number of coefficients, odd and at least 3.
    window_function : WindowFunction
        The taper, which trades transition width for stopband attenuation.

    Raises
    ------
    TypeError
        If ``taps`` is not an integer.
    ValueError
        If ``sampling_rate`` is invalid, a band edge is not below the Nyquist
        frequency, or ``taps`` is not an odd number of at least 3.
    """

    band: Band
    taps: int = 101
    window_function: WindowFunction = WindowFunction.HAMMING

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        self._require_band(self.band)
        require_integer(self.taps, "taps")
        if self.taps < 3 or self.taps % 2 == 0:
            raise ValueError(f"taps must be odd and at least 3, got {self.taps}")

    @override
    def _design(self) -> FloatArray:
        return np.asarray(
            signal.firwin(
                self.taps,
                self._band_edges(self.band),
                window=self._window_name(),
                pass_zero=self._band_name(self.band.band_type),
                fs=self.sampling_rate,
            ),
            dtype=np.float64,
        )

    def _window_name(
        self,
    ) -> Literal["boxcar", "hann", "hamming", "blackman", "blackmanharris"]:
        match self.window_function:
            case WindowFunction.RECTANGULAR:
                return "boxcar"
            case WindowFunction.HANN:
                return "hann"
            case WindowFunction.HAMMING:
                return "hamming"
            case WindowFunction.BLACKMAN:
                return "blackman"
            case WindowFunction.BLACKMAN_HARRIS:
                return "blackmanharris"
