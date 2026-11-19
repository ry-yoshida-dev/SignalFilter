from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import signal

from ...array_types import FloatArray
from ...band import Band
from ..iir_filter import IIRFilter


@dataclass(frozen=True, kw_only=True)
class EllipticFilter(IIRFilter):
    """
    Steepest roll-off for a given order, with ripple in both passband and stopband.

    The gain stays within ``-passband_ripple`` dB in the passband, reaches it at
    each band edge, and stays below ``-stopband_attenuation`` dB in the stopband.

    Attributes
    ----------
    band : Band
        The band to pass or reject.
    order : int
        The design order; bandpass and bandstop filters have twice as many poles.
    passband_ripple : float
        The largest passband loss in decibels.
    stopband_attenuation : float
        The smallest stopband attenuation in decibels, larger than
        ``passband_ripple``.

    Raises
    ------
    TypeError
        If ``order`` is not an integer.
    ValueError
        If ``sampling_rate`` is invalid, a band edge is not below the Nyquist
        frequency, ``order`` is smaller than 1, the ripple or attenuation is not
        positive and finite, or the attenuation does not exceed the ripple.
    """

    band: Band
    order: int = 4
    passband_ripple: float = 1.0
    stopband_attenuation: float = 40.0

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        self._require_band(self.band)
        self._require_order(self.order)
        if not (np.isfinite(self.passband_ripple) and self.passband_ripple > 0.0):
            raise ValueError(
                f"passband_ripple must be positive and finite, got {self.passband_ripple}"
            )
        if not (
            np.isfinite(self.stopband_attenuation)
            and self.stopband_attenuation > self.passband_ripple
        ):
            raise ValueError(
                "stopband_attenuation must be finite and exceed passband_ripple "
                + f"{self.passband_ripple}, got {self.stopband_attenuation}"
            )

    @override
    def _design(self) -> FloatArray:
        return np.asarray(
            signal.ellip(
                self.order,
                self.passband_ripple,
                self.stopband_attenuation,
                self._band_edges(self.band),
                btype=self._band_name(self.band.band_type),
                output="sos",
                fs=self.sampling_rate,
            ),
            dtype=np.float64,
        )
