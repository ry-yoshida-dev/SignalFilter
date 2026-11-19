from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import signal

from ...array_types import FloatArray
from ...band import Band
from ..iir_filter import IIRFilter


@dataclass(frozen=True, kw_only=True)
class ChebyshevType2Filter(IIRFilter):
    """
    Flat passband and a stopband that never rises above ``-stopband_attenuation`` dB.

    Unlike the other designs, the band edges mark where the stopband begins: the
    gain there first reaches ``-stopband_attenuation`` dB, so the passband ends
    somewhat inside the band.

    Attributes
    ----------
    band : Band
        The band to pass or reject, with edges at the start of the stopband.
    order : int
        The design order; bandpass and bandstop filters have twice as many poles.
    stopband_attenuation : float
        The smallest stopband attenuation in decibels.

    Raises
    ------
    TypeError
        If ``order`` is not an integer.
    ValueError
        If ``sampling_rate`` is invalid, a band edge is not below the Nyquist
        frequency, ``order`` is smaller than 1, or ``stopband_attenuation`` is
        not positive and finite.
    """

    band: Band
    order: int = 4
    stopband_attenuation: float = 40.0

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        self._require_band(self.band)
        self._require_order(self.order)
        if not (np.isfinite(self.stopband_attenuation) and self.stopband_attenuation > 0.0):
            raise ValueError(
                "stopband_attenuation must be positive and finite, "
                + f"got {self.stopband_attenuation}"
            )

    @override
    def _design(self) -> FloatArray:
        return np.asarray(
            signal.cheby2(
                self.order,
                self.stopband_attenuation,
                self._band_edges(self.band),
                btype=self._band_name(self.band.band_type),
                output="sos",
                fs=self.sampling_rate,
            ),
            dtype=np.float64,
        )
