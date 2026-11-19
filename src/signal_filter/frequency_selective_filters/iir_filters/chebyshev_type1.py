from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import signal

from ...array_types import FloatArray
from ...band import Band
from ..iir_filter import IIRFilter


@dataclass(frozen=True, kw_only=True)
class ChebyshevType1Filter(IIRFilter):
    """
    Steeper roll-off than Butterworth in exchange for ripple in the passband.

    The gain ripples between 0 and ``-passband_ripple`` dB across the passband
    and reaches ``-passband_ripple`` dB at each band edge.

    Attributes
    ----------
    band : Band
        The band to pass or reject.
    order : int
        The design order; bandpass and bandstop filters have twice as many poles.
    passband_ripple : float
        The largest passband loss in decibels.

    Raises
    ------
    TypeError
        If ``order`` is not an integer.
    ValueError
        If ``sampling_rate`` is invalid, a band edge is not below the Nyquist
        frequency, ``order`` is smaller than 1, or ``passband_ripple`` is not
        positive and finite.
    """

    band: Band
    order: int = 4
    passband_ripple: float = 1.0

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        self._require_band(self.band)
        self._require_order(self.order)
        if not (np.isfinite(self.passband_ripple) and self.passband_ripple > 0.0):
            raise ValueError(
                f"passband_ripple must be positive and finite, got {self.passband_ripple}"
            )

    @override
    def _design(self) -> FloatArray:
        return np.asarray(
            signal.cheby1(
                self.order,
                self.passband_ripple,
                self._band_edges(self.band),
                btype=self._band_name(self.band.band_type),
                output="sos",
                fs=self.sampling_rate,
            ),
            dtype=np.float64,
        )
