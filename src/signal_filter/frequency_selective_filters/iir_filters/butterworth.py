from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import signal

from ...array_types import FloatArray
from ...band import Band
from ..iir_filter import IIRFilter


@dataclass(frozen=True, kw_only=True)
class ButterworthFilter(IIRFilter):
    """
    Maximally flat passband with a moderate roll-off.

    The usual first choice: no ripple anywhere, and the gain is -3 dB at each
    band edge (-6 dB with ``Phase.ZERO``).

    Attributes
    ----------
    band : Band
        The band to pass or reject.
    order : int
        The design order; bandpass and bandstop filters have twice as many poles.
        The roll-off is ``20 * order`` dB per decade.

    Raises
    ------
    TypeError
        If ``order`` is not an integer.
    ValueError
        If ``sampling_rate`` is invalid, a band edge is not below the Nyquist
        frequency, or ``order`` is smaller than 1.
    """

    band: Band
    order: int = 4

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        self._require_band(self.band)
        self._require_order(self.order)

    @override
    def _design(self) -> FloatArray:
        return np.asarray(
            signal.butter(
                self.order,
                self._band_edges(self.band),
                btype=self._band_name(self.band.band_type),
                output="sos",
                fs=self.sampling_rate,
            ),
            dtype=np.float64,
        )
