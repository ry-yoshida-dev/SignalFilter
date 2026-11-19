from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, override

import numpy as np
from scipy import signal

from ...array_types import FloatArray
from ...band import Band
from ..bessel_normalization import BesselNormalization
from ..iir_filter import IIRFilter


@dataclass(frozen=True, kw_only=True)
class BesselFilter(IIRFilter):
    """
    Nearly constant group delay in the passband, with the gentlest roll-off.

    With ``Phase.CAUSAL`` it keeps the shape of pulses and steps with little
    overshoot or ringing, at the cost of selectivity.

    Attributes
    ----------
    band : Band
        The band to pass or reject.
    order : int
        The design order; bandpass and bandstop filters have twice as many poles.
    normalization : BesselNormalization
        What the band edges refer to; ``MAGNITUDE`` puts -3 dB at each edge.

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
    normalization: BesselNormalization = BesselNormalization.MAGNITUDE

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        self._require_band(self.band)
        self._require_order(self.order)

    @override
    def _design(self) -> FloatArray:
        return np.asarray(
            signal.bessel(
                self.order,
                self._band_edges(self.band),
                btype=self._band_name(self.band.band_type),
                output="sos",
                norm=self._normalization_name(),
                fs=self.sampling_rate,
            ),
            dtype=np.float64,
        )

    def _normalization_name(self) -> Literal["mag", "phase", "delay"]:
        match self.normalization:
            case BesselNormalization.MAGNITUDE:
                return "mag"
            case BesselNormalization.PHASE:
                return "phase"
            case BesselNormalization.DELAY:
                return "delay"
