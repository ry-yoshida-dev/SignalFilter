from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import signal

from ...array_types import FloatArray
from ..iir_filter import IIRFilter


@dataclass(frozen=True, kw_only=True)
class NotchFilter(IIRFilter):
    """
    Second-order filter that removes one frequency, such as power-line hum.

    The gain is zero at ``frequency``, and the band where it is below -3 dB
    (-6 dB with ``Phase.ZERO``) is ``frequency / quality_factor`` hertz wide;
    everything else passes nearly unchanged.

    Attributes
    ----------
    frequency : float
        The frequency to remove, in hertz.
    quality_factor : float
        The frequency divided by the -3 dB bandwidth; higher is narrower.

    Raises
    ------
    ValueError
        If ``sampling_rate`` is invalid, ``frequency`` is not strictly between 0
        and the Nyquist frequency, or ``quality_factor`` is not positive and finite.
    """

    frequency: float
    quality_factor: float = 30.0

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        self._require_below_nyquist(self.frequency, "frequency")
        if not (np.isfinite(self.quality_factor) and self.quality_factor > 0.0):
            raise ValueError(
                f"quality_factor must be positive and finite, got {self.quality_factor}"
            )

    @override
    def _design(self) -> FloatArray:
        numerator, denominator = signal.iirnotch(
            self.frequency, self.quality_factor, fs=self.sampling_rate
        )
        return np.asarray(signal.tf2sos(numerator, denominator), dtype=np.float64)
