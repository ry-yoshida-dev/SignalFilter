from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import override

import numpy as np
from scipy import signal

from ...array_types import FloatArray
from ...band import Band
from ..fir_filter import FIRFilter


@dataclass(frozen=True, kw_only=True)
class KaiserFIRFilter(FIRFilter):
    """
    FIR filter designed from its specification with the Kaiser window method.

    The length and the window shape are chosen so that, outside transitions
    ``transition_width`` hertz wide and centered on each band edge (where the
    gain is -6 dB), the stopband gain and the passband ripple stay near
    ``-attenuation`` dB (doubled with ``Phase.ZERO``). The method is an
    estimate: lowpass and highpass designs fall short by about 1.5 dB at most,
    while in bandpass and bandstop designs the ripples of both transitions can
    add up to 6 dB; ask for that much more attenuation when the bound matters.

    Attributes
    ----------
    band : Band
        The band to pass or reject.
    transition_width : float
        The width in hertz of each transition between passband and stopband.
    attenuation : float
        The stopband attenuation in decibels; the passband deviates from 1 by
        about the same fraction.

    Raises
    ------
    ValueError
        If ``sampling_rate`` is invalid, ``attenuation`` is not positive and
        finite, ``transition_width`` is not positive, or a transition does not
        fit between 0, the band edges, and the Nyquist frequency.
    """

    band: Band
    transition_width: float
    attenuation: float = 60.0

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        self._require_band(self.band)
        if not (np.isfinite(self.attenuation) and self.attenuation > 0.0):
            raise ValueError(f"attenuation must be positive and finite, got {self.attenuation}")
        if not (np.isfinite(self.transition_width) and self.transition_width > 0.0):
            raise ValueError(
                f"transition_width must be positive and finite, got {self.transition_width}"
            )
        # Each transition is centered on its edge, so it needs half its width up to 0 and the
        # Nyquist frequency, and its full width up to the next transition.
        gaps = np.diff([0.0, *self.band.edges, self.nyquist_frequency])
        needed = np.full(gaps.shape, self.transition_width)
        needed[[0, -1]] = 0.5 * self.transition_width
        if (gaps < needed).any():
            raise ValueError(
                f"transition_width {self.transition_width} Hz does not fit around the band "
                + f"edges {self.band.edges} within [0, {self.nyquist_frequency}] Hz"
            )

    @property
    def taps(self) -> int:
        """
        Get the number of coefficients the specification needs.

        Returns
        -------
        int
            The filter length, odd.
        """
        return self._kaiser_parameters[0]

    @cached_property
    def _kaiser_parameters(self) -> tuple[int, float]:
        taps, beta = signal.kaiserord(
            self.attenuation, self.transition_width / self.nyquist_frequency
        )
        # An odd length keeps the delay a whole number of samples and allows highpass designs.
        return int(taps) | 1, float(beta)

    @override
    def _design(self) -> FloatArray:
        taps, beta = self._kaiser_parameters
        return np.asarray(
            signal.firwin(
                taps,
                self._band_edges(self.band),
                window=("kaiser", beta),
                pass_zero=self._band_name(self.band.band_type),
                fs=self.sampling_rate,
            ),
            dtype=np.float64,
        )
