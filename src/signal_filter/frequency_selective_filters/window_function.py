"""Window functions that taper a windowed FIR design."""

from __future__ import annotations

from enum import Enum, auto


class WindowFunction(Enum):
    """
    Taper applied to an ideal impulse response, ordered from sharpest to most attenuating.

    Attributes
    ----------
    RECTANGULAR
        No taper: the narrowest transition, but only about 21 dB of attenuation.
    HANN
        About 44 dB of attenuation.
    HAMMING
        About 53 dB of attenuation with a transition as narrow as Hann.
    BLACKMAN
        About 74 dB of attenuation with a wider transition.
    BLACKMAN_HARRIS
        About 92 dB of attenuation with the widest transition.
    """

    RECTANGULAR = auto()
    HANN = auto()
    HAMMING = auto()
    BLACKMAN = auto()
    BLACKMAN_HARRIS = auto()
