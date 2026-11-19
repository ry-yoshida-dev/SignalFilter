"""Kinds of frequency band a filter passes or rejects."""

from __future__ import annotations

from enum import Enum, auto


class BandType(Enum):
    """
    Which frequencies a band filter passes.

    Attributes
    ----------
    LOWPASS
        Below one cutoff.
    HIGHPASS
        Above one cutoff.
    BANDPASS
        Between two edges.
    BANDSTOP
        Outside two edges.
    """

    LOWPASS = auto()
    HIGHPASS = auto()
    BANDPASS = auto()
    BANDSTOP = auto()

    @property
    def edge_count(self) -> int:
        """
        Get the number of edge frequencies the band is defined by.

        Returns
        -------
        int
            One for lowpass and highpass, two for bandpass and bandstop.
        """
        match self:
            case BandType.LOWPASS | BandType.HIGHPASS:
                return 1
            case BandType.BANDPASS | BandType.BANDSTOP:
                return 2
