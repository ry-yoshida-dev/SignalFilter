"""What the cutoff of a Bessel filter refers to."""

from __future__ import annotations

from enum import Enum, auto


class BesselNormalization(Enum):
    """
    Frequency a Bessel design matches to its cutoff.

    Attributes
    ----------
    MAGNITUDE
        The gain is -3 dB at the cutoff, as for the other designs.
    PHASE
        The phase response approaches that of a delay of ``1 / (2 pi cutoff)``
        seconds; the asymptotes of the response meet at the cutoff.
    DELAY
        The group delay in the passband is ``1 / (2 pi cutoff)`` seconds.
    """

    MAGNITUDE = auto()
    PHASE = auto()
    DELAY = auto()
