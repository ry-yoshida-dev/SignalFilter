"""How a frequency-selective filter is run over a signal."""

from __future__ import annotations

from enum import Enum, auto


class Phase(Enum):
    """
    Phase behavior of a frequency-selective filter.

    Attributes
    ----------
    ZERO
        Run the filter forward and then backward. Features stay where they are
        (no delay, no phase distortion), the magnitude response is squared, so the
        attenuation in decibels doubles, and the order is effectively doubled.
        Needs the whole signal, so it suits offline analysis.
    CAUSAL
        Run the filter forward once. Each output depends only on present and past
        samples, as in real-time processing, at the cost of a frequency-dependent
        delay.
    """

    ZERO = auto()
    CAUSAL = auto()
