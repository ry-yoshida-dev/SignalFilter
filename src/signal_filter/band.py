"""Frequency band passed or rejected by a filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .band_type import BandType


@dataclass(frozen=True)
class Band:
    """
    Band of frequencies, in hertz, that a filter passes or rejects.

    Prefer the constructors :meth:`lowpass`, :meth:`highpass`, :meth:`bandpass`,
    and :meth:`bandstop`. What an edge means exactly, such as the -3 dB point,
    depends on the filter design.

    Attributes
    ----------
    band_type : BandType
        Which side of the edges is passed.
    edges : tuple[float, ...]
        The edge frequencies in hertz, increasing: one for lowpass and highpass,
        two for bandpass and bandstop.

    Raises
    ------
    ValueError
        If the number of edges does not match ``band_type``, or the edges are not
        positive, finite, and strictly increasing.
    """

    band_type: BandType
    edges: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.edges) != self.band_type.edge_count:
            raise ValueError(
                f"{self.band_type.name} needs {self.band_type.edge_count} edge(s), "
                + f"got {len(self.edges)}"
            )
        edges = np.asarray(self.edges, dtype=np.float64)
        if not (np.isfinite(edges).all() and (edges > 0.0).all()):
            raise ValueError(f"band edges must be positive and finite, got {self.edges}")
        if (np.diff(edges) <= 0.0).any():
            raise ValueError(f"band edges must be strictly increasing, got {self.edges}")

    @classmethod
    def lowpass(cls, cutoff: float) -> Band:
        """
        Pass frequencies below ``cutoff``.

        Parameters
        ----------
        cutoff : float
            The cutoff frequency in hertz.

        Returns
        -------
        Band
            The lowpass band.
        """
        return cls(BandType.LOWPASS, (cutoff,))

    @classmethod
    def highpass(cls, cutoff: float) -> Band:
        """
        Pass frequencies above ``cutoff``.

        Parameters
        ----------
        cutoff : float
            The cutoff frequency in hertz.

        Returns
        -------
        Band
            The highpass band.
        """
        return cls(BandType.HIGHPASS, (cutoff,))

    @classmethod
    def bandpass(cls, low: float, high: float) -> Band:
        """
        Pass frequencies between ``low`` and ``high``.

        Parameters
        ----------
        low : float
            The lower edge in hertz.
        high : float
            The upper edge in hertz.

        Returns
        -------
        Band
            The bandpass band.
        """
        return cls(BandType.BANDPASS, (low, high))

    @classmethod
    def bandstop(cls, low: float, high: float) -> Band:
        """
        Reject frequencies between ``low`` and ``high``.

        Parameters
        ----------
        low : float
            The lower edge in hertz.
        high : float
            The upper edge in hertz.

        Returns
        -------
        Band
            The bandstop band.
        """
        return cls(BandType.BANDSTOP, (low, high))
