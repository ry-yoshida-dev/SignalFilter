"""Tests for frequency bands."""

from __future__ import annotations

import pytest

from signal_filter import Band, BandType


def test_constructors_set_type_and_edges() -> None:
    assert Band.lowpass(5.0) == Band(BandType.LOWPASS, (5.0,))
    assert Band.highpass(5.0) == Band(BandType.HIGHPASS, (5.0,))
    assert Band.bandpass(1.0, 5.0) == Band(BandType.BANDPASS, (1.0, 5.0))
    assert Band.bandstop(1.0, 5.0) == Band(BandType.BANDSTOP, (1.0, 5.0))


@pytest.mark.parametrize(
    ("band_type", "edges", "message"),
    [
        (BandType.LOWPASS, (1.0, 2.0), "1 edge"),
        (BandType.BANDPASS, (1.0,), "2 edge"),
        (BandType.HIGHPASS, (0.0,), "positive"),
        (BandType.HIGHPASS, (float("nan"),), "finite"),
        (BandType.BANDSTOP, (1.0, float("inf")), "finite"),
        (BandType.BANDPASS, (5.0, 1.0), "increasing"),
        (BandType.BANDPASS, (5.0, 5.0), "increasing"),
    ],
)
def test_invalid_edges_are_rejected(
    band_type: BandType, edges: tuple[float, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Band(band_type, edges)
