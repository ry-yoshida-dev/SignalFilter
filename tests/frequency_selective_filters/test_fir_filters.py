"""Tests for the FIR filter designs."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from signal_filter import (
    Band,
    BandType,
    FloatArray,
    KaiserFIRFilter,
    Phase,
    WindowFIRFilter,
    WindowFunction,
)

type Sinusoid = Callable[[float], FloatArray]

SAMPLING_RATE: float = 200.0
CAUSAL: Phase = Phase.CAUSAL


@pytest.mark.parametrize(
    "band",
    [Band.lowpass(20.0), Band.highpass(20.0), Band.bandpass(20.0, 40.0), Band.bandstop(20.0, 40.0)],
)
def test_window_design_is_linear_phase_and_half_gain_at_edges(band: Band) -> None:
    fir = WindowFIRFilter(band=band, taps=101, sampling_rate=SAMPLING_RATE, phase=CAUSAL)
    coefficients = fir.coefficients
    assert coefficients.shape == (101,)
    np.testing.assert_allclose(coefficients, coefficients[::-1])
    np.testing.assert_allclose(fir.frequency_response(list(band.edges)).magnitude, 0.5, atol=0.01)


def test_coefficients_are_a_copy() -> None:
    fir = WindowFIRFilter(band=Band.lowpass(20.0), sampling_rate=SAMPLING_RATE)
    fir.coefficients[:] = 0.0
    assert fir.coefficients.any()


def test_causal_output_is_the_input_delayed_by_group_delay(sinusoid: Sinusoid) -> None:
    fir = WindowFIRFilter(
        band=Band.lowpass(20.0), taps=41, sampling_rate=SAMPLING_RATE, phase=CAUSAL
    )
    delay = round(fir.group_delay * SAMPLING_RATE)
    assert delay == 20
    signal = sinusoid(3.0)
    gain = fir.frequency_response([3.0]).magnitude[0]
    # After one filter length the start has left the memory of the filter, so the shift is exact.
    taps = fir.coefficients.size
    np.testing.assert_allclose(
        fir.apply(signal)[taps:], gain * signal[taps - delay : -delay], atol=1e-12
    )


@pytest.mark.parametrize(
    ("window_function", "attenuation"),
    [
        (WindowFunction.RECTANGULAR, 21.0),
        (WindowFunction.HANN, 44.0),
        (WindowFunction.HAMMING, 53.0),
        (WindowFunction.BLACKMAN, 74.0),
        (WindowFunction.BLACKMAN_HARRIS, 92.0),
    ],
)
def test_windows_reach_their_nominal_attenuation(
    window_function: WindowFunction, attenuation: float
) -> None:
    fir = WindowFIRFilter(
        band=Band.lowpass(20.0),
        taps=101,
        window_function=window_function,
        sampling_rate=SAMPLING_RATE,
        phase=CAUSAL,
    )
    # 20 Hz past the edge is beyond the widest transition at 101 taps.
    stopband = fir.frequency_response(np.linspace(40.0, 100.0, 2000)).gain_db
    assert stopband.max() < -attenuation


@pytest.mark.parametrize(
    "band",
    [Band.lowpass(30.0), Band.highpass(60.0), Band.bandpass(20.0, 50.0), Band.bandstop(20.0, 50.0)],
)
@pytest.mark.parametrize("attenuation", [40.0, 80.0])
def test_kaiser_design_meets_its_specification(band: Band, attenuation: float) -> None:
    width = 8.0
    fir = KaiserFIRFilter(
        band=band,
        transition_width=width,
        attenuation=attenuation,
        sampling_rate=SAMPLING_RATE,
        phase=CAUSAL,
    )
    assert fir.taps % 2 == 1
    assert fir.coefficients.shape == (fir.taps,)
    frequencies = np.linspace(0.0, SAMPLING_RATE / 2, 4001)
    distance = np.min(np.abs(frequencies[:, np.newaxis] - np.array(band.edges)), axis=1)
    is_away = distance >= width / 2
    response = fir.frequency_response(frequencies[is_away])
    edges = band.edges
    match band.band_type:
        case BandType.LOWPASS:
            is_pass = frequencies < edges[0]
        case BandType.HIGHPASS:
            is_pass = frequencies > edges[0]
        case BandType.BANDPASS:
            is_pass = (frequencies > edges[0]) & (frequencies < edges[1])
        case BandType.BANDSTOP:
            is_pass = (frequencies < edges[0]) | (frequencies > edges[1])
    # The Kaiser method is an estimate, and two transitions can add their ripples.
    shortfall_db = 2.0 if len(band.edges) == 1 else 6.1
    tolerance = 10 ** ((shortfall_db - attenuation) / 20)
    stopband = response.magnitude[~is_pass[is_away]]
    passband = response.magnitude[is_pass[is_away]]
    assert stopband.max() <= tolerance
    assert np.abs(passband - 1.0).max() <= tolerance


def test_kaiser_length_grows_with_sharper_specifications() -> None:
    def taps(width: float, attenuation: float) -> int:
        return KaiserFIRFilter(
            band=Band.lowpass(30.0),
            transition_width=width,
            attenuation=attenuation,
            sampling_rate=SAMPLING_RATE,
        ).taps

    assert taps(4.0, 60.0) > taps(8.0, 60.0)
    assert taps(8.0, 80.0) > taps(8.0, 60.0)


@pytest.mark.parametrize("taps", [0, 1, 2, 100])
def test_window_taps_must_be_odd_and_at_least_three(taps: int) -> None:
    with pytest.raises(ValueError, match="taps"):
        WindowFIRFilter(band=Band.lowpass(20.0), taps=taps, sampling_rate=SAMPLING_RATE)


@pytest.mark.parametrize(
    ("band", "width"),
    [
        (Band.lowpass(3.0), 8.0),
        (Band.lowpass(97.0), 8.0),
        (Band.bandpass(20.0, 25.0), 8.0),
    ],
)
def test_kaiser_transition_must_fit(band: Band, width: float) -> None:
    with pytest.raises(ValueError, match="does not fit"):
        KaiserFIRFilter(band=band, transition_width=width, sampling_rate=SAMPLING_RATE)


@pytest.mark.parametrize(
    ("width", "attenuation", "message"),
    [
        (0.0, 60.0, "transition_width"),
        (np.nan, 60.0, "transition_width"),
        (8.0, 0.0, "attenuation"),
    ],
)
def test_kaiser_specification_must_be_positive(
    width: float, attenuation: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        KaiserFIRFilter(
            band=Band.lowpass(30.0),
            transition_width=width,
            attenuation=attenuation,
            sampling_rate=SAMPLING_RATE,
        )


def test_non_integer_taps_are_rejected() -> None:
    with pytest.raises(TypeError, match="taps must be an integer"):
        WindowFIRFilter(band=Band.lowpass(30.0), taps=cast(int, 101.0), sampling_rate=SAMPLING_RATE)
