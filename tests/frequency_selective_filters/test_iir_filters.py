"""Tests for the IIR filter designs."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from signal_filter import (
    Band,
    BesselFilter,
    BesselNormalization,
    ButterworthFilter,
    ChebyshevType1Filter,
    ChebyshevType2Filter,
    EllipticFilter,
    FloatArray,
    IIRFilter,
    NotchFilter,
    Phase,
)

type Sinusoid = Callable[[float], FloatArray]

SAMPLING_RATE: float = 200.0
CAUSAL: Phase = Phase.CAUSAL


def _gain_db(filter_: IIRFilter, *frequencies: float) -> FloatArray:
    return filter_.frequency_response(list(frequencies)).gain_db


@pytest.mark.parametrize(
    ("band", "edge_frequencies", "pass_frequency", "stop_frequency"),
    [
        (Band.lowpass(20.0), [20.0], 2.0, 60.0),
        (Band.highpass(20.0), [20.0], 60.0, 2.0),
        (Band.bandpass(20.0, 40.0), [20.0, 40.0], 28.3, 80.0),
        (Band.bandstop(20.0, 40.0), [20.0, 40.0], 80.0, 28.3),
    ],
)
def test_butterworth_is_minus_three_db_at_every_edge(
    band: Band, edge_frequencies: list[float], pass_frequency: float, stop_frequency: float
) -> None:
    butterworth = ButterworthFilter(band=band, sampling_rate=SAMPLING_RATE, phase=CAUSAL)
    np.testing.assert_allclose(_gain_db(butterworth, *edge_frequencies), -3.0103, atol=1e-3)
    assert _gain_db(butterworth, pass_frequency)[0] > -0.1
    assert _gain_db(butterworth, stop_frequency)[0] < -30.0


def test_butterworth_is_monotonic() -> None:
    butterworth = ButterworthFilter(band=Band.lowpass(20.0), sampling_rate=SAMPLING_RATE)
    assert (np.diff(butterworth.frequency_response().magnitude) <= 1e-12).all()


@pytest.mark.parametrize(("band", "sections"), [(Band.lowpass(20.0), 3), (Band.bandpass(5, 9), 5)])
def test_second_order_sections(band: Band, sections: int) -> None:
    order = 5
    butterworth = ButterworthFilter(band=band, order=order, sampling_rate=SAMPLING_RATE)
    coefficients = butterworth.second_order_sections
    assert coefficients.shape == (sections, 6)
    coefficients[:] = 0.0
    assert butterworth.second_order_sections.any()


def test_chebyshev_type1_ripples_within_passband_ripple() -> None:
    chebyshev = ChebyshevType1Filter(
        band=Band.lowpass(20.0),
        order=5,
        passband_ripple=0.5,
        sampling_rate=SAMPLING_RATE,
        phase=CAUSAL,
    )
    passband = _gain_db(chebyshev, *np.linspace(0.0, 20.0, 200))
    assert passband.max() <= 1e-9
    assert passband.min() == pytest.approx(-0.5, abs=1e-6)
    assert _gain_db(chebyshev, 20.0)[0] == pytest.approx(-0.5, abs=1e-6)


def test_chebyshev_type2_reaches_attenuation_at_its_edge() -> None:
    chebyshev = ChebyshevType2Filter(
        band=Band.lowpass(30.0),
        order=5,
        stopband_attenuation=50.0,
        sampling_rate=SAMPLING_RATE,
        phase=CAUSAL,
    )
    assert _gain_db(chebyshev, 30.0)[0] == pytest.approx(-50.0, abs=1e-6)
    assert _gain_db(chebyshev, *np.linspace(30.0, 100.0, 200)).max() <= -50.0 + 1e-6
    assert _gain_db(chebyshev, 0.0)[0] == pytest.approx(0.0, abs=1e-9)


def test_elliptic_meets_ripple_and_attenuation() -> None:
    elliptic = EllipticFilter(
        band=Band.highpass(40.0),
        order=6,
        passband_ripple=0.2,
        stopband_attenuation=60.0,
        sampling_rate=SAMPLING_RATE,
        phase=CAUSAL,
    )
    passband = _gain_db(elliptic, *np.linspace(40.0, 100.0, 200))
    assert passband.min() == pytest.approx(-0.2, abs=1e-6)
    assert passband.max() <= 1e-9
    # Once the gain first falls to the attenuation below the edge, it never rises above it.
    frequencies = np.linspace(0.0, 40.0, 4001)
    stopband = _gain_db(elliptic, *frequencies)
    stopband_edge = frequencies[stopband <= -60.0 + 1e-6].max()
    assert 25.0 < stopband_edge < 40.0
    assert stopband[frequencies <= stopband_edge].max() <= -60.0 + 1e-6


def test_elliptic_is_steeper_than_butterworth_of_the_same_order() -> None:
    elliptic = EllipticFilter(band=Band.lowpass(20.0), order=4, sampling_rate=SAMPLING_RATE)
    butterworth = ButterworthFilter(band=Band.lowpass(20.0), order=4, sampling_rate=SAMPLING_RATE)
    assert _gain_db(elliptic, 30.0)[0] < _gain_db(butterworth, 30.0)[0] - 20.0


def _overshoot(filter_: ButterworthFilter | BesselFilter) -> float:
    step = np.concatenate([np.zeros(100), np.ones(400)])
    return float(filter_.apply(step).max() - 1.0)


def test_bessel_step_barely_overshoots_unlike_butterworth() -> None:
    bessel = BesselFilter(band=Band.lowpass(10.0), sampling_rate=SAMPLING_RATE, phase=CAUSAL)
    butterworth = ButterworthFilter(
        band=Band.lowpass(10.0), sampling_rate=SAMPLING_RATE, phase=CAUSAL
    )
    assert _overshoot(bessel) < 0.02
    assert _overshoot(butterworth) > 0.05


def test_bessel_normalization_sets_the_cutoff_meaning() -> None:
    gains = [
        _gain_db(
            BesselFilter(
                band=Band.lowpass(10.0),
                normalization=normalization,
                sampling_rate=SAMPLING_RATE,
                phase=CAUSAL,
            ),
            10.0,
        )[0]
        for normalization in BesselNormalization
    ]
    assert gains[0] == pytest.approx(-3.0103, abs=1e-3)
    assert len({round(gain, 3) for gain in gains}) == len(gains)


def test_notch_removes_only_its_frequency(sinusoid: Sinusoid) -> None:
    notch = NotchFilter(frequency=50.0, quality_factor=10.0, sampling_rate=SAMPLING_RATE)
    signal = sinusoid(5.0) + sinusoid(50.0)
    np.testing.assert_allclose(notch.apply(signal)[200:-200], sinusoid(5.0)[200:-200], atol=1e-3)


def test_notch_bandwidth_follows_quality_factor() -> None:
    notch = NotchFilter(
        frequency=50.0, quality_factor=10.0, sampling_rate=SAMPLING_RATE, phase=CAUSAL
    )
    assert notch.frequency_response([50.0]).magnitude[0] == pytest.approx(0.0, abs=1e-12)
    edges = np.linspace(40.0, 60.0, 20001)
    is_below = notch.frequency_response(edges).gain_db < -3.0103
    assert edges[is_below].max() - edges[is_below].min() == pytest.approx(5.0, abs=2e-3)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: ButterworthFilter(band=Band.lowpass(1.0), order=0, sampling_rate=10.0), "order"),
        (
            lambda: ChebyshevType1Filter(
                band=Band.lowpass(1.0), passband_ripple=0.0, sampling_rate=10.0
            ),
            "passband_ripple",
        ),
        (
            lambda: ChebyshevType2Filter(
                band=Band.lowpass(1.0), stopband_attenuation=np.inf, sampling_rate=10.0
            ),
            "stopband_attenuation",
        ),
        (
            lambda: EllipticFilter(
                band=Band.lowpass(1.0),
                passband_ripple=3.0,
                stopband_attenuation=3.0,
                sampling_rate=10.0,
            ),
            "exceed",
        ),
        (lambda: BesselFilter(band=Band.lowpass(5.0), sampling_rate=10.0), "Nyquist"),
        (lambda: NotchFilter(frequency=0.0, sampling_rate=10.0), "frequency"),
        (lambda: NotchFilter(frequency=1.0, quality_factor=-1.0, sampling_rate=10.0), "quality"),
    ],
)
def test_invalid_designs_are_rejected(build: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build()


@pytest.mark.parametrize("order", [4.0, True, "4"])
def test_non_integer_orders_are_rejected(order: object) -> None:
    with pytest.raises(TypeError, match="order must be an integer"):
        ButterworthFilter(band=Band.lowpass(1.0), order=cast(int, order), sampling_rate=10.0)
