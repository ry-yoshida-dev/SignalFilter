"""Behavior shared by IIR and FIR filters: phases, frequency responses, and validation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import ArrayLike

from signal_filter import (
    Band,
    BesselFilter,
    ButterworthFilter,
    EllipticFilter,
    FloatArray,
    FrequencySelectiveFilter,
    KaiserFIRFilter,
    NotchFilter,
    Phase,
    WindowFIRFilter,
)

type Sinusoid = Callable[[float], FloatArray]
type Design = Callable[[Phase], FrequencySelectiveFilter]

SAMPLING_RATE: float = 200.0

DESIGNS: dict[str, Design] = {
    "butterworth": lambda phase: ButterworthFilter(
        band=Band.lowpass(20.0), sampling_rate=SAMPLING_RATE, phase=phase
    ),
    "elliptic": lambda phase: EllipticFilter(
        band=Band.bandpass(10.0, 30.0), sampling_rate=SAMPLING_RATE, phase=phase
    ),
    "bessel": lambda phase: BesselFilter(
        band=Band.highpass(20.0), sampling_rate=SAMPLING_RATE, phase=phase
    ),
    "notch": lambda phase: NotchFilter(
        frequency=25.0, quality_factor=5.0, sampling_rate=SAMPLING_RATE, phase=phase
    ),
    "window": lambda phase: WindowFIRFilter(
        band=Band.bandstop(15.0, 35.0), taps=51, sampling_rate=SAMPLING_RATE, phase=phase
    ),
    "kaiser": lambda phase: KaiserFIRFilter(
        band=Band.lowpass(20.0), transition_width=10.0, sampling_rate=SAMPLING_RATE, phase=phase
    ),
}


def _measured_response(
    instance: FrequencySelectiveFilter, frequency: float, n_samples: int = 4000
) -> complex:
    """Complex gain on a sinusoid, measured by projection away from the ends."""
    time = np.arange(n_samples) / SAMPLING_RATE
    carrier = np.exp(2j * np.pi * frequency * time)
    filtered = instance.apply(np.cos(2.0 * np.pi * frequency * time))
    middle = slice(n_samples // 4, 3 * n_samples // 4)
    # A whole number of periods is not guaranteed, so project over the middle half.
    return complex(2.0 * np.mean(filtered[middle] * np.conj(carrier[middle])))


@pytest.mark.parametrize("design", DESIGNS.values(), ids=DESIGNS.keys())
@pytest.mark.parametrize("phase", Phase)
@pytest.mark.parametrize("frequency", [3.0, 12.5, 20.0, 25.0, 41.0, 77.0])
def test_response_matches_filtered_sinusoids(
    design: Design, phase: Phase, frequency: float
) -> None:
    instance = design(phase)
    expected = complex(instance.frequency_response([frequency]).response[0])
    assert _measured_response(instance, frequency) == pytest.approx(expected, abs=2e-3)


@pytest.mark.parametrize("design", DESIGNS.values(), ids=DESIGNS.keys())
def test_zero_phase_response_is_squared_magnitude(design: Design) -> None:
    causal = design(Phase.CAUSAL).frequency_response()
    zero = design(Phase.ZERO).frequency_response()
    np.testing.assert_allclose(zero.response.imag, 0.0)
    np.testing.assert_allclose(zero.magnitude, causal.magnitude**2, atol=1e-12)


@pytest.mark.parametrize("design", DESIGNS.values(), ids=DESIGNS.keys())
def test_causal_output_ignores_future_samples(design: Design) -> None:
    instance = design(Phase.CAUSAL)
    signal = np.random.default_rng(0).standard_normal(400)
    changed = signal.copy()
    changed[200:] += 10.0
    np.testing.assert_array_equal(instance.apply(signal)[:200], instance.apply(changed)[:200])


def test_zero_phase_does_not_delay(sinusoid: Sinusoid) -> None:
    signal = sinusoid(2.0)
    filtered = DESIGNS["butterworth"](Phase.ZERO).apply(signal)
    # The padding at the ends is an estimate, so only the interior is exact.
    np.testing.assert_allclose(filtered[50:-50], signal[50:-50], atol=1e-4)
    np.testing.assert_allclose(filtered, signal, atol=1e-2)


def test_causal_run_starts_in_steady_state() -> None:
    offset = np.full((300, 2), [4.0, -1.5])
    lowpass = ButterworthFilter(
        band=Band.lowpass(5.0), sampling_rate=SAMPLING_RATE, phase=Phase.CAUSAL
    )
    highpass = WindowFIRFilter(
        band=Band.highpass(5.0), sampling_rate=SAMPLING_RATE, phase=Phase.CAUSAL
    )
    np.testing.assert_allclose(lowpass.apply(offset), offset)
    # A windowed design leaves a small gain at 0 Hz, the sum of its coefficients.
    direct_current_gain = highpass.coefficients.sum()
    np.testing.assert_allclose(highpass.apply(offset), offset * direct_current_gain)
    assert abs(direct_current_gain) < 1e-2


def test_default_response_spans_zero_to_nyquist() -> None:
    response = DESIGNS["butterworth"](Phase.CAUSAL).frequency_response()
    assert response.frequencies.shape == (512,)
    assert response.frequencies[0] == 0.0
    assert response.frequencies[-1] == SAMPLING_RATE / 2


def test_response_at_given_frequencies() -> None:
    response = DESIGNS["butterworth"](Phase.CAUSAL).frequency_response([0, 20.0, 100.0])
    np.testing.assert_allclose(response.frequencies, [0.0, 20.0, 100.0])
    np.testing.assert_allclose(response.gain_db[:2], [0.0, -3.0103], atol=1e-3)


@pytest.mark.parametrize(
    ("frequencies", "error", "message"),
    [
        ([[1.0, 2.0]], ValueError, "1-D"),
        ([1.0, np.nan], ValueError, "finite"),
        ([-1.0], ValueError, "within"),
        ([100.5], ValueError, "within"),
        (np.array([1j]), TypeError, "real"),
    ],
)
def test_invalid_response_frequencies_are_rejected(
    frequencies: ArrayLike, error: type[Exception], message: str
) -> None:
    instance = DESIGNS["butterworth"](Phase.CAUSAL)
    with pytest.raises(error, match=message):
        instance.frequency_response(frequencies)


@pytest.mark.parametrize("sampling_rate", [0.0, -1.0, np.inf, np.nan])
def test_invalid_sampling_rate_is_rejected(sampling_rate: float) -> None:
    with pytest.raises(ValueError, match="sampling_rate"):
        ButterworthFilter(band=Band.lowpass(1.0), sampling_rate=sampling_rate)


@pytest.mark.parametrize("band", [Band.lowpass(100.0), Band.bandpass(10.0, 150.0)])
def test_band_beyond_nyquist_is_rejected(band: Band) -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        ButterworthFilter(band=band, sampling_rate=SAMPLING_RATE)
    with pytest.raises(ValueError, match="Nyquist"):
        WindowFIRFilter(band=band, sampling_rate=SAMPLING_RATE)


def test_minimum_length_depends_on_phase() -> None:
    assert DESIGNS["butterworth"](Phase.CAUSAL).minimum_length == 1
    assert DESIGNS["butterworth"](Phase.ZERO).minimum_length == 16
    assert DESIGNS["window"](Phase.ZERO).minimum_length == 3 * 51 + 1
