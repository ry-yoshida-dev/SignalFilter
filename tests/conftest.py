"""Shared signals and filters for the test suite."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from signal_filter import (
    Band,
    BesselFilter,
    ButterworthFilter,
    ChebyshevType1Filter,
    ChebyshevType2Filter,
    EllipticFilter,
    FloatArray,
    FrequencySelectiveFilter,
    HampelFilter,
    KaiserFIRFilter,
    MedianFilter,
    MovingAverageFilter,
    NotchFilter,
    Phase,
    SavitzkyGolayFilter,
    SignalFilter,
    WindowFIRFilter,
)

SAMPLING_RATE: float = 200.0


type Sinusoid = Callable[[float], FloatArray]


def _sinusoid(frequency: float) -> FloatArray:
    time = np.arange(2000) / SAMPLING_RATE
    return np.sin(2.0 * np.pi * frequency * time)


@pytest.fixture
def sampling_rate() -> float:
    """Sampling rate of the test signals in hertz."""
    return SAMPLING_RATE


@pytest.fixture
def sinusoid() -> Sinusoid:
    """Unit sine wave ``sinusoid(frequency)`` of 2000 samples at ``SAMPLING_RATE``."""
    return _sinusoid


def _identify(instance: SignalFilter) -> str:
    if isinstance(instance, FrequencySelectiveFilter):
        return f"{type(instance).__name__}-{instance.phase.name}"
    return type(instance).__name__


def _every_filter() -> list[SignalFilter]:
    """One instance of every concrete filter, in both phases where they apply."""
    lowpass = Band.lowpass(20.0)
    frequency_selective: list[SignalFilter] = []
    for phase in Phase:
        frequency_selective += [
            ButterworthFilter(band=lowpass, sampling_rate=SAMPLING_RATE, phase=phase),
            ChebyshevType1Filter(band=lowpass, sampling_rate=SAMPLING_RATE, phase=phase),
            ChebyshevType2Filter(band=lowpass, sampling_rate=SAMPLING_RATE, phase=phase),
            EllipticFilter(band=lowpass, sampling_rate=SAMPLING_RATE, phase=phase),
            BesselFilter(band=lowpass, sampling_rate=SAMPLING_RATE, phase=phase),
            NotchFilter(frequency=50.0, sampling_rate=SAMPLING_RATE, phase=phase),
            WindowFIRFilter(band=lowpass, sampling_rate=SAMPLING_RATE, taps=31, phase=phase),
            KaiserFIRFilter(
                band=lowpass, transition_width=10.0, sampling_rate=SAMPLING_RATE, phase=phase
            ),
        ]
    return [
        *frequency_selective,
        MovingAverageFilter(window_length=5),
        MedianFilter(window_length=5),
        SavitzkyGolayFilter(window_length=7, polynomial_order=3),
        HampelFilter(window_length=7),
    ]


@pytest.fixture(params=_every_filter(), ids=_identify)
def any_filter(request: pytest.FixtureRequest) -> SignalFilter:
    """Each concrete filter in turn."""
    instance: SignalFilter = request.param
    return instance
